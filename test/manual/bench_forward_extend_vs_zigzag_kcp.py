#!/usr/bin/env python3
"""Unit microbench: KDAAttnBackend.forward_extend vs _forward_extend_zigzag_kcp.

Single-GPU, no serving. CP=1 with identity all-gather so KCP still runs the
full prepare→summary→AG→merge→scan path while baseline runs packed
causal_conv1d + FlashKDA/Triton extend on the same local tokens.

Usage (in a CUDA container that has sglang + flash_kda):
  /opt/sglang/bin/python3 bench_forward_extend_vs_zigzag_kcp.py \\
      --lens 128,1024,2048,4096 --heads 64 --warmup 5 --iters 20
"""

from __future__ import annotations

import argparse
import statistics
from types import SimpleNamespace
from typing import List, Sequence

import torch

from sglang.srt.layers.attention.linear.kda_backend import (
    KDAAttnBackend,
    KDAKernelDispatcher,
)
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.layers.cp.base import init_cp_strategy
from sglang.srt.layers.cp.zigzag import ZigzagContextParallelMetadata
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class _IdentityGroup:
    def all_gather_into_tensor(self, output, input_tensor):
        assert output.shape[0] == input_tensor.shape[0], (
            output.shape,
            input_tensor.shape,
        )
        output.copy_(input_tensor)

    @property
    def device_group(self):
        return None


class _Parallel:
    def __init__(self, cp_size: int = 1, cp_rank: int = 0):
        self.attn_cp_size = cp_size
        self.attn_cp_rank = cp_rank
        self.attn_cp_group = _IdentityGroup()


def _install_parallel(cp_size: int = 1, cp_rank: int = 0):
    import sglang.srt.layers.cp.kda_kcp as kcp
    import sglang.srt.runtime_context as rc

    parallel = _Parallel(cp_size=cp_size, cp_rank=cp_rank)
    orig_rc = rc.get_parallel
    orig_kcp = kcp.get_parallel

    def fake():
        return parallel

    rc.get_parallel = fake  # type: ignore[assignment]
    kcp.get_parallel = fake  # type: ignore[assignment]
    return orig_rc, orig_kcp


def _restore_parallel(orig_rc, orig_kcp):
    import sglang.srt.layers.cp.kda_kcp as kcp
    import sglang.srt.runtime_context as rc

    rc.get_parallel = orig_rc  # type: ignore[assignment]
    kcp.get_parallel = orig_kcp  # type: ignore[assignment]


def _pick_prefill_backend() -> LinearAttnKernelBackend:
    try:
        import flash_kda  # noqa: F401

        from sglang.srt.layers.cp.kda_kcp import flash_kda_kcp_available

        print(f"flash_kda={flash_kda.__file__} kcp_ops={flash_kda_kcp_available()}")
        return LinearAttnKernelBackend.FLASHKDA
    except Exception as e:  # pragma: no cover
        print(f"flash_kda unavailable ({e}); using TRITON extend")
        return LinearAttnKernelBackend.TRITON


def _make_layer(heads: int, head_dim: int, device: torch.device):
    dim = 3 * heads * head_dim  # q=k=v heads (no GVA)
    return SimpleNamespace(
        layer_id=0,
        num_q_heads=heads,
        num_k_heads=heads,
        num_v_heads=heads,
        head_q_dim=head_dim,
        head_k_dim=head_dim,
        head_v_dim=head_dim,
        q_dim=heads * head_dim,
        k_dim=heads * head_dim,
        v_dim=heads * head_dim,
        conv_weights=torch.randn(dim, 4, device=device, dtype=torch.bfloat16) * 0.02,
        bias=torch.randn(dim, device=device, dtype=torch.bfloat16) * 0.02,
        A_log=torch.randn(heads, device=device, dtype=torch.float32) * 0.1,
        dt_bias=torch.randn(heads * head_dim, device=device, dtype=torch.float32) * 0.1,
        lower_bound=-5.0,
    )


def _make_backend(
    layer,
    slots: torch.Tensor,
    T: int,
    *,
    prefill_backend: LinearAttnKernelBackend,
):
    num_slots = int(slots.max().item()) + 2
    heads = layer.num_v_heads
    head_dim = layer.head_v_dim
    dim = layer.q_dim + layer.k_dim + layer.v_dim
    state = MambaPool.State(
        conv=[
            torch.zeros(1, num_slots, 3, dim, device=slots.device, dtype=torch.bfloat16)
        ],
        temporal=torch.zeros(
            1,
            num_slots,
            heads,
            head_dim,
            head_dim,
            device=slots.device,
            dtype=torch.float32,
        ),
    )
    backend = KDAAttnBackend.__new__(KDAAttnBackend)
    backend.req_to_token_pool = SimpleNamespace(
        mamba2_layer_cache=state.at_layer_idx,
    )
    backend.forward_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, T], device=slots.device, dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, T], dtype=torch.int32),
        mamba_cache_indices=slots,
        logical_num_tokens=T,
        has_mamba_track_mask=False,
        track_chunk_idx=None,
        track_ssm_h_src=None,
        track_ssm_h_batch_src=None,
        conv_states_mask_indices=None,
        track_conv_indices=None,
    )
    backend.accept_lens_pool = None
    backend.kernel_dispatcher = KDAKernelDispatcher(
        LinearAttnKernelBackend.TRITON,
        prefill_backend,
        LinearAttnKernelBackend.TRITON,
    )
    return backend, state


def _zigzag_meta_cp1(T: int, device: torch.device) -> ZigzagContextParallelMetadata:
    # CP=1 → two chrono slots; local early|late packing.
    early = T - T // 2
    late = T // 2
    return ZigzagContextParallelMetadata(
        per_rank_actual_token=[T],
        per_rank_logical_token=[T],
        actual_seq_q_prev_list=[early],
        actual_seq_q_next_list=[late],
        actual_seq_q_prev_tensor=torch.tensor(
            [early], device=device, dtype=torch.int32
        ),
        actual_seq_q_next_tensor=torch.tensor([late], device=device, dtype=torch.int32),
        total_q_prev_tokens=early,
        total_q_next_tokens=late,
    )


def _make_batch(T: int, device: torch.device, *, zigzag: bool):
    mode = (
        ForwardMode.EXTEND
        if not zigzag
        else ForwardMode.CONTEXT_PARALLEL_EXTEND
        if hasattr(ForwardMode, "CONTEXT_PARALLEL_EXTEND")
        else ForwardMode.EXTEND
    )

    # Some trees use is_context_parallel_extend() on the mode enum member.
    class _Mode:
        def __init__(self, base, cp: bool):
            self._base = base
            self._cp = cp

        def is_target_verify(self):
            return False

        def is_draft_extend_v2(self):
            return False

        def is_context_parallel_extend(self):
            return self._cp

        def __getattr__(self, name):
            return getattr(self._base, name)

    batch = SimpleNamespace(
        forward_mode=_Mode(mode, zigzag),
        extend_prefix_lens=torch.zeros(1, device=device, dtype=torch.int32),
        extend_seq_lens=torch.tensor([T], device=device, dtype=torch.int32),
        extend_seq_lens_cpu=[T],
        attn_cp_metadata=_zigzag_meta_cp1(T, device) if zigzag else None,
        sglang_kda_kcp_local=False,
    )
    return batch


def _cuda_time_ms(fn, *, warmup: int, iters: int) -> List[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _fmt(times: Sequence[float]) -> str:
    return (
        f"med={statistics.median(times):.3f}ms "
        f"mean={statistics.mean(times):.3f}ms "
        f"min={min(times):.3f}ms "
        f"max={max(times):.3f}ms"
    )


def bench_one(
    T: int,
    *,
    heads: int,
    head_dim: int,
    warmup: int,
    iters: int,
    prefill_backend: LinearAttnKernelBackend,
):
    import sglang.srt.layers.cp.kda_kcp as kcp_mod

    device = torch.device("cuda")
    torch.manual_seed(0)
    layer = _make_layer(heads, head_dim, device)
    slots = torch.tensor([1], device=device, dtype=torch.int32)
    dim = layer.q_dim + layer.k_dim + layer.v_dim

    mixed = torch.randn(T, dim, device=device, dtype=torch.bfloat16) * 0.02
    a = torch.randn(1, T, heads * head_dim, device=device, dtype=torch.bfloat16) * 0.02
    b = torch.rand(1, T, heads, device=device, dtype=torch.bfloat16)

    # Container envs may lag the branch; gate via direct monkeypatch.
    kcp_mod.glm_zigzag_kcp_enabled = lambda: False  # type: ignore[assignment]

    # --- baseline forward_extend (KCP off) ---
    init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="zigzag")
    backend_base, state_base = _make_backend(
        layer, slots, T, prefill_backend=prefill_backend
    )
    batch_base = _make_batch(T, device, zigzag=False)

    def run_base():
        # Zero recurrent cache between iters; reuse activation tensors.
        state_base.temporal.zero_()
        state_base.conv[0].zero_()
        return backend_base.forward_extend(layer, batch_base, mixed, a, b)

    # --- zigzag KCP path (call the method directly) ---
    init_cp_strategy(enable_prefill_cp=True, cp_size=1, cp_strategy="zigzag")
    backend_kcp, state_kcp = _make_backend(
        layer, slots, T, prefill_backend=prefill_backend
    )
    batch_kcp = _make_batch(T, device, zigzag=True)

    def run_kcp():
        state_kcp.temporal.zero_()
        state_kcp.conv[0].zero_()
        return backend_kcp._forward_extend_zigzag_kcp(layer, batch_kcp, mixed, a, b)

    def _as_out(x):
        # FlashKDAKernel.extend returns (tensor, None); Triton returns tensor.
        return x[0] if isinstance(x, tuple) else x

    # Smoke once for shapes / correctness-ish
    out_b = _as_out(run_base())
    out_k = _as_out(run_kcp())
    print(
        f"T={T} shapes base={tuple(out_b.shape)} kcp={tuple(out_k.shape)} "
        f"prefill={prefill_backend.value}"
    )

    t_base = _cuda_time_ms(run_base, warmup=warmup, iters=iters)
    t_kcp = _cuda_time_ms(run_kcp, warmup=warmup, iters=iters)
    med_b = statistics.median(t_base)
    med_k = statistics.median(t_kcp)
    print(f"  forward_extend              {_fmt(t_base)}")
    print(f"  _forward_extend_zigzag_kcp  {_fmt(t_kcp)}")
    print(f"  ratio kcp/base={med_k / med_b:.2f}x  delta={med_k - med_b:+.3f}ms")
    return {
        "T": T,
        "base_med_ms": med_b,
        "kcp_med_ms": med_k,
        "ratio": med_k / med_b,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", type=str, default="128,1024,2048,4096")
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    lens = [int(x) for x in args.lens.split(",") if x.strip()]
    prefill = _pick_prefill_backend()
    orig = _install_parallel(cp_size=1, cp_rank=0)
    rows = []
    try:
        for T in lens:
            rows.append(
                bench_one(
                    T,
                    heads=args.heads,
                    head_dim=args.head_dim,
                    warmup=args.warmup,
                    iters=args.iters,
                    prefill_backend=prefill,
                )
            )
    finally:
        _restore_parallel(*orig)
        init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="zigzag")

    print("\n=== summary (median ms) ===")
    print(f"{'T':>8} {'base':>10} {'kcp':>10} {'kcp/base':>10}")
    for r in rows:
        print(
            f"{r['T']:8d} {r['base_med_ms']:10.3f} {r['kcp_med_ms']:10.3f} "
            f"{r['ratio']:10.2f}x"
        )


if __name__ == "__main__":
    main()
