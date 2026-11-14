# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""KDA Kimi Context Parallelism (KCP) for zigzag Prefill-CP.

Five-step prefill path (single summary AG, no dual-part overlap):

  prepare (local early/late) → summary (S_ext, M) → AG → merge → scan

``SGLANG_ENABLE_GLM_ZIGZAG_KCP`` gates the path. Reference affine summary uses
``fused_recurrent_kda`` (FlashKDA prepare/summary kernels can replace it later).
Short-conv halo follows vLLM ``#59304``: gather segment tails → assemble
per-slot windows → ordinary causal conv.

SGLang zigzag local token layout is **all early then all late** across the
batch (not interleaved per request). ``HybridPCPLayout`` / ``KcpPlan`` mirror
vLLM's plan shapes while respecting that packing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    _TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def glm_zigzag_kcp_enabled() -> bool:
    return bool(envs.SGLANG_ENABLE_GLM_ZIGZAG_KCP.get())


def use_zigzag_kcp(forward_batch) -> bool:
    """True when zigzag Prefill-CP extend should take the KDA KCP path."""
    if not glm_zigzag_kcp_enabled():
        return False
    from sglang.srt.layers.cp.zigzag import is_zigzag_extend

    return is_zigzag_extend(forward_batch)


def skip_attn_cp_activation_exchange(forward_batch) -> bool:
    """Layer-boundary AG/RS of activations is skipped for KDA under KCP."""
    return bool(getattr(forward_batch, "sglang_kda_kcp_local", False))


# ---------------------------------------------------------------------------
# HybridPCPLayout / KcpPlan (vLLM #59304-shaped, SGLang zigzag packing)
# ---------------------------------------------------------------------------


@dataclass
class HybridPCPLayout:
    """Where this rank's zigzag prefill tokens sit (SGLang local packing).

    Local tokens are packed as **all early (part 0) across N requests, then
    all late (part 1)** — matching ``ZigzagCPStrategy.build_metadata``.
    Chronological slot ``s`` for request ``n`` has length ``slot_lens[n, s]``;
    rank ``r`` owns slots ``r`` (early) and ``2P-1-r`` (late).
    """

    world: int
    rank: int
    num_prefill_tokens: int
    prefill_lens: List[int]
    # [N, 2P] chronological tokens per (request, slot)
    slot_lens: torch.Tensor
    # Local nonempty segments in token order (early-all then late-all)
    segment_request: List[int]
    segment_slot: List[int]
    segment_start: List[int]  # first token of segment within its request
    segment_local_start: List[int]  # first token in local packed tensor
    scan_cu: List[int]  # host cu; len = num_segments + 1
    scan_cu_seqlens: torch.Tensor  # device int32
    prefill_has_initial_state: torch.Tensor  # [N] bool
    shared: dict = field(default_factory=dict, repr=False)

    @property
    def num_prefill_reqs(self) -> int:
        return len(self.prefill_lens)

    @property
    def num_segments(self) -> int:
        return len(self.segment_request)

    @property
    def num_slots(self) -> int:
        return int(self.world) * 2


@dataclass
class KcpPlan:
    """Per-step KCP indices derived once from ``HybridPCPLayout``.

    Summary AG uses physical ``[P, 2, N, ...]`` (part 0 = slot ``rank``,
    part 1 = slot ``2P-1-rank``). Merge walks chronological slots per request.

    Convolution halo (vLLM ``#59304``): each rank packs ``[N*2*halo]`` raw tail
    rows via ``tail_src_idx``, all-gathers to ``[P*N*2*halo, C]``, then
    ``halo_idx`` / ``final_halo_idx`` select windows from the prefix pool
    (first ``N*halo`` rows) plus gathered tails.
    """

    layout: HybridPCPLayout
    halo_size: int
    # Physical summary row for each local segment: (rank*2 + part)*N + n
    summary_idx: torch.Tensor  # [L] int64
    # [N, 2P] whether any rank has tokens; local segment id or -1
    slot_present: torch.Tensor  # [N, 2P] bool
    segment_of_slot: torch.Tensor  # [N, 2P] int32
    # Local segment bookkeeping
    segment_ids: torch.Tensor  # [L] int32 (1..L for conv scratch rows)
    segment_has_initial_state: torch.Tensor  # [L] bool
    # Conv halo plan (vLLM gather_conv_tails / conv_windows)
    tail_src_idx: torch.Tensor  # [N*2*halo] int64; -1 if absent
    halo_idx: torch.Tensor  # [L, halo] int64 pool rows for initial windows
    final_halo_idx: torch.Tensor  # [N, halo] int64 pool rows for final windows
    # Reusable exchange buffers keyed by shape (summary AG, conv tails)
    buffers: dict = field(default_factory=dict, repr=False)


def build_hybrid_pcp_layout(
    forward_batch,
    *,
    device: Optional[torch.device] = None,
) -> HybridPCPLayout:
    """Build layout from zigzag Prefill-CP metadata on this rank."""
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    extend_lens = [int(x) for x in forward_batch.extend_seq_lens_cpu]
    n_req = len(extend_lens)
    num_slots = cp_size * 2

    slot_lens_host: List[List[int]] = [
        zigzag_slot_lengths_for_request(length, cp_size) for length in extend_lens
    ]
    early_lens, late_lens = local_zigzag_segment_lengths(forward_batch)
    assert len(early_lens) == n_req and len(late_lens) == n_req

    if device is None:
        device = (
            forward_batch.extend_seq_lens.device
            if getattr(forward_batch, "extend_seq_lens", None) is not None
            else torch.device("cpu")
        )

    segment_request: List[int] = []
    segment_slot: List[int] = []
    segment_start: List[int] = []
    segment_local_start: List[int] = []
    scan_cu = [0]
    # early-all then late-all
    local_cursor = 0
    for n, length in enumerate(early_lens):
        length = int(length)
        if length <= 0:
            continue
        slot = cp_rank
        start = sum(slot_lens_host[n][:slot])
        segment_request.append(n)
        segment_slot.append(slot)
        segment_start.append(start)
        segment_local_start.append(local_cursor)
        local_cursor += length
        scan_cu.append(local_cursor)
    for n, length in enumerate(late_lens):
        length = int(length)
        if length <= 0:
            continue
        slot = num_slots - 1 - cp_rank
        start = sum(slot_lens_host[n][:slot])
        segment_request.append(n)
        segment_slot.append(slot)
        segment_start.append(start)
        segment_local_start.append(local_cursor)
        local_cursor += length
        scan_cu.append(local_cursor)

    slot_lens = torch.tensor(slot_lens_host, dtype=torch.int32, device=device)
    prefix = getattr(forward_batch, "extend_prefix_lens", None)
    if prefix is not None:
        has_init = (prefix > 0).to(dtype=torch.bool, device=device)
        if has_init.numel() != n_req:
            has_init = torch.zeros(n_req, dtype=torch.bool, device=device)
    else:
        has_init = torch.zeros(n_req, dtype=torch.bool, device=device)

    return HybridPCPLayout(
        world=cp_size,
        rank=cp_rank,
        num_prefill_tokens=local_cursor,
        prefill_lens=extend_lens,
        slot_lens=slot_lens,
        segment_request=segment_request,
        segment_slot=segment_slot,
        segment_start=segment_start,
        segment_local_start=segment_local_start,
        scan_cu=scan_cu,
        scan_cu_seqlens=torch.tensor(scan_cu, dtype=torch.int32, device=device),
        prefill_has_initial_state=has_init,
    )


def _slot_part(slot: int, world: int) -> int:
    return 0 if slot < world else 1


def _slot_owner(slot: int, world: int) -> int:
    return slot if slot < world else (2 * world - 1 - slot)


def _conv_window_pool_rows(
    n: int,
    end: int,
    *,
    slot_lens_host: Sequence[Sequence[int]],
    halo: int,
    n_req: int,
    world: int,
) -> List[int]:
    """Pool rows for request ``n`` covering chronological ``[end-halo, end)``.

    Pool layout matches vLLM ``#59304``: first ``N*halo`` rows are per-request
    prefix cache columns; remaining rows are rank-major gathered tails
    ``((owner*N + n)*2 + part)*halo + column`` (right-aligned in each slot).
    """
    num_slots = 2 * world
    rows: List[int] = []
    for position in range(end - halo, end):
        if position < 0:
            rows.append(n * halo + position + halo)
            continue
        cum = 0
        found = False
        for slot in range(num_slots):
            length = int(slot_lens_host[n][slot])
            if position < cum + length:
                offset = position - cum
                column = halo - length + offset
                if column < 0:
                    raise RuntimeError(
                        f"conv window column {column} < 0 at n={n} pos={position}"
                    )
                owner = _slot_owner(slot, world)
                part = _slot_part(slot, world)
                tail = ((owner * n_req + n) * 2 + part) * halo
                rows.append(n_req * halo + tail + column)
                found = True
                break
            cum += length
        if not found:
            raise RuntimeError(
                f"conv window position {position} past request {n} length {cum}"
            )
    return rows


def build_kcp_plan(layout: HybridPCPLayout, halo_size: int) -> KcpPlan:
    """Derive per-step KCP indices; cache on ``layout.shared['kcp']``."""
    cached = layout.shared.get("kcp")
    if cached is not None:
        if cached.halo_size != halo_size:
            raise ValueError(
                f"KcpPlan halo mismatch: cached {cached.halo_size} vs {halo_size}"
            )
        return cached

    world, rank = layout.world, layout.rank
    n_req = layout.num_prefill_reqs
    num_slots = layout.num_slots
    halo = int(halo_size)
    device = layout.scan_cu_seqlens.device
    slot_lens_host = [
        [int(x) for x in layout.slot_lens[n].tolist()] for n in range(n_req)
    ]

    summary_idx: List[int] = []
    segment_has_init: List[bool] = []
    # Local [N, 2, halo] pack destinations → source token in local mixed_qkv.
    tail_src = [-1] * (n_req * 2 * halo)
    halo_idx_rows: List[List[int]] = []
    for seg_i, (n, slot, local_start) in enumerate(
        zip(
            layout.segment_request,
            layout.segment_slot,
            layout.segment_local_start,
        )
    ):
        part = _slot_part(int(slot), world)
        summary_idx.append((rank * 2 + part) * n_req + int(n))
        # Windows always come from the pool (prefix zeros or prior tails).
        segment_has_init.append(True)
        length = int(layout.scan_cu[seg_i + 1] - layout.scan_cu[seg_i])
        take = min(halo, length)
        dest = (int(n) * 2 + part) * halo + halo - take
        for i in range(take):
            tail_src[dest + i] = int(local_start) + length - take + i
        halo_idx_rows.append(
            _conv_window_pool_rows(
                int(n),
                int(layout.segment_start[seg_i]),
                slot_lens_host=slot_lens_host,
                halo=halo,
                n_req=n_req,
                world=world,
            )
        )

    final_halo_idx_rows = [
        _conv_window_pool_rows(
            n,
            int(layout.prefill_lens[n]),
            slot_lens_host=slot_lens_host,
            halo=halo,
            n_req=n_req,
            world=world,
        )
        for n in range(n_req)
    ]

    segment_of_slot = torch.full(
        (n_req, num_slots), -1, dtype=torch.int32, device=device
    )
    for seg_i, (n, slot) in enumerate(zip(layout.segment_request, layout.segment_slot)):
        segment_of_slot[n, slot] = seg_i

    slot_present = layout.slot_lens > 0
    num_segments = layout.num_segments
    if num_segments == 0:
        halo_idx = torch.empty(0, halo, dtype=torch.int64, device=device)
    else:
        halo_idx = torch.tensor(halo_idx_rows, dtype=torch.int64, device=device)
    if n_req == 0:
        final_halo_idx = torch.empty(0, halo, dtype=torch.int64, device=device)
    else:
        final_halo_idx = torch.tensor(
            final_halo_idx_rows, dtype=torch.int64, device=device
        )
    plan = KcpPlan(
        layout=layout,
        halo_size=halo,
        summary_idx=torch.tensor(summary_idx, dtype=torch.int64, device=device),
        slot_present=slot_present,
        segment_of_slot=segment_of_slot,
        segment_ids=torch.arange(1, num_segments + 1, dtype=torch.int32, device=device),
        segment_has_initial_state=torch.tensor(
            segment_has_init, dtype=torch.bool, device=device
        ),
        tail_src_idx=torch.tensor(tail_src, dtype=torch.int64, device=device),
        halo_idx=halo_idx,
        final_halo_idx=final_halo_idx,
    )
    layout.shared["kcp"] = plan
    return plan


def plan_for_forward_batch(
    forward_batch, halo_size: int, *, device: Optional[torch.device] = None
) -> KcpPlan:
    """Build (or reuse) layout+plan for this zigzag extend step."""
    shared = getattr(forward_batch, "sglang_kcp_shared", None)
    if shared is None:
        shared = {}
        forward_batch.sglang_kcp_shared = shared
    layout = shared.get("layout")
    if layout is None:
        layout = build_hybrid_pcp_layout(forward_batch, device=device)
        shared["layout"] = layout
    return build_kcp_plan(layout, halo_size)


# ---------------------------------------------------------------------------
# Affine merge (Python + Triton)
# ---------------------------------------------------------------------------


def merge_affine_segment_states(
    s_ext: torch.Tensor,
    m: torch.Tensor,
    s_base: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fold ``2C`` (or N) segment summaries into per-segment initial states.

    Args:
        s_ext: ``[num_slots, ...]`` external state written from a zero input.
        m: ``[num_slots, ...]`` transition; ``S_out = S_ext + M @ S_in``.
        s_base: optional recurrent cache state before slot 0; zeros if None.
    """
    if s_ext.shape[0] != m.shape[0]:
        raise ValueError(f"s_ext/m slot mismatch: {s_ext.shape[0]} vs {m.shape[0]}")
    num_slots = s_ext.shape[0]
    if s_base is None:
        s_in0 = torch.zeros_like(s_ext[0])
    else:
        s_in0 = s_base.to(dtype=s_ext.dtype, device=s_ext.device)

    s_in = s_ext.new_empty(s_ext.shape)
    cur = s_in0
    for i in range(num_slots):
        s_in[i] = cur
        cur = s_ext[i] + _apply_transition(m[i], cur)
    return s_in


def merge_affine_segment_states_batched(
    s_ext: torch.Tensor,
    m: torch.Tensor,
    slot_present: torch.Tensor,
    s_base: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-request chronological fold.

    Args:
        s_ext: ``[S, N, H, dk, dv]`` chronological (or use with remap first)
        m: ``[S, N, H, dk, dk]``
        slot_present: ``[N, S]`` bool
        s_base: optional ``[N, H, dk, dv]``; zeros if None

    Returns:
        ``s_in`` ``[S, N, H, dk, dv]`` initial state before each slot,
        ``s_final`` ``[N, H, dk, dv]`` state after the last present slot.
    """
    num_slots, n_req = s_ext.shape[0], s_ext.shape[1]
    if s_base is None:
        cur = torch.zeros_like(s_ext[0])
    else:
        cur = s_base.to(dtype=s_ext.dtype, device=s_ext.device)
    s_in = s_ext.new_empty(s_ext.shape)
    for slot in range(num_slots):
        s_in[slot] = cur
        present = slot_present[:, slot]
        if not bool(present.any()):
            continue
        # Apply only for present requests; absent keep cur unchanged.
        nxt = s_ext[slot] + torch.matmul(m[slot], cur)
        cur = torch.where(present.view(n_req, *([1] * (cur.ndim - 1))), nxt, cur)
    return s_in, cur


def _apply_transition(m: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Apply ``M @ S`` for state layouts ``[..., dk, dv]`` and ``M`` ``[..., dk, dk]``."""
    if m.ndim == state.ndim and m.shape[:-1] == state.shape[:-1]:
        return torch.matmul(m, state)
    if (
        m.ndim == 2
        and state.ndim == 3
        and m.shape[0] == state.shape[1]
        and m.shape[1] == state.shape[1]
    ):
        return torch.einsum("ij,hjk->hik", m, state)
    raise ValueError(
        f"Unsupported M/S shapes: m={tuple(m.shape)} s={tuple(state.shape)}"
    )


if _TRITON_AVAILABLE:

    @triton.jit
    def _kcp_merge_left_kernel(
        s_ext_ptr,
        m_ptr,
        slot_present_ptr,
        segment_of_slot_ptr,
        initial_ptr,
        final_ptr,
        N,
        S: tl.constexpr,
        H: tl.constexpr,
        DK: tl.constexpr,
        DV: tl.constexpr,
        BDV: tl.constexpr,
    ):
        """Chronological left-multiply merge: S ← S_ext + M @ S.

        ``s_ext`` [S,N,H,DK,DV], ``m`` [S,N,H,DK,DK], writes local-segment
        initials into ``initial`` [L,H,DK,DV] when ``segment_of_slot >= 0``,
        and final states into ``final`` [N,H,DK,DV].
        """
        nh = tl.program_id(0).to(tl.int64)
        n = nh // H
        h = nh % H
        dv = tl.program_id(1) * BDV + tl.arange(0, BDV)[:, None]
        dk = tl.arange(0, DK)[None, :]
        dkk = tl.arange(0, DK)[:, None]
        mask_v = dv < DV

        # state [BDV, DK] stored as rows=dv, cols=dk — we keep [DK, DV] logically
        # by loading/storing with (dk, dv) indexing below.
        # Use [DK, BDV] tiles: rows dk, cols dv.
        state = tl.zeros((DK, BDV), dtype=tl.float32)
        for c in range(0, S):
            present = tl.load(slot_present_ptr + n * S + c)
            segment = tl.load(segment_of_slot_ptr + n * S + c).to(tl.int64)
            if segment >= 0:
                # Store initial [H,DK,DV] for this local segment.
                tl.store(
                    initial_ptr
                    + ((segment * H + h) * DK + dkk) * DV
                    + dv.reshape(1, BDV),
                    state,
                    mask=mask_v.reshape(1, BDV),
                )
            if present != 0:
                # S_ext[c,n,h]: [DK, DV]
                s_block = s_ext_ptr + (((c * N + n) * H + h) * DK) * DV
                s_ext = tl.load(
                    s_block + dkk * DV + dv.reshape(1, BDV),
                    mask=mask_v.reshape(1, BDV),
                    other=0.0,
                ).to(tl.float32)
                # M[c,n,h]: [DK, DK]
                m_block = m_ptr + (((c * N + n) * H + h) * DK) * DK
                m_tile = tl.load(m_block + dkk * DK + dk).to(tl.float32)
                # state' = S_ext + M @ state   (DK,BDV)
                state = s_ext + tl.dot(m_tile, state)
        tl.store(
            final_ptr + ((n * H + h) * DK + dkk) * DV + dv.reshape(1, BDV),
            state,
            mask=mask_v.reshape(1, BDV),
        )


def kcp_merge_left(
    s_ext: torch.Tensor,
    m: torch.Tensor,
    slot_present: torch.Tensor,
    segment_of_slot: torch.Tensor,
    initial: torch.Tensor,
    final: torch.Tensor,
) -> None:
    """Merge chronological summaries into local-segment initials + finals.

    Prefers Triton on CUDA; falls back to the Python batched fold.
    ``s_ext``/``m`` are chronological ``[S, N, H, ...]``.
    """
    num_slots, n_req, heads, d_k, d_v = s_ext.shape
    use_triton = (
        _TRITON_AVAILABLE
        and s_ext.is_cuda
        and d_k <= 128
        and d_v <= 128
        and d_v % 16 == 0
    )
    if use_triton:
        bdv = 64 if d_v >= 64 else max(d_v, 16)
        # Pad DV to block if needed by viewing into padded buffers — require exact.
        if d_v % bdv != 0:
            bdv = d_v
        try:
            _kcp_merge_left_kernel[(n_req * heads, (d_v + bdv - 1) // bdv)](
                s_ext.contiguous(),
                m.contiguous(),
                slot_present.contiguous(),
                segment_of_slot.contiguous(),
                initial,
                final,
                n_req,
                S=num_slots,
                H=heads,
                DK=d_k,
                DV=d_v,
                BDV=bdv,
                num_warps=4,
                num_stages=2,
            )
            return
        except Exception:
            # Fall through to Python reference on kernel issues.
            pass

    s_in_all, s_final = merge_affine_segment_states_batched(
        s_ext, m, slot_present, s_base=None
    )
    # Scatter initials for local segments.
    for n in range(n_req):
        for slot in range(num_slots):
            seg = int(segment_of_slot[n, slot].item())
            if seg >= 0:
                initial[seg].copy_(s_in_all[slot, n])
        final[n].copy_(s_final[n])


# ---------------------------------------------------------------------------
# Summary AG scratch (reusable)
# ---------------------------------------------------------------------------

_AG_SCRATCH: Dict[Tuple, torch.Tensor] = {}


def get_ag_scratch(
    key: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Reuse a flat AG scratch tensor across layers/steps when shape matches."""
    cache_key = (key, shape, dtype, str(device))
    buf = _AG_SCRATCH.get(cache_key)
    if buf is None or buf.shape != shape or buf.dtype != dtype or buf.device != device:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _AG_SCRATCH[cache_key] = buf
    return buf


def clear_ag_scratch() -> None:
    _AG_SCRATCH.clear()


def zigzag_slot_order(cp_size: int) -> List[int]:
    """Chronological block ids ``0 .. 2C-1`` (merge walks this order)."""
    return list(range(cp_size * 2))


def pack_summary_buffer_elems(
    cp_size: int,
    num_requests: int,
    num_heads: int,
    d_k: int,
    d_v: int,
) -> int:
    """Element count of the BF16 ``[S; M]`` all-gather buffer (vLLM formula)."""
    return 2 * cp_size * num_requests * num_heads * (d_v + d_k) * d_k


def summary_ag_bytes(
    cp_size: int,
    num_requests: int,
    num_heads: int,
    d_k: int,
    d_v: int,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    return (
        pack_summary_buffer_elems(cp_size, num_requests, num_heads, d_k, d_v)
        * dtype.itemsize
    )


# ---------------------------------------------------------------------------
# Zigzag segment lengths / conv halo
# ---------------------------------------------------------------------------


def local_zigzag_segment_lengths(forward_batch) -> Tuple[List[int], List[int]]:
    """Per-request (early, late) token counts on this zigzag CP rank."""
    meta = forward_batch.attn_cp_metadata
    early = list(meta.actual_seq_q_prev_list or [])
    late = list(meta.actual_seq_q_next_list or [])
    if not early or not late:
        n = (
            int(forward_batch.extend_seq_lens_cpu[0])
            if forward_batch.extend_seq_lens_cpu
            else 0
        )
        cp_size = get_parallel().attn_cp_size
        cp_rank = get_parallel().attn_cp_rank
        from sglang.srt.layers.cp.zigzag import zigzag_rows_per_request

        if meta.actual_seq_q_prev_tensor is not None:
            early = [int(x) for x in meta.actual_seq_q_prev_tensor.tolist()]
            late = [int(x) for x in meta.actual_seq_q_next_tensor.tolist()]
        else:
            total = zigzag_rows_per_request(
                forward_batch.extend_seq_lens_cpu, cp_rank, cp_size
            )
            early = [t - t // 2 for t in total]
            late = [t // 2 for t in total]
    return early, late


def conv_halo_size(conv_weights: torch.Tensor) -> int:
    """``kernel_size - 1`` rows the short conv needs before each segment."""
    return int(conv_weights.shape[-1]) - 1


def pack_local_conv_tails(
    mixed_qkv: torch.Tensor,
    early_len: int,
    late_len: int,
    halo: int,
) -> torch.Tensor:
    """Pack this rank's early/late raw tails as ``[2, C, halo]`` (bs=1 helper)."""
    batch = pack_local_conv_tails_batch(mixed_qkv, [early_len], [late_len], halo)
    return batch[0]


def pack_local_conv_tails_batch(
    mixed_qkv: torch.Tensor,
    early_lens: Sequence[int],
    late_lens: Sequence[int],
    halo: int,
) -> torch.Tensor:
    """Pack local tails as ``[N, 2, C, halo]`` (right-aligned; SGLang packing).

    Local ``mixed_qkv`` is all-early then all-late. Short/empty parts leave
    leading columns zero.
    """
    if halo <= 0:
        raise ValueError("halo must be positive")
    n_req = len(early_lens)
    if len(late_lens) != n_req:
        raise ValueError("early/late lens length mismatch")
    channels = mixed_qkv.shape[-1]
    tails = mixed_qkv.new_zeros(n_req, 2, channels, halo)
    early_total = sum(int(x) for x in early_lens)
    # early region
    cursor = 0
    for n, length in enumerate(early_lens):
        length = int(length)
        take = min(halo, length)
        if take > 0:
            piece = mixed_qkv[cursor + length - take : cursor + length].transpose(0, 1)
            tails[n, 0, :, halo - take : halo] = piece
        cursor += length
    # late region
    cursor = early_total
    for n, length in enumerate(late_lens):
        length = int(length)
        take = min(halo, length)
        if take > 0:
            piece = mixed_qkv[cursor + length - take : cursor + length].transpose(0, 1)
            tails[n, 1, :, halo - take : halo] = piece
        cursor += length
    return tails.contiguous()


def all_gather_conv_tails(local_tails: torch.Tensor) -> torch.Tensor:
    """AG ``[2, C, halo]`` (bs=1) → chrono ``[2C, C, halo]``."""
    if local_tails.ndim == 3:
        gathered = all_gather_conv_tails_batch(local_tails.unsqueeze(0))
        return gathered[0]
    return all_gather_conv_tails_batch(local_tails)


def all_gather_conv_tails_batch(local_tails: torch.Tensor) -> torch.Tensor:
    """AG ``[N, 2, C, halo]`` per rank → chrono ``[N, 2C, C, halo]``.

    Rank-major gather is ``[early_r, late_r]``; remap to zigzag slot order
    ``slot r = early of rank r``, ``slot 2C-1-r = late of rank r``.
    """
    if local_tails.ndim != 4 or local_tails.shape[1] != 2:
        raise ValueError(
            f"local_tails must be [N, 2, C, halo], got {tuple(local_tails.shape)}"
        )
    n_req, _, channels, halo = local_tails.shape
    cp_size = get_parallel().attn_cp_size
    group = get_parallel().attn_cp_group
    # Pack [N*2, C*halo] for AG across ranks → [P, N*2, C*halo]
    flat = local_tails.reshape(n_req * 2, channels * halo).contiguous()
    scratch = get_ag_scratch(
        "conv_tails",
        (cp_size * n_req * 2, channels * halo),
        flat.dtype,
        flat.device,
    )
    if hasattr(group, "all_gather_into_tensor"):
        group.all_gather_into_tensor(scratch, flat)
        gathered = scratch
    else:
        parts = [torch.empty_like(flat) for _ in range(cp_size)]
        torch.distributed.all_gather(parts, flat, group=group.device_group)
        gathered = torch.cat(parts, dim=0)
    rm = gathered.view(cp_size, n_req, 2, channels, halo)
    slot_tails = local_tails.new_empty(n_req, 2 * cp_size, channels, halo)
    for r in range(cp_size):
        slot_tails[:, r] = rm[r, :, 0]
        slot_tails[:, 2 * cp_size - 1 - r] = rm[r, :, 1]
    return slot_tails


# ---------------------------------------------------------------------------
# Triton conv halo (vLLM #59304 gather_conv_tails / conv_windows)
# ---------------------------------------------------------------------------


if _TRITON_AVAILABLE:

    @triton.jit
    def _pack_conv_tail_rows(
        x,
        source_indices,
        out,
        stride_token: tl.int64,
        stride_channel: tl.constexpr,
        C: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel_tiles: tl.constexpr = tl.cdiv(C, BLOCK)
        token = (tl.program_id(0) // channel_tiles).to(tl.int64)
        channel = (tl.program_id(0) % channel_tiles).to(tl.int64) * BLOCK
        channel += tl.arange(0, BLOCK)
        source = tl.load(source_indices + token).to(tl.int64)
        value = tl.load(
            x + tl.maximum(source, 0) * stride_token + channel * stride_channel,
            (source >= 0) & (channel < C),
            other=0.0,
        )
        tl.store(out + token * C + channel, value, channel < C)

    @triton.jit
    def _conv_windows_kernel(
        conv_state,
        state_indices,
        has_initial_state,
        tails,
        pool_rows,
        out,
        stride_state_slot,
        stride_state_channel,
        stride_state_column,
        stride_out_row,
        stride_out_channel,
        stride_out_column,
        num_prefix_rows,
        C: tl.constexpr,
        HALO: tl.constexpr,
        OUT_BY_SLOT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        channel = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = channel < C
        values = ()  # type: tuple
        for column in tl.static_range(HALO):
            source = tl.load(pool_rows + row * HALO + column).to(tl.int64)
            from_prefix = source < num_prefix_rows
            request = tl.where(from_prefix, source // HALO, 0)
            slot = tl.load(state_indices + request).to(tl.int64)
            continued = tl.load(has_initial_state + request).to(tl.int1)
            prefix = tl.load(
                conv_state
                + slot * stride_state_slot
                + channel * stride_state_channel
                + (source % HALO) * stride_state_column,
                mask=mask & from_prefix & continued,
                other=0.0,
            )
            tail = tl.load(
                tails + tl.maximum(source - num_prefix_rows, 0) * C + channel,
                mask=mask & (source >= num_prefix_rows),
                other=0.0,
            )
            values += (tl.where(from_prefix, prefix, tail.to(prefix.dtype)),)
        destination = (
            tl.load(state_indices + row).to(tl.int64) if OUT_BY_SLOT else row + 1
        )
        for column in tl.static_range(HALO):
            tl.store(
                out
                + destination * stride_out_row
                + channel * stride_out_channel
                + column * stride_out_column,
                values[column],
                mask=mask,
            )


def gather_conv_tails(plan: KcpPlan, qkv: torch.Tensor) -> torch.Tensor:
    """Pack local tails with Triton and AG to ``[P*N*2*halo, C]`` (vLLM layout)."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("gather_conv_tails requires triton")
    rows = int(plan.tail_src_idx.numel())
    channels = int(qkv.shape[-1])
    cp_size = get_parallel().attn_cp_size
    group = get_parallel().attn_cp_group
    local = qkv.new_empty(rows, channels)
    block = 256
    _pack_conv_tail_rows[(rows * triton.cdiv(channels, block),)](
        qkv,
        plan.tail_src_idx,
        local,
        qkv.stride(0),
        qkv.stride(1),
        channels,
        BLOCK=block,
    )
    scratch = get_ag_scratch(
        "conv_tails_flat",
        (cp_size * rows, channels),
        local.dtype,
        local.device,
    )
    flat = local.contiguous()
    if hasattr(group, "all_gather_into_tensor"):
        group.all_gather_into_tensor(scratch, flat)
        return scratch
    parts = [torch.empty_like(flat) for _ in range(cp_size)]
    torch.distributed.all_gather(parts, flat, group=group.device_group)
    return torch.cat(parts, dim=0)


def conv_windows(
    plan: KcpPlan,
    tails: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    final: bool = False,
) -> torch.Tensor:
    """Assemble halo windows from prefix cache + gathered ``tails``.

    Default: segment initial windows as ``[1+L, C, halo]`` (row 0 unused).
    ``final=True`` writes each request's final window into ``conv_state[slots]``.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("conv_windows requires triton")
    channels = int(conv_state.shape[1])
    halo = int(plan.halo_size)
    if conv_state.shape[-1] != halo:
        raise ValueError(f"conv_state halo {conv_state.shape[-1]} != plan halo {halo}")
    if tails.shape[-1] != channels or not tails.is_contiguous():
        raise ValueError("tails must be contiguous [..., C]")
    n_req = plan.layout.num_prefill_reqs
    if final:
        rows, out = plan.final_halo_idx, conv_state
    else:
        rows = plan.halo_idx
        shape = (plan.layout.num_segments + 1, channels, halo)
        if conv_state.stride(-1) == 1:
            out = conv_state.new_empty(shape)
        else:
            out = conv_state.new_empty(shape[0], shape[2], shape[1]).transpose(1, 2)
    if rows.shape[0] == 0:
        return out
    block = 1024
    _conv_windows_kernel[(rows.shape[0], triton.cdiv(channels, block))](
        conv_state,
        state_indices,
        plan.layout.prefill_has_initial_state,
        tails,
        rows,
        out,
        *conv_state.stride(),
        *out.stride(),
        n_req * halo,
        C=channels,
        HALO=halo,
        OUT_BY_SLOT=final,
        BLOCK=block,
    )
    return out


def run_local_kcp_conv(
    plan: KcpPlan,
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    conv_states: torch.Tensor,
    cache_indices: torch.Tensor,
) -> torch.Tensor:
    """One batched short-conv over local zigzag segments (vLLM ``_forward_kcp``).

    Packs and all-gathers tails, builds initial windows, runs ``causal_conv1d_fn`` once
    over ``scan_cu_seqlens``, then publishes request-final windows into
    ``conv_states[cache_indices]``.
    """
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

    layout = plan.layout
    slots = cache_indices.reshape(-1)[: layout.num_prefill_reqs]
    tails = gather_conv_tails(plan, mixed_qkv)
    windows = conv_windows(plan, tails, conv_states, slots, final=False)
    if layout.num_segments == 0:
        conv_windows(plan, tails, conv_states, slots, final=True)
        return mixed_qkv.new_empty(0, mixed_qkv.shape[-1])
    seq_lens_cpu = [
        int(layout.scan_cu[i + 1] - layout.scan_cu[i])
        for i in range(layout.num_segments)
    ]
    qkv = causal_conv1d_fn(
        mixed_qkv.transpose(0, 1),
        conv_weights,
        conv_bias,
        activation="silu",
        conv_states=windows,
        has_initial_state=plan.segment_has_initial_state,
        cache_indices=plan.segment_ids,
        query_start_loc=layout.scan_cu_seqlens,
        seq_lens_cpu=seq_lens_cpu,
    ).transpose(0, 1)
    conv_windows(plan, tails, conv_states, slots, final=True)
    return qkv


def initial_conv_window_for_slot(
    slot_tails: torch.Tensor,
    slot_id: int,
) -> torch.Tensor:
    """Initial ``[C, halo]`` window for chrono ``slot_id`` (zeros for slot 0).

    ``slot_tails`` may be ``[2C, C, halo]`` (bs=1) or ``[N, 2C, C, halo]``
    (caller indexes request first).
    """
    if slot_id <= 0:
        return torch.zeros_like(
            slot_tails[0] if slot_tails.ndim == 3 else slot_tails[0, 0]
        )
    if slot_tails.ndim == 3:
        return slot_tails[slot_id - 1]
    raise ValueError("use initial_conv_window_for_request_slot for batched tails")


def initial_conv_window_for_request_slot(
    slot_tails: torch.Tensor,
    req_id: int,
    slot_id: int,
) -> torch.Tensor:
    """Initial ``[C, halo]`` for request ``req_id``, chrono ``slot_id``."""
    if slot_id <= 0:
        return torch.zeros(
            slot_tails.shape[-2],
            slot_tails.shape[-1],
            device=slot_tails.device,
            dtype=slot_tails.dtype,
        )
    return slot_tails[req_id, slot_id - 1]


def final_conv_window_from_slots(
    slot_tails: torch.Tensor,
    slot_lengths: Sequence[int],
) -> torch.Tensor:
    """Request-final ``[C, halo]`` = last nonempty slot's packed tail (bs=1)."""
    last = -1
    for slot_id, length in enumerate(slot_lengths):
        if int(length) > 0:
            last = slot_id
    if last < 0:
        return torch.zeros_like(slot_tails[0])
    return slot_tails[last]


def final_conv_windows_batch(
    slot_tails: torch.Tensor,
    slot_lens: torch.Tensor,
) -> torch.Tensor:
    """``[N, C, halo]`` final windows from chrono ``[N, 2C, C, halo]`` tails."""
    n_req, num_slots, channels, halo = slot_tails.shape
    out = slot_tails.new_zeros(n_req, channels, halo)
    for n in range(n_req):
        last = -1
        for slot_id in range(num_slots):
            if int(slot_lens[n, slot_id].item()) > 0:
                last = slot_id
        if last >= 0:
            out[n] = slot_tails[n, last]
    return out


def zigzag_slot_lengths_for_request(extend_len: int, cp_size: int) -> List[int]:
    """Chronological per-slot token counts for one request (length ``2C``)."""
    num_slots = cp_size * 2
    base = int(extend_len) // num_slots
    rem = int(extend_len) % num_slots
    return [base + 1 if s < rem else base for s in range(num_slots)]


# ---------------------------------------------------------------------------
# Reference summary / scan / AG
# ---------------------------------------------------------------------------


def _flash_state_to_merge(state: torch.Tensor) -> torch.Tensor:
    """FlashKDA ``[H, V, K]`` → merge layout ``[H, dk=K, dv=V]``."""
    return state.transpose(-1, -2).contiguous()


def _merge_state_to_flash(state: torch.Tensor) -> torch.Tensor:
    """Merge layout ``[H, dk, dv]`` → FlashKDA ``[H, V, K]``."""
    return state.transpose(-1, -2).contiguous()


def reference_affine_summary(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference ``(S_ext, M)`` via ``fused_recurrent_kda`` (V==K identity trick).

    Args:
        q,k,v: ``[1, T, H, D]``; g: ``[1, T, H, K]``; beta: ``[1, T, H]``.
    Returns:
        ``s_ext`` ``[H, dk, dv]``, ``m`` ``[H, dk, dk]`` (fp32).
    """
    from sglang.kernels.ops.attention.fla.kda import fused_recurrent_kda

    assert q.shape[0] == 1, "reference summary expects packed B=1 varlen slice"
    _B, _T, H, Kdim = k.shape
    Vdim = v.shape[-1]
    if Vdim != Kdim:
        raise NotImplementedError(
            f"reference KCP summary identity trick needs V==K, got V={Vdim} K={Kdim}"
        )
    device = q.device
    zero = torch.zeros(1, H, Vdim, Kdim, device=device, dtype=torch.float32)
    _, s_ext = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=zero,
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=None,
    )
    eye = (
        torch.eye(Kdim, device=device, dtype=torch.float32)
        .expand(1, H, Kdim, Kdim)
        .contiguous()
    )
    v0 = torch.zeros_like(v)
    _, m_flash = fused_recurrent_kda(
        q=q,
        k=k,
        v=v0,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=eye,
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=None,
    )
    s_ext_m = _flash_state_to_merge(s_ext[0].float())
    m = m_flash[0].float().transpose(-1, -2).contiguous()
    return s_ext_m, m


def _empty_summary(H: int, Kdim: int, Vdim: int, device, dtype=torch.float32):
    s_ext = torch.zeros(H, Kdim, Vdim, device=device, dtype=dtype)
    m = torch.eye(Kdim, device=device, dtype=dtype).expand(H, Kdim, Kdim).contiguous()
    return s_ext, m


def all_gather_slot_summaries(
    s_ext_local: torch.Tensor,
    m_local: torch.Tensor,
    *,
    cp_size: int,
    cp_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """All-gather per-rank local ``[2, ...]`` summaries into ``[2C, ...]`` (N=1)."""
    s_slots, m_slots = all_gather_slot_summaries_batch(
        s_ext_local.unsqueeze(1), m_local.unsqueeze(1), cp_size=cp_size, cp_rank=cp_rank
    )
    return s_slots.squeeze(1), m_slots.squeeze(1)


def all_gather_slot_summaries_batch(
    s_ext_local: torch.Tensor,
    m_local: torch.Tensor,
    *,
    cp_size: int,
    cp_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """AG local ``[2, N, ...]`` → chronological ``[2C, N, ...]``.

    Uses a reusable scratch for the packed gather buffer.
    """
    if s_ext_local.shape[0] != 2 or m_local.shape[0] != 2:
        raise ValueError("local summaries must be [2, N, ...] (early, late)")
    group = get_parallel().attn_cp_group
    n_req = s_ext_local.shape[1]
    s_flat = s_ext_local.reshape(2, n_req, -1).contiguous()
    m_flat = m_local.reshape(2, n_req, -1).contiguous()
    s_elems = s_flat.shape[-1]
    packed = torch.cat([s_flat, m_flat], dim=-1)  # [2, N, s+m]
    width = packed.shape[-1]
    flat = packed.reshape(2 * n_req, width).contiguous()
    scratch = get_ag_scratch(
        "summary_ag",
        (2 * cp_size * n_req, width),
        flat.dtype,
        flat.device,
    )
    if hasattr(group, "all_gather_into_tensor"):
        group.all_gather_into_tensor(scratch, flat)
        gathered = scratch
    else:
        parts = [torch.empty_like(flat) for _ in range(cp_size)]
        torch.distributed.all_gather(parts, flat, group=group.device_group)
        gathered = torch.cat(parts, dim=0)

    # Rank-major [P, 2, N, W] → chrono [2C, N, ...]
    rm = gathered.view(cp_size, 2, n_req, width)
    s_rm = rm[..., :s_elems].reshape(cp_size, 2, n_req, *s_ext_local.shape[2:])
    m_rm = rm[..., s_elems:].reshape(cp_size, 2, n_req, *m_local.shape[2:])
    s_slots = s_ext_local.new_empty(2 * cp_size, n_req, *s_ext_local.shape[2:])
    m_slots = m_local.new_empty(2 * cp_size, n_req, *m_local.shape[2:])
    for r in range(cp_size):
        s_slots[r] = s_rm[r, 0]
        m_slots[r] = m_rm[r, 0]
        s_slots[2 * cp_size - 1 - r] = s_rm[r, 1]
        m_slots[2 * cp_size - 1 - r] = m_rm[r, 1]
    return s_slots, m_slots


def scan_segment(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scan one segment with ``fused_recurrent_kda``.

    ``initial_state`` is merge-layout ``[H, dk, dv]``; returns output ``[1,T,H,V]``
    and final flash-layout state ``[H, V, K]``.
    """
    from sglang.kernels.ops.attention.fla.kda import fused_recurrent_kda

    s0 = _merge_state_to_flash(initial_state.float()).unsqueeze(0)
    out, final = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=s0,
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=None,
    )
    return out, final[0]


def split_local_early_late(
    tensor: torch.Tensor, early_len: int, late_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a local zigzag packed tensor ``[early|late, ...]`` (bs=1)."""
    total = early_len + late_len
    if tensor.shape[0] < total:
        raise RuntimeError(f"local zigzag rows {tensor.shape[0]} < early+late {total}")
    return tensor[:early_len], tensor[early_len:total]


# ---------------------------------------------------------------------------
# FlashKDA kcp_flash_* path (vLLM #59304 / FlashKDA #14)
# ---------------------------------------------------------------------------

# FlashKDA workspace bytes per (head, 16-token tile) for head_dim 128.
WORKSPACE_BYTES_PER_TILE = 13824


def flash_kda_kcp_available() -> bool:
    """True when ``flash_kda.kcp_flash_*`` ops are importable."""
    try:
        import flash_kda  # noqa: F401
        import torch

        ops = torch.ops.flash_kda
        return all(
            hasattr(ops, name)
            for name in (
                "kcp_flash_prepare",
                "kcp_flash_summary",
                "kcp_flash_scan",
                "get_workspace_size",
            )
        )
    except Exception:
        return False


def _chrono_to_physical_slot(chrono: int, cp_size: int) -> int:
    """Map chronological slot → physical ``[rank][part]`` index in AG buffer."""
    num_slots = 2 * cp_size
    if chrono < cp_size:
        return 2 * chrono
    return 2 * (num_slots - 1 - chrono) + 1


def merge_packed_flash_summaries(
    packed: torch.Tensor,
    slot_present: torch.Tensor,
    segment_of_slot: torch.Tensor,
    *,
    cp_size: int,
    s_base_flash: Optional[torch.Tensor] = None,
    num_segments: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fold physical ``[2P, N, H, V+K, K]`` BF16 ``[S^T; M^T]`` summaries.

    Converts flash value-first packed blocks into chronological key-first
    ``(s_ext, m)`` and reuses Triton ``kcp_merge_left`` (no per-slot
    ``.item()`` sync storm). Prefill-with-prefix is rejected upstream, so
    ``s_base_flash`` must be ``None`` for the fast path.

    Returns:
        ``initial`` value-first FP32 ``[L, H, V, K]`` for local segments,
        ``final`` value-first FP32 ``[N, H, V, K]`` per request.
    """
    num_phys, n_req, heads, vk, d_k = packed.shape
    d_v = vk - d_k
    num_slots = 2 * cp_size
    if num_phys != num_slots:
        raise ValueError(f"packed phys {num_phys} != 2*cp {num_slots}")
    if num_segments is None:
        # Single host sync only when the caller did not pass layout.num_segments.
        if bool((segment_of_slot >= 0).any()):
            num_segments = int(segment_of_slot.max().item()) + 1
        else:
            num_segments = 0

    device = packed.device
    dtype = torch.float32
    # Physical → chronological without Python matmul / .item() in the fold.
    phys_idx = torch.tensor(
        [_chrono_to_physical_slot(c, cp_size) for c in range(num_slots)],
        device=device,
        dtype=torch.long,
    )
    chrono_packed = packed.float().index_select(0, phys_idx)  # [S,N,H,V+K,K]
    s_t = chrono_packed[:, :, :, :d_v, :]  # [S,N,H,V,K] = S^T
    m_t = chrono_packed[:, :, :, d_v:, :]  # [S,N,H,K,K] = M^T
    # key-first: S = (S^T)^T, M = (M^T)^T, state' = S + M @ state
    s_ext = s_t.transpose(-2, -1).contiguous()  # [S,N,H,K,V]
    m = m_t.transpose(-2, -1).contiguous()  # [S,N,H,K,K]

    if s_base_flash is not None:
        # Triton kcp_merge_left currently starts from zeros; keep a rare
        # Python fallback if a nonempty base state is ever passed.
        s_base = (
            s_base_flash.to(device=device, dtype=dtype).transpose(-2, -1).contiguous()
        )
        s_in_all, s_final = merge_affine_segment_states_batched(
            s_ext, m, slot_present, s_base=s_base
        )
        initial = s_ext.new_zeros(max(num_segments, 1), heads, d_k, d_v)
        for n in range(n_req):
            for slot in range(num_slots):
                seg = int(segment_of_slot[n, slot].item())
                if seg >= 0:
                    initial[seg].copy_(s_in_all[slot, n])
        final_kf = s_final
        if num_segments == 0:
            initial = initial[:0]
        else:
            initial = initial[:num_segments]
        return initial.transpose(-2, -1).contiguous(), final_kf.transpose(
            -2, -1
        ).contiguous()

    initial_kf = torch.zeros(
        max(num_segments, 1), heads, d_k, d_v, device=device, dtype=dtype
    )
    final_kf = torch.zeros(n_req, heads, d_k, d_v, device=device, dtype=dtype)
    kcp_merge_left(
        s_ext,
        m,
        slot_present,
        segment_of_slot,
        initial_kf,
        final_kf,
    )
    if num_segments == 0:
        initial_kf = initial_kf[:0]
    else:
        initial_kf = initial_kf[:num_segments]
    # flash scan wants value-first [L,H,V,K] / [N,H,V,K]
    return initial_kf.transpose(-2, -1).contiguous(), final_kf.transpose(
        -2, -1
    ).contiguous()


def run_zigzag_kcp_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    plan: KcpPlan,
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    lower_bound: float,
    s_base_flash: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """KCP five-step on FlashKDA ``kcp_flash_{prepare,summary,scan}``."""
    import flash_kda

    layout = plan.layout
    cp_size = layout.world
    n_req = layout.num_prefill_reqs
    _, tokens, heads, dim = q.shape
    if dim != 128:
        raise NotImplementedError(f"FlashKDA KCP requires head_dim=128, got {dim}")
    device = q.device
    # beta_t: [H, T]
    beta_t = beta[0].transpose(0, 1).contiguous()
    tiles = (tokens + 15) // 16 + max(layout.num_segments, 1)
    workspace = torch.empty(
        heads * tiles * WORKSPACE_BYTES_PER_TILE,
        dtype=torch.uint8,
        device=device,
    )
    cu = layout.scan_cu_seqlens.to(device=device, dtype=torch.int32).contiguous()
    flash_kda.kcp_flash_prepare(
        q.contiguous(),
        k.contiguous(),
        g.contiguous(),
        beta_t,
        A_log.reshape(-1).contiguous(),
        dt_bias.reshape(heads, dim).contiguous(),
        cu,
        workspace,
        float(scale),
        float(lower_bound),
    )

    # Local summary rows [2, N] → flat [2N]; AG → [2P N].
    local_rows = 2 * n_req
    local_sum = torch.zeros(
        local_rows, heads, dim + dim, dim, device=device, dtype=torch.bfloat16
    )
    if layout.num_segments > 0:
        # out_rows: local index part*N + n for each segment in scan order.
        out_rows = []
        for n, slot in zip(layout.segment_request, layout.segment_slot):
            part = 0 if slot < cp_size else 1
            out_rows.append(part * n_req + n)
        out_rows_t = torch.tensor(out_rows, device=device, dtype=torch.int64)
        flash_kda.kcp_flash_summary(
            v.contiguous(),
            beta_t,
            workspace,
            cu,
            out_rows_t,
            local_sum,
        )

    group = get_parallel().attn_cp_group
    flat = local_sum.reshape(local_rows, -1).contiguous()
    gathered = get_ag_scratch(
        "kcp_flash_summary",
        (local_rows * cp_size, flat.shape[1]),
        flat.dtype,
        flat.device,
    )
    if hasattr(group, "all_gather_into_tensor"):
        group.all_gather_into_tensor(gathered, flat)
    else:
        parts = [torch.empty_like(flat) for _ in range(cp_size)]
        torch.distributed.all_gather(parts, flat, group=group.device_group)
        gathered = torch.cat(parts, dim=0)
    packed = gathered.view(cp_size, 2, n_req, heads, dim + dim, dim)
    # Physical [2P, N, H, V+K, K]: slot 2r = rank r part0, 2r+1 = rank r part1.
    phys = local_sum.new_empty(2 * cp_size, n_req, heads, dim + dim, dim)
    for r in range(cp_size):
        phys[2 * r] = packed[r, 0]
        phys[2 * r + 1] = packed[r, 1]

    initial, final_flash = merge_packed_flash_summaries(
        phys,
        plan.slot_present,
        plan.segment_of_slot,
        cp_size=cp_size,
        s_base_flash=s_base_flash,
        num_segments=layout.num_segments,
    )

    out = torch.empty_like(v)
    if layout.num_segments > 0:
        flash_kda.kcp_flash_scan(
            v.contiguous(),
            beta_t,
            workspace,
            cu,
            initial.contiguous(),
            out,
        )
    return out, final_flash


# ---------------------------------------------------------------------------
# Five-step runners
# ---------------------------------------------------------------------------


def run_zigzag_kcp_from_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    plan: KcpPlan,
    *,
    scale: float,
    s_base: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full KCP five-step driven by ``KcpPlan`` (supports bs>=1).

    Prefers FlashKDA ``kcp_flash_*`` when the extension is installed and
    ``A_log``/``dt_bias``/``lower_bound`` are provided; otherwise falls back to
    the ``fused_recurrent_kda`` reference path.

    Args:
        q,k,v,g,beta: ``[1, T_local, H, ...]`` packed all-early then all-late.
        s_base: optional merge-layout ``[N, H, dk, dv]`` prefix (reference path)
            or ignored when Flash path uses ``s_base`` converted to flash layout.

    Returns:
        ``core_out`` ``[1, T_local, H, V]``, ``final_states`` flash ``[N, H, V, K]``.
    """
    use_flash = (
        flash_kda_kcp_available()
        and A_log is not None
        and dt_bias is not None
        and lower_bound is not None
        and q.shape[-1] == 128
    )
    if use_flash:
        s_base_flash = None
        if s_base is not None:
            s_base_flash = _merge_state_to_flash(s_base.float())
        return run_zigzag_kcp_flash(
            q,
            k,
            v,
            g,
            beta,
            plan,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            lower_bound=float(lower_bound),
            s_base_flash=s_base_flash,
        )

    layout = plan.layout
    cp_size, cp_rank = layout.world, layout.rank
    n_req = layout.num_prefill_reqs
    H, Kdim, Vdim = k.shape[2], k.shape[3], v.shape[3]
    device = q.device

    # Local [2, N, H, ...] summaries (identity/zero for empty parts).
    s_local = torch.zeros(2, n_req, H, Kdim, Vdim, device=device, dtype=torch.float32)
    m_local = (
        torch.eye(Kdim, device=device, dtype=torch.float32)
        .expand(2, n_req, H, Kdim, Kdim)
        .contiguous()
        .clone()
    )
    part_tensors: List[Optional[Tuple]] = [None] * layout.num_segments

    for seg_i, (n, slot, local_start) in enumerate(
        zip(
            layout.segment_request,
            layout.segment_slot,
            layout.segment_local_start,
        )
    ):
        length = int(layout.scan_cu[seg_i + 1] - layout.scan_cu[seg_i])
        part = 0 if slot < cp_size else 1
        if length <= 0:
            continue
        qs = q[:, local_start : local_start + length].contiguous()
        ks = k[:, local_start : local_start + length].contiguous()
        vs = v[:, local_start : local_start + length].contiguous()
        gs = g[:, local_start : local_start + length].contiguous()
        bs = beta[:, local_start : local_start + length].contiguous()
        s_ext, m = reference_affine_summary(qs, ks, vs, gs, bs, scale=scale)
        s_local[part, n] = s_ext
        m_local[part, n] = m
        part_tensors[seg_i] = (qs, ks, vs, gs, bs)

    s_slots, m_slots = all_gather_slot_summaries_batch(
        s_local, m_local, cp_size=cp_size, cp_rank=cp_rank
    )
    # [S, N, H, ...] chronological
    if s_base is not None:
        # Seed fold by rewriting slot-0 path via Python when prefix present.
        s_in_all, s_final = merge_affine_segment_states_batched(
            s_slots, m_slots, plan.slot_present, s_base=s_base
        )
        initial = s_slots.new_empty(layout.num_segments, H, Kdim, Vdim)
        for seg_i, (n, slot) in enumerate(
            zip(layout.segment_request, layout.segment_slot)
        ):
            initial[seg_i] = s_in_all[slot, n]
        final_merge = s_final
    else:
        initial = s_slots.new_empty(layout.num_segments, H, Kdim, Vdim)
        final_merge = s_slots.new_empty(n_req, H, Kdim, Vdim)
        kcp_merge_left(
            s_slots,
            m_slots,
            plan.slot_present,
            plan.segment_of_slot,
            initial,
            final_merge,
        )

    outs = []
    for seg_i, part in enumerate(part_tensors):
        if part is None:
            continue
        qs, ks, vs, gs, bs = part
        out, _ = scan_segment(
            qs, ks, vs, gs, bs, scale=scale, initial_state=initial[seg_i]
        )
        outs.append((layout.segment_local_start[seg_i], out))

    if not outs:
        raise RuntimeError("zigzag KCP produced no local segments")
    outs.sort(key=lambda x: x[0])
    core = torch.cat([o for _, o in outs], dim=1)
    final_flash = _merge_state_to_flash(final_merge)
    return core, final_flash


def run_zigzag_kcp_on_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    early_len: int,
    late_len: int,
    scale: float,
    s_base: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """bs=1 convenience wrapper around ``run_zigzag_kcp_from_plan``."""
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    total = early_len + late_len
    device = q.device
    layout = HybridPCPLayout(
        world=cp_size,
        rank=cp_rank,
        num_prefill_tokens=total,
        prefill_lens=[early_len + late_len],  # approximate; slot_lens set below
        slot_lens=torch.zeros(1, 2 * cp_size, dtype=torch.int32, device=device),
        segment_request=[],
        segment_slot=[],
        segment_start=[],
        segment_local_start=[],
        scan_cu=[0],
        scan_cu_seqlens=torch.tensor([0], dtype=torch.int32, device=device),
        prefill_has_initial_state=torch.zeros(1, dtype=torch.bool, device=device),
    )
    # Fill slot lens from early/late only on owned slots; others stay 0 so merge
    # still walks but AG fills real values from peers. For the bs=1 wrapper we
    # only need local segments correct; present flags come from gathered lens
    # approximated as owned lengths (peer slots marked present if length>0 unknown).
    # Prefer the legacy single-req path for exact peer presence:
    return _run_zigzag_kcp_on_qkv_legacy(
        q,
        k,
        v,
        g,
        beta,
        early_len=early_len,
        late_len=late_len,
        scale=scale,
        s_base=s_base,
    )


def _run_zigzag_kcp_on_qkv_legacy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    early_len: int,
    late_len: int,
    scale: float,
    s_base: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Original bs=1 five-step (kept for API compatibility)."""
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank

    def _slice_bt(t, start, length):
        return t[:, start : start + length].contiguous()

    parts = []
    summaries_s = []
    summaries_m = []
    for start, length in ((0, early_len), (early_len, late_len)):
        if length == 0:
            H, Kdim, Vdim = k.shape[2], k.shape[3], v.shape[3]
            s_ext, m = _empty_summary(H, Kdim, Vdim, q.device)
            summaries_s.append(s_ext)
            summaries_m.append(m)
            parts.append(None)
            continue
        qs = _slice_bt(q, start, length)
        ks = _slice_bt(k, start, length)
        vs = _slice_bt(v, start, length)
        gs = _slice_bt(g, start, length)
        bs = _slice_bt(beta, start, length)
        s_ext, m = reference_affine_summary(qs, ks, vs, gs, bs, scale=scale)
        summaries_s.append(s_ext)
        summaries_m.append(m)
        parts.append((qs, ks, vs, gs, bs))

    s_local = torch.stack(summaries_s, dim=0)
    m_local = torch.stack(summaries_m, dim=0)
    s_slots, m_slots = all_gather_slot_summaries(
        s_local, m_local, cp_size=cp_size, cp_rank=cp_rank
    )
    s_base_m = None
    if s_base is not None:
        s_base_m = _flash_state_to_merge(s_base.float())
    s_in_all = merge_affine_segment_states(s_slots, m_slots, s_base=s_base_m)

    local_slot_ids = [cp_rank, 2 * cp_size - 1 - cp_rank]
    outs = []
    final_state = None
    for part, slot_id in zip(parts, local_slot_ids):
        if part is None:
            continue
        qs, ks, vs, gs, bs = part
        out, final_state = scan_segment(
            qs, ks, vs, gs, bs, scale=scale, initial_state=s_in_all[slot_id]
        )
        outs.append(out)
    if not outs:
        raise RuntimeError("zigzag KCP produced no local segments")
    core = torch.cat(outs, dim=1)
    return core, final_state
