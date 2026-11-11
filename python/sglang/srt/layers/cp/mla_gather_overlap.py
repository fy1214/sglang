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

"""Prefill-CP: overlap MLA latent KV all-gather with kpool compress→topk.

After ``materialize_compression_inputs`` finishes the compression-input
all-gather, launch **only** the attn-CP ``all_gather_into_tensor`` on
``alt_stream`` while compress / mqa_logits / topk continue on the current
stream. Pad / reorder / rope stay off ``alt_stream``.

Gated by ``SGLANG_ENABLE_CP_MLA_KV_GATHER_OVERLAP``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.cp.base import get_cp_strategy
from sglang.srt.layers.cp.interleave import is_interleave_extend
from sglang.srt.layers.cp.utils import is_cp_active
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    is_allocation_symmetric,
)
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def _prepare_interleave_ag_buffers(
    x: torch.Tensor, forward_batch: ForwardBatch, strategy: Any
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], dict[str, Any]]:
    """Pad + allocate gather output on the *current* stream; no collective."""
    metadata = getattr(forward_batch, "attn_cp_metadata", None)
    if metadata is None:
        raise RuntimeError("Interleave CP gather requires attn_cp_metadata.")

    total_tokens = int(metadata.total_seq_lens)
    if total_tokens < 0:
        raise RuntimeError(
            f"Invalid interleave CP total_seq_lens={total_tokens}; expected >= 0."
        )

    logical_rank_lens = (
        metadata.per_rank_logical_token or metadata.per_rank_actual_token
    )
    local_logical_len = logical_rank_lens[strategy.cp_rank]
    if x.shape[0] < local_logical_len:
        raise RuntimeError(
            "Interleave CP gather received an unexpected local token count: "
            f"rank={strategy.cp_rank}, got={x.shape[0]}, "
            f"expected_at_least={local_logical_len}, "
            f"total={total_tokens}, cp_size={strategy.cp_size}."
        )

    physical_rank_len = max(metadata.per_rank_actual_token)
    ctx = {
        "metadata": metadata,
        "total_tokens": total_tokens,
        "physical_rank_len": physical_rank_len,
        "cp_size": strategy.cp_size,
        "tail_shape": tuple(x.shape[1:]),
        "device": x.device,
    }
    if physical_rank_len == 0:
        return None, None, ctx

    padded_x = x.new_zeros((physical_rank_len, *x.shape[1:]))
    padded_x[:local_logical_len] = x[:local_logical_len]
    with use_symmetric_memory(
        get_parallel().attn_cp_group, disabled=not is_allocation_symmetric()
    ):
        gathered = x.new_empty((strategy.cp_size * physical_rank_len, *x.shape[1:]))
    return padded_x, gathered, ctx


def _finish_interleave_ag(
    gathered: Optional[torch.Tensor], ctx: dict[str, Any]
) -> torch.Tensor:
    """Reorder AG output into sequence order on the *current* stream."""
    metadata = ctx["metadata"]
    total_tokens = ctx["total_tokens"]
    physical_rank_len = ctx["physical_rank_len"]
    cp_size = ctx["cp_size"]
    tail_shape = ctx["tail_shape"]
    device = ctx["device"]

    if physical_rank_len == 0 or gathered is None:
        return (
            gathered.new_empty((0, *tail_shape))
            if gathered is not None
            else (torch.empty((0, *tail_shape), device=device))
        )

    if metadata.gather_index is not None:
        return gathered.index_select(0, metadata.gather_index)

    actual = metadata.per_rank_actual_token
    if total_tokens == cp_size * physical_rank_len and all(
        int(n) == physical_rank_len for n in actual
    ):
        return (
            gathered.view(cp_size, physical_rank_len, *tail_shape)
            .transpose(0, 1)
            .reshape(total_tokens, *tail_shape)
        )

    flat_indices = torch.arange(total_tokens, device=device)
    gather_indices = (
        flat_indices % cp_size
    ) * physical_rank_len + flat_indices // cp_size
    return gathered.index_select(0, gather_indices)


@dataclass
class CpMlaGatherOverlapState:
    """Per-layer handshake: compression-input gather done → MLA KV all-gather on alt_stream."""

    attn_mla: Any
    forward_batch: ForwardBatch
    k_nope: torch.Tensor
    k_pe: torch.Tensor
    q_pe: torch.Tensor
    positions: torch.Tensor
    alt_stream: torch.cuda.Stream
    launched: bool = False
    pending: bool = False
    roped: bool = False
    _consumer_stream: Optional[torch.cuda.Stream] = field(default=None, repr=False)
    _kv_lora_rank: int = 0
    _ag_gathered: Optional[torch.Tensor] = field(default=None, repr=False)
    _ag_ctx: Optional[dict[str, Any]] = field(default=None, repr=False)

    def launch_after_compression_gather(self) -> None:
        """Enqueue **only** attn-CP all-gather on ``alt_stream``.

        Pad/alloc run on the current stream before the fork. No rope, no
        reorder, no ``prepare_kv_for_attention`` on alt. Caller must rope
        after ``wait_and_take`` when ``roped`` is False.
        """
        if self.launched:
            return
        self.launched = True
        current = torch.cuda.current_stream()
        self._consumer_stream = current

        strategy = get_cp_strategy()
        assert strategy is not None
        self._kv_lora_rank = int(self.k_nope.shape[-1])
        latent = torch.cat([self.k_nope, self.k_pe], dim=-1).squeeze(1).contiguous()
        padded, gathered, ctx = _prepare_interleave_ag_buffers(
            latent, self.forward_batch, strategy
        )
        self._ag_ctx = ctx
        self._ag_gathered = gathered
        self.roped = False

        self.alt_stream.wait_stream(current)
        if padded is not None and gathered is not None:
            with torch.cuda.stream(self.alt_stream):
                # Sole kernel/collective on alt_stream.
                attn_cp_all_gather_into_tensor(gathered, padded.contiguous())
                gathered.record_stream(current)
                padded.record_stream(current)
        self.pending = True

    def wait_and_take(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Join alt_stream; reorder AG output; return pre-rope (k_nope, k_pe, q_pe, False)."""
        if self.pending:
            torch.cuda.current_stream().wait_stream(self.alt_stream)
            self.pending = False

        assert self._ag_ctx is not None
        full_latent = _finish_interleave_ag(self._ag_gathered, self._ag_ctx)
        rank = self._kv_lora_rank
        k_nope = full_latent[..., :rank].unsqueeze(1)
        k_pe = full_latent[..., rank:].unsqueeze(1)
        # q_pe stays the local pre-rope view; rope on the consumer stream.
        return k_nope, k_pe, self.q_pe, False


def cp_mla_gather_overlap_enabled() -> bool:
    return bool(envs.SGLANG_ENABLE_CP_MLA_KV_GATHER_OVERLAP.get())


def maybe_install_cp_mla_gather_overlap(
    attn_mla: Any,
    forward_batch: ForwardBatch,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    q_pe: torch.Tensor,
    positions: torch.Tensor,
) -> Optional[CpMlaGatherOverlapState]:
    """Install overlap state on ``forward_batch`` when the gate is open.

    BCG / cuda-graph capture / trtllm fused-rope defer / missing alt_stream
    stay on the serial path. Caller must pass pre-rope local ``k_pe``/``q_pe``.
    """
    forward_batch.cp_mla_gather_overlap = None
    if not cp_mla_gather_overlap_enabled():
        return None
    if attn_mla.alt_stream is None:
        return None
    if not dsa_use_prefill_cp(forward_batch) or not is_cp_active(forward_batch):
        return None
    if not is_interleave_extend(forward_batch):
        return None
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
        is_in_breakable_cuda_graph,
    )

    if get_is_capture_mode() or is_in_breakable_cuda_graph():
        return None
    fuse_rope = getattr(attn_mla, "_fuse_rope_for_trtllm_mla", None)
    if fuse_rope is not None and fuse_rope(forward_batch):
        return None

    state = CpMlaGatherOverlapState(
        attn_mla=attn_mla,
        forward_batch=forward_batch,
        k_nope=k_nope,
        k_pe=k_pe,
        q_pe=q_pe,
        positions=positions,
        alt_stream=attn_mla.alt_stream,
    )
    forward_batch.cp_mla_gather_overlap = state
    return state


def clear_cp_mla_gather_overlap(forward_batch: ForwardBatch) -> None:
    forward_batch.cp_mla_gather_overlap = None
