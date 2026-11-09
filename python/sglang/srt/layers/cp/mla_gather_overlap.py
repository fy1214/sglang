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
all-gather, launch rope(k) + ``materialize_full_mla_kv`` on ``alt_stream``
while compress/logits/topk continue on the current stream.
Gated by ``SGLANG_ENABLE_CP_MLA_KV_GATHER_OVERLAP``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.attention.dsa_backend import prepare_kv_for_attention
from sglang.srt.layers.cp.interleave import is_interleave_extend
from sglang.srt.layers.cp.utils import is_cp_active

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass
class CpMlaGatherOverlapState:
    """Per-layer handshake: compression-input gather done → MLA KV gather on alt_stream."""

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

    def launch_after_compression_gather(self) -> None:
        """Enqueue rope(q/k pe) + MLA latent KV all-gather on ``alt_stream``.

        Must run on the stream that just finished ``materialize_compression_inputs``
        (typically current) so ``alt.wait_stream(current)`` sees that gather complete.
        """
        if self.launched:
            return
        self.launched = True
        current = torch.cuda.current_stream()
        self._consumer_stream = current
        self.alt_stream.wait_stream(current)
        attn = self.attn_mla
        with torch.cuda.stream(self.alt_stream):
            q_pe, k_pe = self.q_pe, self.k_pe
            if attn.rotary_emb is not None:
                q_pe, k_pe = attn.rotary_emb(self.positions, q_pe, k_pe)
                self.roped = True
            k_nope, k_pe = prepare_kv_for_attention(
                attn,
                self.forward_batch,
                self.k_nope,
                k_pe,
                defer_materialization=False,
                stream=self.alt_stream,
            )
            # Cross-stream lifetime for the consumer (current) stream.
            for t in (k_nope, k_pe, q_pe):
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    t.record_stream(current)
            self.k_nope = k_nope
            self.k_pe = k_pe
            self.q_pe = q_pe
        self.pending = True

    def wait_and_take(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Join alt_stream; return (k_nope, k_pe, q_pe, roped)."""
        if self.pending:
            torch.cuda.current_stream().wait_stream(self.alt_stream)
            self.pending = False
        return self.k_nope, self.k_pe, self.q_pe, self.roped


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
