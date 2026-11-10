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

"""KDA Kimi Context Parallelism (KCP) helpers for zigzag Prefill-CP.

First local cut: affine merge of per-segment summaries ``(S_ext, M)``.
FlashKDA prepare / summary / scan kernels are wired later behind
``SGLANG_ENABLE_GLM_ZIGZAG_KCP``.

For a segment affine map ``S_out = S_ext + M @ S_in`` (column-vector state),
folding segments 0..k-1 into the initial state of segment k is:

    S_in[0] = S_base
    S_in[i+1] = S_ext[i] + M[i] @ S_in[i]
"""

from __future__ import annotations

from typing import List, Optional

import torch

from sglang.srt.environ import envs


def glm_zigzag_kcp_enabled() -> bool:
    return bool(envs.SGLANG_ENABLE_GLM_ZIGZAG_KCP.get())


def merge_affine_segment_states(
    s_ext: torch.Tensor,
    m: torch.Tensor,
    s_base: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fold ``2C`` (or N) segment summaries into per-segment initial states.

    Args:
        s_ext: ``[num_slots, ...]`` external state written from a zero input.
            Trailing dims are the state matrix (typically ``[H, dk, dv]`` or
            ``[dk, dv]``).
        m: ``[num_slots, ...]`` transition; trailing dims ``[H, dk, dk]`` /
            ``[dk, dk]`` matching ``S_out = S_ext + M @ S_in``.
        s_base: optional recurrent cache state before slot 0; zeros if None.

    Returns:
        ``s_in`` with the same leading layout as ``s_ext``: initial state for
        each slot before that slot's local scan.
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
        # S_out = S_ext + M @ S_in  (apply M on the state leading axis)
        cur = s_ext[i] + _apply_transition(m[i], cur)
    return s_in


def _apply_transition(m: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Apply ``M @ S`` for state layouts ``[..., dk, dv]`` and ``M`` ``[..., dk, dk]``."""
    if m.ndim == state.ndim and m.shape[:-1] == state.shape[:-1]:
        # [..., dk, dk] @ [..., dk, dv]
        return torch.matmul(m, state)
    if m.ndim + 1 == state.ndim:
        # broadcast head-free M over an extra leading head dim on state
        return torch.matmul(m, state)
    raise ValueError(
        f"Unsupported M/S shapes: m={tuple(m.shape)} s={tuple(state.shape)}"
    )


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
