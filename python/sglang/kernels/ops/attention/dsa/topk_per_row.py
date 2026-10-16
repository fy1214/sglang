"""vLLM ``top_k_per_row_{prefill,decode}`` (topKPerRow) JIT bindings for kpool select."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _jit_topk_per_row_module():
    return load_jit(
        "topk_per_row",
        cuda_files=["dsa/topk_per_row.cuh"],
        cuda_wrappers=[
            (
                "top_k_per_row_prefill",
                "sglang::topk_per_row::TopKPerRowKernel::prefill",
            ),
            ("top_k_per_row_decode", "sglang::topk_per_row::TopKPerRowKernel::decode"),
        ],
    )


def top_k_per_row_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    out_indices: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Select top-``topk`` column indices per row in ``[row_starts, row_ends)``.

    Indices are relative to the row start (same contract as vLLM
    ``top_k_per_row_prefill``). ``out_indices`` is int32 ``[B, topk]``, ``-1`` padded.
    """
    assert logits.dtype == torch.float32 and logits.is_cuda
    assert logits.stride(1) == 1
    _jit_topk_per_row_module().top_k_per_row_prefill(
        logits, row_starts, row_ends, out_indices, topk
    )
    return out_indices


def top_k_per_row_decode(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    out_indices: torch.Tensor,
    topk: int,
    next_n: int = 1,
) -> torch.Tensor:
    """Decode-shaped topKPerRow (vLLM ``top_k_per_row_decode``)."""
    assert logits.dtype == torch.float32 and logits.is_cuda
    assert logits.stride(1) == 1
    aux_indices = None
    aux_logits = None
    # Long-row path needs split/merge workspace (same threshold as vLLM).
    if logits.shape[1] >= 200_000:
        aux_indices = torch.empty(
            (logits.shape[0], 10, topk), dtype=torch.int32, device=logits.device
        )
        aux_logits = torch.empty(
            (logits.shape[0], 10, topk), dtype=torch.float32, device=logits.device
        )
    _jit_topk_per_row_module().top_k_per_row_decode(
        logits,
        seq_lens,
        out_indices,
        aux_indices,
        aux_logits,
        next_n,
        topk,
    )
    return out_indices


def top_k_per_row_select(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    *,
    row_starts: Optional[torch.Tensor] = None,
    next_n: int = 1,
) -> torch.Tensor:
    """Unified select for kpool: prefill when ``row_starts`` given, else decode."""
    out = logits.new_empty((logits.shape[0], topk), dtype=torch.int32)
    lengths = lengths.to(dtype=torch.int32, device=logits.device)
    if row_starts is not None:
        row_starts = row_starts.to(dtype=torch.int32, device=logits.device)
        row_ends = row_starts + lengths
        return top_k_per_row_prefill(logits, row_starts, row_ends, out, topk)
    return top_k_per_row_decode(logits, lengths, out, topk, next_n=next_n)
