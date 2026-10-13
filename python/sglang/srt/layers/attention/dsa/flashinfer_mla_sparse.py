"""DSA sparse path via FlashInfer ``BatchMLAPagedAttentionWrapper`` (SM90 FA3).

Mirrors vLLM ``FLASHINFER_MLA_SPARSE_SM90``: with ``page_size=1`` the per-token
top-k slot indices are the page table; causality is already encoded by the
indexer, so ``causal=False``. Distinct from SM120 ``flashinfer_sparse_mla``.
"""

from __future__ import annotations

from typing import Optional

import torch


def pack_topk_page_table_to_csr(
    page_table: torch.Tensor,
    kv_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack dense ``[T, W]`` top-k slots into CSR ``(kv_indptr, kv_indices, kv_len)``.

    Valid tokens are assumed to occupy the leading ``kv_lens[t]`` columns of
    each row (same layout as DSA ``dsa_cache_seqlens_int32`` + ``-1``-padded
    tails). Negative slots are clamped to 0 so over-reads are not out of range;
    planned ``kv_len`` keeps the kernel on the valid prefix.
    """
    if page_table.ndim != 2:
        raise ValueError(f"page_table must be [T, W], got {tuple(page_table.shape)}")
    num_tokens, width = page_table.shape
    if kv_lens.numel() != num_tokens:
        raise ValueError(
            f"kv_lens length {kv_lens.numel()} != page_table rows {num_tokens}"
        )
    kv_len_arr = kv_lens.to(dtype=torch.int32).clamp(min=0, max=width)
    kv_indptr = torch.empty(num_tokens + 1, dtype=torch.int32, device=page_table.device)
    kv_indptr[0] = 0
    torch.cumsum(kv_len_arr, dim=0, out=kv_indptr[1:])
    cols = torch.arange(width, device=page_table.device, dtype=torch.int32).expand(
        num_tokens, width
    )
    mask = cols < kv_len_arr.unsqueeze(1)
    kv_indices = page_table.to(torch.int32).clamp(min=0)[mask]
    return kv_indptr, kv_indices, kv_len_arr


class DSAFlashInferMLASparseRunner:
    """Owns workspace + wrapper for DSA ``flashinfer_mla_sparse``."""

    def __init__(
        self,
        workspace: torch.Tensor,
        *,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        backend: str = "fa3",
    ) -> None:
        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        self.workspace = workspace
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.wrapper = BatchMLAPagedAttentionWrapper(
            workspace,
            backend=backend,
        )
        self._planned_num_tokens: int = -1

    def plan(
        self,
        num_tokens: int,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        sm_scale: float,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
    ) -> None:
        """Plan outside CUDA-graph capture (FlashInfer plan does host copies)."""
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "flashinfer_mla_sparse plan() must run outside CUDA graph capture"
            )
        qo_indptr = torch.arange(
            num_tokens + 1, dtype=torch.int32, device=kv_indptr.device
        )
        # Prefer CPU tensors when already host-side to avoid extra D2H inside plan.
        if kv_indptr.device.type == "cpu":
            qo_cpu = torch.arange(num_tokens + 1, dtype=torch.int32)
            self.wrapper.plan(
                qo_cpu,
                kv_indptr,
                kv_indices,
                kv_len_arr,
                self.num_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                1,  # page_size: top-k slots are the page table
                False,  # causal: encoded by indexer selection
                sm_scale,
                q_dtype,
                kv_dtype,
            )
        else:
            self.wrapper.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_len_arr,
                self.num_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                1,
                False,
                sm_scale,
                q_dtype,
                kv_dtype,
            )
        self._planned_num_tokens = num_tokens

    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv: torch.Tensor,
        kpe: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.wrapper.run(q_nope, q_pe, ckv, kpe, out=out)

    def forward(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        cache_seqlens: torch.Tensor,
        sm_scale: float,
        v_head_dim: int,
    ) -> torch.Tensor:
        """Eager plan+pack+run for one DSA sparse step.

        ``kv_cache`` is the MLA key buffer ``[..., kv_lora_rank + rope]`` (page
        size 1 token layout as used by DSA ``fa3``).
        """
        num_tokens = q_nope.shape[0]
        if page_table_1.shape[0] != num_tokens:
            raise ValueError(
                f"page_table rows {page_table_1.shape[0]} != q tokens {num_tokens}"
            )
        if cache_seqlens.shape[0] != num_tokens:
            # Metadata may be padded to batch capacity; trim to live query rows.
            cache_seqlens = cache_seqlens[:num_tokens]

        kv_indptr, kv_indices, kv_len_arr = pack_topk_page_table_to_csr(
            page_table_1, cache_seqlens
        )
        # Empty CSR (all-zero lengths) still needs a dummy index buffer for plan.
        if kv_indices.numel() == 0:
            kv_indices = torch.zeros(1, dtype=torch.int32, device=page_table_1.device)

        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        if capturing:
            # Reserved-buffer plan/pack (vLLM-style) not wired yet; fail closed
            # so stale plan indices cannot silently corrupt graph replay.
            raise NotImplementedError(
                "flashinfer_mla_sparse does not yet support CUDA-graph capture "
                "(need use_cuda_graph reserved buffers + metadata-time plan). "
                "Use eager decode or disable CUDA graph for this backend; "
                "prefill/eager decode plan+run is supported."
            )
        self.plan(
            num_tokens,
            kv_indptr,
            kv_indices,
            kv_len_arr,
            sm_scale,
            q_nope.dtype,
            kv_cache.dtype,
        )

        # DSA MLA cache: [num_tokens_or_pages, 1?, dim] or [N, dim]
        flat = kv_cache
        if flat.ndim == 2:
            flat = flat.unsqueeze(1)
        elif flat.ndim == 4:
            # [num_blocks, page_size, 1, dim] → flatten pages to tokens
            flat = flat.reshape(-1, flat.shape[-2], flat.shape[-1])
        ckv = flat[..., :v_head_dim]
        kpe = flat[..., v_head_dim:]
        if self.qk_rope_head_dim == 0:
            # NoPE: keep a zero-width rope tensor with matching layout.
            kpe = flat.new_empty(*flat.shape[:-1], 0)
            if q_pe.numel() == 0 or q_pe.shape[-1] == 0:
                q_pe = q_nope.new_empty(num_tokens, q_nope.shape[1], 0)

        out = q_nope.new_empty(q_nope.shape)
        return self.run(q_nope, q_pe, ckv, kpe, out=out)
