"""DSA sparse path via FlashInfer ``BatchMLAPagedAttentionWrapper`` (SM90 FA3).

Mirrors vLLM ``FLASHINFER_MLA_SPARSE_SM90``:

- ``page_size=1`` top-k slots are the page table; ``causal=False``
- ``plan()`` runs outside CUDA-graph capture (metadata build / replay)
- ``pack_indices()`` + ``run()`` are capture-safe and update reserved buffers
- Distinct from SM120 ``flashinfer_sparse_mla``
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


def pack_topk_page_table_to_csr(
    page_table: torch.Tensor,
    kv_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack dense ``[T, W]`` top-k slots into CSR ``(kv_indptr, kv_indices, kv_len)``.

    Valid tokens occupy the leading ``kv_lens[t]`` columns (DSA
    ``dsa_cache_seqlens_int32`` + ``-1``-padded tails). Negative slots clamp to 0.
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


if _TRITON_AVAILABLE:

    @triton.jit
    def _pack_topk_indices_kernel(
        slots,
        indptr,
        indices,
        WIDTH: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        start = tl.load(indptr + row)
        end = tl.load(indptr + row + 1)
        mask = (cols < end - start) & (cols < WIDTH)
        values = tl.load(slots + row * WIDTH + cols, mask=mask, other=0)
        tl.store(indices + start + cols, tl.maximum(values, 0), mask=mask)


class DSAFlashInferMLASparseRunner:
    """vLLM-style BatchMLA sparse state: reserved buffers, plan outside graph."""

    def __init__(
        self,
        workspace: torch.Tensor,
        *,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        sm_scale: float,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        max_tokens: int,
        topk_width: int,
        backend: str = "fa3",
    ) -> None:
        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        if max_tokens < 1 or topk_width < 1:
            raise ValueError(
                f"max_tokens/topk_width must be >=1, got {max_tokens=}, {topk_width=}"
            )

        self.workspace = workspace
        self.device = workspace.device
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.sm_scale = sm_scale
        self.q_dtype = q_dtype
        self.kv_dtype = kv_dtype
        self.backend = backend
        self.max_tokens = max_tokens
        self.topk_width = topk_width

        # Capture-stable device buffers (plan() refreshes via copy_).
        self.kv_indices = torch.zeros(
            max_tokens * topk_width, dtype=torch.int32, device=self.device
        )
        self.kv_len_arr = torch.zeros(max_tokens, dtype=torch.int32, device=self.device)
        self.kv_indptr = torch.zeros(
            max_tokens + 1, dtype=torch.int32, device=self.device
        )
        self.qo_indptr = torch.zeros(
            max_tokens + 1, dtype=torch.int32, device=self.device
        )
        self.wrapper = BatchMLAPagedAttentionWrapper(
            workspace,
            qo_indptr=self.qo_indptr,
            kv_indptr=self.kv_indptr,
            kv_indices=self.kv_indices,
            kv_len_arr=self.kv_len_arr,
            use_cuda_graph=True,
            backend=backend,
        )

        # Host staging so FlashInfer plan()'s internal .to("cpu") is a no-op.
        self._arange_cpu = torch.arange(max_tokens + 1, dtype=torch.int32)
        self._qo_cpu = torch.empty(max_tokens + 1, dtype=torch.int32)
        self._kv_cpu = torch.empty(max_tokens + 1, dtype=torch.int32)
        self._lens_cpu = torch.zeros(max_tokens, dtype=torch.int32)
        self._planned_num_tokens: int = -1
        # After CUDA-graph init, buffers are capture-stable; growing would
        # rebuild the wrapper and leave captured graphs on dangling pointers.
        self._capacity_frozen: bool = False

    def freeze_capacity(self) -> None:
        """Lock reserved-buffer sizes after ``init_cuda_graph_state``."""
        self._capacity_frozen = True

    def ensure_capacity(self, max_tokens: int, topk_width: int) -> None:
        """Grow reserved buffers before CUDA-graph capture. Fail closed after freeze."""
        if max_tokens <= self.max_tokens and topk_width <= self.topk_width:
            return
        if self._capacity_frozen:
            raise RuntimeError(
                "flashinfer_mla_sparse capacity is frozen after CUDA-graph init; "
                f"need max_tokens>={max_tokens}, topk_width>={topk_width}, "
                f"have max_tokens={self.max_tokens}, topk_width={self.topk_width}. "
                "Size buffers in init_cuda_graph_state; do not grow at runtime."
            )
        new_tokens = max(max_tokens, self.max_tokens)
        new_width = max(topk_width, self.topk_width)
        # Re-init in place (wrapper must be rebuilt with new reserved buffers).
        frozen = self._capacity_frozen
        self.__init__(
            self.workspace,
            num_heads=self.num_heads,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            sm_scale=self.sm_scale,
            q_dtype=self.q_dtype,
            kv_dtype=self.kv_dtype,
            max_tokens=new_tokens,
            topk_width=new_width,
            backend=self.backend,
        )
        self._capacity_frozen = frozen

    def plan(self, num_tokens: int, kv_lens: torch.Tensor) -> None:
        """Plan per-row KV lengths. Must run outside CUDA-graph capture.

        ``kv_lens`` is ``[num_tokens]`` (CPU or GPU int32). Host staging avoids
        an extra D2H inside FlashInfer when the tensor is already on CPU.
        """
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "flashinfer_mla_sparse plan() must run outside CUDA graph capture"
            )
        if num_tokens > self.max_tokens:
            self.ensure_capacity(num_tokens, self.topk_width)
        if num_tokens < 0 or num_tokens > self.max_tokens:
            raise ValueError(
                f"num_tokens={num_tokens} out of range for max_tokens={self.max_tokens}"
            )

        # Callers must pass CPU lengths (metadata host twin). Converting a GPU
        # tensor here would re-introduce per-step D2H on the decode replay path.
        if kv_lens.device.type != "cpu":
            raise ValueError(
                "flashinfer_mla_sparse plan() requires CPU kv_lens; "
                "build dsa_cache_seqlens_cpu in metadata (no per-step .cpu())."
            )
        kv_lens = (
            kv_lens[:num_tokens].to(dtype=torch.int32).clamp(min=0, max=self.topk_width)
        )

        # Full-size host buffers: use_cuda_graph plan does exact-size copy_.
        torch.clamp(self._arange_cpu, max=num_tokens, out=self._qo_cpu)
        self._lens_cpu.zero_()
        self._lens_cpu[:num_tokens] = kv_lens
        self._kv_cpu[0] = 0
        torch.cumsum(self._lens_cpu, dim=0, out=self._kv_cpu[1:])

        # Pass the full reserved kv_indices buffer (dense BatchMLA cuda-graph
        # replay style). Effective span is described by kv_len_arr / kv_indptr;
        # pack_indices() refreshes contents before run().
        self.wrapper.plan(
            self._qo_cpu,
            self._kv_cpu,
            self.kv_indices,
            self._lens_cpu,
            self.num_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            1,  # page_size: top-k slots are the page table
            False,  # causal: encoded by indexer selection
            self.sm_scale,
            self.q_dtype,
            self.kv_dtype,
        )
        self._planned_num_tokens = num_tokens

    def pack_indices(self, page_table: torch.Tensor) -> None:
        """Pack top-k page table into reserved ``kv_indices`` (graph-safe).

        Uses the ``kv_indptr`` refreshed by the latest ``plan()``.
        """
        num_tokens, width = page_table.shape
        if self._planned_num_tokens < 0:
            raise RuntimeError(
                "flashinfer_mla_sparse pack_indices() called before plan()"
            )
        if num_tokens != self._planned_num_tokens:
            raise RuntimeError(
                "flashinfer_mla_sparse pack_indices num_tokens "
                f"{num_tokens} != planned {self._planned_num_tokens}"
            )
        if width > self.topk_width:
            raise RuntimeError(
                f"flashinfer_mla_sparse page_table width {width} > "
                f"topk_width {self.topk_width}; bump capacity and re-plan"
            )

        page_table = page_table.to(dtype=torch.int32)
        if _TRITON_AVAILABLE and page_table.is_cuda:
            block = 256
            _pack_topk_indices_kernel[(num_tokens, triton.cdiv(width, block))](
                page_table,
                self.kv_indptr,
                self.kv_indices,
                width,
                block,
            )
            return

        # CPU / no-triton fallback (eager only; not used under CUDA graph).
        kv_lens = self.kv_len_arr[:num_tokens]
        if kv_lens.device.type != page_table.device.type:
            kv_lens = self._lens_cpu[:num_tokens]
        _, packed, _ = pack_topk_page_table_to_csr(page_table, kv_lens)
        if packed.numel() == 0:
            return
        self.kv_indices[: packed.numel()].copy_(packed)

    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run BatchMLA using the planned reserved buffers (no plan here)."""
        if self._planned_num_tokens < 0:
            raise RuntimeError("flashinfer_mla_sparse run() called before plan()")

        # DSA MLA key buffer is 3D: [num_pages, 1, dim] (page_size=1 token layout).
        if kv_cache.ndim == 2:
            flat = kv_cache.unsqueeze(1)
        else:
            flat = kv_cache
        ckv = flat[..., :v_head_dim]
        kpe = flat[..., v_head_dim:]
        if self.qk_rope_head_dim == 0:
            kpe = flat.new_empty(*flat.shape[:-1], 0)
            if q_pe.shape[-1] == 0:
                q_pe = q_nope.new_empty(q_nope.shape[0], q_nope.shape[1], 0)

        if out is None:
            out = q_nope.new_empty(q_nope.shape)
        return self.wrapper.run(q_nope, q_pe, ckv, kpe, out=out)
