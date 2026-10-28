"""DSA sparse path via FlashInfer ``BatchMLAPagedAttentionWrapper`` (SM90 FA3).

Mirrors vLLM ``FLASHINFER_MLA_SPARSE_SM90``:

- ``page_size=1`` top-k slots are the page table; ``causal=False``
- ``plan()`` runs outside CUDA-graph capture (metadata build / replay)
- ``pack_indices()`` + ``run()`` are capture-safe and update reserved buffers
- FP8 KV:
  - **G=1 packed** (preferred for DSA): row
    ``[kv_lora e4m3 | fp32 scale | 12B pad]`` (=528 for 512-d NoPE) fed via
    ``run_packed_kv`` (zero-copy in-row scale; no contiguous scale gather).
  - **raw E4M3** (legacy / vLLM-style): last dim ``kv_lora(+rope)`` +
    per-tensor ``ckv_scale`` / ``kpe_scale``.
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

_FP8_KV_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)

# G=1 packed scale slot: 4B live fp32 + 12B pad (16B aligned).
_G1_SCALE_SLOT_BYTES = 16


def normalize_batch_mla_kv_dtype(kv_dtype: torch.dtype) -> torch.dtype:
    """FlashInfer plan rejects ``uint8``; map storage dtype to logical E4M3.

    Same fix as vLLM #55222 for ``FLASHINFER_MLA_SPARSE_SM90``.
    """
    if kv_dtype == torch.uint8:
        return torch.float8_e4m3fn
    return kv_dtype


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
        packed_fp8_kv: bool = False,
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
        # plan()/run() need logical E4M3; pool may store uint8 (vLLM #55222).
        self.kv_dtype = normalize_batch_mla_kv_dtype(kv_dtype)
        self.use_fp8_kv_cache = self.kv_dtype in _FP8_KV_DTYPES
        self.packed_fp8_kv = bool(packed_fp8_kv) and self.use_fp8_kv_cache
        self.backend = backend
        self.max_tokens = max_tokens
        self.topk_width = topk_width
        # G=1 packed row width: nope + 16B scale slot (+ rope bf16 bytes if any).
        self.packed_row_bytes = (
            kv_lora_rank
            + _G1_SCALE_SLOT_BYTES
            + qk_rope_head_dim * 2  # rope stored bf16 in DSA pool
        )

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
        # Pin so wrapper.plan()'s reserved-buffer copy_ is Pinned→Device DMA
        # (not Pageable→Device). Same pattern as dense flashinfer_mla_backend
        # fast_plan_*_cpu buffers.
        pin = torch.cuda.is_available()
        self._arange_cpu = torch.arange(
            max_tokens + 1, dtype=torch.int32, device="cpu", pin_memory=pin
        )
        self._qo_cpu = torch.empty(
            max_tokens + 1, dtype=torch.int32, device="cpu", pin_memory=pin
        )
        self._kv_cpu = torch.empty(
            max_tokens + 1, dtype=torch.int32, device="cpu", pin_memory=pin
        )
        self._lens_cpu = torch.zeros(
            max_tokens, dtype=torch.int32, device="cpu", pin_memory=pin
        )
        self._planned_num_tokens: int = -1
        self._last_kv_lens: Optional[torch.Tensor] = None
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
            packed_fp8_kv=self.packed_fp8_kv,
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

        # Steady-state shortcut: the C++ plan is a pure function of
        # (num_tokens, kv_lens) — a greedy host-side cluster scheduling. For
        # DSA sparse decode/verify/draft_extend every active row clamps to
        # the same topk width once contexts exceed it, so an unchanged
        # (num_tokens, kv_lens) replans to the identical schedule. Skip the
        # host scheduling, the H2D buffer DMAs and the C++ call; the reserved
        # buffers and _plan_info still hold exactly the right values. Rows
        # with shorter contexts (new requests) change kv_lens and take the
        # full path. The equality check is a few us over a small CPU tensor.
        if (
            self._planned_num_tokens == num_tokens
            and self._last_kv_lens is not None
            and self._last_kv_lens.shape[0] == num_tokens
            and torch.equal(self._last_kv_lens, kv_lens)
        ):
            return

        # qo_indptr is clamp(arange), a pure function of num_tokens — rebuild
        # only when the span changes. The C++ plan infers the span from the
        # qo_indptr contents, so only the first num_tokens entries of
        # kv_len_arr and the first num_tokens+1 of kv_indptr are ever read;
        # the tail of the reserved buffers is dead weight. Skip the
        # full-width zero_/cumsum (over up to max_tokens entries) in favor of
        # exact-span writes — CPU time scales with num_tokens, not capacity.
        if num_tokens != self._planned_num_tokens:
            torch.clamp(self._arange_cpu, max=num_tokens, out=self._qo_cpu)
        self._lens_cpu[:num_tokens] = kv_lens
        self._kv_cpu[0] = 0
        torch.cumsum(
            self._lens_cpu[:num_tokens], dim=0, out=self._kv_cpu[1 : 1 + num_tokens]
        )

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
        self._last_kv_lens = kv_lens.clone()
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
        ckv_scale: Optional[float] = None,
        kpe_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Run BatchMLA using the planned reserved buffers (no plan here).

        FP8 KV:
        - G=1 packed rows → ``run_packed_kv`` (in-row scale at offset
          ``kv_lora_rank``; no contiguous scale gather).
        - raw E4M3 rows → view + per-tensor ``ckv_scale`` / ``kpe_scale``
          (vLLM ``forward_mqa`` style).
        """
        if self._planned_num_tokens < 0:
            raise RuntimeError("flashinfer_mla_sparse run() called before plan()")

        # DSA MLA key buffer is 3D: [num_pages, 1, dim] (page_size=1 token layout).
        if kv_cache.ndim == 2:
            flat = kv_cache.unsqueeze(1)
        else:
            flat = kv_cache

        if out is None:
            out = q_nope.new_empty(q_nope.shape)

        if self.use_fp8_kv_cache or flat.dtype in (
            torch.uint8,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ):
            if flat.dtype == torch.uint8:
                flat = flat.view(torch.float8_e4m3fn)
            elif flat.dtype not in _FP8_KV_DTYPES:
                raise TypeError(
                    "flashinfer_mla_sparse FP8 path expected uint8/float8_e4m3 "
                    f"KV cache, got dtype={flat.dtype}"
                )

            last_dim = flat.shape[-1]
            raw_dim = self.kv_lora_rank + self.qk_rope_head_dim
            packed_dim = self.packed_row_bytes

            if last_dim == packed_dim or self.packed_fp8_kv:
                return self._run_packed_fp8(q_nope, q_pe, flat, out=out)

            if last_dim != raw_dim:
                raise ValueError(
                    "flashinfer_mla_sparse FP8 expected raw MLA dim "
                    f"{raw_dim} or G=1 packed dim {packed_dim}; got "
                    f"last dim={last_dim}."
                )

            ckv = flat[..., :v_head_dim]
            kpe = flat[..., v_head_dim:]
            if self.qk_rope_head_dim == 0:
                kpe = flat.new_empty(*flat.shape[:-1], 0)
                if q_pe.shape[-1] == 0:
                    q_pe = q_nope.new_empty(q_nope.shape[0], q_nope.shape[1], 0)

            return self.wrapper.run(
                q_nope,
                q_pe,
                ckv,
                kpe,
                out=out,
                ckv_scale=float(1.0 if ckv_scale is None else ckv_scale),
                kpe_scale=float(1.0 if kpe_scale is None else kpe_scale),
            )

        ckv = flat[..., :v_head_dim]
        kpe = flat[..., v_head_dim:]
        if self.qk_rope_head_dim == 0:
            kpe = flat.new_empty(*flat.shape[:-1], 0)
            if q_pe.shape[-1] == 0:
                q_pe = q_nope.new_empty(q_nope.shape[0], q_nope.shape[1], 0)

        return self.wrapper.run(q_nope, q_pe, ckv, kpe, out=out)

    def _run_packed_fp8(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        flat: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """G=1 packed path: zero-copy in-row scale via ``run_packed_kv``."""
        if flat.shape[-1] != self.packed_row_bytes:
            raise ValueError(
                "flashinfer_mla_sparse G=1 packed FP8 expected last dim "
                f"{self.packed_row_bytes} "
                f"(=kv_lora {self.kv_lora_rank} + 16B scale slot"
                f"{'' if self.qk_rope_head_dim == 0 else f' + rope*{self.qk_rope_head_dim}*2'}); "
                f"got {flat.shape[-1]}."
            )
        if not hasattr(self.wrapper, "run_packed_kv"):
            raise RuntimeError(
                "flashinfer_mla_sparse G=1 packed FP8 requires FlashInfer "
                "BatchMLAPagedAttentionWrapper.run_packed_kv "
                "(glm5.3-flash / squack G=1 build)."
            )

        # Rope stays BF16 in the DSA pool tail; FlashInfer G=1 kpe expects FP8
        # sharing the same per-token scale. GLM-5.3-Flash DSA is NoPE
        # (qk_rope_head_dim=0). Fail closed if rope is present until a FP8
        # rope-tail store path exists.
        if self.qk_rope_head_dim != 0:
            raise NotImplementedError(
                "flashinfer_mla_sparse G=1 packed FP8 currently supports "
                "NoPE only (qk_rope_head_dim=0). Rope bf16 tail in the packed "
                "row is not yet re-quantized for run_packed_kv."
            )

        if q_pe.shape[-1] == 0:
            q_pe = q_nope.new_empty(q_nope.shape[0], q_nope.shape[1], 0)

        return self.wrapper.run_packed_kv(
            q_nope,
            q_pe,
            flat,
            scale_offset=self.kv_lora_rank,
            out=out,
        )
