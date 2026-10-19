"""Fused Hadamard-128 (FWHT) + per-row FP8 ue8m0 quant for DSA IndexerKPool query.

Ported from vLLM ``vllm.models.glm5next.nvidia.ops.kpool_compress.fwht128_quant_fp8``
(``_fwht_quant_kernel``): one Triton launch does fp32 butterflies + absmax FP8,
avoiding the bf16 intermediate that ``act_quant(rotate_activation(q))`` materializes.

Unlike AMD PR #44661 (``fused_hadamard_act_quant``, Sylvester matmul + optional
weight fold, gfx950-gated), this keeps vLLM's in-register FWHT and leaves
head-gate scaling to the caller.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

INDEX_HEAD_DIM = 128


@triton.jit
def _fwht_stage(x, N: tl.constexpr, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (N,))


@triton.jit
def _fwht_quant_kernel(
    q_ptr,
    qout_ptr,
    sout_ptr,
    n_rows,
    BLOCK_R: tl.constexpr,
):
    """Fused Hadamard-128 rotation + per-row absmax FP8 (ue8m0) quant."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < n_rows
    offs = tl.arange(0, 128)
    x = tl.load(
        q_ptr + rows[:, None] * 128 + offs[None, :], mask=rmask[:, None], other=0.0
    ).to(tl.float32)

    N: tl.constexpr = BLOCK_R * 128
    x = tl.reshape(x, (N,))
    x = _fwht_stage(x, N, BLOCK_R * 64, 1)
    x = _fwht_stage(x, N, BLOCK_R * 32, 2)
    x = _fwht_stage(x, N, BLOCK_R * 16, 4)
    x = _fwht_stage(x, N, BLOCK_R * 8, 8)
    x = _fwht_stage(x, N, BLOCK_R * 4, 16)
    x = _fwht_stage(x, N, BLOCK_R * 2, 32)
    x = _fwht_stage(x, N, BLOCK_R, 64)
    x = x * 0.08838834764831845  # 1/sqrt(128)

    # Match unfused path: materialize as bf16 before quantizing.
    x = x.to(tl.bfloat16).to(tl.float32)
    x = tl.reshape(x, (BLOCK_R, 128))

    absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    scale = tl.exp2(tl.ceil(tl.log2(absmax * (1.0 / 448.0))))
    y = tl.minimum(tl.maximum(x / scale[:, None], -448.0), 448.0)

    tl.store(qout_ptr + rows[:, None] * 128 + offs[None, :], y, mask=rmask[:, None])
    tl.store(sout_ptr + rows, scale, mask=rmask)


def fwht128_quant_fp8(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate each 128-wide row by FWHT-128, then FP8-quant (ue8m0).

    Args:
        q: ``[..., 128]`` contiguous bf16 (e.g. ``[tokens, heads, 128]``).

    Returns:
        ``(q_fp8, scale)`` with ``q_fp8`` same shape as ``q`` (float8_e4m3fn)
        and ``scale`` shape ``[..., 1]`` float32.
    """
    assert q.dtype == torch.bfloat16, q.dtype
    assert q.is_contiguous(), "input must be contiguous"
    assert q.shape[-1] == INDEX_HEAD_DIM, q.shape

    leading = q.shape[:-1]
    n_rows = q.numel() // INDEX_HEAD_DIM
    q2 = q.view(n_rows, INDEX_HEAD_DIM)
    q_fp8 = torch.empty(
        (n_rows, INDEX_HEAD_DIM), dtype=torch.float8_e4m3fn, device=q.device
    )
    q_scale = torch.empty((n_rows, 1), dtype=torch.float32, device=q.device)
    if n_rows == 0:
        return q_fp8.view(*leading, INDEX_HEAD_DIM), q_scale.view(*leading, 1)

    BLOCK_R = 32
    _fwht_quant_kernel[(triton.cdiv(n_rows, BLOCK_R),)](
        q2,
        q_fp8,
        q_scale,
        n_rows,
        BLOCK_R=BLOCK_R,
        num_warps=2,
    )
    return q_fp8.view(*leading, INDEX_HEAD_DIM), q_scale.view(*leading, 1)
