from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_glm5_router_gemm_module() -> Module:
    return load_jit(
        "glm5_router_gemm",
        cuda_files=["gemm/glm5_router_gemm.cuh"],
        cuda_wrappers=[("glm5_router_gemm", "glm5_router_gemm")],
    )


@debug_kernel_api
def glm5_router_gemm(
    hidden_states: torch.Tensor, router_weight: torch.Tensor
) -> torch.Tensor:
    """BF16 [M, 6144] x BF16 [256, 6144]^T -> FP32 [M, 256]."""
    output = torch.empty(
        (hidden_states.shape[0], 256),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    _jit_glm5_router_gemm_module().glm5_router_gemm(
        output, hidden_states, router_weight
    )
    return output
