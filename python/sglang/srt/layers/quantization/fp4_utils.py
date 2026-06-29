from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Dict, NamedTuple, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.utils.common import is_sm120_supported

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NVFP4 per-token quantize + grouped GEMM interface
# Kernels: miniTransformer fp4_gemm C++ module (temporary, to be replaced by
# sgl-kernel native implementations).
# ---------------------------------------------------------------------------

_MMA_TILE = 128
_FP4_BLOCK_K = 16
_FP4_DATA_DTYPE = torch.uint8
_FP4_SCALE_DTYPE = torch.float8_e4m3fn


def compute_padded_expert_offsets(
    expert_offsets: torch.Tensor,
    tile: int = _MMA_TILE,
) -> torch.Tensor:
    """Compute 128-row-padded cumulative row offsets from expert_offsets.

    Args:
        expert_offsets: [E+1] int32 GPU, cumulative token counts per expert.
        tile: MMA tile height for padding (default 128).

    Returns:
        padded_offsets: [E+1] int32 GPU.
    """
    m_raw = expert_offsets[1:] - expert_offsets[:-1]
    m_padded = ((m_raw + tile - 1) // tile) * tile
    padded = F.pad(torch.cumsum(m_padded, dim=0), (1, 0), value=0).to(torch.int32)
    return padded


def compute_act_scale_offsets(
    padded_offsets: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Compute activation blockscale offsets for a given hidden dimension k.

    Formula: scale_offsets[i+1] = scale_offsets[i] + m_padded[i] * (k // 16).

    Args:
        padded_offsets: [E+1] int32 GPU.
        k: hidden dimension (must be divisible by 16).

    Returns:
        scale_offsets: [E+1] int64 GPU.
    """
    m_padded = padded_offsets[1:] - padded_offsets[:-1]
    sf_per_expert = m_padded.to(torch.int64) * (k // _FP4_BLOCK_K)
    return F.pad(torch.cumsum(sf_per_expert, dim=0), (1, 0), value=0)


class NvFp4QuantResult(NamedTuple):
    data: torch.Tensor
    scale_flat: torch.Tensor
    global_scale: torch.Tensor


def nvfp4_quantize_pertoken(
    x: torch.Tensor,
    padded_offsets: torch.Tensor,
    unpadded_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
    total_padded: int,
    k: int,
) -> NvFp4QuantResult:
    """Grouped per-token NVFP4 quantization via miniTE C++ kernels.

    Args:
        x: [total_m, k] bf16 activations.
        padded_offsets: [E+1] int32 GPU, padded row cumsum.
        unpadded_offsets: [E+1] int32 GPU, unpadded row cumsum.
            Pass the *same object* as padded_offsets when x is already padded.
        scale_offsets: [E+1] int64 GPU.
        total_padded: total padded rows (sum of all padded group sizes).
        k: hidden dimension.

    Returns:
        NvFp4QuantResult(data, scale_flat, global_scale).
    """
    import fp4_gemm

    device = x.device
    data = torch.empty((total_padded, k // 2), dtype=_FP4_DATA_DTYPE, device=device)
    scale_flat_size = total_padded * (k // _FP4_BLOCK_K)
    scale_flat = torch.zeros(scale_flat_size, dtype=_FP4_SCALE_DTYPE, device=device)
    global_scale = torch.empty(total_padded, dtype=torch.float32, device=device)

    if padded_offsets is not unpadded_offsets:
        fp4_gemm.rtn_fp4_group_pertoken_pad_fuse(
            data, scale_flat, global_scale, x,
            padded_offsets, unpadded_offsets, scale_offsets, 1.0,
        )
    else:
        fp4_gemm.rtn_fp4_group_pertoken(
            data, scale_flat, global_scale, x,
            padded_offsets, scale_offsets, 1.0,
        )

    return NvFp4QuantResult(data, scale_flat, global_scale)


# ---------------------------------------------------------------------------
# Grouped GEMM v2 — persistent buffer cache + wrapper
# ---------------------------------------------------------------------------

_v2_buf_cache: Dict[Tuple[int, str], Tuple[torch.Tensor, ...]] = {}


def _get_v2_buffers(
    num_groups: int, device: torch.device
) -> Tuple[torch.Tensor, ...]:
    import fp4_gemm

    key = (num_groups, str(device))
    if key not in _v2_buf_cache:
        ptrs_b, strides_b, layouts_b = (
            fp4_gemm.grouped_cutlass_gemm_v2_buffer_sizes(num_groups)
        )
        _v2_buf_cache[key] = (
            torch.empty(ptrs_b, dtype=torch.uint8, device=device),
            torch.empty(strides_b, dtype=torch.uint8, device=device),
            torch.empty(layouts_b, dtype=torch.uint8, device=device),
            torch.empty(num_groups * 3, dtype=torch.int32, device=device),
            torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device),
        )
    return _v2_buf_cache[key]


_prob_host_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}


def get_prob_host(num_groups: int, max_m: int, n: int, k: int) -> torch.Tensor:
    """Return a cached CPU prob_host tensor for CUTLASS initialize().

    Only the first call to grouped_cutlass_gemm_v2 per (num_groups, device)
    actually reads prob_host (for workspace planning); subsequent calls use
    the cached CUTLASS Gemm object.  We use conservative max_m so the
    workspace is always large enough.
    """
    key = (num_groups, n, k)
    if key not in _prob_host_cache:
        _prob_host_cache[key] = torch.tensor(
            [[max_m, n, k]] * num_groups, dtype=torch.int32
        ).reshape(-1)
    return _prob_host_cache[key]


def nvfp4_grouped_gemm(
    out: torch.Tensor,
    act_data: torch.Tensor,
    wgt_data: torch.Tensor,
    act_sf: torch.Tensor,
    wgt_sf: torch.Tensor,
    a_rows: torch.Tensor,
    b_rows: torch.Tensor,
    a_sf_offsets: torch.Tensor,
    b_sf_offsets: torch.Tensor,
    num_groups: int,
    per_token_scale: torch.Tensor,
    prob_host: torch.Tensor,
) -> None:
    """Grouped FP4 GEMM v2 with per-token scale epilogue.

    Writes result into pre-allocated ``out``.
    """
    import fp4_gemm

    buf_ptrs, buf_strides, buf_layouts, buf_psizes, workspace = _get_v2_buffers(
        num_groups, out.device
    )
    fp4_gemm.grouped_cutlass_gemm_v2(
        out,
        act_data, wgt_data,
        act_sf, wgt_sf,
        a_rows, b_rows,
        a_sf_offsets, b_sf_offsets,
        num_groups,
        buf_ptrs, buf_strides, buf_layouts, buf_psizes,
        prob_host, workspace, per_token_scale,
    )


# ---------------------------------------------------------------------------
# Online (dynamic) NVFP4 input-scale helpers
# ---------------------------------------------------------------------------

_FP4_E2M1_MAX = 6.0
_nvfp4_online_scale: bool | None = None


def nvfp4_online_scale_enabled() -> bool:
    global _nvfp4_online_scale
    if _nvfp4_online_scale is None:
        from sglang.srt.environ import envs

        _nvfp4_online_scale = envs.SGLANG_NVFP4_ONLINE_SCALE.get()
    return _nvfp4_online_scale


def nvfp4_compute_input_scale_and_inv(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    amax = x.abs().max()
    input_scale = amax / _FP4_E2M1_MAX
    input_scale_inv = _FP4_E2M1_MAX / amax.clamp(min=1e-12)
    return input_scale, input_scale_inv


class Fp4GemmRunnerBackend(Enum):
    """Enum for FP4 GEMM runner backend selection."""

    AUTO = "auto"
    CUTLASS = "cutlass"
    FLASHINFER_CUDNN = "flashinfer_cudnn"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_TRTLLM = "flashinfer_trtllm"

    def is_auto(self) -> bool:
        return self == Fp4GemmRunnerBackend.AUTO

    def is_cutlass(self) -> bool:
        return self == Fp4GemmRunnerBackend.CUTLASS

    def is_flashinfer_cudnn(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUDNN

    def is_flashinfer_cutlass(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUTLASS

    def is_flashinfer_trtllm(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_TRTLLM

    def is_flashinfer(self) -> bool:
        return self.value.startswith("flashinfer_")

    def get_flashinfer_backend(self) -> str:
        """Get the backend string to pass to FlashInfer's mm_fp4 API.

        This remaps SGLang's user-facing backend names to FlashInfer's API names.
        Examples:
            'flashinfer_trtllm' -> 'trtllm'
            'flashinfer_cutlass' -> 'cutlass'
            'flashinfer_cudnn' -> 'cudnn'
        """
        if self.value.startswith("flashinfer_"):
            return self.value.removeprefix("flashinfer_")
        else:
            return self.value


FP4_GEMM_RUNNER_BACKEND: Fp4GemmRunnerBackend | None = None


def initialize_fp4_gemm_config(server_args: ServerArgs) -> None:
    """Initialize FP4 GEMM configuration from server args."""
    global FP4_GEMM_RUNNER_BACKEND

    backend = server_args.fp4_gemm_runner_backend
    if backend == "auto":
        if is_sm120_supported():
            # flashinfer_cutlass produces NaN in dense MLP layers with
            # heterogeneous batches on SM120 (Blackwell).  cudnn is stable.
            # See: https://github.com/sgl-project/sglang/issues/20043
            backend = "flashinfer_cudnn"
            logger.info(
                "SM120 (Blackwell) detected: auto-selecting "
                "fp4-gemm-backend=flashinfer_cudnn"
            )
        else:
            backend = "flashinfer_cutlass"

    FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend(backend)


def get_fp4_gemm_runner_backend() -> Fp4GemmRunnerBackend:
    """Get the current FP4 GEMM runner backend."""
    global FP4_GEMM_RUNNER_BACKEND
    if FP4_GEMM_RUNNER_BACKEND is None:
        FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend.AUTO
    return FP4_GEMM_RUNNER_BACKEND
