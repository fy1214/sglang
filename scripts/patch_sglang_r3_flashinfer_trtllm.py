#!/usr/bin/env python3
"""Enable R3 routed-experts capture for flashinfer_trtllm NVFP4 MoE.

Surgically patches sglang in place (no whole-file overwrite):

- ``topk.py`` / ``flashinfer_trtllm.py``: unified diff via ``patch -p1``
- ``fused_moe_triton/layer.py``: idempotent string replacements (robust to line drift)

Typical container usage::

    python3 /sgl-workspace/sglang/scripts/patch_sglang_r3_flashinfer_trtllm.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PATCH_FILE = SCRIPT_DIR / "patches" / "r3_flashinfer_trtllm_core.patch"
DEFAULT_ROOT = Path("/sgl-workspace/sglang")
MARKER = "def _accepts_topk_output(topk_output: TopKOutput) -> bool:"

LAYER_PATCHES: list[tuple[str, str]] = [
    (
        """    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        assert TopKOutputChecker.format_is_bypassed(
            topk_output
        ), "Only bypassed topk output is supported for flashinfer fp4 moe"

        if is_in_piecewise_cuda_graph():
            return flashinfer_fp4_moe_forward_piecewise_cuda_graph_impl(
                hidden_states,
                topk_output.router_logits,
                topk_output.topk_config.top_k,
                topk_output.topk_config.topk_group,
                topk_output.topk_config.num_expert_group,
                topk_output.topk_config.correction_bias,
                self.layer_id,
            )
        else:
            return self.forward_impl(hidden_states, topk_output)
""",
        """    @staticmethod
    def _accepts_topk_output(topk_output: TopKOutput) -> bool:
        if TopKOutputChecker.format_is_bypassed(topk_output):
            return True
        return (
            TopKOutputChecker.format_is_standard(topk_output)
            and get_global_server_args().enable_return_routed_experts
        )

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        assert self._accepts_topk_output(topk_output), (
            "Only bypassed topk output is supported for flashinfer fp4 moe, "
            "unless --enable-return-routed-experts is set"
        )

        if is_in_piecewise_cuda_graph():
            if TopKOutputChecker.format_is_bypassed(topk_output):
                return flashinfer_fp4_moe_forward_piecewise_cuda_graph_impl(
                    hidden_states,
                    topk_output.router_logits,
                    topk_output.topk_config.top_k,
                    topk_output.topk_config.topk_group,
                    topk_output.topk_config.num_expert_group,
                    topk_output.topk_config.correction_bias,
                    self.layer_id,
                )
            return self.forward_impl(hidden_states, topk_output)
        else:
            return self.forward_impl(hidden_states, topk_output)
""",
    ),
    (
        """            topk_output: TopKOutput object with Bypassed format
        """,
        """            topk_output: TopKOutput object with Bypassed or Standard (R3) format
        """,
    ),
    (
        """        ), "Only gated MoEs are supported for flashinfer fp4 moe"

        assert TopKOutputChecker.format_is_bypassed(topk_output)

        if (
            NVFP4_PERTOKEN_SCALE
            and get_moe_runner_backend().is_flashinfer_trtllm()
            and hasattr(self, "gemm1_weights_fp4_shuffled")
            and hasattr(self, "pertoken_g1_scale_c")
        ):
            return FusedMoE.forward_impl(self, hidden_states, topk_output)

        router_logits = topk_output.router_logits
""",
        """        ), "Only gated MoEs are supported for flashinfer fp4 moe"

        if (
            NVFP4_PERTOKEN_SCALE
            and get_moe_runner_backend().is_flashinfer_trtllm()
            and hasattr(self, "gemm1_weights_fp4_shuffled")
            and hasattr(self, "pertoken_g1_scale_c")
        ):
            return FusedMoE.forward_impl(self, hidden_states, topk_output)

        assert TopKOutputChecker.format_is_bypassed(topk_output)

        router_logits = topk_output.router_logits
""",
    ),
]


def _layer_file(root: Path) -> Path:
    return root / "python/sglang/srt/layers/moe/fused_moe_triton/layer.py"


def is_layer_applied(root: Path) -> bool:
    layer = _layer_file(root)
    return layer.exists() and MARKER in layer.read_text()


def is_core_applied(root: Path) -> bool:
    topk = root / "python/sglang/srt/layers/moe/topk.py"
    flashinfer = (
        root / "python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py"
    )
    if not topk.exists() or not flashinfer.exists():
        return False
    return (
        "enable_return_routed_experts" in topk.read_text()
        and "use_routed_topk" in flashinfer.read_text()
    )


def apply_layer_patches(root: Path) -> str:
    if is_layer_applied(root):
        return "layer already applied"

    layer = _layer_file(root)
    source = layer.read_text()
    for idx, (old, new) in enumerate(LAYER_PATCHES, start=1):
        if new in source:
            continue
        if old not in source:
            raise RuntimeError(
                f"R3_FLASHINFER_TRTLLM_LAYER_PATCH block {idx}: "
                "could not find expected source block"
            )
        source = source.replace(old, new, 1)

    if not is_layer_applied_text(source):
        raise RuntimeError(
            "R3_FLASHINFER_TRTLLM_LAYER_PATCH: verification failed after edits"
        )
    layer.write_text(source)
    return "layer applied"


def is_layer_applied_text(source: str) -> bool:
    return MARKER in source


def apply_core_patch(root: Path, *, dry_run: bool = False) -> str:
    if is_core_applied(root):
        return "core already applied"
    if not PATCH_FILE.exists():
        raise FileNotFoundError(f"patch file not found: {PATCH_FILE}")

    cmd = ["patch", "-p1", "--forward"]
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend(["-i", str(PATCH_FILE)])

    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if result.returncode != 0 and "previously applied" not in result.stdout:
        # patch may partially apply; verify by marker instead of return code
        if not is_core_applied(root):
            raise RuntimeError(
                "R3_FLASHINFER_TRTLLM_CORE_PATCH failed:\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
    if dry_run:
        return "core dry-run ok"
    if not is_core_applied(root):
        raise RuntimeError("R3_FLASHINFER_TRTLLM_CORE_PATCH verification failed")
    return "core applied"


def apply(root: Path, *, dry_run: bool = False) -> list[str]:
    statuses: list[str] = []
    statuses.append(apply_core_patch(root, dry_run=dry_run))
    if not dry_run:
        statuses.append(apply_layer_patches(root))
    elif is_layer_applied(root):
        statuses.append("layer already applied")
    else:
        statuses.append("layer pending")
    return statuses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"sglang source root (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run core patch --dry-run without modifying files",
    )
    args = parser.parse_args()

    statuses = apply(args.root, dry_run=args.dry_run)
    print(
        "R3_FLASHINFER_TRTLLM_PATCH: "
        + "; ".join(statuses)
        + f" ({args.root})"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"R3_FLASHINFER_TRTLLM_PATCH: error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
