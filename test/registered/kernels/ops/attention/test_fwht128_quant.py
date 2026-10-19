"""Correctness: fused FWHT+FP8 vs production rotate_activation + act_quant."""

from __future__ import annotations

import unittest

import torch

from sglang.kernels.ops.attention.dsa.fwht128_quant import fwht128_quant_fp8
from sglang.kernels.ops.attention.dsa.triton_kernel import act_quant
from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFwht128QuantFp8(CustomTestCase):
    def _input(self, rows: int, heads: int, seed: int) -> torch.Tensor:
        g = torch.Generator(device="cuda").manual_seed(seed)
        return (
            torch.randn(
                rows, heads, 128, device="cuda", dtype=torch.bfloat16, generator=g
            )
            * 0.3
        ).contiguous()

    def test_matches_two_pass_ue8m0(self):
        for rows in (1, 64, 4096):
            for heads in (16, 32):
                for seed in (0, 41):
                    with self.subTest(rows=rows, heads=heads, seed=seed):
                        x = self._input(rows, heads, seed)
                        y, s = fwht128_quant_fp8(x)
                        y_ref, s_ref = act_quant(
                            rotate_activation(x), 128, scale_fmt="ue8m0"
                        )
                        torch.testing.assert_close(s, s_ref, rtol=0, atol=0)
                        dequant = y.float() * s
                        dequant_ref = y_ref.float() * s_ref
                        cos = torch.nn.functional.cosine_similarity(
                            dequant.flatten(), dequant_ref.flatten(), dim=0
                        ).item()
                        self.assertGreater(cos, 0.999)
                        exact = (
                            (y.view(torch.uint8) == y_ref.view(torch.uint8))
                            .float()
                            .mean()
                            .item()
                        )
                        self.assertGreater(exact, 0.95)


if __name__ == "__main__":
    unittest.main()
