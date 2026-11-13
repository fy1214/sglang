"""Gates for colocated CP-TP group sharing + DCP on GLM-5.3-Flash."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.cp_tp_group_sharing_hook import (
    resolve_cp_tp_group_sharing,
)
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.layers.cp.cp_decode_attn_tp import CP_DECODE_ATTN_TP_SUPPORTED_ARCHS
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCpTpSharingDcpGates(CustomTestCase):
    def test_glm5_next_is_on_decode_attn_tp_whitelist(self):
        self.assertIn(
            "Glm5NextForConditionalGeneration", CP_DECODE_ATTN_TP_SUPPORTED_ARCHS
        )
        self.assertIn(
            "Glm5NextForConditionalGenerationNextN",
            CP_DECODE_ATTN_TP_SUPPORTED_ARCHS,
        )

    def test_cp_tp_group_sharing_allows_dcp(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_prefill_cp=True,
            cp_strategy="interleave",
            attn_cp_size=8,
            tp_size=8,
            dcp_size=8,
            moe_dp_size=1,
            disaggregation_mode="null",
            enable_mixed_chunk=False,
            enable_dsa_cache_layer_split=False,
        )
        model = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Glm5NextForConditionalGeneration"]
            )
        )
        linear_cfg = SimpleNamespace(
            linear_num_key_heads=16,
            linear_num_value_heads=16,
        )

        with patch(
            "sglang.srt.arg_groups.cp_tp_group_sharing_hook.mambaish_config",
            return_value=linear_cfg,
        ):
            resolve_cp_tp_group_sharing(server_args, model)

        self.assertTrue(resolved_view(server_args).enable_cp_tp_group_sharing)
        self.assertEqual(server_args.dcp_size, 8)


if __name__ == "__main__":
    unittest.main()
