import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.cp.base import (
    CPAttentionBackendKind,
    init_cp_strategy,
)
from sglang.srt.layers.cp.interleave_kpool import (
    local_query_inputs,
    materialize_compression_inputs,
)
from sglang.srt.layers.cp.kda_kcp import (
    merge_affine_segment_states,
    pack_summary_buffer_elems,
    summary_ag_bytes,
)
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy, zigzag_rows_per_request
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _ExtendMode:
    def is_context_parallel_extend(self):
        return True


class TestGlmZigzagDsaKpoolOrder(CustomTestCase):
    def tearDown(self):
        init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="zigzag")

    def test_zigzag_rows_per_request_cp2(self):
        # N=8, C=2 → 4 blocks of 2. Rank0 owns blocks 0+3 → 4 tokens; rank1 → 4.
        self.assertEqual(zigzag_rows_per_request([8], 0, 2), [4])
        self.assertEqual(zigzag_rows_per_request([8], 1, 2), [4])
        # Uneven: N=10 → blocks [3,3,2,2]. Rank0: 3+2=5; rank1: 3+2=5.
        self.assertEqual(zigzag_rows_per_request([10], 0, 2), [5])
        self.assertEqual(zigzag_rows_per_request([10], 1, 2), [5])

    def test_zigzag_declares_dsa_backend(self):
        backends = ZigzagCPStrategy(cp_size=2).get_supported_attention_backend()
        self.assertIn(CPAttentionBackendKind.DSA, backends)

    def test_zigzag_shard_per_request_filters_empty(self):
        init_cp_strategy(enable_prefill_cp=True, cp_size=2, cp_strategy="zigzag")
        strategy = ZigzagCPStrategy(cp_size=2)
        extend = torch.tensor([8, 8], dtype=torch.int32)
        with get_parallel().override(attn_cp_rank=0, attn_cp_size=2):
            q_cpu, q, bs_cpu, bs = strategy.shard_per_request([8, 8], extend)
        self.assertEqual(q_cpu, [4, 4])
        self.assertEqual(bs_cpu, [0, 1])
        self.assertTrue(torch.equal(q, torch.tensor([4, 4], dtype=torch.int32)))

    def test_materialize_compression_inputs_uses_strategy_gather(self):
        init_cp_strategy(enable_prefill_cp=True, cp_size=2, cp_strategy="zigzag")
        strategy = ZigzagCPStrategy(cp_size=2)
        metadata = strategy.build_metadata(8, [8], [8])
        key = torch.arange(4, dtype=torch.float32).view(4, 1)
        score = key + 10
        positions = torch.arange(4)
        forward_batch = SimpleNamespace(
            attn_cp_metadata=metadata,
            forward_mode=_ExtendMode(),
            positions=positions,
            extend_seq_lens_cpu=[8],
        )
        packed_full = torch.arange(16, dtype=torch.float32).view(8, 2)
        with (
            patch(
                "sglang.srt.layers.cp.interleave_kpool.get_cp_strategy",
                return_value=strategy,
            ),
            patch.object(
                strategy, "gather_kv_cache", return_value=packed_full
            ) as gather,
        ):
            out_key, out_score, _ = materialize_compression_inputs(
                key, score, positions, forward_batch
            )
        gather.assert_called_once()
        self.assertTrue(torch.equal(out_key, packed_full[:, :1]))
        self.assertTrue(torch.equal(out_score, packed_full[:, 1:]))

    def test_local_query_inputs_zigzag_counts(self):
        init_cp_strategy(enable_prefill_cp=True, cp_size=2, cp_strategy="zigzag")
        strategy = ZigzagCPStrategy(cp_size=2)
        metadata = strategy.build_metadata(8, [8], [8])
        metadata.real_page_table = object()
        metadata.dsa_seqlens_expanded = object()
        metadata.dsa_extend_seq_lens_list = [4]
        metadata.indexer_seq_lens_cpu = torch.tensor([8])
        forward_batch = SimpleNamespace(
            attn_cp_metadata=metadata,
            forward_mode=_ExtendMode(),
            extend_seq_lens_cpu=[8],
            req_pool_indices=torch.tensor([3]),
        )
        with patch(
            "sglang.srt.layers.cp.interleave_kpool.get_parallel",
            return_value=SimpleNamespace(attn_cp_rank=0, attn_cp_size=2),
        ):
            out = local_query_inputs(forward_batch, metadata)
        self.assertIn("local_req_pool_indices", out)
        self.assertTrue(torch.equal(out["local_req_pool_indices"], torch.tensor([3])))

    def test_kcp_affine_merge_matches_serial_scan(self):
        # Toy state: S_out = b + m * S_in  (1x1 matrices)
        s_ext = torch.tensor([1.0, 3.0, 4.0]).view(3, 1, 1)
        m = torch.tensor([0.5, 0.8, 0.2]).view(3, 1, 1)
        s_in = merge_affine_segment_states(s_ext, m)
        self.assertAlmostEqual(float(s_in[0]), 0.0)
        self.assertAlmostEqual(float(s_in[1]), 1.0)
        self.assertAlmostEqual(float(s_in[2]), 3.8)

    def test_summary_buffer_bytes_cp8(self):
        elems = pack_summary_buffer_elems(8, 1, 64, 128, 128)
        self.assertEqual(elems, 2 * 8 * 1 * 64 * 256 * 128)
        self.assertEqual(summary_ag_bytes(8, 1, 64, 128, 128), 64 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
