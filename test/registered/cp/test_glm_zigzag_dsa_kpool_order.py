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
    _apply_transition,
    merge_affine_segment_states,
    pack_summary_buffer_elems,
    summary_ag_bytes,
)
from sglang.srt.layers.cp.zigzag import (
    ZigzagCPStrategy,
    cp_sequence_to_zigzag_gathered_order,
    cp_zigzag_gathered_to_sequence_order,
    zigzag_rows_per_request,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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

    def test_kcp_affine_merge_real_head_shape(self):
        # Flash-like [H,dk,dv] with matched M=[H,dk,dk]
        h, dk, dv, slots = 4, 3, 2, 3
        torch.manual_seed(0)
        s_ext = torch.randn(slots, h, dk, dv)
        m = torch.randn(slots, h, dk, dk)
        s_in = merge_affine_segment_states(s_ext, m)
        cur = torch.zeros(h, dk, dv)
        for i in range(slots):
            self.assertTrue(torch.allclose(s_in[i], cur))
            cur = s_ext[i] + torch.matmul(m[i], cur)

    def test_apply_transition_head_free_m(self):
        h, dk, dv = 4, 3, 2
        m = torch.eye(dk)
        state = torch.randn(h, dk, dv)
        out = _apply_transition(m, state)
        self.assertTrue(torch.allclose(out, state))
        with self.assertRaises(ValueError):
            # Wrong: would have been the old broken matmul([dk,dk],[H,dk,dv]) path
            # if shapes were accepted without einsum — force unsupported ndim combo.
            _apply_transition(torch.randn(dk, dk, dv), state)

    def test_zigzag_gathered_sequence_order_roundtrip(self):
        """KDA path: boundary AG layout → chrono → back must be identity on logical rows."""
        cp_size = 2
        init_cp_strategy(enable_prefill_cp=True, cp_size=cp_size, cp_strategy="zigzag")
        strategy = ZigzagCPStrategy(cp_size=cp_size)
        n = 8
        chrono = torch.arange(n, dtype=torch.float32).unsqueeze(-1)
        # Build padded rank-major AG layout like attn_cp_interleave_gather.
        rank_shards = []
        meta0 = None
        for rank in range(cp_size):
            with get_parallel().override(attn_cp_rank=rank, attn_cp_size=cp_size):
                meta = strategy.build_metadata(n, [n], [n])
                if rank == 0:
                    meta0 = meta
                fb = SimpleNamespace(attn_cp_metadata=meta, forward_mode=_ExtendMode())
                local = strategy.shard_hidden_states(chrono, fb)
            physical = max(meta.per_rank_actual_token)
            logical = (meta.per_rank_logical_token or meta.per_rank_actual_token)[rank]
            piece = local[:logical]
            if piece.shape[0] < physical:
                piece = torch.nn.functional.pad(
                    piece, [0, 0, 0, physical - piece.shape[0]]
                )
            rank_shards.append(piece)
        gathered = torch.cat(rank_shards, dim=0)
        physical = max(meta0.per_rank_actual_token)
        per_rank = meta0.per_rank_logical_token or meta0.per_rank_actual_token
        forward_batch = SimpleNamespace(
            attn_cp_metadata=meta0, forward_mode=_ExtendMode()
        )

        with get_parallel().override(attn_cp_rank=0, attn_cp_size=cp_size):
            restored = cp_zigzag_gathered_to_sequence_order(gathered, forward_batch)
            back = cp_sequence_to_zigzag_gathered_order(
                restored, forward_batch, gathered.shape[0]
            )
        self.assertTrue(torch.equal(restored.squeeze(-1), torch.arange(n).float()))
        for rank in range(cp_size):
            sl = slice(rank * physical, rank * physical + per_rank[rank])
            self.assertTrue(torch.equal(back[sl], gathered[sl]))

    def test_summary_buffer_bytes_cp8(self):
        # External vLLM #59304 literal: 64 MiB for C=8,R=1,H=64,K=V=128,BF16.
        self.assertEqual(summary_ag_bytes(8, 1, 64, 128, 128), 64 * 1024 * 1024)
        elems = pack_summary_buffer_elems(8, 1, 64, 128, 128)
        self.assertEqual(elems, 33554432)


if __name__ == "__main__":
    unittest.main()
