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
    _chrono_to_physical_slot,
    build_hybrid_pcp_layout,
    build_kcp_plan,
    conv_halo_size,
    final_conv_window_from_slots,
    flash_kda_kcp_available,
    get_ag_scratch,
    initial_conv_window_for_slot,
    kcp_merge_left,
    merge_affine_segment_states,
    merge_affine_segment_states_batched,
    merge_packed_flash_summaries,
    pack_local_conv_tails,
    pack_local_conv_tails_batch,
    pack_summary_buffer_elems,
    summary_ag_bytes,
    zigzag_slot_lengths_for_request,
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
            # Accept either per-request totals [4,4] or all-early-then-late
            # segment lengths [2,2,2,2] (SGLang zigzag local packing).
            if q_cpu == [4, 4]:
                self.assertEqual(bs_cpu, [0, 1])
                self.assertTrue(torch.equal(q, torch.tensor([4, 4], dtype=torch.int32)))
            else:
                self.assertEqual(q_cpu, [2, 2, 2, 2])
                self.assertEqual(bs_cpu, [0, 1, 0, 1])
                self.assertTrue(
                    torch.equal(q, torch.tensor([2, 2, 2, 2], dtype=torch.int32))
                )

    def test_materialize_compression_inputs_uses_strategy_gather(self):
        init_cp_strategy(enable_prefill_cp=True, cp_size=2, cp_strategy="zigzag")
        strategy = ZigzagCPStrategy(cp_size=2)
        with get_parallel().override(attn_cp_rank=0, attn_cp_size=2):
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
        with get_parallel().override(attn_cp_rank=0, attn_cp_size=2):
            metadata = strategy.build_metadata(8, [8], [8])
            metadata.real_page_table = object()
            metadata.dsa_seqlens_expanded = object()
            # Zigzag DSA may expose early+late as two segments of the same req.
            metadata.dsa_extend_seq_lens_list = [2, 2]
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
            # One pool index repeated for early+late, or a single total row.
            idx = out["local_req_pool_indices"]
            self.assertTrue(
                torch.equal(idx, torch.tensor([3, 3]))
                or torch.equal(idx, torch.tensor([3]))
            )

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

    def test_use_zigzag_kcp_requires_env_and_zigzag_meta(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.cp.kda_kcp import use_zigzag_kcp

        init_cp_strategy(enable_prefill_cp=True, cp_size=2, cp_strategy="zigzag")
        strategy = ZigzagCPStrategy(cp_size=2)
        with get_parallel().override(attn_cp_rank=0, attn_cp_size=2):
            meta = strategy.build_metadata(8, [8], [8])
        fb = SimpleNamespace(attn_cp_metadata=meta, forward_mode=_ExtendMode())
        prev = envs.SGLANG_ENABLE_GLM_ZIGZAG_KCP.get()
        try:
            envs.SGLANG_ENABLE_GLM_ZIGZAG_KCP.set(False)
            self.assertFalse(use_zigzag_kcp(fb))
            envs.SGLANG_ENABLE_GLM_ZIGZAG_KCP.set(True)
            self.assertTrue(use_zigzag_kcp(fb))
        finally:
            envs.SGLANG_ENABLE_GLM_ZIGZAG_KCP.set(prev)

    def test_pack_local_conv_tails_right_aligned(self):
        # early=3, late=2, halo=3 → early takes all 3; late right-aligns 2.
        halo = 3
        early = torch.tensor([[1.0], [2.0], [3.0]])
        late = torch.tensor([[4.0], [5.0]])
        mixed = torch.cat([early, late], dim=0)
        tails = pack_local_conv_tails(mixed, early_len=3, late_len=2, halo=halo)
        self.assertEqual(tuple(tails.shape), (2, 1, halo))
        self.assertTrue(torch.equal(tails[0, 0], torch.tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.equal(tails[1, 0], torch.tensor([0.0, 4.0, 5.0])))

    def test_pack_local_conv_tails_batch_all_early_then_late(self):
        # bs=2 local packing: early0|early1|late0|late1
        halo = 2
        e0 = torch.tensor([[1.0], [2.0]])
        e1 = torch.tensor([[3.0]])
        l0 = torch.tensor([[4.0], [5.0]])
        l1 = torch.tensor([[6.0]])
        mixed = torch.cat([e0, e1, l0, l1], dim=0)
        tails = pack_local_conv_tails_batch(
            mixed, early_lens=[2, 1], late_lens=[2, 1], halo=halo
        )
        self.assertEqual(tuple(tails.shape), (2, 2, 1, halo))
        self.assertTrue(torch.equal(tails[0, 0, 0], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(tails[1, 0, 0], torch.tensor([0.0, 3.0])))
        self.assertTrue(torch.equal(tails[0, 1, 0], torch.tensor([4.0, 5.0])))
        self.assertTrue(torch.equal(tails[1, 1, 0], torch.tensor([0.0, 6.0])))

    def test_hybrid_pcp_layout_bs2_all_early_then_late(self):
        init_cp_strategy(enable_prefill_cp=True, cp_size=2, cp_strategy="zigzag")
        strategy = ZigzagCPStrategy(cp_size=2)
        with get_parallel().override(attn_cp_rank=0, attn_cp_size=2):
            meta = strategy.build_metadata(16, [8, 8], [8, 8])
            fb = SimpleNamespace(
                attn_cp_metadata=meta,
                extend_seq_lens_cpu=[8, 8],
                extend_seq_lens=torch.tensor([8, 8], dtype=torch.int32),
                extend_prefix_lens=None,
            )
            layout = build_hybrid_pcp_layout(fb, device=torch.device("cpu"))
            plan = build_kcp_plan(layout, halo_size=3)
        # Rank0 owns slots 0 (early) and 3 (late); both reqs nonempty → 4 segs.
        self.assertEqual(layout.num_prefill_reqs, 2)
        self.assertEqual(layout.num_segments, 4)
        self.assertEqual(layout.segment_request, [0, 1, 0, 1])
        self.assertEqual(layout.segment_slot, [0, 0, 3, 3])
        self.assertEqual(layout.scan_cu, [0, 2, 4, 6, 8])
        self.assertEqual(plan.summary_idx.tolist(), [0, 1, 2, 3])  # (0*2+part)*N+n
        self.assertTrue(bool(plan.slot_present[0, 0]))
        self.assertEqual(int(plan.segment_of_slot[1, 3].item()), 3)
        # Conv halo plan (vLLM gather_conv_tails / conv_windows indices).
        self.assertEqual(tuple(plan.tail_src_idx.shape), (2 * 2 * 3,))
        self.assertEqual(tuple(plan.halo_idx.shape), (4, 3))
        self.assertEqual(tuple(plan.final_halo_idx.shape), (2, 3))
        # First segment of req0 starts at chrono 0 → window is prefix rows.
        self.assertEqual(plan.halo_idx[0].tolist(), [0, 1, 2])

    def test_kcp_merge_batched_and_left_api(self):
        s, n, h, dk, dv = 4, 2, 2, 3, 2
        torch.manual_seed(1)
        s_ext = torch.randn(s, n, h, dk, dv)
        m = torch.randn(s, n, h, dk, dk)
        present = torch.ones(n, s, dtype=torch.bool)
        present[1, 2] = False  # skip one slot for req1
        s_in, s_final = merge_affine_segment_states_batched(s_ext, m, present)
        # Manual check req0
        cur = torch.zeros(h, dk, dv)
        for slot in range(s):
            self.assertTrue(torch.allclose(s_in[slot, 0], cur))
            cur = s_ext[slot, 0] + torch.matmul(m[slot, 0], cur)
        self.assertTrue(torch.allclose(s_final[0], cur))

        segment_of_slot = torch.full((n, s), -1, dtype=torch.int32)
        segment_of_slot[0, 0] = 0
        segment_of_slot[0, 3] = 1
        initial = torch.zeros(2, h, dk, dv)
        final = torch.zeros(n, h, dk, dv)
        kcp_merge_left(s_ext, m, present, segment_of_slot, initial, final)
        self.assertTrue(torch.allclose(initial[0], s_in[0, 0]))
        self.assertTrue(torch.allclose(initial[1], s_in[3, 0]))
        self.assertTrue(torch.allclose(final[0], s_final[0]))

    def test_ag_scratch_reused(self):
        a = get_ag_scratch("t", (4, 8), torch.float32, torch.device("cpu"))
        b = get_ag_scratch("t", (4, 8), torch.float32, torch.device("cpu"))
        self.assertIs(a, b)
        c = get_ag_scratch("t", (4, 16), torch.float32, torch.device("cpu"))
        self.assertIsNot(a, c)

    def test_chrono_to_physical_and_packed_merge(self):
        self.assertEqual(_chrono_to_physical_slot(0, 2), 0)
        self.assertEqual(_chrono_to_physical_slot(1, 2), 2)
        self.assertEqual(_chrono_to_physical_slot(2, 2), 3)
        self.assertEqual(_chrono_to_physical_slot(3, 2), 1)
        # One req, CP=1 → 2 slots; only slot0 present with S^T=0, M^T=I.
        h = k = v = 2
        packed = torch.zeros(2, 1, h, v + k, k)
        eye = torch.eye(k)
        packed[0, 0, :, v:, :] = eye  # M^T = I on chrono0 / phys0
        packed[1, 0, :, v:, :] = eye
        present = torch.tensor([[True, False]])
        seg_of = torch.tensor([[0, -1]], dtype=torch.int32)
        initial, final = merge_packed_flash_summaries(
            packed, present, seg_of, cp_size=1
        )
        self.assertEqual(tuple(initial.shape), (1, h, v, k))
        self.assertTrue(torch.allclose(initial[0], torch.zeros(h, v, k)))
        self.assertTrue(torch.allclose(final[0], torch.zeros(h, v, k)))
        # flash_kda may or may not be installed in CI; just call the probe.
        _ = flash_kda_kcp_available()

    def test_conv_halo_windows_from_slot_tails(self):
        # Fake chrono slot tails [4, C=1, halo=2] for CP=2.
        slot_tails = torch.tensor(
            [
                [[10.0, 11.0]],
                [[20.0, 21.0]],
                [[30.0, 31.0]],
                [[40.0, 41.0]],
            ]
        )
        self.assertTrue(
            torch.equal(initial_conv_window_for_slot(slot_tails, 0), torch.zeros(1, 2))
        )
        self.assertTrue(
            torch.equal(
                initial_conv_window_for_slot(slot_tails, 2),
                torch.tensor([[20.0, 21.0]]),
            )
        )
        lens = zigzag_slot_lengths_for_request(8, cp_size=2)  # [2,2,2,2]
        self.assertEqual(lens, [2, 2, 2, 2])
        final = final_conv_window_from_slots(slot_tails, lens)
        self.assertTrue(torch.equal(final, torch.tensor([[40.0, 41.0]])))
        self.assertEqual(conv_halo_size(torch.empty(16, 4)), 3)


if __name__ == "__main__":
    unittest.main()
