"""CPU unit tests for DSA flashinfer_mla_sparse CSR packing (no GPU / FlashInfer)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

import torch


class TestPackTopkPageTableToCsr(unittest.TestCase):
    def test_packs_valid_prefix(self):
        from sglang.srt.layers.attention.dsa.flashinfer_mla_sparse import (
            pack_topk_page_table_to_csr,
        )

        page_table = torch.tensor(
            [
                [10, 11, 12, -1],
                [20, -1, -1, -1],
                [30, 31, -1, -1],
            ],
            dtype=torch.int32,
        )
        kv_lens = torch.tensor([3, 1, 2], dtype=torch.int32)
        kv_indptr, kv_indices, kv_len_arr = pack_topk_page_table_to_csr(
            page_table, kv_lens
        )
        self.assertEqual(kv_indptr.tolist(), [0, 3, 4, 6])
        self.assertEqual(kv_indices.tolist(), [10, 11, 12, 20, 30, 31])
        self.assertEqual(kv_len_arr.tolist(), [3, 1, 2])

    def test_clamps_negative_and_overwidth(self):
        from sglang.srt.layers.attention.dsa.flashinfer_mla_sparse import (
            pack_topk_page_table_to_csr,
        )

        page_table = torch.tensor([[5, -1], [7, 8]], dtype=torch.int32)
        kv_lens = torch.tensor([2, 99], dtype=torch.int32)
        _, kv_indices, kv_len_arr = pack_topk_page_table_to_csr(page_table, kv_lens)
        self.assertEqual(kv_len_arr.tolist(), [2, 2])
        self.assertEqual(kv_indices.tolist(), [5, 0, 7, 8])


class TestCliChoiceRegistered(unittest.TestCase):
    def test_flashinfer_mla_sparse_in_dsa_choices(self):
        import argparse

        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        actions = {
            option: action
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertIn("flashinfer_mla_sparse", actions["--dsa-prefill-backend"].choices)
        self.assertIn("flashinfer_mla_sparse", actions["--dsa-decode-backend"].choices)
        # Must not collide with SM120 name.
        self.assertIn("flashinfer_sparse_mla", actions["--dsa-prefill-backend"].choices)
        args = parser.parse_args(
            [
                "--model",
                "dummy",
                "--dsa-prefill-backend",
                "flashinfer_mla_sparse",
                "--dsa-decode-backend",
                "flashinfer_mla_sparse",
            ]
        )
        self.assertEqual(args.dsa_prefill_backend, "flashinfer_mla_sparse")
        self.assertEqual(args.dsa_decode_backend, "flashinfer_mla_sparse")


class TestFp8KvPlanDtype(unittest.TestCase):
    def test_normalize_uint8_to_e4m3(self):
        from sglang.srt.layers.attention.dsa.flashinfer_mla_sparse import (
            normalize_batch_mla_kv_dtype,
        )

        self.assertEqual(normalize_batch_mla_kv_dtype(torch.uint8), torch.float8_e4m3fn)
        self.assertEqual(
            normalize_batch_mla_kv_dtype(torch.float8_e4m3fn), torch.float8_e4m3fn
        )
        self.assertEqual(normalize_batch_mla_kv_dtype(torch.bfloat16), torch.bfloat16)


class TestFp8KvRunScales(unittest.TestCase):
    def test_run_views_uint8_and_passes_scales(self):
        """Mirror vLLM forward_mqa: view(e4m3) + ckv_scale/kpe_scale kwargs."""
        from sglang.srt.layers.attention.dsa import flashinfer_mla_sparse as mod

        class _FakeWrapper:
            def __init__(self):
                self.kwargs = None
                self.ckv_dtype = None

            def run(self, q_nope, q_pe, ckv, kpe, out=None, **kwargs):
                self.kwargs = kwargs
                self.ckv_dtype = ckv.dtype
                return out if out is not None else q_nope

        runner = object.__new__(mod.DSAFlashInferMLASparseRunner)
        runner._planned_num_tokens = 1
        runner.use_fp8_kv_cache = True
        runner.packed_fp8_kv = False
        runner.kv_lora_rank = 512
        runner.qk_rope_head_dim = 0
        runner.packed_row_bytes = 528
        runner.wrapper = _FakeWrapper()

        q = torch.zeros(1, 2, 512, dtype=torch.bfloat16)
        q_pe = torch.zeros(1, 2, 0, dtype=torch.bfloat16)
        kv = torch.zeros(4, 1, 512, dtype=torch.uint8)
        out = runner.run(q, q_pe, kv, 512, ckv_scale=0.5, kpe_scale=1.0)
        self.assertEqual(out.shape, q.shape)
        self.assertEqual(runner.wrapper.ckv_dtype, torch.float8_e4m3fn)
        self.assertEqual(runner.wrapper.kwargs, {"ckv_scale": 0.5, "kpe_scale": 1.0})

    def test_run_packed_fp8_calls_run_packed_kv(self):
        """G=1 packed 528B rows go through run_packed_kv (scale_offset=512)."""
        from sglang.srt.layers.attention.dsa import flashinfer_mla_sparse as mod

        class _FakeWrapper:
            def __init__(self):
                self.packed_kwargs = None

            def run_packed_kv(self, q_nope, q_pe, kv, *, scale_offset, out=None, **kw):
                self.packed_kwargs = {
                    "scale_offset": scale_offset,
                    "kv_shape": tuple(kv.shape),
                    "kv_dtype": kv.dtype,
                }
                return out if out is not None else q_nope

        runner = object.__new__(mod.DSAFlashInferMLASparseRunner)
        runner._planned_num_tokens = 1
        runner.use_fp8_kv_cache = True
        runner.packed_fp8_kv = True
        runner.kv_lora_rank = 512
        runner.qk_rope_head_dim = 0
        runner.packed_row_bytes = 528
        runner.wrapper = _FakeWrapper()

        q = torch.zeros(1, 2, 512, dtype=torch.bfloat16)
        q_pe = torch.zeros(1, 2, 0, dtype=torch.bfloat16)
        kv = torch.zeros(4, 1, 528, dtype=torch.uint8)
        out = runner.run(q, q_pe, kv, 512)
        self.assertEqual(out.shape, q.shape)
        self.assertEqual(
            runner.wrapper.packed_kwargs,
            {
                "scale_offset": 512,
                "kv_shape": (4, 1, 528),
                "kv_dtype": torch.float8_e4m3fn,
            },
        )


class TestFp8G1QuantLayout(unittest.TestCase):
    def test_g1_scale_slot_and_row_bytes(self):
        from sglang.kernels.ops.attention.dsa.quant_k_cache import (
            FP8_G1_SCALE_SLOT_BYTES,
            fp8_kv_packed_row_bytes,
            fp8_kv_scale_slot_bytes,
        )

        self.assertEqual(FP8_G1_SCALE_SLOT_BYTES, 16)
        self.assertEqual(fp8_kv_scale_slot_bytes(512, 512), 16)
        self.assertEqual(fp8_kv_scale_slot_bytes(512, 128), 16)
        self.assertEqual(fp8_kv_packed_row_bytes(512, 0, 512), 528)
        self.assertEqual(fp8_kv_packed_row_bytes(512, 0, 128), 528)


if __name__ == "__main__":
    unittest.main()
