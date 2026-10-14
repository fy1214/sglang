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


if __name__ == "__main__":
    unittest.main()
