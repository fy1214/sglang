from pathlib import Path

import torch
from torch import nn

from sglang.srt.debug_utils.tensor_dump_forward_hook import (
    register_forward_hook_for_model,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.Sequential(nn.Linear(2, 2, bias=False))
        self.shared_experts = nn.Sequential(nn.Linear(2, 2, bias=False))

    def forward(self, value):
        return self.experts(value) + self.shared_experts(value)


class _InnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(), _Layer()])

    def forward(self, value, forward_batch):
        del forward_batch
        for layer in self.layers:
            value = layer(value)
        return value


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _InnerModel()

    def forward(self, value, forward_batch):
        return self.model(value, forward_batch)


def _make_forward_batch() -> ForwardBatch:
    batch = object.__new__(ForwardBatch)
    batch.input_ids = torch.tensor([11, 12])
    batch.seq_lens = torch.tensor([2], dtype=torch.int32)
    batch.positions = torch.tensor([0, 1])
    batch.req_pool_indices = torch.tensor([3], dtype=torch.int32)
    batch.extend_seq_lens = torch.tensor([2], dtype=torch.int32)
    batch.extend_prefix_lens = torch.tensor([0], dtype=torch.int32)
    batch.rids = ["request-0"]
    batch.forward_mode = "extend"
    return batch


def test_layer_outputs_only_dump(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SGLANG_TENSOR_DUMP_LAYER_OUTPUTS_ONLY", "1")
    monkeypatch.setenv("SGLANG_TENSOR_DUMP_CHUNK_SIZE", "16")
    model = _Model()
    dumper = register_forward_hook_for_model(
        model,
        str(tmp_path),
        dump_layers=[0, 1],
        tp_size=1,
        tp_rank=0,
        pp_rank=0,
    )

    model(torch.ones((2, 2)), _make_forward_batch())

    (dump_file,) = list(Path(dumper.get_dump_dir()).glob("Chunk*.pt"))
    (values,) = torch.load(dump_file, weights_only=False)
    assert "model.layers.0" in values
    assert "model.layers.1" in values
    assert not any("experts" in name for name in values)
    assert values["model.forward_batch_info.rids"] == ["request-0"]
    torch.testing.assert_close(
        values["model.forward_batch_info.positions"], torch.tensor([0, 1])
    )


def test_layer_outputs_only_can_select_nested_modules(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SGLANG_TENSOR_DUMP_LAYER_OUTPUTS_ONLY", "1")
    monkeypatch.setenv(
        "SGLANG_TENSOR_DUMP_MODULE_SUFFIXES", "mlp.experts,experts,shared_experts"
    )
    model = _Model()
    dumper = register_forward_hook_for_model(
        model,
        str(tmp_path),
        dump_layers=[0],
        tp_size=1,
        tp_rank=0,
        pp_rank=0,
    )

    model(torch.ones((2, 2)), _make_forward_batch())

    data = torch.load(Path(dumper.get_dump_dir()) / "Pass00000.pt", weights_only=False)
    assert "model.layers.0" in data
    assert "model.layers.0.experts" in data
    assert "model.layers.0.shared_experts" in data
    assert not any(name.startswith("model.layers.1") for name in data)
