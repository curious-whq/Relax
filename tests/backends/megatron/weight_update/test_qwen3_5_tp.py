# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.weight_update import common
from relax.distributed.checkpoint_service import utils as checkpoint_utils


def _make_tp_param(tensor: torch.Tensor) -> torch.nn.Parameter:
    param = torch.nn.Parameter(tensor.clone())
    param.tensor_model_parallel = True
    param.partition_dim = 0
    param.partition_stride = 1
    return param


@pytest.mark.parametrize(
    ("name", "segment_sizes"),
    [
        ("module.module.decoder.layers.0.self_attention.conv1d.weight", [4, 4, 6]),
        ("module.module.decoder.layers.0.self_attention.in_proj.weight", [4, 4, 6, 6, 2, 2]),
    ],
)
def test_qwen3_5_gdn_gather_and_chunk_are_inverse(monkeypatch, name, segment_sizes):
    tp_size = 2
    config = SimpleNamespace(
        linear_key_head_dim=2,
        linear_value_head_dim=3,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
    )
    hf_config = SimpleNamespace(text_config=config)
    args = SimpleNamespace(hf_checkpoint="unused")

    monkeypatch.setattr(common, "get_hf_config", lambda _checkpoint: hf_config)
    monkeypatch.setattr(checkpoint_utils, "get_hf_config", lambda _checkpoint: hf_config)
    monkeypatch.setattr(common.mpu, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(common.mpu, "get_tensor_model_parallel_group", lambda: object())
    monkeypatch.setattr(checkpoint_utils.mpu, "get_tensor_model_parallel_world_size", lambda: tp_size)

    full_param = torch.arange(sum(segment_sizes) * 3, dtype=torch.float32).reshape(sum(segment_sizes), 3)
    segments = torch.split(full_param, segment_sizes, dim=0)
    segment_shards = [[chunk.clone() for chunk in torch.chunk(segment, tp_size, dim=0)] for segment in segments]
    local_params = [torch.cat([shards[tp_rank] for shards in segment_shards], dim=0) for tp_rank in range(tp_size)]
    collective_index = 0

    def fake_all_gather(outputs, tensor, group=None):
        nonlocal collective_index
        del group
        expected_shards = segment_shards[collective_index]
        torch.testing.assert_close(tensor, expected_shards[0])
        for output, shard in zip(outputs, expected_shards, strict=True):
            output.copy_(shard)
        collective_index += 1

    monkeypatch.setattr(common.dist, "all_gather", fake_all_gather)

    gathered = common.all_gather_param(args, name, _make_tp_param(local_params[0]))
    torch.testing.assert_close(gathered, full_param)
    assert collective_index == len(segment_sizes)

    for tp_rank, expected_local in enumerate(local_params):
        monkeypatch.setattr(checkpoint_utils.mpu, "get_tensor_model_parallel_rank", lambda rank=tp_rank: rank)
        actual_local = checkpoint_utils.chunk_param(args, name, _make_tp_param(expected_local), gathered)
        torch.testing.assert_close(actual_local, expected_local)
