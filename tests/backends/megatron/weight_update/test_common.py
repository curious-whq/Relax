# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.weight_update import common


def _make_tp_param() -> torch.nn.Parameter:
    param = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(4, 3))
    param.tensor_model_parallel = True
    param.partition_dim = 0
    param.partition_stride = 1
    return param


@pytest.mark.parametrize(
    "name",
    [
        "module.module.embedding.word_embeddings.weight",
        "module.module.decoder.layers.0.self_attention.conv1d.weight",
        "module.module.decoder.layers.0.self_attention.in_proj.weight",
    ],
)
def test_all_gather_param_skips_collective_for_tp_size_one(monkeypatch, name):
    param = _make_tp_param()
    monkeypatch.setattr(common.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        common.mpu,
        "get_tensor_model_parallel_group",
        lambda: pytest.fail("TP group is not needed when TP size is one"),
    )
    monkeypatch.setattr(common.dist, "all_gather", lambda *_args, **_kwargs: pytest.fail("unexpected collective"))
    monkeypatch.setattr(common, "get_hf_config", lambda *_args, **_kwargs: pytest.fail("unexpected config load"))

    gathered = common.all_gather_param(SimpleNamespace(hf_checkpoint="unused"), name, param)

    assert gathered.data_ptr() == param.data_ptr()
    torch.testing.assert_close(gathered, param)


def test_all_gather_param_uses_expert_tp_size_for_tp_size_one(monkeypatch):
    param = _make_tp_param()
    monkeypatch.setattr(common.mpu, "get_expert_tensor_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        common.mpu,
        "get_expert_tensor_parallel_group",
        lambda: pytest.fail("expert TP group is not needed when expert TP size is one"),
    )
    monkeypatch.setattr(
        common.mpu,
        "get_tensor_model_parallel_world_size",
        lambda: pytest.fail("regular TP size must not be used for expert parameters"),
    )
    monkeypatch.setattr(common.dist, "all_gather", lambda *_args, **_kwargs: pytest.fail("unexpected collective"))

    gathered = common.all_gather_param(
        SimpleNamespace(hf_checkpoint="unused"),
        "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight",
        param,
    )

    assert gathered.data_ptr() == param.data_ptr()
    torch.testing.assert_close(gathered, param)


@pytest.mark.parametrize(
    ("name", "world_size_getter", "group_getter"),
    [
        (
            "module.module.embedding.word_embeddings.weight",
            "get_tensor_model_parallel_world_size",
            "get_tensor_model_parallel_group",
        ),
        (
            "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight",
            "get_expert_tensor_parallel_world_size",
            "get_expert_tensor_parallel_group",
        ),
    ],
)
def test_all_gather_params_async_skips_collective_for_tp_size_one(monkeypatch, name, world_size_getter, group_getter):
    param = _make_tp_param()
    info = SimpleNamespace(name=name)
    monkeypatch.setattr(common.mpu, world_size_getter, lambda: 1)
    monkeypatch.setattr(
        common.mpu,
        group_getter,
        lambda: pytest.fail("TP group is not needed when TP size is one"),
    )
    monkeypatch.setattr(common.dist, "all_gather", lambda *_args, **_kwargs: pytest.fail("unexpected collective"))

    gathered = common.all_gather_params_async([(info, param)])

    assert len(gathered) == 1
    assert gathered[0].data_ptr() == param.data_ptr()
    torch.testing.assert_close(gathered[0], param)
