# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
import pathlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


_MODULE_NAME = "relax.backends.megatron.weight_conversion.qwen3_5"
_SOURCE = pathlib.Path(__file__).resolve().parents[4] / "relax/backends/megatron/weight_conversion/qwen3_5.py"


def _load_qwen3_5_module() -> ModuleType:
    misc_module = ModuleType("relax.utils.misc")
    misc_module.get_hf_config = lambda _checkpoint: None
    saved_misc = sys.modules.get("relax.utils.misc")
    sys.modules["relax.utils.misc"] = misc_module
    try:
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if saved_misc is None:
            sys.modules.pop("relax.utils.misc", None)
        else:
            sys.modules["relax.utils.misc"] = saved_misc


qwen3_5 = _load_qwen3_5_module()


@pytest.fixture
def vision_config(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    config = SimpleNamespace(num_heads=2, hidden_size=4)
    monkeypatch.setattr(
        qwen3_5,
        "get_hf_config",
        lambda _checkpoint: SimpleNamespace(vision_config=config),
    )
    return config


@pytest.mark.parametrize(
    ("megatron_name", "hf_name"),
    [
        (
            "module.module.vision_model.patch_embed.proj.weight",
            "model.visual.patch_embed.proj.weight",
        ),
        (
            "module.module.vision_model.pos_embed.weight",
            "model.visual.pos_embed.weight",
        ),
        (
            "module.module.vision_model.merger.patch_norm.weight",
            "model.visual.merger.norm.weight",
        ),
        (
            "module.module.vision_model.merger.linear_fc1.weight",
            "model.visual.merger.linear_fc1.weight",
        ),
        (
            "module.module.vision_model.merger.linear_fc2.weight",
            "model.visual.merger.linear_fc2.weight",
        ),
    ],
)
def test_convert_qwen3_5_vision_passthrough_names(vision_config, megatron_name, hf_name):
    param = torch.randn(2, 2)

    assert qwen3_5.convert_qwen3_5_vision_to_hf(SimpleNamespace(hf_checkpoint="unused"), megatron_name, param) == [
        (hf_name, param)
    ]


def test_convert_qwen3_5_vision_qkv_reorders_interleaved_heads(vision_config):
    # Megatron stores each vision head as [Q, K, V]. Hugging Face stores all
    # query heads first, followed by all key heads and then all value heads.
    qkv = torch.arange(12 * 3, dtype=torch.float32).reshape(12, 3)
    grouped = qkv.reshape(2, 3, 2, 3)
    expected = torch.cat(
        [
            grouped[:, 0].reshape(-1, 3),
            grouped[:, 1].reshape(-1, 3),
            grouped[:, 2].reshape(-1, 3),
        ],
        dim=0,
    )

    converted = qwen3_5.convert_qwen3_5_vision_to_hf(
        SimpleNamespace(hf_checkpoint="unused"),
        "module.module.vision_model.decoder.layers.3.self_attention.linear_qkv.weight",
        qkv,
    )

    assert converted[0][0] == "model.visual.blocks.3.attn.qkv.weight"
    torch.testing.assert_close(converted[0][1], expected)


def test_qwen3_5_gdn_conversion_round_trip():
    config = SimpleNamespace(
        hidden_size=4,
        linear_key_head_dim=2,
        linear_value_head_dim=3,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
    )
    qk_dim = config.linear_key_head_dim * config.linear_num_key_heads
    v_dim = config.linear_value_head_dim * config.linear_num_value_heads
    weights = (
        torch.randn(2 * qk_dim + v_dim, config.hidden_size),
        torch.randn(v_dim, config.hidden_size),
        torch.randn(config.linear_num_value_heads, config.hidden_size),
        torch.randn(config.linear_num_value_heads, config.hidden_size),
    )

    megatron_weight = qwen3_5.gdn_hf_to_mca(config, weights)
    restored = qwen3_5.gdn_mca_to_hf(config, megatron_weight)

    for expected, actual in zip(weights, restored, strict=True):
        torch.testing.assert_close(actual, expected)


def test_convert_qwen3_5_vision_rejects_unknown_parameter(vision_config):
    with pytest.raises(ValueError, match="Unknown parameter name"):
        qwen3_5.convert_qwen3_5_vision_to_hf(
            SimpleNamespace(hf_checkpoint="unused"),
            "module.module.vision_model.unsupported.weight",
            torch.randn(2, 2),
        )
