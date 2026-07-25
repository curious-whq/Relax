# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest
import torch

from relax.core.registry import get_algorithm
from relax.engine.filters.dynamic_sampling_filters import check_reward_nonzero_std
from relax.utils.training.algorithm_ops import (
    GDPO_EPSILON,
    AdvantageContext,
    distributed_standardize_sequence_batch,
    gdpo_reward_processor,
    standardize_sequence_batch,
    validate_gdpo_config,
)
from relax.utils.types import Sample


def _args(**kwargs):
    defaults = {
        "advantage_estimator": "gdpo",
        "reward_keys": ["correctness", "format"],
        "reward_weights": None,
        "reward_key": "correctness",
        "n_samples_per_prompt": 2,
        "rewards_normalization": True,
        "grpo_std_normalization": True,
        "normalize_advantages": False,
        "custom_reward_post_process_path": None,
        "global_batch_size": 4,
        "rollout_batch_size": 2,
        "fully_async": False,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


def _samples(reward_vectors):
    return [
        Sample(
            index=position,
            group_index=position // 2,
            reward={"correctness": correctness, "format": format_reward},
        )
        for position, (correctness, format_reward) in enumerate(reward_vectors)
    ]


def test_gdpo_formula_group_normalize_sum_then_batch_normalize():
    args = _args()
    samples = _samples([(0.0, 0.0), (0.0, 1.0), (0.0, 0.0), (1.0, 1.0)])
    raw_rewards, pre_batch = gdpo_reward_processor(args, samples)

    unit = 1.0 / (torch.tensor([0.0, 1.0]).std() + GDPO_EPSILON)
    expected_pre_batch = torch.tensor([-0.5 * unit, 0.5 * unit, -unit, unit])
    assert raw_rewards == [0.0, 0.0, 0.0, 1.0]
    torch.testing.assert_close(torch.tensor(pre_batch), expected_pre_batch)

    result = get_algorithm("gdpo").resolve_advantage_estimator()(
        AdvantageContext(
            args=args,
            rewards=pre_batch,
            kl=[torch.zeros(length) for length in (1, 2, 3, 4)],
            values=None,
            response_lengths=[1, 2, 3, 4],
            loss_masks=[torch.ones(length) for length in (1, 2, 3, 4)],
            total_lengths=[2, 3, 4, 5],
            batch_standardize=standardize_sequence_batch,
        )
    )
    expected_final = standardize_sequence_batch(expected_pre_batch)
    for advantage, expected, length in zip(result.advantages, expected_final, (1, 2, 3, 4), strict=True):
        torch.testing.assert_close(advantage, expected.expand(length))
    assert result.advantages is not result.returns


def test_gdpo_preserves_signal_levels_that_grpo_collapses():
    args = _args()
    vectors = [(0.0, 0.0), (0.0, 1.0), (0.0, 0.0), (1.0, 1.0)]
    _, gdpo_pre_batch = gdpo_reward_processor(args, _samples(vectors))

    grpo_args = Namespace(
        reward_key=None,
        rewards_normalization=True,
        grpo_std_normalization=True,
        n_samples_per_prompt=2,
    )
    summed_samples = [
        Sample(index=position, group_index=position // 2, reward=sum(vector))
        for position, vector in enumerate(vectors)
    ]
    _, collapsed = get_algorithm("grpo").resolve_reward_processor()(grpo_args, summed_samples)

    torch.testing.assert_close(torch.tensor(collapsed[:2]), torch.tensor(collapsed[2:]))
    assert abs(gdpo_pre_batch[2]) > abs(gdpo_pre_batch[0])
    final = standardize_sequence_batch(torch.tensor(gdpo_pre_batch))
    assert abs(final[2]) > abs(final[0])


def test_gdpo_reward_weights_apply_after_independent_normalization():
    args = _args(reward_weights=[2.0, 0.5])
    samples = _samples([(0.0, 0.0), (1.0, 1.0)])
    _, pre_batch = gdpo_reward_processor(args, samples)
    unit = 0.5 / (torch.tensor([0.0, 1.0]).std() + GDPO_EPSILON)
    expected = torch.tensor([-2.5 * unit, 2.5 * unit])
    torch.testing.assert_close(torch.tensor(pre_batch), expected)


def test_gdpo_zero_variance_reward_dimension_contributes_zero():
    args = _args()
    _, pre_batch = gdpo_reward_processor(args, _samples([(1.0, 0.0), (1.0, 1.0)]))
    assert pre_batch[0] < 0 < pre_batch[1]
    assert torch.isfinite(torch.tensor(pre_batch)).all()


def test_gdpo_all_zero_variance_returns_finite_zeros():
    args = _args()
    _, pre_batch = gdpo_reward_processor(args, _samples([(1.0, 2.0), (1.0, 2.0)]))
    assert pre_batch == [0.0, 0.0]
    torch.testing.assert_close(standardize_sequence_batch(torch.tensor(pre_batch)), torch.zeros(2))


def test_gdpo_dynamic_filter_only_collapses_when_all_reward_dimensions_are_constant():
    varying = _samples([(1.0, 0.0), (1.0, 1.0)])
    collapsed = _samples([(1.0, 2.0), (1.0, 2.0)])
    assert check_reward_nonzero_std(_args(), varying).keep
    assert not check_reward_nonzero_std(_args(), collapsed).keep


@pytest.mark.parametrize("bad_value", [True, "1", None, [], float("nan"), float("inf"), -float("inf")])
def test_gdpo_rejects_non_numeric_or_non_finite_rewards(bad_value):
    samples = _samples([(0.0, 0.0), (1.0, 1.0)])
    samples[0].reward["format"] = bad_value
    error = TypeError if bad_value is True or not isinstance(bad_value, float) else ValueError
    with pytest.raises(error, match="format"):
        gdpo_reward_processor(_args(), samples)


def test_gdpo_rejects_missing_reward():
    samples = _samples([(0.0, 0.0), (1.0, 1.0)])
    del samples[0].reward["format"]
    with pytest.raises(KeyError, match="format"):
        gdpo_reward_processor(_args(), samples)


def test_gdpo_rejects_non_mapping_reward():
    samples = _samples([(0.0, 0.0), (1.0, 1.0)])
    samples[0].reward = 1.0
    with pytest.raises(TypeError, match="mapping"):
        gdpo_reward_processor(_args(), samples)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reward_keys": ["correctness"]}, "at least 2"),
        ({"reward_keys": ["correctness", "correctness"]}, "unique"),
        ({"reward_keys": ["correctness", ""]}, "non-empty"),
        ({"reward_weights": [1.0]}, "expected 2"),
        ({"reward_weights": [1.0, float("nan")]}, "finite"),
        ({"reward_key": "score"}, "must be included"),
        ({"n_samples_per_prompt": 1}, ">= 2"),
        ({"rewards_normalization": False}, "requires reward normalization"),
        ({"grpo_std_normalization": False}, "standard deviation"),
        ({"normalize_advantages": True}, "remove --normalize-advantages"),
        ({"custom_reward_post_process_path": "custom.fn"}, "does not support"),
        ({"global_batch_size": None}, "positive --global-batch-size"),
        ({"global_batch_size": 0}, "positive --global-batch-size"),
        ({"rollout_batch_size": None}, "positive --rollout-batch-size"),
        ({"rollout_batch_size": 0}, "positive --rollout-batch-size"),
        ({"global_batch_size": 3}, "divisible"),
        ({"global_batch_size": 4, "rollout_batch_size": 3}, "exactly one"),
    ],
)
def test_gdpo_config_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        get_algorithm("gdpo").validate(_args(**kwargs))


def test_gdpo_defaults_primary_reward_key_to_first_configured_key():
    args = _args(reward_key=None)
    validate_gdpo_config(args)
    assert args.reward_key == "correctness"


def test_gdpo_requires_complete_groups():
    samples = _samples([(0.0, 0.0), (1.0, 1.0)])
    samples[1].group_index = 1
    with pytest.raises(ValueError, match="has 1 samples"):
        gdpo_reward_processor(_args(), samples)


def test_distributed_batch_standardization_uses_global_sample_moments(monkeypatch):
    local = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    remote = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64)
    global_mean = torch.cat((local, remote)).mean()
    call_index = 0

    def fake_all_reduce(statistic, group):
        nonlocal call_index
        del group
        if call_index == 0:
            statistic += torch.tensor([remote.sum(), remote.numel()], dtype=torch.float64)
        else:
            statistic += ((remote - global_mean) ** 2).sum()
        call_index += 1

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    actual = distributed_standardize_sequence_batch(local, process_group=object())
    expected = standardize_sequence_batch(torch.cat((local, remote)))[:2]
    assert call_index == 2
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "values",
    [
        torch.tensor([3.0]),
        torch.tensor([3.0, 3.0, 3.0]),
    ],
)
def test_distributed_batch_standardization_returns_zero_for_undefined_or_zero_variance(
    monkeypatch,
    values,
):
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda statistic, group: None)
    actual = distributed_standardize_sequence_batch(values, process_group=object())
    torch.testing.assert_close(actual, torch.zeros_like(values))


def test_distributed_batch_standardization_accumulates_moments_in_float64(monkeypatch):
    local = torch.tensor([100_000_000.0, 100_000_001.0], dtype=torch.float64)
    seen_dtypes = []

    def fake_all_reduce(statistic, group):
        del group
        seen_dtypes.append(statistic.dtype)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    actual = distributed_standardize_sequence_batch(local, process_group=object())

    assert seen_dtypes == [torch.float64, torch.float64]
    torch.testing.assert_close(actual, standardize_sequence_batch(local))
