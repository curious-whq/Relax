# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from argparse import Namespace
from types import ModuleType

import pytest
import torch

from relax.core.registry import get_algorithm
from relax.utils.training.algorithm_ops import (
    AdvantageContext,
    PolicyObjectiveContext,
    RatioContext,
    group_centered_reward_processor,
    group_reward_processor,
    identity_reward_processor,
)
from relax.utils.training.ppo_utils import (
    compute_cispo_loss,
    compute_gspo_kl,
    compute_policy_loss,
    compute_sapo_loss,
    get_advantages_and_returns_batch,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)
from relax.utils.types import Sample


@pytest.fixture
def single_cp_megatron(monkeypatch):
    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    mpu = ModuleType("megatron.core.mpu")
    mpu.get_context_parallel_world_size = lambda: 1
    core.mpu = mpu
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu)


def _args(**kwargs):
    defaults = {
        "reward_key": None,
        "rewards_normalization": True,
        "grpo_std_normalization": True,
        "n_samples_per_prompt": 2,
        "kl_coef": 0.2,
        "gamma": 0.9,
        "eps_clip": 0.2,
        "eps_clip_high": 0.3,
        "sapo_tau_pos": 1.0,
        "sapo_tau_neg": 1.05,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


def _samples(rewards):
    return [Sample(index=i, group_index=i // 2, reward=reward) for i, reward in enumerate(rewards)]


@pytest.mark.parametrize("name", ["grpo", "gspo", "sapo", "cispo"])
def test_group_algorithms_preserve_legacy_reward_normalization(name):
    args = _args()
    samples = _samples([1.0, 3.0, 10.0, 14.0])
    raw, processed = get_algorithm(name).resolve_reward_processor()(args, samples)

    rewards = torch.tensor(raw)
    expected = torch.empty_like(rewards)
    expected[:2] = (rewards[:2] - rewards[:2].mean()) / (rewards[:2].std() + 1e-6)
    expected[2:] = (rewards[2:] - rewards[2:].mean()) / (rewards[2:].std() + 1e-6)
    torch.testing.assert_close(torch.tensor(processed), expected)


def test_group_reward_processor_preserves_disable_std_behavior():
    args = _args(grpo_std_normalization=False)
    raw, processed = group_reward_processor(args, _samples([1.0, 3.0, 10.0, 14.0]))
    assert raw == [1.0, 3.0, 10.0, 14.0]
    torch.testing.assert_close(torch.tensor(processed), torch.tensor([-1.0, 1.0, -2.0, 2.0]))


def test_group_reward_processor_preserves_disable_normalization_behavior():
    args = _args(rewards_normalization=False)
    raw, processed = group_reward_processor(args, _samples([1.0, 3.0]))
    assert processed is raw


def test_reinforce_baseline_reward_processor_only_centers_groups():
    raw, processed = group_centered_reward_processor(_args(), _samples([1.0, 3.0, 10.0, 14.0]))
    assert raw == [1.0, 3.0, 10.0, 14.0]
    torch.testing.assert_close(torch.tensor(processed), torch.tensor([-1.0, 1.0, -2.0, 2.0]))


def test_identity_reward_processor_returns_original_list():
    raw, processed = identity_reward_processor(_args(), _samples([1.0, 3.0]))
    assert processed is raw


def test_outcome_advantage_dispatch_preserves_scalar_broadcast_and_list_copy():
    spec = get_algorithm("grpo")
    kl = [torch.tensor([0.1, 0.2]), torch.tensor([0.3])]
    result = spec.resolve_advantage_estimator()(
        AdvantageContext(
            args=_args(),
            rewards=[-1.0, 2.0],
            kl=kl,
            values=None,
            response_lengths=[2, 1],
            loss_masks=[torch.ones(2), torch.ones(1)],
            total_lengths=[3, 2],
        )
    )
    assert result.advantages is not result.returns
    torch.testing.assert_close(result.advantages[0], torch.tensor([-1.0, -1.0]))
    torch.testing.assert_close(result.advantages[1], torch.tensor([2.0]))


def test_reinforce_estimators_match_existing_helpers(single_cp_megatron):
    del single_cp_megatron
    args = _args()
    rewards = [1.0, -0.5]
    kl = [torch.tensor([0.1, 0.2]), torch.tensor([0.3])]
    masks = [torch.ones(2), torch.ones(1)]
    context = AdvantageContext(
        args=args,
        rewards=rewards,
        kl=[item.clone() for item in kl],
        values=None,
        response_lengths=[2, 1],
        loss_masks=masks,
        total_lengths=[3, 2],
    )

    reinforce = get_algorithm("reinforce_plus_plus").resolve_advantage_estimator()(context)
    expected_returns = get_reinforce_plus_plus_returns(
        rewards=torch.tensor(rewards),
        kl=kl,
        loss_masks=masks,
        response_lengths=[2, 1],
        total_lengths=[3, 2],
        kl_coef=args.kl_coef,
        gamma=args.gamma,
    )
    for actual, expected in zip(reinforce.returns, expected_returns, strict=True):
        torch.testing.assert_close(actual, expected)

    baseline = get_algorithm("reinforce_plus_plus_baseline").resolve_advantage_estimator()(context)
    expected_baseline = get_reinforce_plus_plus_baseline_advantages(
        rewards=torch.tensor(rewards),
        kl=context.kl,
        loss_masks=masks,
        kl_coef=args.kl_coef,
    )
    for actual, expected in zip(baseline.advantages, expected_baseline, strict=True):
        torch.testing.assert_close(actual, expected)
    assert baseline.advantages is baseline.returns


def test_ppo_advantage_dispatch_matches_existing_helper(single_cp_megatron):
    del single_cp_megatron
    args = _args(lambd=0.8)
    rewards = [1.0, -0.5]
    kl = [torch.tensor([0.1, 0.2]), torch.tensor([0.3])]
    values = [torch.tensor([0.2, 0.4]), torch.tensor([-0.1])]
    context = AdvantageContext(
        args=args,
        rewards=rewards,
        kl=[item.clone() for item in kl],
        values=values,
        response_lengths=[2, 1],
        loss_masks=[torch.ones(2), torch.ones(1)],
        total_lengths=[3, 2],
    )

    actual = get_algorithm("ppo").resolve_advantage_estimator()(context)
    token_rewards = [item.clone().mul(-args.kl_coef) for item in kl]
    token_rewards[0][-1] += rewards[0]
    token_rewards[1][-1] += rewards[1]
    expected_advantages, expected_returns = get_advantages_and_returns_batch(
        context.total_lengths,
        context.response_lengths,
        values,
        token_rewards,
        args.gamma,
        args.lambd,
    )

    for actual_item, expected_item in zip(actual.advantages, expected_advantages, strict=True):
        torch.testing.assert_close(actual_item, expected_item)
    for actual_item, expected_item in zip(actual.returns, expected_returns, strict=True):
        torch.testing.assert_close(actual_item, expected_item)


@pytest.mark.parametrize("name", ["grpo", "reinforce_plus_plus", "ppo"])
def test_token_ratio_and_clipped_objective_dispatch_match_existing_helpers(name):
    args = _args()
    log_probs = [torch.tensor([-0.2, -0.4]), torch.tensor([-0.1])]
    old_log_probs = [torch.tensor([-0.3, -0.5]), torch.tensor([-0.2])]
    ratio = get_algorithm(name).resolve_ratio_builder()(
        RatioContext(
            args=args,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            full_log_probs=None,
            full_old_log_probs=None,
            loss_masks=[torch.ones(2), torch.ones(1)],
        )
    )
    expected_kl = torch.cat(old_log_probs) - torch.cat(log_probs)
    torch.testing.assert_close(ratio.ppo_kl, expected_kl)

    advantages = torch.tensor([1.0, -1.0, 0.5])
    actual = get_algorithm(name).resolve_policy_objective()(
        PolicyObjectiveContext(args=args, ppo_kl=ratio.ppo_kl, log_probs=ratio.log_probs, advantages=advantages)
    )
    expected = compute_policy_loss(expected_kl, advantages, args.eps_clip, args.eps_clip_high)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_gspo_ratio_dispatch_matches_existing_helper():
    args = _args()
    full_log_probs = [torch.tensor([-0.2, -0.4]), torch.tensor([-0.1])]
    full_old_log_probs = [torch.tensor([-0.3, -0.5]), torch.tensor([-0.2])]
    local_log_probs = [item.clone() for item in full_log_probs]
    masks = [torch.ones(2), torch.ones(1)]
    actual = get_algorithm("gspo").resolve_ratio_builder()(
        RatioContext(
            args=args,
            log_probs=local_log_probs,
            old_log_probs=[item.clone() for item in full_old_log_probs],
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            loss_masks=masks,
        )
    )
    expected = compute_gspo_kl(full_log_probs, full_old_log_probs, local_log_probs, masks)
    torch.testing.assert_close(actual.ppo_kl, expected)


@pytest.mark.parametrize("name", ["sapo", "cispo"])
def test_specialized_policy_objective_dispatch_matches_existing_helper(name):
    args = _args()
    ppo_kl = torch.tensor([0.1, -0.2])
    log_probs = torch.tensor([-0.5, -0.7], requires_grad=True)
    advantages = torch.tensor([1.0, -1.0])
    context = PolicyObjectiveContext(args=args, ppo_kl=ppo_kl, log_probs=log_probs, advantages=advantages)
    actual = get_algorithm(name).resolve_policy_objective()(context)
    if name == "sapo":
        expected = compute_sapo_loss(ppo_kl, advantages, args.sapo_tau_pos, args.sapo_tau_neg)
    else:
        expected = compute_cispo_loss(log_probs, ppo_kl, advantages, args.eps_clip, args.eps_clip_high)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
