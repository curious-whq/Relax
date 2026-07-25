# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import math
from argparse import Namespace
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch
import torch.distributed as dist

from relax.utils.training.ppo_utils import (
    compute_cispo_loss,
    compute_gspo_kl,
    compute_policy_loss,
    compute_sapo_loss,
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)
from relax.utils.types import Sample


GDPO_EPSILON = 1e-4
LEGACY_REWARD_EPSILON = 1e-6


@dataclass
class AdvantageContext:
    args: Namespace
    rewards: list[float]
    kl: list[torch.Tensor]
    values: list[torch.Tensor] | None
    response_lengths: list[int]
    loss_masks: list[torch.Tensor]
    total_lengths: list[int]
    cp_rank: int = 0
    batch_standardize: Callable[[torch.Tensor], torch.Tensor] | None = None


@dataclass
class AdvantageResult:
    advantages: list[torch.Tensor]
    returns: list[torch.Tensor]


@dataclass
class RatioContext:
    args: Namespace
    log_probs: list[torch.Tensor]
    old_log_probs: list[torch.Tensor]
    full_log_probs: list[torch.Tensor] | None
    full_old_log_probs: list[torch.Tensor] | None
    loss_masks: list[torch.Tensor]


@dataclass
class RatioResult:
    ppo_kl: torch.Tensor
    log_probs: torch.Tensor
    old_log_probs: torch.Tensor


@dataclass
class PolicyObjectiveContext:
    args: Namespace
    ppo_kl: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor


def extract_scalar_reward(args: Namespace, sample: Sample, sample_position: int | None = None) -> tuple[Any]:
    del sample_position
    return (sample.get_reward_value(args),)


def _validate_reward_value(value: Any, *, key: str, sample: Sample, sample_position: int | None) -> float:
    location = f"sample position {sample_position}" if sample_position is not None else "sample"
    if sample.group_index is not None:
        location += f", group {sample.group_index}"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"GDPO reward {key!r} at {location} must be a real number, got {type(value).__name__}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"GDPO reward {key!r} at {location} must be finite, got {value}.")
    return value


def extract_multi_reward(
    args: Namespace,
    sample: Sample,
    sample_position: int | None = None,
) -> tuple[float, ...]:
    reward_keys = getattr(args, "reward_keys", None) or ()
    if not isinstance(sample.reward, Mapping):
        location = f" at sample position {sample_position}" if sample_position is not None else ""
        raise TypeError(f"GDPO requires sample.reward to be a mapping{location}, got {type(sample.reward).__name__}.")

    values = []
    for key in reward_keys:
        if key not in sample.reward:
            location = f"sample position {sample_position}" if sample_position is not None else "sample"
            raise KeyError(f"Missing GDPO reward key {key!r} at {location}, group {sample.group_index}.")
        values.append(
            _validate_reward_value(
                sample.reward[key],
                key=key,
                sample=sample,
                sample_position=sample_position,
            )
        )
    return tuple(values)


def reward_group_has_variance(
    args: Namespace,
    samples: list[Sample],
    reward_extractor: Callable,
) -> bool:
    if not samples:
        return False
    reward_vectors = [reward_extractor(args, sample, position) for position, sample in enumerate(samples)]
    first = reward_vectors[0]
    return any(any(vector[index] != first[index] for vector in reward_vectors[1:]) for index in range(len(first)))


def reward_vector_label(args: Namespace, sample: Sample, reward_extractor: Callable) -> str:
    vector = reward_extractor(args, sample)
    if len(vector) == 1:
        return str(round(vector[0], 1))
    return "_".join(f"{key}={round(value, 1)}" for key, value in zip(args.reward_keys, vector, strict=True))


def _extract_scalar_rewards(args: Namespace, samples: list[Sample]) -> list[Any]:
    return [extract_scalar_reward(args, sample, position)[0] for position, sample in enumerate(samples)]


def _positions_by_group(args: Namespace, samples: list[Sample]) -> dict[int, list[int]]:
    positions_by_group: dict[int, list[int]] = {}
    for position, sample in enumerate(samples):
        if sample.group_index is None:
            raise ValueError("Sample.group_index is required for group reward normalization.")
        positions_by_group.setdefault(sample.group_index, []).append(position)

    expected_size = args.n_samples_per_prompt
    for group_index, positions in positions_by_group.items():
        if len(positions) != expected_size:
            raise ValueError(f"Reward group {group_index} has {len(positions)} samples, expected {expected_size}.")
    return positions_by_group


def identity_reward_processor(args: Namespace, samples: list[Sample]) -> tuple[list[Any], list[Any]]:
    raw_rewards = _extract_scalar_rewards(args, samples)
    return raw_rewards, raw_rewards


def _legacy_group_normalize(
    args: Namespace,
    samples: list[Sample],
    *,
    divide_by_std: bool,
) -> tuple[list[Any], list[float]]:
    raw_rewards = _extract_scalar_rewards(args, samples)
    if not args.rewards_normalization:
        return raw_rewards, raw_rewards

    rewards = torch.tensor(raw_rewards, dtype=torch.float)
    normalized_rewards = torch.empty_like(rewards)
    for positions in _positions_by_group(args, samples).values():
        group_rewards = rewards[positions]
        group_rewards = group_rewards - group_rewards.mean()
        if divide_by_std:
            group_rewards = group_rewards / (group_rewards.std() + LEGACY_REWARD_EPSILON)
        normalized_rewards[positions] = group_rewards
    return raw_rewards, normalized_rewards.tolist()


def group_reward_processor(args: Namespace, samples: list[Sample]) -> tuple[list[Any], list[float]]:
    return _legacy_group_normalize(
        args,
        samples,
        divide_by_std=args.grpo_std_normalization,
    )


def group_centered_reward_processor(args: Namespace, samples: list[Sample]) -> tuple[list[Any], list[float]]:
    return _legacy_group_normalize(args, samples, divide_by_std=False)


def gdpo_reward_processor(args: Namespace, samples: list[Sample]) -> tuple[list[float], list[float]]:
    reward_keys = tuple(args.reward_keys)
    extracted_rewards = [extract_multi_reward(args, sample, position) for position, sample in enumerate(samples)]
    reward_vectors = torch.tensor(
        extracted_rewards,
        dtype=torch.float64,
    )
    reward_weights = getattr(args, "reward_weights", None)
    weights = (
        torch.ones(len(reward_keys), dtype=torch.float64)
        if reward_weights is None
        else torch.tensor(reward_weights, dtype=torch.float64)
    )

    pre_batch_advantages = torch.empty(len(samples), dtype=torch.float64)
    for positions in _positions_by_group(args, samples).values():
        group_rewards = reward_vectors[positions]
        centered = group_rewards - group_rewards.mean(dim=0)
        std = group_rewards.std(dim=0)
        normalized = torch.where(
            std > 0,
            centered / (std + GDPO_EPSILON),
            torch.zeros_like(centered),
        )
        pre_batch_advantages[positions] = (normalized * weights).sum(dim=1)

    primary_reward_key = args.reward_key or reward_keys[0]
    primary_index = reward_keys.index(primary_reward_key)
    raw_rewards = [reward_vector[primary_index] for reward_vector in extracted_rewards]
    return raw_rewards, pre_batch_advantages.tolist()


def standardize_sequence_batch(values: torch.Tensor, epsilon: float = GDPO_EPSILON) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    values_accumulator = values.to(torch.float64)
    centered = values_accumulator - values_accumulator.mean()
    std = values_accumulator.std()
    normalized = torch.where(std > 0, centered / (std + epsilon), torch.zeros_like(centered))
    return normalized.to(values.dtype)


def distributed_standardize_sequence_batch(
    values: torch.Tensor,
    *,
    process_group: dist.ProcessGroup,
    epsilon: float = GDPO_EPSILON,
) -> torch.Tensor:
    values_accumulator = values.to(torch.float64)
    sum_and_count = torch.stack(
        (
            values_accumulator.sum(),
            values_accumulator.new_tensor(values_accumulator.numel()),
        )
    )
    dist.all_reduce(sum_and_count, group=process_group)
    total_sum, count = sum_and_count.unbind()
    safe_count = torch.clamp_min(count, 1.0)
    mean = total_sum / safe_count
    squared_deviation_sum = (values_accumulator - mean).square().sum()
    dist.all_reduce(squared_deviation_sum, group=process_group)
    variance = squared_deviation_sum / torch.clamp_min(count - 1.0, 1.0)
    std = torch.sqrt(torch.clamp_min(variance, 0.0))
    normalized = (values_accumulator - mean) / (std + epsilon)
    valid = (count > 1.0) & (std > 0)
    return torch.where(valid, normalized, torch.zeros_like(normalized)).to(values.dtype)


def compute_outcome_advantages(context: AdvantageContext) -> AdvantageResult:
    rewards = torch.tensor(context.rewards, dtype=torch.float32, device=context.kl[0].device)
    returns = get_grpo_returns(rewards, context.kl)
    return AdvantageResult(advantages=list(returns), returns=returns)


def compute_gdpo_advantages(context: AdvantageContext) -> AdvantageResult:
    if context.batch_standardize is None:
        raise RuntimeError("GDPO advantage computation requires a sequence-wise batch standardizer.")
    rewards = torch.tensor(context.rewards, dtype=torch.float32, device=context.kl[0].device)
    rewards = context.batch_standardize(rewards)
    returns = get_grpo_returns(rewards, context.kl)
    return AdvantageResult(advantages=list(returns), returns=returns)


def compute_ppo_advantages(context: AdvantageContext) -> AdvantageResult:
    token_rewards = []
    for reward, kl in zip(context.rewards, context.kl, strict=False):
        kl *= -context.args.kl_coef
        if context.cp_rank == 0:
            kl[-1] += reward
        token_rewards.append(kl)
    advantages, returns = get_advantages_and_returns_batch(
        context.total_lengths,
        context.response_lengths,
        context.values,
        token_rewards,
        context.args.gamma,
        context.args.lambd,
    )
    return AdvantageResult(advantages=advantages, returns=returns)


def compute_reinforce_plus_plus_advantages(context: AdvantageContext) -> AdvantageResult:
    rewards = torch.tensor(context.rewards, dtype=torch.float32, device=context.kl[0].device)
    returns = get_reinforce_plus_plus_returns(
        rewards=rewards,
        kl=context.kl,
        loss_masks=context.loss_masks,
        response_lengths=context.response_lengths,
        total_lengths=context.total_lengths,
        kl_coef=context.args.kl_coef,
        gamma=context.args.gamma,
    )
    return AdvantageResult(advantages=list(returns), returns=returns)


def compute_reinforce_plus_plus_baseline_advantages(context: AdvantageContext) -> AdvantageResult:
    rewards = torch.tensor(context.rewards, dtype=torch.float32, device=context.kl[0].device)
    advantages = get_reinforce_plus_plus_baseline_advantages(
        rewards=rewards,
        kl=context.kl,
        loss_masks=context.loss_masks,
        kl_coef=context.args.kl_coef,
    )
    return AdvantageResult(advantages=advantages, returns=advantages)


def build_token_ratio(context: RatioContext) -> RatioResult:
    old_log_probs = torch.cat(context.old_log_probs, dim=0)
    log_probs = torch.cat(context.log_probs, dim=0)
    return RatioResult(
        ppo_kl=old_log_probs - log_probs,
        log_probs=log_probs,
        old_log_probs=old_log_probs,
    )


def build_gspo_ratio(context: RatioContext) -> RatioResult:
    if context.full_log_probs is None or context.full_old_log_probs is None:
        raise RuntimeError("GSPO ratio computation requires full sequence log probabilities.")
    ppo_kl = compute_gspo_kl(
        full_log_probs=context.full_log_probs,
        full_old_log_probs=context.full_old_log_probs,
        local_log_probs=context.log_probs,
        loss_masks=context.loss_masks,
    )
    return RatioResult(
        ppo_kl=ppo_kl,
        log_probs=torch.cat(context.log_probs, dim=0),
        old_log_probs=torch.cat(context.old_log_probs, dim=0),
    )


def compute_clipped_policy_objective(context: PolicyObjectiveContext) -> tuple[torch.Tensor, torch.Tensor]:
    return compute_policy_loss(
        context.ppo_kl,
        context.advantages,
        context.args.eps_clip,
        context.args.eps_clip_high,
    )


def compute_sapo_policy_objective(context: PolicyObjectiveContext) -> tuple[torch.Tensor, torch.Tensor]:
    return compute_sapo_loss(
        ppo_kl=context.ppo_kl,
        advantages=context.advantages,
        tau_pos=getattr(context.args, "sapo_tau_pos", 1.0),
        tau_neg=getattr(context.args, "sapo_tau_neg", 1.05),
    )


def compute_cispo_policy_objective(context: PolicyObjectiveContext) -> tuple[torch.Tensor, torch.Tensor]:
    return compute_cispo_loss(
        log_probs=context.log_probs,
        ppo_kl=context.ppo_kl,
        advantages=context.advantages,
        eps_clip=context.args.eps_clip,
        eps_clip_high=context.args.eps_clip_high,
    )


def validate_gdpo_config(args: Namespace) -> None:
    reward_keys = args.reward_keys
    if any(not isinstance(key, str) or not key for key in reward_keys):
        raise ValueError("GDPO reward keys must be non-empty strings.")
    if len(set(reward_keys)) != len(reward_keys):
        raise ValueError(f"GDPO reward keys must be unique, got {reward_keys}.")

    reward_weights = getattr(args, "reward_weights", None)
    if reward_weights is not None:
        if len(reward_weights) != len(reward_keys):
            raise ValueError(
                f"--reward-weights has {len(reward_weights)} values, expected {len(reward_keys)} "
                "to match --reward-keys."
            )
        if any(
            isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(float(weight))
            for weight in reward_weights
        ):
            raise ValueError("GDPO reward weights must be finite real numbers.")

    if getattr(args, "reward_key", None) is None:
        args.reward_key = reward_keys[0]
    elif args.reward_key not in reward_keys:
        raise ValueError(f"--reward-key {args.reward_key!r} must be included in --reward-keys for GDPO.")
    if args.n_samples_per_prompt < 2:
        raise ValueError("GDPO requires --n-samples-per-prompt >= 2.")
    if not args.rewards_normalization:
        raise ValueError("GDPO requires reward normalization; remove --disable-rewards-normalization.")
    if not args.grpo_std_normalization:
        raise ValueError("GDPO requires per-reward standard deviation normalization.")
    if args.normalize_advantages:
        raise ValueError("GDPO already performs sequence-wise batch normalization; remove --normalize-advantages.")
    if getattr(args, "custom_reward_post_process_path", None) is not None:
        raise ValueError("GDPO does not support --custom-reward-post-process-path.")

    global_batch_size = getattr(args, "global_batch_size", None)
    rollout_batch_size = getattr(args, "rollout_batch_size", None)
    if global_batch_size is None or global_batch_size <= 0:
        raise ValueError("GDPO requires a positive --global-batch-size.")
    if rollout_batch_size is None or rollout_batch_size <= 0:
        raise ValueError("GDPO requires a positive --rollout-batch-size.")
    if global_batch_size % args.n_samples_per_prompt != 0:
        raise ValueError(
            "GDPO requires --global-batch-size to be divisible by --n-samples-per-prompt "
            "so reward groups are never split across normalization batches."
        )

    rollout_samples = rollout_batch_size * args.n_samples_per_prompt
    if rollout_samples != global_batch_size:
        raise ValueError(
            "GDPO currently requires rollout_batch_size * n_samples_per_prompt to equal "
            "global_batch_size so each rollout is exactly one sequence-wise normalization batch."
        )
