# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.core.registry import get_algorithm
from relax.engine.filters.base_types import DynamicFilterOutput
from relax.utils.training.algorithm_ops import reward_group_has_variance, reward_vector_label
from relax.utils.types import Sample


__all__ = ["check_reward_nonzero_std"]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    del kwargs
    spec = get_algorithm(args.advantage_estimator)
    reward_extractor = spec.resolve_reward_extractor()
    keep = reward_group_has_variance(args, samples, reward_extractor)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{reward_vector_label(args, samples[0], reward_extractor)}",
    )
