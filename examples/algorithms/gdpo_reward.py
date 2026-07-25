# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Minimal two-dimensional reward used by the GDPO training example."""

import asyncio
from argparse import Namespace
from typing import Any

from relax.engine.rewards.math_dapo_utils import last_boxed_only_string
from relax.engine.rewards.math_utils import grade_answer_verl
from relax.utils.types import Sample


def _score_sample(sample: Sample) -> dict[str, float]:
    if sample.label is None:
        raise ValueError("The GDPO example requires each sample to have a label.")

    return {
        "correctness": float(grade_answer_verl(sample.response, sample.label)),
        "format": float(last_boxed_only_string(sample.response) is not None),
    }


async def reward_func(
    args: Namespace,
    sample_or_samples: Sample | list[Sample],
    **kwargs: Any,
) -> dict[str, float] | list[dict[str, float]]:
    """Return independent correctness and format rewards."""
    del args, kwargs
    loop = asyncio.get_running_loop()
    if isinstance(sample_or_samples, Sample):
        return await loop.run_in_executor(None, _score_sample, sample_or_samples)
    if not isinstance(sample_or_samples, list) or not all(isinstance(sample, Sample) for sample in sample_or_samples):
        raise TypeError("The GDPO reward function expects a Sample or a list of Samples.")

    tasks = [loop.run_in_executor(None, _score_sample, sample) for sample in sample_or_samples]
    return await asyncio.gather(*tasks)
