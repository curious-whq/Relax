# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio

import pytest

from examples.algorithms import gdpo_reward
from relax.utils.types import Sample


class _InlineExecutorLoop:
    def run_in_executor(self, executor, function, *args):
        del executor

        async def run():
            return function(*args)

        return run()


def test_gdpo_reward_supports_single_and_batched_calls(monkeypatch):
    monkeypatch.setattr(gdpo_reward, "grade_answer_verl", lambda response, label: label in response)
    monkeypatch.setattr(gdpo_reward.asyncio, "get_running_loop", _InlineExecutorLoop)
    boxed = Sample(response="reasoning \\\\boxed{42}", label="42")
    plain = Sample(response="42", label="42")

    single_reward = asyncio.run(gdpo_reward.reward_func(None, boxed))
    batched_rewards = asyncio.run(gdpo_reward.reward_func(None, [boxed, plain]))

    assert single_reward == {"correctness": 1.0, "format": 1.0}
    assert batched_rewards == [
        {"correctness": 1.0, "format": 1.0},
        {"correctness": 1.0, "format": 0.0},
    ]


def test_gdpo_reward_rejects_invalid_input():
    with pytest.raises(TypeError, match="Sample or a list of Samples"):
        asyncio.run(gdpo_reward.reward_func(None, "invalid"))
