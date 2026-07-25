# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

from relax.utils import utils


class _RecordingSpec:
    def __init__(self) -> None:
        self.called = False

    def resolve_reward_processor(self):
        def process(args, samples):
            self.called = True
            return [args.advantage_estimator], list(samples)

        return process


def test_post_process_rewards_dispatches_through_algorithm_registry(monkeypatch):
    spec = _RecordingSpec()
    requested_names = []

    def fake_get_algorithm(name):
        requested_names.append(name)
        return spec

    monkeypatch.setattr(utils, "get_algorithm", fake_get_algorithm)
    args = Namespace(advantage_estimator="registered_test", custom_reward_post_process_path=None)

    raw_rewards, processed_rewards = utils.post_process_rewards(args, [1.0, 2.0])

    assert requested_names == ["registered_test"]
    assert spec.called
    assert raw_rewards == ["registered_test"]
    assert processed_rewards == [1.0, 2.0]
