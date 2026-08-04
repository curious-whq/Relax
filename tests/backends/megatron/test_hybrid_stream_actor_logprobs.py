# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture()
def actor_module():
    pytest.importorskip("torch")
    pytest.importorskip("megatron.core")
    from relax.backends.megatron import actor

    return actor


def _make_actor(actor_module, *, partition_complete: bool | None = None):
    instance = object.__new__(actor_module.MegatronTrainRayActor)
    instance.args = Namespace(
        compute_advantages_and_returns=True,
        use_dynamic_batch_size=False,
        log_probs_max_tokens_per_gpu=1,
        max_tokens_per_gpu=1,
        use_rollout_routing_replay=False,
        use_routing_replay=False,
        keep_old_actor=False,
        use_rollout_logprobs=False,
        get_mismatch_metrics=False,
        debug_train_only=False,
        hybrid_stream_actor_logprobs=True,
    )
    instance.model = []
    instance.weights_backuper = SimpleNamespace(backup_tags=set())
    instance._switch_model = Mock()
    instance.compute_log_prob = Mock(return_value={"log_probs": []})
    if partition_complete is not None:
        instance.data_system_client = SimpleNamespace(
            async_check_production_completed=Mock(return_value=partition_complete)
        )
    return instance


@pytest.mark.parametrize(("precomputed", "expected_calls"), [(False, 1), (True, 0)])
def test_hybrid_forward_skips_only_precomputed_actor_logprobs(actor_module, monkeypatch, precomputed, expected_calls):
    instance = _make_actor(actor_module)
    monkeypatch.setattr(actor_module, "get_data_iterator", lambda *_args, **_kwargs: ([object()], [1]))

    instance._hybrid_forward_subbatch({}, actor_log_probs_precomputed=precomputed)

    assert instance.compute_log_prob.call_count == expected_calls


def test_hybrid_stream_falls_back_when_partition_is_complete(actor_module, monkeypatch):
    instance = _make_actor(actor_module, partition_complete=True)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: object())
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda **_kwargs: 0)
    monkeypatch.setattr(actor_module.dist, "broadcast_object_list", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module, "run", lambda value: value)

    assert instance._should_stream_hybrid_actor_logprobs(3) is False
    instance.data_system_client.async_check_production_completed.assert_called_once_with("train_3")
