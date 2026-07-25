# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from dataclasses import replace

import pytest

from relax.core.registry import (
    ALGORITHM_REGISTRY,
    ALGOS,
    AlgorithmRegistry,
    AlgorithmSpec,
    get_algorithm,
    registered_algorithm_names,
)


EXPECTED_ALGORITHMS = {
    "grpo",
    "gspo",
    "sapo",
    "cispo",
    "reinforce_plus_plus",
    "reinforce_plus_plus_baseline",
    "ppo",
    "gdpo",
}


def test_registered_algorithm_names_are_single_source_for_controller_view():
    assert set(registered_algorithm_names()) == EXPECTED_ALGORITHMS
    assert EXPECTED_ALGORITHMS <= set(ALGOS)
    assert "sft" in ALGOS


def test_algorithm_capabilities_are_registered():
    assert get_algorithm("gspo").capabilities.needs_full_log_probs
    assert get_algorithm("gdpo").capabilities.min_reward_keys == 2
    assert get_algorithm("gdpo").capabilities.performs_batch_advantage_normalization
    assert get_algorithm("reinforce_plus_plus").capabilities.requires_advantage_whitening
    assert get_algorithm("ppo").capabilities.uses_critic
    assert not get_algorithm("ppo").capabilities.enabled


def test_registry_rejects_duplicate_names():
    registry = AlgorithmRegistry()
    spec = get_algorithm("grpo")
    registry.register(spec)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec)


def test_registry_unknown_name_lists_available_algorithms():
    with pytest.raises(ValueError, match="Registered algorithms.*grpo"):
        ALGORITHM_REGISTRY.get("missing")


def test_registry_rejects_non_lowercase_name():
    registry = AlgorithmRegistry()
    with pytest.raises(ValueError, match="lowercase"):
        registry.register(replace(get_algorithm("grpo"), name="GRPO"))


def test_registry_reserves_sft_service_name():
    registry = AlgorithmRegistry()
    with pytest.raises(ValueError, match="reserved"):
        registry.register(replace(get_algorithm("grpo"), name="sft"))


def test_live_controller_view_sees_runtime_registration(monkeypatch):
    spec = replace(get_algorithm("grpo"), name="test_runtime_algorithm")
    monkeypatch.setitem(ALGORITHM_REGISTRY._specs, spec.name, spec)

    assert spec.name in ALGOS
    assert ALGOS.get(spec.name) is not None
    assert ALGOS.get(spec.name).copy()


def test_all_specs_have_explicit_dispatch_hooks():
    for spec in ALGORITHM_REGISTRY.specs():
        assert isinstance(spec, AlgorithmSpec)
        assert spec.reward_extractor.path
        assert spec.reward_processor.path
        assert spec.advantage_estimator.path
        assert spec.ratio_builder.path
        assert spec.policy_objective.path


def test_algos_copy_returns_mutable_isolated_role_mapping():
    first = ALGOS["grpo"].copy()
    second = ALGOS["grpo"].copy()
    first["extra"] = object()
    assert "extra" not in second
