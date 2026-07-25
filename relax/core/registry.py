# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


try:
    from enum import StrEnum
except ImportError:
    # Python 3.10 compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return self.value


@dataclass(frozen=True)
class SymbolRef:
    """Lazy reference to a Python symbol."""

    path: str

    def resolve(self) -> Any:
        return _resolve_symbol(self.path)


@lru_cache(maxsize=None)
def _resolve_symbol(path: str) -> Any:
    module_path, separator, attr = path.rpartition(".")
    if not separator:
        raise ValueError(f"Symbol path must include a module and attribute: {path!r}")
    return getattr(importlib.import_module(module_path), attr)


@dataclass(frozen=True)
class AlgorithmCapabilities:
    """Properties used by orchestration without checking algorithm names."""

    grouped_rewards: bool = False
    reports_group_reward_variance: bool = True
    min_reward_keys: int = 0
    requires_complete_groups: bool = False
    performs_batch_advantage_normalization: bool = False
    requires_advantage_whitening: bool = False
    needs_full_log_probs: bool = False
    uses_critic: bool = False
    supports_fully_async: bool = True
    enabled: bool = True
    disabled_reason: str | None = None


@dataclass(frozen=True)
class AlgorithmSpec:
    """Declarative algorithm registration."""

    name: str
    reward_extractor: SymbolRef
    reward_processor: SymbolRef
    advantage_estimator: SymbolRef
    ratio_builder: SymbolRef
    policy_objective: SymbolRef
    capabilities: AlgorithmCapabilities = AlgorithmCapabilities()
    validator: SymbolRef | None = None
    service_topology: str = "rl"
    cli_visible: bool = True

    def resolve_reward_extractor(self):
        return self.reward_extractor.resolve()

    def resolve_reward_processor(self):
        return self.reward_processor.resolve()

    def resolve_advantage_estimator(self):
        return self.advantage_estimator.resolve()

    def resolve_ratio_builder(self):
        return self.ratio_builder.resolve()

    def resolve_policy_objective(self):
        return self.policy_objective.resolve()

    def validate(self, args) -> None:
        if not self.capabilities.enabled:
            raise ValueError(self.capabilities.disabled_reason or f"Algorithm {self.name!r} is disabled.")
        if self.capabilities.min_reward_keys:
            reward_keys = getattr(args, "reward_keys", None)
            if not isinstance(reward_keys, (list, tuple)) or len(reward_keys) < self.capabilities.min_reward_keys:
                raise ValueError(
                    f"Algorithm {self.name!r} requires at least {self.capabilities.min_reward_keys} "
                    "reward keys via --reward-keys."
                )
        if self.validator is not None:
            self.validator.resolve()(args)


class AlgorithmRegistry:
    """Name-to-spec registry with duplicate protection."""

    def __init__(self) -> None:
        self._specs: dict[str, AlgorithmSpec] = {}

    def register(self, spec: AlgorithmSpec) -> AlgorithmSpec:
        if not spec.name or spec.name != spec.name.lower():
            raise ValueError(f"Algorithm name must be a non-empty lowercase string, got {spec.name!r}.")
        if spec.name == "sft":
            raise ValueError("Algorithm name 'sft' is reserved for the supervised-fine-tuning service topology.")
        if spec.name in self._specs:
            raise ValueError(f"Algorithm {spec.name!r} is already registered.")
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> AlgorithmSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            available = ", ".join(self.names())
            raise ValueError(f"Unknown algorithm {name!r}. Registered algorithms: {available}.") from exc

    def names(self, *, cli_visible: bool | None = None) -> tuple[str, ...]:
        return tuple(
            name for name, spec in self._specs.items() if cli_visible is None or spec.cli_visible is cli_visible
        )

    def specs(self) -> tuple[AlgorithmSpec, ...]:
        return tuple(self._specs.values())


ALGORITHM_REGISTRY = AlgorithmRegistry()

_OPS = "relax.utils.training.algorithm_ops"
_SCALAR_EXTRACTOR = SymbolRef(f"{_OPS}.extract_scalar_reward")
_MULTI_REWARD_EXTRACTOR = SymbolRef(f"{_OPS}.extract_multi_reward")
_IDENTITY_REWARDS = SymbolRef(f"{_OPS}.identity_reward_processor")
_GROUP_REWARDS = SymbolRef(f"{_OPS}.group_reward_processor")
_GROUP_CENTERED_REWARDS = SymbolRef(f"{_OPS}.group_centered_reward_processor")
_GDPO_REWARDS = SymbolRef(f"{_OPS}.gdpo_reward_processor")
_OUTCOME_ADVANTAGE = SymbolRef(f"{_OPS}.compute_outcome_advantages")
_GDPO_ADVANTAGE = SymbolRef(f"{_OPS}.compute_gdpo_advantages")
_PPO_ADVANTAGE = SymbolRef(f"{_OPS}.compute_ppo_advantages")
_REINFORCE_ADVANTAGE = SymbolRef(f"{_OPS}.compute_reinforce_plus_plus_advantages")
_REINFORCE_BASELINE_ADVANTAGE = SymbolRef(f"{_OPS}.compute_reinforce_plus_plus_baseline_advantages")
_TOKEN_RATIO = SymbolRef(f"{_OPS}.build_token_ratio")
_GSPO_RATIO = SymbolRef(f"{_OPS}.build_gspo_ratio")
_CLIPPED_OBJECTIVE = SymbolRef(f"{_OPS}.compute_clipped_policy_objective")
_SAPO_OBJECTIVE = SymbolRef(f"{_OPS}.compute_sapo_policy_objective")
_CISPO_OBJECTIVE = SymbolRef(f"{_OPS}.compute_cispo_policy_objective")


def _register_builtin_algorithms() -> None:
    grouped = AlgorithmCapabilities(grouped_rewards=True, requires_complete_groups=True)
    for name, objective in (
        ("grpo", _CLIPPED_OBJECTIVE),
        ("sapo", _SAPO_OBJECTIVE),
        ("cispo", _CISPO_OBJECTIVE),
    ):
        ALGORITHM_REGISTRY.register(
            AlgorithmSpec(
                name=name,
                reward_extractor=_SCALAR_EXTRACTOR,
                reward_processor=_GROUP_REWARDS,
                advantage_estimator=_OUTCOME_ADVANTAGE,
                ratio_builder=_TOKEN_RATIO,
                policy_objective=objective,
                capabilities=grouped,
            )
        )

    ALGORITHM_REGISTRY.register(
        AlgorithmSpec(
            name="gspo",
            reward_extractor=_SCALAR_EXTRACTOR,
            reward_processor=_GROUP_REWARDS,
            advantage_estimator=_OUTCOME_ADVANTAGE,
            ratio_builder=_GSPO_RATIO,
            policy_objective=_CLIPPED_OBJECTIVE,
            capabilities=AlgorithmCapabilities(
                grouped_rewards=True,
                requires_complete_groups=True,
                needs_full_log_probs=True,
            ),
        )
    )
    ALGORITHM_REGISTRY.register(
        AlgorithmSpec(
            name="reinforce_plus_plus",
            reward_extractor=_SCALAR_EXTRACTOR,
            reward_processor=_IDENTITY_REWARDS,
            advantage_estimator=_REINFORCE_ADVANTAGE,
            ratio_builder=_TOKEN_RATIO,
            policy_objective=_CLIPPED_OBJECTIVE,
            capabilities=AlgorithmCapabilities(requires_advantage_whitening=True),
        )
    )
    ALGORITHM_REGISTRY.register(
        AlgorithmSpec(
            name="reinforce_plus_plus_baseline",
            reward_extractor=_SCALAR_EXTRACTOR,
            reward_processor=_GROUP_CENTERED_REWARDS,
            advantage_estimator=_REINFORCE_BASELINE_ADVANTAGE,
            ratio_builder=_TOKEN_RATIO,
            policy_objective=_CLIPPED_OBJECTIVE,
            capabilities=AlgorithmCapabilities(
                grouped_rewards=True,
                requires_complete_groups=True,
                requires_advantage_whitening=True,
            ),
        )
    )
    ALGORITHM_REGISTRY.register(
        AlgorithmSpec(
            name="ppo",
            reward_extractor=_SCALAR_EXTRACTOR,
            reward_processor=_IDENTITY_REWARDS,
            advantage_estimator=_PPO_ADVANTAGE,
            ratio_builder=_TOKEN_RATIO,
            policy_objective=_CLIPPED_OBJECTIVE,
            capabilities=AlgorithmCapabilities(
                uses_critic=True,
                reports_group_reward_variance=False,
                enabled=False,
                disabled_reason=(
                    "PPO (Proximal Policy Optimization) is no longer supported in Relax. "
                    "Please use one of the registered enabled advantage estimators instead."
                ),
            ),
        )
    )
    ALGORITHM_REGISTRY.register(
        AlgorithmSpec(
            name="gdpo",
            reward_extractor=_MULTI_REWARD_EXTRACTOR,
            reward_processor=_GDPO_REWARDS,
            advantage_estimator=_GDPO_ADVANTAGE,
            ratio_builder=_TOKEN_RATIO,
            policy_objective=_CLIPPED_OBJECTIVE,
            validator=SymbolRef(f"{_OPS}.validate_gdpo_config"),
            capabilities=AlgorithmCapabilities(
                grouped_rewards=True,
                min_reward_keys=2,
                requires_complete_groups=True,
                performs_batch_advantage_normalization=True,
            ),
        )
    )


_register_builtin_algorithms()


def register_algorithm(spec: AlgorithmSpec) -> AlgorithmSpec:
    return ALGORITHM_REGISTRY.register(spec)


def get_algorithm(name: str) -> AlgorithmSpec:
    return ALGORITHM_REGISTRY.get(name)


def registered_algorithm_names() -> tuple[str, ...]:
    return ALGORITHM_REGISTRY.names(cli_visible=True)


# NOTE(dev): Use StrEnum and keep visiting order with definition order
class ROLES(StrEnum):
    actor: str = "actor"
    critic: str = "critic"
    rollout: str = "rollout"
    advantages: str = "advantages"
    reference: str = "reference"
    actor_fwd: str = "actor_fwd"
    sft: str = "sft"


class ROLES_TRAIN_ONLY(StrEnum):
    actor: str = "actor"


class ROLES_ROLLOUT_ONLY(StrEnum):
    rollout: str = "rollout"


class ROLES_COLOCATE(StrEnum):
    actor: str = "actor"
    critic: str = "critic"
    rollout: str = "rollout"


class ROLES_SFT_ONLY(StrEnum):
    sft: str = "sft"
    actor: str = "actor"


class ROLES_FULLY_ASYNC_ON_POLICY(StrEnum):
    actor: str = "actor"
    critic: str = "critic"
    rollout: str = "rollout"
    advantages: str = "advantages"
    reference: str = "reference"


class _LazyRoleMapping(Mapping):
    def __init__(self, refs: Mapping[Any, SymbolRef]) -> None:
        self._refs = dict(refs)

    def __getitem__(self, key: Any) -> Any:
        return self._refs[key].resolve()

    def __iter__(self) -> Iterator[Any]:
        return iter(self._refs)

    def __len__(self) -> int:
        return len(self._refs)

    def copy(self) -> dict:
        return {role: ref.resolve() for role, ref in self._refs.items()}


_RL_SERVICE_REFS = {
    ROLES.rollout: SymbolRef("relax.components.rollout.Rollout"),
    ROLES.actor: SymbolRef("relax.components.actor.Actor"),
    ROLES.advantages: SymbolRef("relax.components.advantages.Advantages"),
    ROLES.reference: SymbolRef("relax.components.actor_fwd.ActorFwd"),
    ROLES.actor_fwd: SymbolRef("relax.components.actor_fwd.ActorFwd"),
}
_SFT_SERVICE_REFS = {
    ROLES.sft: SymbolRef("relax.components.sft.SFT"),
    ROLES.actor: SymbolRef("relax.components.actor.Actor"),
}
_SERVICE_TOPOLOGIES = {
    "rl": _RL_SERVICE_REFS,
    "sft": _SFT_SERVICE_REFS,
}


class _AlgorithmServiceMapping(Mapping):
    """Backward-compatible live view used by the Controller."""

    def __getitem__(self, name: str) -> _LazyRoleMapping:
        if name == "sft":
            topology = "sft"
        else:
            try:
                topology = ALGORITHM_REGISTRY.get(name).service_topology
            except ValueError as exc:
                raise KeyError(name) from exc
        try:
            return _LazyRoleMapping(_SERVICE_TOPOLOGIES[topology])
        except KeyError as exc:
            raise KeyError(f"Unknown service topology {topology!r} for algorithm {name!r}.") from exc

    def __iter__(self) -> Iterator[str]:
        yield from ALGORITHM_REGISTRY.names()
        yield "sft"

    def __len__(self) -> int:
        return len(ALGORITHM_REGISTRY.names()) + 1


# Backward-compatible live Controller view, derived from the algorithm registry.
ALGOS: Mapping[str, _LazyRoleMapping] = _AlgorithmServiceMapping()


def process_role(config):
    if config.debug_rollout_only:
        return ROLES_ROLLOUT_ONLY
    if config.debug_train_only:
        return ROLES_TRAIN_ONLY
    if getattr(config, "loss_type", None) == "sft":
        return ROLES_SFT_ONLY
    if config.hybrid:
        # hybrid mode: actor handles ref/actor_fwd internally
        # via _switch_model, only need actor + rollout services
        return ROLES_COLOCATE
    if config.fully_async:
        if getattr(config, "true_on_policy_mode", False):
            # actor_fwd's log_probs equal the train forward's log_probs in this regime
            # (same weights, deterministic Megatron forward), so we recompute inline.
            return ROLES_FULLY_ASYNC_ON_POLICY
        return ROLES
    return ROLES_COLOCATE
