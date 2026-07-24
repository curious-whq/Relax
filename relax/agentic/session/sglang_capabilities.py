# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SGLangRouterKind(str, Enum):
    NATIVE = "native"
    SLIME = "slime"
    EXTERNAL = "external"


class SGLangTopology(str, Enum):
    REGULAR = "regular"
    PD = "pd"


class SGLangCacheKind(str, Enum):
    DEVICE_RADIX = "device_radix"
    DISABLED = "disabled"
    HIERARCHICAL = "hierarchical"


class SGLangSessionWireFormat(str, Enum):
    NONE = "none"
    SESSION_PARAMS_V1 = "session_params_v1"


class SGLangSessionCapabilityReason(str, Enum):
    LIFECYCLE_NOT_READY = "lifecycle_not_ready"
    EXTERNAL_ROUTER_UNVERIFIED = "external_router_unverified"
    SLIME_ROUTER_UNSUPPORTED = "slime_router_unsupported"
    PD_UNSUPPORTED = "pd_unsupported"
    RADIX_CACHE_DISABLED = "radix_cache_disabled"
    HIERARCHICAL_CACHE_UNSUPPORTED = "hierarchical_cache_unsupported"


@dataclass(frozen=True, kw_only=True)
class SGLangCapabilityProfile:
    router_kind: SGLangRouterKind
    topology: SGLangTopology
    cache_kind: SGLangCacheKind
    session_wire_format: SGLangSessionWireFormat
    supports_worker_registry: bool
    supports_session_control: bool
    lifecycle_ready: bool
    unavailable_reason: SGLangSessionCapabilityReason | None

    def __post_init__(self) -> None:
        if not isinstance(self.router_kind, SGLangRouterKind):
            raise ValueError("router_kind must be an SGLangRouterKind")
        if not isinstance(self.topology, SGLangTopology):
            raise ValueError("topology must be an SGLangTopology")
        if not isinstance(self.cache_kind, SGLangCacheKind):
            raise ValueError("cache_kind must be an SGLangCacheKind")
        if not isinstance(self.session_wire_format, SGLangSessionWireFormat):
            raise ValueError("session_wire_format must be an SGLangSessionWireFormat")
        if not isinstance(self.supports_worker_registry, bool):
            raise ValueError("supports_worker_registry must be a boolean")
        if not isinstance(self.supports_session_control, bool):
            raise ValueError("supports_session_control must be a boolean")
        if not isinstance(self.lifecycle_ready, bool):
            raise ValueError("lifecycle_ready must be a boolean")
        if self.unavailable_reason is not None and not isinstance(
            self.unavailable_reason,
            SGLangSessionCapabilityReason,
        ):
            raise ValueError("unavailable_reason must be an SGLangSessionCapabilityReason")
        if self.lifecycle_ready and not self.supports_session_control:
            raise ValueError("lifecycle_ready requires session control support")
        if self.lifecycle_ready and self.session_wire_format == SGLangSessionWireFormat.NONE:
            raise ValueError("lifecycle_ready requires a session wire format")
        if self.lifecycle_ready and self.unavailable_reason is not None:
            raise ValueError("lifecycle_ready capability must not have an unavailable reason")

    @property
    def session_wire_enabled(self) -> bool:
        return (
            self.lifecycle_ready
            and self.supports_session_control
            and self.session_wire_format != SGLangSessionWireFormat.NONE
        )


def _cache_kind(*, radix_cache_disabled: bool, hierarchical_cache_enabled: bool) -> SGLangCacheKind:
    if hierarchical_cache_enabled:
        return SGLangCacheKind.HIERARCHICAL
    if radix_cache_disabled:
        return SGLangCacheKind.DISABLED
    return SGLangCacheKind.DEVICE_RADIX


def _unavailable_profile(
    *,
    router_kind: SGLangRouterKind,
    topology: SGLangTopology,
    cache_kind: SGLangCacheKind,
    supports_worker_registry: bool,
    reason: SGLangSessionCapabilityReason,
) -> SGLangCapabilityProfile:
    return SGLangCapabilityProfile(
        router_kind=router_kind,
        topology=topology,
        cache_kind=cache_kind,
        session_wire_format=SGLangSessionWireFormat.NONE,
        supports_worker_registry=supports_worker_registry,
        supports_session_control=False,
        lifecycle_ready=False,
        unavailable_reason=reason,
    )


def resolve_sglang_capability_profile(
    *,
    router_managed: bool,
    use_slime_router: bool,
    has_pd_disaggregation: bool,
    radix_cache_disabled: bool,
    hierarchical_cache_enabled: bool,
    lifecycle_ready: bool = False,
) -> SGLangCapabilityProfile:
    topology = SGLangTopology.PD if has_pd_disaggregation else SGLangTopology.REGULAR
    cache_kind = _cache_kind(
        radix_cache_disabled=radix_cache_disabled,
        hierarchical_cache_enabled=hierarchical_cache_enabled,
    )
    if not router_managed:
        return _unavailable_profile(
            router_kind=SGLangRouterKind.EXTERNAL,
            topology=topology,
            cache_kind=cache_kind,
            supports_worker_registry=False,
            reason=SGLangSessionCapabilityReason.EXTERNAL_ROUTER_UNVERIFIED,
        )
    if use_slime_router:
        return _unavailable_profile(
            router_kind=SGLangRouterKind.SLIME,
            topology=topology,
            cache_kind=cache_kind,
            supports_worker_registry=False,
            reason=SGLangSessionCapabilityReason.SLIME_ROUTER_UNSUPPORTED,
        )
    if has_pd_disaggregation:
        return _unavailable_profile(
            router_kind=SGLangRouterKind.NATIVE,
            topology=SGLangTopology.PD,
            cache_kind=cache_kind,
            supports_worker_registry=True,
            reason=SGLangSessionCapabilityReason.PD_UNSUPPORTED,
        )
    if hierarchical_cache_enabled:
        return _unavailable_profile(
            router_kind=SGLangRouterKind.NATIVE,
            topology=SGLangTopology.REGULAR,
            cache_kind=SGLangCacheKind.HIERARCHICAL,
            supports_worker_registry=True,
            reason=SGLangSessionCapabilityReason.HIERARCHICAL_CACHE_UNSUPPORTED,
        )
    if radix_cache_disabled:
        return _unavailable_profile(
            router_kind=SGLangRouterKind.NATIVE,
            topology=SGLangTopology.REGULAR,
            cache_kind=SGLangCacheKind.DISABLED,
            supports_worker_registry=True,
            reason=SGLangSessionCapabilityReason.RADIX_CACHE_DISABLED,
        )
    return SGLangCapabilityProfile(
        router_kind=SGLangRouterKind.NATIVE,
        topology=SGLangTopology.REGULAR,
        cache_kind=SGLangCacheKind.DEVICE_RADIX,
        session_wire_format=SGLangSessionWireFormat.SESSION_PARAMS_V1,
        supports_worker_registry=True,
        supports_session_control=True,
        lifecycle_ready=lifecycle_ready,
        unavailable_reason=(None if lifecycle_ready else SGLangSessionCapabilityReason.LIFECYCLE_NOT_READY),
    )


def unavailable_sglang_capability_profile() -> SGLangCapabilityProfile:
    return resolve_sglang_capability_profile(
        router_managed=False,
        use_slime_router=False,
        has_pd_disaggregation=False,
        radix_cache_disabled=False,
        hierarchical_cache_enabled=False,
    )
