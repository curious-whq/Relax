# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import pytest

from relax.agentic.session.sglang_capabilities import (
    SGLangCacheKind,
    SGLangCapabilityProfile,
    SGLangRouterKind,
    SGLangSessionCapabilityReason,
    SGLangSessionWireFormat,
    SGLangTopology,
    resolve_sglang_capability_profile,
    unavailable_sglang_capability_profile,
)


def _profile(**overrides) -> SGLangCapabilityProfile:
    values = {
        "router_managed": True,
        "use_slime_router": False,
        "has_pd_disaggregation": False,
        "radix_cache_disabled": False,
        "hierarchical_cache_enabled": False,
        "lifecycle_ready": False,
    }
    values.update(overrides)
    return resolve_sglang_capability_profile(**values)


def test_native_regular_radix_profile_exposes_dormant_session_wire() -> None:
    profile = _profile()

    assert profile.router_kind == SGLangRouterKind.NATIVE
    assert profile.topology == SGLangTopology.REGULAR
    assert profile.cache_kind == SGLangCacheKind.DEVICE_RADIX
    assert profile.session_wire_format == SGLangSessionWireFormat.SESSION_PARAMS_V1
    assert profile.supports_worker_registry
    assert profile.supports_session_control
    assert not profile.lifecycle_ready
    assert not profile.session_wire_enabled
    assert profile.unavailable_reason == SGLangSessionCapabilityReason.LIFECYCLE_NOT_READY


def test_native_regular_radix_profile_can_be_armed_after_lifecycle_prerequisites() -> None:
    profile = _profile(lifecycle_ready=True)

    assert profile.lifecycle_ready
    assert profile.session_wire_enabled
    assert profile.unavailable_reason is None


@pytest.mark.parametrize(
    ("overrides", "router_kind", "topology", "cache_kind", "reason"),
    [
        (
            {"router_managed": False},
            SGLangRouterKind.EXTERNAL,
            SGLangTopology.REGULAR,
            SGLangCacheKind.DEVICE_RADIX,
            SGLangSessionCapabilityReason.EXTERNAL_ROUTER_UNVERIFIED,
        ),
        (
            {"use_slime_router": True},
            SGLangRouterKind.SLIME,
            SGLangTopology.REGULAR,
            SGLangCacheKind.DEVICE_RADIX,
            SGLangSessionCapabilityReason.SLIME_ROUTER_UNSUPPORTED,
        ),
        (
            {"has_pd_disaggregation": True},
            SGLangRouterKind.NATIVE,
            SGLangTopology.PD,
            SGLangCacheKind.DEVICE_RADIX,
            SGLangSessionCapabilityReason.PD_UNSUPPORTED,
        ),
        (
            {"radix_cache_disabled": True},
            SGLangRouterKind.NATIVE,
            SGLangTopology.REGULAR,
            SGLangCacheKind.DISABLED,
            SGLangSessionCapabilityReason.RADIX_CACHE_DISABLED,
        ),
        (
            {"hierarchical_cache_enabled": True},
            SGLangRouterKind.NATIVE,
            SGLangTopology.REGULAR,
            SGLangCacheKind.HIERARCHICAL,
            SGLangSessionCapabilityReason.HIERARCHICAL_CACHE_UNSUPPORTED,
        ),
    ],
)
def test_unsupported_deployment_profiles_fail_open(
    overrides: dict[str, bool],
    router_kind: SGLangRouterKind,
    topology: SGLangTopology,
    cache_kind: SGLangCacheKind,
    reason: SGLangSessionCapabilityReason,
) -> None:
    profile = _profile(**overrides, lifecycle_ready=True)

    assert profile.router_kind == router_kind
    assert profile.topology == topology
    assert profile.cache_kind == cache_kind
    assert profile.session_wire_format == SGLangSessionWireFormat.NONE
    assert not profile.supports_session_control
    assert not profile.lifecycle_ready
    assert not profile.session_wire_enabled
    assert profile.unavailable_reason == reason


def test_hierarchical_cache_reason_takes_precedence_over_disabled_radix() -> None:
    profile = _profile(
        radix_cache_disabled=True,
        hierarchical_cache_enabled=True,
    )

    assert profile.cache_kind == SGLangCacheKind.HIERARCHICAL
    assert profile.unavailable_reason == SGLangSessionCapabilityReason.HIERARCHICAL_CACHE_UNSUPPORTED


def test_default_unavailable_profile_never_enables_session_wire() -> None:
    profile = unavailable_sglang_capability_profile()

    assert profile.router_kind == SGLangRouterKind.EXTERNAL
    assert not profile.session_wire_enabled


def test_capability_profile_rejects_ready_without_session_control() -> None:
    with pytest.raises(ValueError, match="requires session control"):
        SGLangCapabilityProfile(
            router_kind=SGLangRouterKind.NATIVE,
            topology=SGLangTopology.REGULAR,
            cache_kind=SGLangCacheKind.DEVICE_RADIX,
            session_wire_format=SGLangSessionWireFormat.SESSION_PARAMS_V1,
            supports_worker_registry=True,
            supports_session_control=False,
            lifecycle_ready=True,
            unavailable_reason=None,
        )
