# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class AdmissionAction(str, Enum):
    BYPASS = "bypass"
    ADMIT = "admit"
    DEFER = "defer"


class AdmissionReason(str, Enum):
    FEATURE_DISABLED = "feature_disabled"
    MISSING_IDENTITY = "missing_identity"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    DEGRADED = "degraded"
    CAPACITY_AVAILABLE = "capacity_available"
    CAPACITY_EXHAUSTED = "capacity_exhausted"
    PRESSURE_GUARD = "pressure_guard"
    FAIRNESS_RESERVE = "fairness_reserve"


class BudgetAcquireStatus(str, Enum):
    ACQUIRED = "acquired"
    CAPACITY_EXHAUSTED = "capacity_exhausted"
    BYPASS = "bypass"
    UNKNOWN = "unknown"


class LeaseReleaseOutcome(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REQUEUED = "requeued"
    STALE = "stale"


def _require_non_empty(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_optional_non_empty(value: str | None, *, field_name: str) -> None:
    if value is not None:
        _require_non_empty(value, field_name=field_name)


def _require_non_negative_int(value: int, *, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


@dataclass(frozen=True, kw_only=True)
class AgenticIdentity:
    program_id: str
    program_owner_key: str
    root_session_id: str
    engine_session_id: str
    parent_engine_session_id: str | None

    def __post_init__(self) -> None:
        for field_name in (
            "program_id",
            "program_owner_key",
            "root_session_id",
            "engine_session_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name=field_name)
        _require_optional_non_empty(
            self.parent_engine_session_id,
            field_name="parent_engine_session_id",
        )

    def to_payload(self) -> dict[str, str | None]:
        return {
            "program_id": self.program_id,
            "program_owner_key": self.program_owner_key,
            "root_session_id": self.root_session_id,
            "engine_session_id": self.engine_session_id,
            "parent_engine_session_id": self.parent_engine_session_id,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "AgenticIdentity":
        return cls(
            program_id=payload["program_id"],
            program_owner_key=payload["program_owner_key"],
            root_session_id=payload["root_session_id"],
            engine_session_id=payload["engine_session_id"],
            parent_engine_session_id=payload.get("parent_engine_session_id"),
        )


@dataclass(frozen=True, kw_only=True)
class SessionControlRef:
    program_owner_key: str
    engine_session_id: str
    owner_epoch: int

    def __post_init__(self) -> None:
        _require_non_empty(self.program_owner_key, field_name="program_owner_key")
        _require_non_empty(self.engine_session_id, field_name="engine_session_id")
        _require_non_negative_int(self.owner_epoch, field_name="owner_epoch")


@dataclass(frozen=True, kw_only=True)
class SessionRegistrationGrant:
    control_ref: SessionControlRef
    credential: str = field(repr=False)
    event_seq: int

    def __post_init__(self) -> None:
        if not isinstance(self.control_ref, SessionControlRef):
            raise ValueError("control_ref must be a SessionControlRef")
        _require_non_empty(self.credential, field_name="credential")
        _require_non_negative_int(self.event_seq, field_name="event_seq")


@dataclass(frozen=True, kw_only=True)
class AdmissionDecision:
    action: AdmissionAction
    reason_code: AdmissionReason
    reservation_tokens: int
    admission_decision_id: str
    owner_epoch: int

    def __post_init__(self) -> None:
        if not isinstance(self.action, AdmissionAction):
            raise ValueError("action must be an AdmissionAction")
        if not isinstance(self.reason_code, AdmissionReason):
            raise ValueError("reason_code must be an AdmissionReason")
        _require_non_negative_int(self.reservation_tokens, field_name="reservation_tokens")
        _require_non_empty(self.admission_decision_id, field_name="admission_decision_id")
        _require_non_negative_int(self.owner_epoch, field_name="owner_epoch")


@dataclass(frozen=True, kw_only=True)
class AdmissionLease:
    owner_epoch: int
    dispatch_id: str
    admission_decision_id: str
    reservation_tokens: int
    ttl_s: float
    # This deadline is normalized by the shard-side budget port against the
    # shard's local monotonic clock. Coordinator monotonic timestamps must not
    # be compared across nodes.
    expires_at_local_monotonic: float

    def __post_init__(self) -> None:
        _require_non_negative_int(self.owner_epoch, field_name="owner_epoch")
        if self.owner_epoch == 0:
            raise ValueError("owner_epoch must be positive for an admission lease")
        _require_non_empty(self.dispatch_id, field_name="dispatch_id")
        _require_non_empty(self.admission_decision_id, field_name="admission_decision_id")
        _require_non_negative_int(self.reservation_tokens, field_name="reservation_tokens")
        if (
            not isinstance(self.ttl_s, (int, float))
            or isinstance(self.ttl_s, bool)
            or not math.isfinite(self.ttl_s)
            or self.ttl_s <= 0
        ):
            raise ValueError("ttl_s must be positive")
        if not isinstance(self.expires_at_local_monotonic, (int, float)) or isinstance(
            self.expires_at_local_monotonic, bool
        ):
            raise ValueError("expires_at_local_monotonic must be a number")
        if not math.isfinite(self.expires_at_local_monotonic) or self.expires_at_local_monotonic <= 0:
            raise ValueError("expires_at_local_monotonic must be positive")


@dataclass(frozen=True, kw_only=True)
class BudgetAcquireResult:
    status: BudgetAcquireStatus
    reason_code: AdmissionReason
    lease: AdmissionLease | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, BudgetAcquireStatus):
            raise ValueError("status must be a BudgetAcquireStatus")
        if not isinstance(self.reason_code, AdmissionReason):
            raise ValueError("reason_code must be an AdmissionReason")
        if self.status == BudgetAcquireStatus.ACQUIRED and self.lease is None:
            raise ValueError("acquired budget result requires a lease")
        if self.status != BudgetAcquireStatus.ACQUIRED and self.lease is not None:
            raise ValueError("only an acquired budget result may carry a lease")


@dataclass(frozen=True, kw_only=True)
class AdmissionGrant:
    decision: AdmissionDecision
    lease: AdmissionLease | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, AdmissionDecision):
            raise ValueError("decision must be an AdmissionDecision")
        if self.decision.action == AdmissionAction.ADMIT and self.lease is None:
            raise ValueError("admit grant requires a lease")
        if self.decision.action != AdmissionAction.ADMIT and self.lease is not None:
            raise ValueError("only an admit grant may carry a lease")
        if self.lease is not None and (
            self.lease.owner_epoch != self.decision.owner_epoch
            or self.lease.admission_decision_id != self.decision.admission_decision_id
            or self.lease.reservation_tokens != self.decision.reservation_tokens
        ):
            raise ValueError("admission lease does not match its decision")


@dataclass(frozen=True, kw_only=True)
class RoutingContext:
    request_id: str
    dispatch_id: str
    owner_epoch: int
    program_id: str
    root_session_id: str
    engine_session_id: str
    parent_engine_session_id: str | None
    attempt_id: str
    context_version_id: str
    serving_weight_version: str | None
    prompt_tokens: int
    expected_decode_tokens: int
    priority: int
    affinity_key: str | None

    def __post_init__(self) -> None:
        for field_name in (
            "request_id",
            "dispatch_id",
            "program_id",
            "root_session_id",
            "engine_session_id",
            "attempt_id",
            "context_version_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name=field_name)
        _require_non_negative_int(self.owner_epoch, field_name="owner_epoch")
        _require_optional_non_empty(
            self.parent_engine_session_id,
            field_name="parent_engine_session_id",
        )
        _require_optional_non_empty(
            self.serving_weight_version,
            field_name="serving_weight_version",
        )
        _require_non_negative_int(self.prompt_tokens, field_name="prompt_tokens")
        _require_non_negative_int(self.expected_decode_tokens, field_name="expected_decode_tokens")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise ValueError("priority must be an integer")
        _require_optional_non_empty(self.affinity_key, field_name="affinity_key")

    @property
    def reservation_tokens(self) -> int:
        return self.prompt_tokens + self.expected_decode_tokens


@dataclass(frozen=True, kw_only=True)
class RouteObservation:
    request_id: str
    dispatch_id: str
    owner_epoch: int
    route_decision_id: str | None
    selected_worker_id: str | None
    selected_engine_epoch: str | None
    serving_weight_version: str | None
    actual_cached_tokens: int | None
    prompt_tokens: int | None

    def __post_init__(self) -> None:
        _require_non_empty(self.request_id, field_name="request_id")
        _require_non_empty(self.dispatch_id, field_name="dispatch_id")
        _require_non_negative_int(self.owner_epoch, field_name="owner_epoch")
        for field_name in (
            "route_decision_id",
            "selected_worker_id",
            "selected_engine_epoch",
            "serving_weight_version",
        ):
            _require_optional_non_empty(getattr(self, field_name), field_name=field_name)
        if self.actual_cached_tokens is not None:
            _require_non_negative_int(self.actual_cached_tokens, field_name="actual_cached_tokens")
        if self.prompt_tokens is not None:
            _require_non_negative_int(self.prompt_tokens, field_name="prompt_tokens")

    @property
    def has_complete_worker_receipt(self) -> bool:
        return self.selected_worker_id is not None and self.selected_engine_epoch is not None
