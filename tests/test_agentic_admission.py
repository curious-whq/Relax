# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import time
from collections import deque
from types import SimpleNamespace

import pytest

from relax.agentic.session.admission import (
    AdmissionBudgetUnavailableBeforeCommit,
    AdmissionBudgetUnknownCommitOutcome,
    ProgramScheduler,
)
from relax.agentic.session.contracts import (
    AdmissionAction,
    AdmissionLease,
    AdmissionReason,
    AgenticIdentity,
    BudgetAcquireResult,
    BudgetAcquireStatus,
    LeaseReleaseOutcome,
    RoutingContext,
    SessionControlRef,
)
from relax.agentic.session.service import AgenticSessionShard, _ProgramRecord, _SessionRecord
from relax.agentic.session.state import InflightRequest, RequestKind


def _routing_context(**overrides) -> RoutingContext:
    values = {
        "request_id": "request-1",
        "dispatch_id": "dispatch-1",
        "owner_epoch": 17,
        "program_id": "program-1",
        "root_session_id": "root-1",
        "engine_session_id": "engine-1",
        "parent_engine_session_id": None,
        "attempt_id": "attempt-1",
        "context_version_id": "context-1",
        "serving_weight_version": None,
        "prompt_tokens": 32,
        "expected_decode_tokens": 8,
        "priority": 0,
        "affinity_key": "engine-1",
    }
    values.update(overrides)
    return RoutingContext(**values)


class _FakeBudgetPort:
    def __init__(
        self,
        *,
        capacity: int,
        events: list[str] | None = None,
        revalidate_results: list[bool] | None = None,
    ) -> None:
        self.capacity = capacity
        self.events = events if events is not None else []
        self.revalidate_results = deque(revalidate_results or [True])
        self.active_lease_ids: set[tuple[int, str, str]] = set()
        self.contexts: list[RoutingContext] = []
        self.successful_programs: list[str] = []
        self.release_outcomes: list[LeaseReleaseOutcome] = []
        self.try_status_override: BudgetAcquireStatus | None = None
        self.reconcile_status = BudgetAcquireStatus.BYPASS
        self.cancel_unknown_calls = 0
        self.try_exceptions: deque[Exception] = deque()
        self.cancel_unknown_error: Exception | None = None
        self.max_reservation_tokens: int | None = None
        self.block_emergency_started: asyncio.Event | None = None
        self.unblock_emergency: asyncio.Event | None = None
        self.release_started: asyncio.Event | None = None
        self.unblock_release: asyncio.Event | None = None
        self.release_calls = 0
        self.release_exceptions_after_commit: deque[Exception] = deque()

    def _acquired_result(
        self,
        *,
        routing_context: RoutingContext,
        admission_decision_id: str,
    ) -> BudgetAcquireResult:
        lease = AdmissionLease(
            owner_epoch=routing_context.owner_epoch,
            dispatch_id=routing_context.dispatch_id,
            admission_decision_id=admission_decision_id,
            reservation_tokens=routing_context.reservation_tokens,
            ttl_s=60.0,
            expires_at_local_monotonic=time.monotonic() + 60,
        )
        self.active_lease_ids.add((lease.owner_epoch, lease.dispatch_id, lease.admission_decision_id))
        self.successful_programs.append(routing_context.program_id)
        return BudgetAcquireResult(
            status=BudgetAcquireStatus.ACQUIRED,
            reason_code=AdmissionReason.CAPACITY_AVAILABLE,
            lease=lease,
        )

    async def try_acquire(
        self,
        *,
        routing_context: RoutingContext,
        admission_decision_id: str,
        emergency: bool,
    ) -> BudgetAcquireResult:
        self.events.append(f"budget:acquire:{'emergency' if emergency else 'normal'}")
        self.contexts.append(routing_context)
        if emergency and self.block_emergency_started is not None and self.unblock_emergency is not None:
            self.block_emergency_started.set()
            await self.unblock_emergency.wait()
        if self.try_exceptions:
            raise self.try_exceptions.popleft()
        if self.try_status_override == BudgetAcquireStatus.UNKNOWN:
            return BudgetAcquireResult(
                status=BudgetAcquireStatus.UNKNOWN,
                reason_code=AdmissionReason.DEGRADED,
            )
        if (
            len(self.active_lease_ids) >= self.capacity
            or (
                self.max_reservation_tokens is not None
                and routing_context.reservation_tokens > self.max_reservation_tokens
            )
        ):
            return BudgetAcquireResult(
                status=BudgetAcquireStatus.CAPACITY_EXHAUSTED,
                reason_code=AdmissionReason.CAPACITY_EXHAUSTED,
            )
        return self._acquired_result(
            routing_context=routing_context,
            admission_decision_id=admission_decision_id,
        )

    async def reconcile(
        self,
        *,
        routing_context: RoutingContext,
        admission_decision_id: str,
    ) -> BudgetAcquireResult:
        self.events.append("budget:reconcile")
        if self.reconcile_status == BudgetAcquireStatus.ACQUIRED:
            return self._acquired_result(
                routing_context=routing_context,
                admission_decision_id=admission_decision_id,
            )
        return BudgetAcquireResult(
            status=self.reconcile_status,
            reason_code=AdmissionReason.DEGRADED,
        )

    async def cancel_unknown(
        self,
        *,
        routing_context: RoutingContext,
        admission_decision_id: str,
    ) -> None:
        del routing_context, admission_decision_id
        self.cancel_unknown_calls += 1
        self.events.append("budget:cancel_unknown")
        if self.cancel_unknown_error is not None:
            raise self.cancel_unknown_error

    async def revalidate(self, *, lease: AdmissionLease) -> bool:
        del lease
        self.events.append("budget:revalidate")
        if len(self.revalidate_results) > 1:
            return self.revalidate_results.popleft()
        return self.revalidate_results[0]

    async def release(
        self,
        *,
        lease: AdmissionLease,
        outcome: LeaseReleaseOutcome,
    ) -> None:
        self.release_calls += 1
        if self.release_started is not None and self.unblock_release is not None:
            self.release_started.set()
            await self.unblock_release.wait()
        self.events.append("budget:release")
        self.active_lease_ids.discard((lease.owner_epoch, lease.dispatch_id, lease.admission_decision_id))
        if self.release_exceptions_after_commit:
            raise self.release_exceptions_after_commit.popleft()
        self.release_outcomes.append(outcome)


@pytest.mark.asyncio
async def test_scheduler_bypasses_without_identity_or_budget_capability() -> None:
    scheduler = ProgramScheduler(decision_id_factory=lambda: "decision-1")

    missing_ticket = await scheduler.submit(
        routing_context=None,
        dispatch_id="dispatch-1",
        request_kind=RequestKind.FRESH,
    )
    missing_grant = await missing_ticket.wait()
    assert missing_grant.decision.action == AdmissionAction.BYPASS
    assert missing_grant.decision.reason_code == AdmissionReason.MISSING_IDENTITY

    disabled_ticket = await scheduler.submit(
        routing_context=_routing_context(),
        dispatch_id="dispatch-1",
        request_kind=RequestKind.FRESH,
    )
    disabled_grant = await disabled_ticket.wait()
    assert disabled_grant.decision.action == AdmissionAction.BYPASS
    assert disabled_grant.decision.reason_code == AdmissionReason.FEATURE_DISABLED
    assert disabled_grant.lease is None


@pytest.mark.asyncio
async def test_scheduler_defers_without_consuming_capacity_then_resumes() -> None:
    port = _FakeBudgetPort(capacity=1)
    scheduler = ProgramScheduler(budget_port=port)
    first_context = _routing_context()
    second_context = _routing_context(
        request_id="request-2",
        dispatch_id="dispatch-2",
        engine_session_id="engine-2",
    )

    first_ticket = await scheduler.submit(
        routing_context=first_context,
        dispatch_id=first_context.dispatch_id,
        request_kind=RequestKind.FRESH,
    )
    first_grant = await first_ticket.wait()
    second_ticket = await scheduler.submit(
        routing_context=second_context,
        dispatch_id=second_context.dispatch_id,
        request_kind=RequestKind.FRESH,
    )

    assert second_ticket.initial_decision.action == AdmissionAction.DEFER
    assert scheduler.snapshot() == {"deferred_requests": 1, "ready_programs": 1}

    await scheduler.release(
        lease=first_grant.lease,
        outcome=LeaseReleaseOutcome.COMPLETED,
    )
    second_grant = await second_ticket.wait()
    assert second_grant.decision.action == AdmissionAction.ADMIT
    assert second_grant.lease.dispatch_id == "dispatch-2"
    assert scheduler.snapshot() == {"deferred_requests": 0, "ready_programs": 0}


@pytest.mark.asyncio
async def test_scheduler_round_robins_waiting_programs() -> None:
    port = _FakeBudgetPort(capacity=0)
    scheduler = ProgramScheduler(budget_port=port)
    contexts = [
        _routing_context(request_id="a-1", dispatch_id="a-1"),
        _routing_context(request_id="a-2", dispatch_id="a-2", engine_session_id="engine-a-2"),
        _routing_context(
            request_id="b-1",
            dispatch_id="b-1",
            program_id="program-2",
            root_session_id="root-2",
            engine_session_id="engine-b-1",
        ),
    ]
    tickets = [
        await scheduler.submit(
            routing_context=context,
            dispatch_id=context.dispatch_id,
            request_kind=RequestKind.FRESH,
        )
        for context in contexts
    ]
    assert all(ticket.initial_decision.action == AdmissionAction.DEFER for ticket in tickets)

    port.capacity = 3
    await scheduler.notify_capacity_changed()
    grants = await asyncio.gather(*(ticket.wait() for ticket in tickets))

    assert all(grant.decision.action == AdmissionAction.ADMIT for grant in grants)
    assert set(port.successful_programs[-3:-1]) == {"program-1", "program-2"}
    assert port.successful_programs[-1] == "program-1"


@pytest.mark.asyncio
async def test_scheduler_large_waiter_does_not_block_smaller_program() -> None:
    port = _FakeBudgetPort(capacity=2)
    port.max_reservation_tokens = 50
    scheduler = ProgramScheduler(budget_port=port)
    large_context = _routing_context(
        prompt_tokens=80,
        expected_decode_tokens=20,
    )
    small_context = _routing_context(
        request_id="request-2",
        dispatch_id="dispatch-2",
        program_id="program-2",
        root_session_id="root-2",
        engine_session_id="engine-2",
        prompt_tokens=10,
        expected_decode_tokens=10,
    )

    large_ticket = await scheduler.submit(
        routing_context=large_context,
        dispatch_id=large_context.dispatch_id,
        request_kind=RequestKind.FRESH,
    )
    small_ticket = await scheduler.submit(
        routing_context=small_context,
        dispatch_id=small_context.dispatch_id,
        request_kind=RequestKind.FRESH,
    )

    assert large_ticket.initial_decision.action == AdmissionAction.DEFER
    assert (await small_ticket.wait()).decision.action == AdmissionAction.ADMIT
    assert scheduler.snapshot() == {"deferred_requests": 1, "ready_programs": 1}
    await large_ticket.cancel()


@pytest.mark.asyncio
async def test_scheduler_protected_wait_uses_emergency_then_bypasses() -> None:
    port = _FakeBudgetPort(capacity=0)
    scheduler = ProgramScheduler(
        budget_port=port,
        protected_max_wait_s=0.001,
    )
    context = _routing_context(priority=100)

    ticket = await scheduler.submit(
        routing_context=context,
        dispatch_id=context.dispatch_id,
        request_kind=RequestKind.PROTECTED,
    )
    grant = await ticket.wait()

    assert grant.decision.action == AdmissionAction.BYPASS
    assert grant.decision.reason_code == AdmissionReason.FAIRNESS_RESERVE
    assert "budget:acquire:emergency" in port.events


@pytest.mark.asyncio
async def test_scheduler_ordinary_wait_timeout_does_not_use_emergency_budget() -> None:
    port = _FakeBudgetPort(capacity=0)
    scheduler = ProgramScheduler(
        budget_port=port,
        max_wait_s=0.001,
    )
    context = _routing_context()

    ticket = await scheduler.submit(
        routing_context=context,
        dispatch_id=context.dispatch_id,
        request_kind=RequestKind.FRESH,
    )
    grant = await ticket.wait()

    assert grant.decision.action == AdmissionAction.BYPASS
    assert grant.decision.reason_code == AdmissionReason.FAIRNESS_RESERVE
    assert port.events.count("budget:acquire:normal") == 2
    assert "budget:acquire:emergency" not in port.events


@pytest.mark.asyncio
async def test_cancelled_timeout_releases_late_emergency_lease() -> None:
    port = _FakeBudgetPort(capacity=0)
    port.block_emergency_started = asyncio.Event()
    port.unblock_emergency = asyncio.Event()
    scheduler = ProgramScheduler(
        budget_port=port,
        protected_max_wait_s=0.001,
    )
    context = _routing_context(priority=100)
    ticket = await scheduler.submit(
        routing_context=context,
        dispatch_id=context.dispatch_id,
        request_kind=RequestKind.PROTECTED,
    )
    wait_task = asyncio.create_task(ticket.wait())
    await asyncio.wait_for(port.block_emergency_started.wait(), timeout=1)

    wait_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await wait_task
    port.capacity = 1
    port.unblock_emergency.set()
    for _ in range(100):
        if port.release_outcomes:
            break
        await asyncio.sleep(0)

    assert port.active_lease_ids == set()
    assert port.release_outcomes == [LeaseReleaseOutcome.CANCELLED]


@pytest.mark.asyncio
async def test_scheduler_reconciles_unknown_outcome_before_bypass() -> None:
    port = _FakeBudgetPort(capacity=0)
    port.try_status_override = BudgetAcquireStatus.UNKNOWN
    port.reconcile_status = BudgetAcquireStatus.UNKNOWN
    scheduler = ProgramScheduler(budget_port=port)
    context = _routing_context()

    ticket = await scheduler.submit(
        routing_context=context,
        dispatch_id=context.dispatch_id,
        request_kind=RequestKind.FRESH,
    )
    grant = await ticket.wait()

    assert grant.decision.action == AdmissionAction.BYPASS
    assert grant.decision.reason_code == AdmissionReason.DEGRADED
    assert port.events[:3] == [
        "budget:acquire:normal",
        "budget:reconcile",
        "budget:cancel_unknown",
    ]


@pytest.mark.asyncio
async def test_scheduler_classifies_precommit_unavailability_as_degraded_bypass() -> None:
    port = _FakeBudgetPort(capacity=1)
    port.try_exceptions.append(AdmissionBudgetUnavailableBeforeCommit("unavailable"))
    scheduler = ProgramScheduler(budget_port=port)
    context = _routing_context()

    ticket = await scheduler.submit(
        routing_context=context,
        dispatch_id=context.dispatch_id,
        request_kind=RequestKind.FRESH,
    )
    grant = await ticket.wait()

    assert grant.decision.action == AdmissionAction.BYPASS
    assert grant.decision.reason_code == AdmissionReason.DEGRADED
    assert "budget:reconcile" not in port.events


@pytest.mark.asyncio
async def test_scheduler_reconciles_unknown_exception_to_acquired_lease() -> None:
    port = _FakeBudgetPort(capacity=1)
    port.try_exceptions.append(AdmissionBudgetUnknownCommitOutcome("unknown"))
    port.reconcile_status = BudgetAcquireStatus.ACQUIRED
    scheduler = ProgramScheduler(budget_port=port)
    context = _routing_context()

    ticket = await scheduler.submit(
        routing_context=context,
        dispatch_id=context.dispatch_id,
        request_kind=RequestKind.FRESH,
    )
    grant = await ticket.wait()

    assert grant.decision.action == AdmissionAction.ADMIT
    assert port.events[:2] == ["budget:acquire:normal", "budget:reconcile"]
    await ticket.cancel()


@pytest.mark.asyncio
async def test_scheduler_never_bypasses_unknown_commit_when_cancel_fails() -> None:
    port = _FakeBudgetPort(capacity=1)
    port.try_exceptions.append(AdmissionBudgetUnknownCommitOutcome("unknown"))
    port.reconcile_status = BudgetAcquireStatus.UNKNOWN
    port.cancel_unknown_error = RuntimeError("cancel failed")
    scheduler = ProgramScheduler(budget_port=port)
    context = _routing_context()

    with pytest.raises(RuntimeError, match="cancel failed"):
        await scheduler.submit(
            routing_context=context,
            dispatch_id=context.dispatch_id,
            request_kind=RequestKind.FRESH,
        )


@pytest.mark.asyncio
async def test_scheduler_cancelled_defer_cannot_be_revived_by_late_wake() -> None:
    port = _FakeBudgetPort(capacity=0)
    scheduler = ProgramScheduler(budget_port=port)
    context = _routing_context()
    ticket = await scheduler.submit(
        routing_context=context,
        dispatch_id=context.dispatch_id,
        request_kind=RequestKind.FRESH,
    )
    wait_task = asyncio.create_task(ticket.wait())
    await asyncio.sleep(0)

    wait_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await wait_task
    assert scheduler.snapshot() == {"deferred_requests": 0, "ready_programs": 0}

    port.capacity = 1
    await scheduler.notify_capacity_changed()
    assert port.successful_programs == []


@pytest.mark.asyncio
async def test_cancelling_immediate_admit_releases_lease_once() -> None:
    port = _FakeBudgetPort(capacity=1)
    scheduler = ProgramScheduler(budget_port=port)
    context = _routing_context()
    ticket = await scheduler.submit(
        routing_context=context,
        dispatch_id=context.dispatch_id,
        request_kind=RequestKind.FRESH,
    )

    assert ticket.initial_decision.action == AdmissionAction.ADMIT
    await asyncio.gather(ticket.cancel(), ticket.cancel())

    assert port.active_lease_ids == set()
    assert port.release_outcomes == [LeaseReleaseOutcome.CANCELLED]


@pytest.mark.asyncio
async def test_cancel_cleanup_survives_caller_cancellation_and_releases_once() -> None:
    port = _FakeBudgetPort(capacity=1)
    port.release_started = asyncio.Event()
    port.unblock_release = asyncio.Event()
    scheduler = ProgramScheduler(budget_port=port)
    context = _routing_context()
    ticket = await scheduler.submit(
        routing_context=context,
        dispatch_id=context.dispatch_id,
        request_kind=RequestKind.FRESH,
    )
    cancel_task = asyncio.create_task(ticket.cancel())
    await asyncio.wait_for(port.release_started.wait(), timeout=1)

    cancel_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancel_task
    port.unblock_release.set()
    await ticket.cancel()

    assert port.active_lease_ids == set()
    assert port.release_outcomes == [LeaseReleaseOutcome.CANCELLED]


@pytest.mark.asyncio
async def test_cancel_can_retry_idempotent_release_after_unknown_response() -> None:
    port = _FakeBudgetPort(capacity=1)
    port.release_exceptions_after_commit.append(RuntimeError("response lost"))
    scheduler = ProgramScheduler(budget_port=port)
    context = _routing_context()
    ticket = await scheduler.submit(
        routing_context=context,
        dispatch_id=context.dispatch_id,
        request_kind=RequestKind.FRESH,
    )

    with pytest.raises(RuntimeError, match="response lost"):
        await ticket.cancel()
    await ticket.cancel()

    assert port.active_lease_ids == set()
    assert port.release_calls == 2
    assert port.release_outcomes == [LeaseReleaseOutcome.CANCELLED]


@pytest.mark.asyncio
async def test_scheduler_rejects_expired_lease_before_budget_rpc() -> None:
    port = _FakeBudgetPort(capacity=1)
    scheduler = ProgramScheduler(budget_port=port, clock=lambda: 100.0)
    lease = AdmissionLease(
        owner_epoch=17,
        dispatch_id="dispatch-1",
        admission_decision_id="decision-1",
        reservation_tokens=40,
        ttl_s=60.0,
        expires_at_local_monotonic=99.0,
    )

    assert await scheduler.revalidate(lease=lease) is False
    assert "budget:revalidate" not in port.events


class _TracingSemaphore:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def acquire(self, *, blocking: bool) -> bool:
        assert blocking is False
        self.events.append("permit:acquire")
        return True

    def release(self) -> None:
        self.events.append("permit:release")


def _service_shard(*, port: _FakeBudgetPort, events: list[str]):
    shard_cls = AgenticSessionShard.__ray_metadata__.modified_class
    shard = object.__new__(shard_cls)
    shard.args = SimpleNamespace(
        rollout_max_response_len=8,
        rollout_max_context_len=64,
        partial_rollout=False,
    )
    shard._owner_epoch = 17
    shard._program_scheduler = ProgramScheduler(budget_port=port)
    shard._sglang_request_semaphore = _TracingSemaphore(events)
    shard._sglang_request_limiter = None
    identity = AgenticIdentity(
        program_id="program-1",
        program_owner_key="owner-1",
        root_session_id="engine-1",
        engine_session_id="engine-1",
        parent_engine_session_id=None,
    )
    record = _SessionRecord(scope_id="train", identity=identity)
    request = InflightRequest(
        request_id="request-1",
        parent_state_hash="context-1",
        rollout_id=1,
        kind=RequestKind.FRESH,
        abort_count=0,
        sampling_params={"max_new_tokens": 32},
        history_rollout_token_prefix=[11, 12],
        pending_rollout_token_delta=[13],
        runner_epoch=1,
    )
    record.irs_by_id[request.request_id] = request
    shard._session_records = {identity.engine_session_id: record}
    shard._session_locks = {identity.engine_session_id: asyncio.Lock()}
    shard._program_locks = {}
    shard._credential_session_ids = {}
    shard._terminal_program_owner_keys = deque()
    shard._program_records = {
        identity.program_owner_key: _ProgramRecord(
            program_id=identity.program_id,
            program_owner_key=identity.program_owner_key,
            root_session_id=identity.root_session_id,
            owner_epoch=shard._owner_epoch,
            engine_session_ids={identity.engine_session_id},
        )
    }
    return shard_cls, shard, record, request


@pytest.mark.asyncio
async def test_shard_admits_before_permit_and_uses_dispatch_identity() -> None:
    events: list[str] = []
    port = _FakeBudgetPort(capacity=1, events=events)
    shard_cls, shard, record, request = _service_shard(port=port, events=events)
    record.active_ir_runner_tasks[request.request_id] = asyncio.current_task()
    backend_request_ids: list[str] = []

    async def generate(**kwargs):
        events.append("backend")
        backend_request_ids.append(kwargs["request_id"])
        raise RuntimeError("stop after capture")

    shard.backend = SimpleNamespace(generate=generate)

    await shard_cls._run_ir(
        shard,
        session_id="engine-1",
        ir_id=request.request_id,
        runner_epoch=1,
    )

    assert events == [
        "budget:acquire:normal",
        "permit:acquire",
        "budget:revalidate",
        "backend",
        "permit:release",
        "budget:release",
    ]
    assert backend_request_ids[0].startswith("dispatch-")
    assert port.contexts[0].request_id == "request-1"
    assert port.contexts[0].dispatch_id == backend_request_ids[0]
    assert port.contexts[0].prompt_tokens == 3
    assert port.contexts[0].expected_decode_tokens == 8
    assert port.release_outcomes == [LeaseReleaseOutcome.FAILED]


@pytest.mark.asyncio
async def test_shard_defer_holds_no_permit_and_resumes_once() -> None:
    events: list[str] = []
    port = _FakeBudgetPort(capacity=0, events=events)
    shard_cls, shard, record, request = _service_shard(port=port, events=events)
    backend_started = asyncio.Event()

    async def generate(**kwargs):
        del kwargs
        events.append("backend")
        backend_started.set()
        raise RuntimeError("stop after capture")

    shard.backend = SimpleNamespace(generate=generate)
    task = asyncio.create_task(
        shard_cls._run_ir(
            shard,
            session_id="engine-1",
            ir_id=request.request_id,
            runner_epoch=1,
        )
    )
    record.active_ir_runner_tasks[request.request_id] = task
    for _ in range(50):
        if request.admission_action == AdmissionAction.DEFER.value:
            break
        await asyncio.sleep(0)

    assert request.admission_action == AdmissionAction.DEFER.value
    assert shard._program_scheduler.snapshot()["deferred_requests"] == 1
    assert "permit:acquire" not in events
    assert "backend" not in events
    assert request.backend_started is False

    port.capacity = 1
    await shard._program_scheduler.notify_capacity_changed()
    await asyncio.wait_for(backend_started.wait(), timeout=1)
    await task

    assert events.count("permit:acquire") == 1
    assert events.count("backend") == 1
    program_events = shard._program_records["owner-1"].event_seq_by_id
    assert any(event_id.startswith("admission_deferred:") for event_id in program_events), program_events
    assert any(event_id.startswith("admission_resumed:") for event_id in program_events), program_events


@pytest.mark.asyncio
async def test_shard_revalidation_failure_retries_with_new_dispatch() -> None:
    events: list[str] = []
    port = _FakeBudgetPort(
        capacity=1,
        events=events,
        revalidate_results=[False, True],
    )
    shard_cls, shard, record, request = _service_shard(port=port, events=events)
    record.active_ir_runner_tasks[request.request_id] = asyncio.current_task()

    async def generate(**kwargs):
        events.append("backend")
        raise RuntimeError(kwargs["request_id"])

    shard.backend = SimpleNamespace(generate=generate)

    await shard_cls._run_ir(
        shard,
        session_id="engine-1",
        ir_id=request.request_id,
        runner_epoch=1,
    )

    dispatch_ids = [context.dispatch_id for context in port.contexts]
    assert len(dispatch_ids) == 2
    assert len(set(dispatch_ids)) == 2
    assert len({context.attempt_id for context in port.contexts}) == 1
    assert events.count("backend") == 1
    assert port.release_outcomes == [
        LeaseReleaseOutcome.STALE,
        LeaseReleaseOutcome.FAILED,
    ]


@pytest.mark.asyncio
async def test_shard_bounds_revalidation_retries_then_degrades_to_bypass() -> None:
    events: list[str] = []
    port = _FakeBudgetPort(
        capacity=1,
        events=events,
        revalidate_results=[False],
    )
    shard_cls, shard, record, request = _service_shard(port=port, events=events)
    record.active_ir_runner_tasks[request.request_id] = asyncio.current_task()
    backend_request_ids: list[str] = []

    async def generate(**kwargs):
        events.append("backend")
        backend_request_ids.append(kwargs["request_id"])
        raise RuntimeError("stop after bounded retries")

    shard.backend = SimpleNamespace(generate=generate)

    await shard_cls._run_ir(
        shard,
        session_id="engine-1",
        ir_id=request.request_id,
        runner_epoch=1,
    )

    assert len(port.contexts) == 3
    assert len({context.dispatch_id for context in port.contexts}) == 3
    assert events.count("budget:revalidate") == 3
    assert events.count("backend") == 1
    assert backend_request_ids[0] not in {context.dispatch_id for context in port.contexts}
    assert request.admission_action == AdmissionAction.BYPASS.value
    assert port.release_outcomes == [
        LeaseReleaseOutcome.STALE,
        LeaseReleaseOutcome.STALE,
        LeaseReleaseOutcome.STALE,
    ]


@pytest.mark.asyncio
async def test_shard_health_does_not_read_scheduler_state_from_control_loop() -> None:
    events: list[str] = []
    port = _FakeBudgetPort(capacity=0, events=events)
    shard_cls, shard, _, request = _service_shard(port=port, events=events)
    request.admission_action = AdmissionAction.DEFER.value

    def fail_snapshot():
        raise AssertionError("scheduler state belongs to the default actor loop")

    shard._program_scheduler.snapshot = fail_snapshot
    health = await shard_cls.health(shard)

    assert health["admission"] == {
        "deferred_requests": 1,
        "ready_programs": 1,
        "permit_cleanup_pending": 0,
    }


@pytest.mark.asyncio
async def test_discarded_deferred_ir_releases_scheduler_ownership_without_backend() -> None:
    events: list[str] = []
    port = _FakeBudgetPort(capacity=0, events=events)
    shard_cls, shard, record, request = _service_shard(port=port, events=events)

    async def generate(**kwargs):
        del kwargs
        events.append("backend")
        raise RuntimeError("must not run")

    shard.backend = SimpleNamespace(generate=generate)
    task = asyncio.create_task(
        shard_cls._run_ir(
            shard,
            session_id="engine-1",
            ir_id=request.request_id,
            runner_epoch=1,
        )
    )
    record.active_ir_runner_tasks[request.request_id] = task
    for _ in range(50):
        if request.admission_action == AdmissionAction.DEFER.value:
            break
        await asyncio.sleep(0)

    discarded = await shard_cls.discard_session(
        shard,
        control_ref=SessionControlRef(
            program_owner_key="owner-1",
            engine_session_id="engine-1",
            owner_epoch=17,
        ),
    )
    await asyncio.sleep(0)

    assert discarded is True
    assert task.cancelled()
    assert shard._program_scheduler.snapshot() == {
        "deferred_requests": 0,
        "ready_programs": 0,
    }
    assert "permit:acquire" not in events
    assert "backend" not in events


@pytest.mark.asyncio
async def test_gate_closed_during_defer_requeues_without_permit_or_backend() -> None:
    events: list[str] = []
    port = _FakeBudgetPort(capacity=0, events=events)
    shard_cls, shard, record, request = _service_shard(port=port, events=events)

    async def generate(**kwargs):
        del kwargs
        events.append("backend")
        raise RuntimeError("must not run")

    shard.backend = SimpleNamespace(generate=generate)
    task = asyncio.create_task(
        shard_cls._run_ir(
            shard,
            session_id="engine-1",
            ir_id=request.request_id,
            runner_epoch=1,
        )
    )
    record.active_ir_runner_tasks[request.request_id] = task
    for _ in range(50):
        if request.admission_action == AdmissionAction.DEFER.value:
            break
        await asyncio.sleep(0)

    record.gate_reason = "partial_resume"
    port.capacity = 1
    await shard._program_scheduler.notify_capacity_changed()
    await task

    assert "permit:acquire" not in events
    assert "backend" not in events
    assert list(record.ir_queue) == [request.request_id]
    assert record.active_ir_runner_tasks == {}
    assert port.release_outcomes == [LeaseReleaseOutcome.REQUEUED]


@pytest.mark.asyncio
async def test_sibling_sessions_can_hold_admission_leases_concurrently() -> None:
    events: list[str] = []
    port = _FakeBudgetPort(capacity=2, events=events)
    shard_cls, shard, first_record, first_request = _service_shard(port=port, events=events)
    sibling_identity = AgenticIdentity(
        program_id="program-1",
        program_owner_key="owner-1",
        root_session_id="engine-1",
        engine_session_id="engine-2",
        parent_engine_session_id="engine-1",
    )
    sibling_record = _SessionRecord(scope_id="train", identity=sibling_identity)
    sibling_request = InflightRequest(
        request_id="request-2",
        parent_state_hash="context-2",
        rollout_id=1,
        kind=RequestKind.FRESH,
        abort_count=0,
        sampling_params={"max_new_tokens": 8},
        history_rollout_token_prefix=[21, 22],
        runner_epoch=1,
    )
    sibling_record.irs_by_id[sibling_request.request_id] = sibling_request
    shard._session_records[sibling_identity.engine_session_id] = sibling_record
    shard._session_locks[sibling_identity.engine_session_id] = asyncio.Lock()
    shard._program_records["owner-1"].engine_session_ids.add(sibling_identity.engine_session_id)
    both_started = asyncio.Event()
    release_backend = asyncio.Event()
    active_backend = 0
    max_active_backend = 0

    async def generate(**kwargs):
        nonlocal active_backend, max_active_backend
        del kwargs
        active_backend += 1
        max_active_backend = max(max_active_backend, active_backend)
        if active_backend == 2:
            both_started.set()
        try:
            await release_backend.wait()
            raise RuntimeError("stop after overlap")
        finally:
            active_backend -= 1

    shard.backend = SimpleNamespace(generate=generate)
    first_task = asyncio.create_task(
        shard_cls._run_ir(
            shard,
            session_id="engine-1",
            ir_id=first_request.request_id,
            runner_epoch=1,
        )
    )
    sibling_task = asyncio.create_task(
        shard_cls._run_ir(
            shard,
            session_id="engine-2",
            ir_id=sibling_request.request_id,
            runner_epoch=1,
        )
    )
    first_record.active_ir_runner_tasks[first_request.request_id] = first_task
    sibling_record.active_ir_runner_tasks[sibling_request.request_id] = sibling_task

    await asyncio.wait_for(both_started.wait(), timeout=1)
    assert max_active_backend == 2
    assert len(port.active_lease_ids) == 2
    release_backend.set()
    await asyncio.gather(first_task, sibling_task)

    assert len(port.active_lease_ids) == 0
    assert port.release_outcomes == [
        LeaseReleaseOutcome.FAILED,
        LeaseReleaseOutcome.FAILED,
    ]
