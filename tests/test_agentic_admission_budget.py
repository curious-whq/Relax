# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest

from relax.agentic.session import service as session_service
from relax.agentic.session.admission import AdmissionBudgetUnknownCommitOutcome, ProgramScheduler
from relax.agentic.session.admission_budget import (
    AdmissionBudgetCoordinatorCore,
    RayAdmissionBudgetPort,
)
from relax.agentic.session.contracts import (
    AdmissionAction,
    AdmissionLease,
    AdmissionReason,
    BudgetAcquireStatus,
    LeaseReleaseOutcome,
    RoutingContext,
    WorkerPressureState,
    WorkerSnapshot,
    WorkerSnapshotBatch,
)
from relax.agentic.session.sglang_capabilities import resolve_sglang_capability_profile
from relax.agentic.session.state import RequestKind


class _Clock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, delta_s: float) -> None:
        self.value += delta_s


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
        "serving_weight_version": "weight-1",
        "prompt_tokens": 30,
        "expected_decode_tokens": 10,
        "priority": 0,
        "affinity_key": "engine-1",
    }
    values.update(overrides)
    return RoutingContext(**values)


def _snapshot_batch(
    *,
    capacity: int = 100,
    version: str = "weight-1",
    publisher_epoch: str = "publisher-epoch-1",
    batch_seq: int = 1,
    source_open: bool = True,
    complete: bool = True,
    healthy: bool = True,
    pressure_state: WorkerPressureState = WorkerPressureState.NORMAL,
) -> WorkerSnapshotBatch:
    return WorkerSnapshotBatch(
        source_id="router-1",
        publisher_epoch=publisher_epoch,
        batch_seq=batch_seq,
        source_open=source_open,
        complete=complete,
        snapshots=(
            WorkerSnapshot(
                worker_id="worker-1",
                engine_epoch="engine-epoch-1",
                serving_weight_version=version,
                safe_execution_capacity_tokens=capacity,
                healthy=healthy,
                pressure_state=pressure_state,
            ),
        ),
    )


def _core(*, clock: _Clock | None = None, **kwargs) -> AdmissionBudgetCoordinatorCore:
    core = AdmissionBudgetCoordinatorCore(
        clock=clock or _Clock(),
        coordinator_epoch_factory=lambda: "coordinator-1",
        **kwargs,
    )
    core.register_owner(shard_id=0, owner_epoch=17)
    return core


def _lease_from_reply(*, context: RoutingContext, decision_id: str, reply, clock: _Clock) -> AdmissionLease:
    return AdmissionLease(
        owner_epoch=context.owner_epoch,
        dispatch_id=context.dispatch_id,
        admission_decision_id=decision_id,
        reservation_tokens=context.reservation_tokens,
        ttl_s=reply.ttl_remaining_s,
        expires_at_local_monotonic=clock() + reply.ttl_remaining_s,
        coordinator_epoch=reply.coordinator_epoch,
        capacity_generation=reply.capacity_generation,
    )


def test_coordinator_fails_open_without_complete_fresh_versioned_snapshot() -> None:
    clock = _Clock()
    core = _core(clock=clock, snapshot_ttl_s=2.0)
    context = _routing_context()

    missing = core.try_acquire(
        routing_context=context,
        admission_decision_id="decision-missing",
        emergency=False,
    )
    assert missing.status == BudgetAcquireStatus.BYPASS
    assert missing.reason_code == AdmissionReason.DEGRADED

    core.replace_worker_snapshots(batch=_snapshot_batch(complete=False, batch_seq=1))
    incomplete = core.try_acquire(
        routing_context=context,
        admission_decision_id="decision-incomplete",
        emergency=False,
    )
    assert incomplete.status == BudgetAcquireStatus.BYPASS

    core.replace_worker_snapshots(batch=_snapshot_batch(version="weight-other", batch_seq=2))
    mismatch = core.try_acquire(
        routing_context=context,
        admission_decision_id="decision-mismatch",
        emergency=False,
    )
    assert mismatch.status == BudgetAcquireStatus.BYPASS

    core.replace_worker_snapshots(batch=_snapshot_batch(healthy=False, batch_seq=3))
    unhealthy = core.try_acquire(
        routing_context=context,
        admission_decision_id="decision-unhealthy",
        emergency=False,
    )
    assert unhealthy.status == BudgetAcquireStatus.BYPASS

    core.replace_worker_snapshots(batch=_snapshot_batch(batch_seq=4))
    clock.advance(2.1)
    stale = core.try_acquire(
        routing_context=context,
        admission_decision_id="decision-stale",
        emergency=False,
    )
    assert stale.status == BudgetAcquireStatus.BYPASS

    missing_version = core.try_acquire(
        routing_context=_routing_context(serving_weight_version=None),
        admission_decision_id="decision-no-version",
        emergency=False,
    )
    assert missing_version.status == BudgetAcquireStatus.BYPASS


def test_coordinator_snapshot_expiry_notifies_deferred_capacity_watch_once() -> None:
    clock = _Clock()
    core = _core(clock=clock, snapshot_ttl_s=2.0)
    core.replace_worker_snapshots(batch=_snapshot_batch())
    before_expiry = core.availability_seq()

    clock.advance(2.1)

    assert core.availability_seq() == before_expiry + 1
    assert core.availability_seq() == before_expiry + 1


def test_coordinator_atomically_bounds_shared_capacity_and_deduplicates_acquire() -> None:
    clock = _Clock()
    core = _core(
        clock=clock,
        safety_headroom_ratio=0,
        emergency_reserve_ratio=0,
    )
    core.register_owner(shard_id=1, owner_epoch=23)
    core.replace_worker_snapshots(batch=_snapshot_batch(capacity=100))
    first_context = _routing_context(prompt_tokens=50, expected_decode_tokens=10)
    second_context = _routing_context(
        request_id="request-2",
        dispatch_id="dispatch-2",
        owner_epoch=23,
        program_id="program-2",
        root_session_id="root-2",
        engine_session_id="engine-2",
        prompt_tokens=35,
        expected_decode_tokens=10,
    )

    first = core.try_acquire(
        routing_context=first_context,
        admission_decision_id="decision-1",
        emergency=False,
    )
    duplicate = core.try_acquire(
        routing_context=first_context,
        admission_decision_id="decision-1",
        emergency=False,
    )
    second = core.try_acquire(
        routing_context=second_context,
        admission_decision_id="decision-2",
        emergency=False,
    )

    assert first.status == BudgetAcquireStatus.ACQUIRED
    assert duplicate.status == BudgetAcquireStatus.ACQUIRED
    assert duplicate.ttl_remaining_s == first.ttl_remaining_s
    assert second.status == BudgetAcquireStatus.CAPACITY_EXHAUSTED
    assert core.snapshot()["reserved_tokens"] == 60


def test_coordinator_rejects_conflicting_or_tombstoned_decision() -> None:
    clock = _Clock()
    core = _core(
        clock=clock,
        safety_headroom_ratio=0,
        emergency_reserve_ratio=0,
    )
    core.replace_worker_snapshots(batch=_snapshot_batch())
    context = _routing_context()
    reply = core.try_acquire(
        routing_context=context,
        admission_decision_id="decision-1",
        emergency=False,
    )
    conflicting = core.try_acquire(
        routing_context=_routing_context(prompt_tokens=31),
        admission_decision_id="decision-1",
        emergency=False,
    )
    duplicate_dispatch = core.try_acquire(
        routing_context=context,
        admission_decision_id="decision-other",
        emergency=False,
    )
    lease = _lease_from_reply(
        context=context,
        decision_id="decision-1",
        reply=reply,
        clock=clock,
    )
    core.release(lease=lease, outcome=LeaseReleaseOutcome.COMPLETED)
    late = core.try_acquire(
        routing_context=context,
        admission_decision_id="decision-1",
        emergency=False,
    )

    assert conflicting.status == BudgetAcquireStatus.REJECTED
    assert duplicate_dispatch.status == BudgetAcquireStatus.REJECTED
    assert late.status == BudgetAcquireStatus.REJECTED
    assert core.snapshot()["active_leases"] == 0


def test_coordinator_reserves_emergency_partition_for_protected_retry() -> None:
    core = _core(
        safety_headroom_ratio=0,
        emergency_reserve_ratio=0.2,
    )
    core.replace_worker_snapshots(batch=_snapshot_batch(capacity=100))
    normal_context = _routing_context(prompt_tokens=70, expected_decode_tokens=10)
    reserve_context = _routing_context(
        request_id="request-2",
        dispatch_id="dispatch-2",
        engine_session_id="engine-2",
        prompt_tokens=10,
        expected_decode_tokens=10,
    )
    first = core.try_acquire(
        routing_context=normal_context,
        admission_decision_id="decision-1",
        emergency=False,
    )
    normal_retry = core.try_acquire(
        routing_context=reserve_context,
        admission_decision_id="decision-2",
        emergency=False,
    )
    emergency_retry = core.try_acquire(
        routing_context=reserve_context,
        admission_decision_id="decision-2",
        emergency=True,
    )

    assert first.status == BudgetAcquireStatus.ACQUIRED
    assert normal_retry.status == BudgetAcquireStatus.CAPACITY_EXHAUSTED
    assert emergency_retry.status == BudgetAcquireStatus.ACQUIRED
    assert core.snapshot()["reserved_tokens"] == 100


def test_emergency_reservation_does_not_consume_normal_partition() -> None:
    core = _core(
        safety_headroom_ratio=0,
        emergency_reserve_ratio=0.2,
    )
    core.replace_worker_snapshots(batch=_snapshot_batch(capacity=100))
    emergency_context = _routing_context(prompt_tokens=10, expected_decode_tokens=10)
    normal_context = _routing_context(
        request_id="request-2",
        dispatch_id="dispatch-2",
        engine_session_id="engine-2",
        prompt_tokens=70,
        expected_decode_tokens=10,
    )

    emergency = core.try_acquire(
        routing_context=emergency_context,
        admission_decision_id="decision-1",
        emergency=True,
    )
    normal = core.try_acquire(
        routing_context=normal_context,
        admission_decision_id="decision-2",
        emergency=False,
    )

    assert emergency.status == BudgetAcquireStatus.ACQUIRED
    assert normal.status == BudgetAcquireStatus.ACQUIRED
    assert core.snapshot()["reserved_tokens"] == 100


def test_coordinator_pressure_guard_defers_without_issuing_lease() -> None:
    core = _core()
    core.replace_worker_snapshots(
        batch=_snapshot_batch(pressure_state=WorkerPressureState.CRITICAL),
    )

    reply = core.try_acquire(
        routing_context=_routing_context(),
        admission_decision_id="decision-1",
        emergency=False,
    )

    assert reply.status == BudgetAcquireStatus.CAPACITY_EXHAUSTED
    assert reply.reason_code == AdmissionReason.PRESSURE_GUARD
    assert core.snapshot()["active_leases"] == 0


def test_snapshot_generation_revokes_reserved_but_not_in_flight_lease() -> None:
    clock = _Clock()
    core = _core(
        clock=clock,
        safety_headroom_ratio=0,
        emergency_reserve_ratio=0,
    )
    core.replace_worker_snapshots(batch=_snapshot_batch(capacity=100, batch_seq=1))
    reserved_context = _routing_context()
    in_flight_context = _routing_context(
        request_id="request-2",
        dispatch_id="dispatch-2",
        engine_session_id="engine-2",
    )
    reserved_reply = core.try_acquire(
        routing_context=reserved_context,
        admission_decision_id="decision-1",
        emergency=False,
    )
    in_flight_reply = core.try_acquire(
        routing_context=in_flight_context,
        admission_decision_id="decision-2",
        emergency=False,
    )
    reserved_lease = _lease_from_reply(
        context=reserved_context,
        decision_id="decision-1",
        reply=reserved_reply,
        clock=clock,
    )
    in_flight_lease = _lease_from_reply(
        context=in_flight_context,
        decision_id="decision-2",
        reply=in_flight_reply,
        clock=clock,
    )
    assert core.activate(lease=in_flight_lease).valid is True

    core.replace_worker_snapshots(batch=_snapshot_batch(capacity=40, batch_seq=2))

    assert core.activate(lease=reserved_lease).valid is False
    assert core.renew(lease=in_flight_lease).valid is True
    assert core.snapshot()["active_leases"] == 1
    assert core.snapshot()["in_flight_leases"] == 1


def test_snapshot_batches_reject_source_switch_and_ignore_reordered_sequence() -> None:
    core = _core()
    assert core.replace_worker_snapshots(batch=_snapshot_batch(capacity=100, batch_seq=2)) == 1
    assert core.replace_worker_snapshots(batch=_snapshot_batch(capacity=100, batch_seq=1)) == 1
    assert core.replace_worker_snapshots(batch=_snapshot_batch(capacity=100, batch_seq=3)) == 1

    with pytest.raises(ValueError, match="source fence"):
        core.replace_worker_snapshots(
            batch=_snapshot_batch(
                publisher_epoch="publisher-epoch-2",
                batch_seq=3,
            )
        )

    assert (
        core.fence_worker_snapshot_source(
            source_id="router-1",
            publisher_epoch="publisher-epoch-2",
        )
        == 2
    )
    assert (
        core.replace_worker_snapshots(
            batch=_snapshot_batch(
                publisher_epoch="publisher-epoch-2",
                batch_seq=1,
            )
        )
        == 3
    )


def test_coordinator_ttl_owner_fencing_and_renewal() -> None:
    clock = _Clock()
    core = _core(
        clock=clock,
        lease_ttl_s=3.0,
        snapshot_ttl_s=30.0,
        safety_headroom_ratio=0,
        emergency_reserve_ratio=0,
    )
    core.replace_worker_snapshots(batch=_snapshot_batch())
    context = _routing_context()
    reply = core.try_acquire(
        routing_context=context,
        admission_decision_id="decision-1",
        emergency=False,
    )
    lease = _lease_from_reply(
        context=context,
        decision_id="decision-1",
        reply=reply,
        clock=clock,
    )
    activated = core.activate(lease=lease)
    assert activated.valid is True

    clock.advance(2.0)
    renewed = core.renew(lease=lease)
    assert renewed.valid is True
    assert renewed.ttl_remaining_s == 3.0

    core.register_owner(shard_id=0, owner_epoch=19)
    assert core.revalidate(lease=lease).valid is False
    stale_owner = core.try_acquire(
        routing_context=_routing_context(dispatch_id="dispatch-2"),
        admission_decision_id="decision-2",
        emergency=False,
    )
    assert stale_owner.status == BudgetAcquireStatus.REJECTED

    core.register_owner(shard_id=0, owner_epoch=17)
    fresh = core.try_acquire(
        routing_context=_routing_context(dispatch_id="dispatch-3"),
        admission_decision_id="decision-3",
        emergency=False,
    )
    assert fresh.status == BudgetAcquireStatus.ACQUIRED
    clock.advance(3.1)
    assert core.snapshot()["active_leases"] == 0


def test_cancel_unknown_tombstones_missing_decision() -> None:
    core = _core()
    core.replace_worker_snapshots(batch=_snapshot_batch())
    context = _routing_context()

    core.cancel_unknown(
        routing_context=context,
        admission_decision_id="decision-1",
    )
    reconcile = core.reconcile(
        routing_context=context,
        admission_decision_id="decision-1",
    )
    late = core.try_acquire(
        routing_context=context,
        admission_decision_id="decision-1",
        emergency=False,
    )

    assert reconcile.status == BudgetAcquireStatus.REJECTED
    assert late.status == BudgetAcquireStatus.REJECTED


def test_tombstone_saturation_fails_open_until_replay_window_expires() -> None:
    clock = _Clock()
    core = _core(
        clock=clock,
        lease_ttl_s=1.0,
        tombstone_ttl_s=3.0,
        tombstone_limit=1,
    )
    core.replace_worker_snapshots(batch=_snapshot_batch())
    context = _routing_context()
    reply = core.try_acquire(
        routing_context=context,
        admission_decision_id="decision-1",
        emergency=False,
    )
    lease = _lease_from_reply(
        context=context,
        decision_id="decision-1",
        reply=reply,
        clock=clock,
    )
    core.release(lease=lease, outcome=LeaseReleaseOutcome.COMPLETED)
    next_context = _routing_context(
        request_id="request-2",
        dispatch_id="dispatch-2",
        engine_session_id="engine-2",
    )

    saturated = core.try_acquire(
        routing_context=next_context,
        admission_decision_id="decision-2",
        emergency=False,
    )
    clock.advance(3.1)
    after_window = core.try_acquire(
        routing_context=next_context,
        admission_decision_id="decision-2",
        emergency=False,
    )

    assert saturated.status == BudgetAcquireStatus.BYPASS
    assert after_window.status == BudgetAcquireStatus.ACQUIRED


class _RemoteMethod:
    def __init__(self, function) -> None:
        self._function = function

    def remote(self, **kwargs):
        async def invoke():
            await asyncio.sleep(0)
            return self._function(**kwargs)

        return invoke()


class _LocalCoordinatorHandle:
    def __init__(self, core: AdmissionBudgetCoordinatorCore) -> None:
        self.register_owner = _RemoteMethod(core.register_owner)
        self.try_acquire = _RemoteMethod(core.try_acquire)
        self.reconcile = _RemoteMethod(core.reconcile)
        self.cancel_unknown = _RemoteMethod(core.cancel_unknown)
        self.revalidate = _RemoteMethod(core.revalidate)
        self.activate = _RemoteMethod(core.activate)
        self.renew = _RemoteMethod(core.renew)
        self.release = _RemoteMethod(core.release)
        self.availability_seq = _RemoteMethod(lambda: core.availability_seq())


@pytest.mark.asyncio
async def test_ray_port_bounds_acquire_rpc_and_reports_unknown_outcome() -> None:
    class _SlowAcquire:
        def remote(self, **kwargs):
            del kwargs

            async def invoke():
                await asyncio.sleep(0.02)

            return invoke()

    handle = SimpleNamespace(try_acquire=_SlowAcquire())
    port = RayAdmissionBudgetPort(coordinator=handle, rpc_timeout_s=0.001)

    with pytest.raises(AdmissionBudgetUnknownCommitOutcome):
        await port.try_acquire(
            routing_context=_routing_context(),
            admission_decision_id="decision-1",
            emergency=False,
        )
    await asyncio.sleep(0.03)


@pytest.mark.asyncio
async def test_ray_port_normalizes_remote_ttl_to_shard_clock() -> None:
    coordinator_clock = _Clock(100.0)
    shard_clock = _Clock(9000.0)
    core = _core(
        clock=coordinator_clock,
        lease_ttl_s=20.0,
        snapshot_ttl_s=30.0,
        safety_headroom_ratio=0,
        emergency_reserve_ratio=0,
    )
    core.replace_worker_snapshots(batch=_snapshot_batch())
    port = RayAdmissionBudgetPort(
        coordinator=_LocalCoordinatorHandle(core),
        clock=shard_clock,
        deadline_safety_ratio=0.1,
    )
    context = _routing_context()

    result = await port.try_acquire(
        routing_context=context,
        admission_decision_id="decision-1",
        emergency=False,
    )

    assert result.status == BudgetAcquireStatus.ACQUIRED
    assert result.lease.ttl_s == 18.0
    assert result.lease.expires_at_local_monotonic == 9018.0
    assert result.lease.coordinator_epoch == "coordinator-1"
    activated = await port.activate(lease=result.lease)
    assert activated is not None
    renewed = await port.renew(lease=activated)
    assert renewed.expires_at_local_monotonic == 9018.0


@pytest.mark.asyncio
async def test_release_on_one_shard_wakes_deferred_scheduler_on_another() -> None:
    clock = _Clock()
    core = _core(
        clock=clock,
        snapshot_ttl_s=30.0,
        safety_headroom_ratio=0,
        emergency_reserve_ratio=0,
    )
    core.register_owner(shard_id=1, owner_epoch=23)
    core.replace_worker_snapshots(batch=_snapshot_batch(capacity=40))
    handle = _LocalCoordinatorHandle(core)
    first_scheduler = ProgramScheduler(
        budget_port=RayAdmissionBudgetPort(coordinator=handle, clock=clock),
        capacity_poll_s=0.001,
    )
    second_scheduler = ProgramScheduler(
        budget_port=RayAdmissionBudgetPort(coordinator=handle, clock=clock),
        capacity_poll_s=0.001,
    )
    first_context = _routing_context()
    second_context = _routing_context(
        request_id="request-2",
        dispatch_id="dispatch-2",
        owner_epoch=23,
        program_id="program-2",
        root_session_id="root-2",
        engine_session_id="engine-2",
    )
    first_ticket = await first_scheduler.submit(
        routing_context=first_context,
        dispatch_id=first_context.dispatch_id,
        request_kind=RequestKind.FRESH,
    )
    first_grant = await first_ticket.wait()
    second_ticket = await second_scheduler.submit(
        routing_context=second_context,
        dispatch_id=second_context.dispatch_id,
        request_kind=RequestKind.FRESH,
    )
    assert second_ticket.initial_decision.action == AdmissionAction.DEFER

    await first_scheduler.release(
        lease=first_grant.lease,
        outcome=LeaseReleaseOutcome.COMPLETED,
    )
    second_grant = await asyncio.wait_for(second_ticket.wait(), timeout=1)

    assert second_grant.decision.action == AdmissionAction.ADMIT
    await second_ticket.cancel()


@pytest.mark.asyncio
async def test_session_shard_registers_its_owner_with_injected_coordinator(monkeypatch) -> None:
    core = AdmissionBudgetCoordinatorCore(
        coordinator_epoch_factory=lambda: "coordinator-1",
    )
    handle = _LocalCoordinatorHandle(core)
    monkeypatch.setattr(session_service, "init_http_client", lambda args: None)
    monkeypatch.setattr(
        session_service,
        "get_agentic_runtime_resources",
        lambda args: SimpleNamespace(compiler=None),
    )
    monkeypatch.setattr(
        session_service,
        "SGLangBackendAdapter",
        lambda args, compiler_resources, capability_profile: SimpleNamespace(),
    )
    shard_cls = session_service.AgenticSessionShard.__ray_metadata__.modified_class
    shard = shard_cls(
        Namespace(),
        admission_budget_coordinator=handle,
        shard_index=3,
        shard_count=4,
    )

    await shard.register_admission_budget_owner()

    assert core.snapshot()["registered_owners"] == 1
    assert shard._program_scheduler._budget_port is shard._admission_budget_port


def test_shard_launcher_fails_open_when_coordinator_startup_fails(monkeypatch) -> None:
    remote_calls: list[dict[str, object]] = []
    capability_profile = resolve_sglang_capability_profile(
        router_managed=True,
        use_slime_router=False,
        has_pd_disaggregation=False,
        radix_cache_disabled=False,
        hierarchical_cache_enabled=False,
    )

    class _ShardOptions:
        def remote(self, config, **kwargs):
            del config
            remote_calls.append(kwargs)
            return SimpleNamespace()

    def missing_actor(name):
        del name
        raise ValueError("missing")

    monkeypatch.setattr(session_service, "_DEFAULT_SESSION_SHARD_COUNT", 2)
    monkeypatch.setattr(session_service, "_STALE_SESSION_SHARD_CLEANUP_LIMIT", 2)
    monkeypatch.setattr(session_service.ray, "get_actor", missing_actor)
    monkeypatch.setattr(
        session_service,
        "create_admission_budget_coordinator",
        lambda: (_ for _ in ()).throw(RuntimeError("coordinator unavailable")),
    )
    monkeypatch.setattr(
        session_service.AgenticSessionShard,
        "options",
        lambda **kwargs: _ShardOptions(),
    )
    handles = session_service.create_agentic_session_shards(
        Namespace(
            sglang_server_concurrency=8,
            rollout_num_gpus=2,
            rollout_num_gpus_per_engine=1,
        ),
        sglang_capability_profile=capability_profile,
    )

    assert len(handles) == 2
    assert len(remote_calls) == 2
    assert all(call["admission_budget_coordinator"] is None for call in remote_calls)
    assert all(call["sglang_capability_profile"] is capability_profile for call in remote_calls)
    assert all(call["owner_epoch"] > 0 for call in remote_calls)


def test_shard_launcher_cleans_partial_deployment_on_creation_failure(monkeypatch) -> None:
    first_shard = SimpleNamespace(name="shard-0")
    coordinator = SimpleNamespace(
        register_owner=SimpleNamespace(remote=lambda **kwargs: kwargs),
    )
    killed: list[object] = []
    shutdown_calls: list[bool] = []
    remote_call_count = 0

    class _ShardOptions:
        def remote(self, config, **kwargs):
            del config, kwargs
            nonlocal remote_call_count
            remote_call_count += 1
            if remote_call_count == 1:
                return first_shard
            raise RuntimeError("second shard failed")

    def missing_actor(name):
        del name
        raise ValueError("missing")

    monkeypatch.setattr(session_service, "_DEFAULT_SESSION_SHARD_COUNT", 2)
    monkeypatch.setattr(session_service, "_STALE_SESSION_SHARD_CLEANUP_LIMIT", 2)
    monkeypatch.setattr(session_service.ray, "get_actor", missing_actor)
    monkeypatch.setattr(session_service.ray, "get", lambda refs, timeout: None)
    monkeypatch.setattr(
        session_service.ray,
        "kill",
        lambda handle, no_restart: killed.append(handle),
    )
    monkeypatch.setattr(
        session_service,
        "create_admission_budget_coordinator",
        lambda: coordinator,
    )
    monkeypatch.setattr(
        session_service,
        "shutdown_admission_budget_coordinator",
        lambda: shutdown_calls.append(True),
    )
    monkeypatch.setattr(
        session_service.AgenticSessionShard,
        "options",
        lambda **kwargs: _ShardOptions(),
    )

    with pytest.raises(RuntimeError, match="second shard failed"):
        session_service.create_agentic_session_shards(
            Namespace(
                sglang_server_concurrency=8,
                rollout_num_gpus=2,
                rollout_num_gpus_per_engine=1,
            )
        )

    assert killed == [first_shard]
    assert shutdown_calls == [True]
