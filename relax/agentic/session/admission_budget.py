# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import secrets
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import ray

from relax.agentic.session.admission import (
    AdmissionBudgetUnavailableBeforeCommit,
    AdmissionBudgetUnknownCommitOutcome,
    AdmissionBudgetUnknownRenewalOutcome,
)
from relax.agentic.session.contracts import (
    AdmissionLease,
    AdmissionReason,
    BudgetAcquireResult,
    BudgetAcquireStatus,
    LeaseReleaseOutcome,
    RoutingContext,
    WorkerPressureState,
    WorkerSnapshotBatch,
)


AGENTIC_ADMISSION_BUDGET_COORDINATOR_NAME = "agentic_admission_budget_coordinator"


@dataclass(frozen=True, kw_only=True)
class CoordinatorAcquireReply:
    status: BudgetAcquireStatus
    reason_code: AdmissionReason
    coordinator_epoch: str
    capacity_generation: int
    availability_seq: int
    ttl_remaining_s: float | None = None


@dataclass(frozen=True, kw_only=True)
class CoordinatorLeaseValidation:
    valid: bool
    coordinator_epoch: str
    capacity_generation: int
    availability_seq: int
    ttl_remaining_s: float | None = None


@dataclass
class _LeaseRecord:
    routing_context: RoutingContext
    admission_decision_id: str
    expires_at: float
    capacity_generation: int
    emergency: bool
    activated: bool = False

    @property
    def key(self) -> tuple[int, str, str]:
        return (
            self.routing_context.owner_epoch,
            self.routing_context.dispatch_id,
            self.admission_decision_id,
        )


class AdmissionBudgetCoordinatorCore:
    """Await-free single-writer execution-budget state machine."""

    def __init__(
        self,
        *,
        lease_ttl_s: float = 30.0,
        snapshot_ttl_s: float = 5.0,
        safety_headroom_ratio: float = 0.1,
        emergency_reserve_ratio: float = 0.1,
        clock: Callable[[], float] = time.monotonic,
        coordinator_epoch_factory: Callable[[], str] | None = None,
        tombstone_limit: int = 16384,
        tombstone_ttl_s: float | None = None,
    ) -> None:
        if lease_ttl_s <= 0:
            raise ValueError("lease_ttl_s must be positive")
        if snapshot_ttl_s <= 0:
            raise ValueError("snapshot_ttl_s must be positive")
        if not 0 <= safety_headroom_ratio < 1:
            raise ValueError("safety_headroom_ratio must be in [0, 1)")
        if not 0 <= emergency_reserve_ratio < 1:
            raise ValueError("emergency_reserve_ratio must be in [0, 1)")
        if tombstone_limit <= 0:
            raise ValueError("tombstone_limit must be positive")
        if tombstone_ttl_s is not None and tombstone_ttl_s < lease_ttl_s:
            raise ValueError("tombstone_ttl_s must be at least lease_ttl_s")
        self._lease_ttl_s = float(lease_ttl_s)
        self._snapshot_ttl_s = float(snapshot_ttl_s)
        self._safety_headroom_ratio = float(safety_headroom_ratio)
        self._emergency_reserve_ratio = float(emergency_reserve_ratio)
        self._clock = clock
        self._coordinator_epoch = (coordinator_epoch_factory or (lambda: f"budget-{secrets.token_hex(16)}"))()
        self._tombstone_limit = int(tombstone_limit)
        self._tombstone_ttl_s = float(tombstone_ttl_s or max(60.0, lease_ttl_s * 2.0))
        self._snapshot_batch: WorkerSnapshotBatch | None = None
        self._snapshot_received_at: float | None = None
        self._snapshot_stale_observed = False
        self._snapshot_source_fence: tuple[str, str] | None = None
        self._capacity_generation = 0
        self._availability_seq = 0
        self._owner_epoch_by_shard: dict[int, int] = {}
        self._leases_by_key: dict[tuple[int, str, str], _LeaseRecord] = {}
        self._lease_key_by_dispatch: dict[tuple[int, str], tuple[int, str, str]] = {}
        self._tombstone_expiry_by_key: dict[tuple[int, str, str], float] = {}
        self._tombstone_expiry_by_dispatch: dict[tuple[int, str], float] = {}
        self._tombstone_order: deque[tuple[float, tuple[int, str, str]]] = deque()

    @property
    def coordinator_epoch(self) -> str:
        return self._coordinator_epoch

    def _remember_tombstone(self, key: tuple[int, str, str]) -> None:
        self._prune_tombstones()
        if key in self._tombstone_expiry_by_key:
            return
        expires_at = self._clock() + self._tombstone_ttl_s
        dispatch_key = (key[0], key[1])
        self._tombstone_expiry_by_key[key] = expires_at
        self._tombstone_expiry_by_dispatch[dispatch_key] = expires_at
        self._tombstone_order.append((expires_at, key))

    def _prune_tombstones(self) -> None:
        now = self._clock()
        while self._tombstone_order and self._tombstone_order[0][0] <= now:
            expires_at, key = self._tombstone_order.popleft()
            if self._tombstone_expiry_by_key.get(key) != expires_at:
                continue
            self._tombstone_expiry_by_key.pop(key, None)
            dispatch_key = (key[0], key[1])
            if self._tombstone_expiry_by_dispatch.get(dispatch_key) == expires_at:
                self._tombstone_expiry_by_dispatch.pop(dispatch_key, None)

    def _remove_lease(self, key: tuple[int, str, str]) -> bool:
        if self._leases_by_key.pop(key, None) is None:
            return False
        self._lease_key_by_dispatch.pop((key[0], key[1]), None)
        self._remember_tombstone(key)
        return True

    def _sweep_expired(self) -> None:
        self._prune_tombstones()
        now = self._clock()
        expired_keys = [key for key, lease in self._leases_by_key.items() if lease.expires_at <= now]
        if not expired_keys:
            return
        for key in expired_keys:
            self._remove_lease(key)
        self._availability_seq += 1

    def register_owner(self, *, shard_id: int, owner_epoch: int) -> None:
        if shard_id < 0:
            raise ValueError("shard_id must be non-negative")
        if owner_epoch <= 0:
            raise ValueError("owner_epoch must be positive")
        previous_epoch = self._owner_epoch_by_shard.get(shard_id)
        if previous_epoch == owner_epoch:
            return
        if any(
            registered_shard != shard_id and registered_epoch == owner_epoch
            for registered_shard, registered_epoch in self._owner_epoch_by_shard.items()
        ):
            raise ValueError("owner_epoch is already registered to another shard")
        self._owner_epoch_by_shard[shard_id] = owner_epoch
        if previous_epoch is None:
            return
        revoked_keys = [key for key in self._leases_by_key if key[0] == previous_epoch]
        for key in revoked_keys:
            self._remove_lease(key)
        if revoked_keys:
            self._availability_seq += 1

    def replace_worker_snapshots(self, *, batch: WorkerSnapshotBatch) -> int:
        if not isinstance(batch, WorkerSnapshotBatch):
            raise ValueError("batch must be a WorkerSnapshotBatch")
        current = self._snapshot_batch
        next_source = (batch.source_id, batch.publisher_epoch)
        if self._snapshot_source_fence is not None and next_source != self._snapshot_source_fence:
            raise ValueError("snapshot batch does not match the active source fence")
        if current is not None:
            current_source = (current.source_id, current.publisher_epoch)
            if next_source != current_source:
                raise ValueError("snapshot source changed without an explicit source fence")
            if batch.batch_seq <= current.batch_seq:
                return self._capacity_generation
            if (
                batch.source_open == current.source_open
                and batch.complete == current.complete
                and batch.snapshots == current.snapshots
            ):
                self._snapshot_batch = batch
                self._snapshot_received_at = self._clock()
                self._snapshot_stale_observed = False
                return self._capacity_generation
        self._snapshot_batch = batch
        self._snapshot_source_fence = next_source
        self._snapshot_received_at = self._clock()
        self._snapshot_stale_observed = False
        self._capacity_generation += 1
        self._availability_seq += 1
        revoked_keys = [key for key, lease in self._leases_by_key.items() if not lease.activated]
        for key in revoked_keys:
            self._remove_lease(key)
        return self._capacity_generation

    def fence_worker_snapshot_source(self, *, source_id: str, publisher_epoch: str) -> int:
        if not source_id or not publisher_epoch:
            raise ValueError("snapshot source fence fields must be non-empty")
        self._snapshot_source_fence = (source_id, publisher_epoch)
        self._snapshot_batch = None
        self._snapshot_received_at = None
        self._snapshot_stale_observed = False
        self._capacity_generation += 1
        self._availability_seq += 1
        revoked_keys = [key for key, lease in self._leases_by_key.items() if not lease.activated]
        for key in revoked_keys:
            self._remove_lease(key)
        return self._capacity_generation

    def _owner_is_active(self, owner_epoch: int) -> bool:
        return owner_epoch in self._owner_epoch_by_shard.values()

    def _observe_snapshot_staleness(self) -> None:
        if (
            self._snapshot_batch is not None
            and self._snapshot_received_at is not None
            and not self._snapshot_stale_observed
            and self._clock() - self._snapshot_received_at > self._snapshot_ttl_s
        ):
            self._snapshot_stale_observed = True
            self._availability_seq += 1

    @staticmethod
    def _key(
        *,
        routing_context: RoutingContext,
        admission_decision_id: str,
    ) -> tuple[int, str, str]:
        return (
            routing_context.owner_epoch,
            routing_context.dispatch_id,
            admission_decision_id,
        )

    def _capacity_view(
        self,
        *,
        serving_weight_version: str | None,
    ) -> tuple[int, int, bool] | None:
        self._observe_snapshot_staleness()
        if serving_weight_version is None:
            return None
        batch = self._snapshot_batch
        received_at = self._snapshot_received_at
        if (
            batch is None
            or received_at is None
            or not batch.source_open
            or not batch.complete
            or self._clock() - received_at > self._snapshot_ttl_s
        ):
            return None
        eligible = [
            snapshot
            for snapshot in batch.snapshots
            if snapshot.healthy and snapshot.serving_weight_version == serving_weight_version
        ]
        if not eligible:
            return None
        total_capacity = sum(snapshot.safe_execution_capacity_tokens for snapshot in eligible)
        hard_ceiling = int(total_capacity * (1.0 - self._safety_headroom_ratio))
        emergency_reserve = int(hard_ceiling * self._emergency_reserve_ratio)
        if hard_ceiling > 0 and self._emergency_reserve_ratio > 0:
            emergency_reserve = max(1, emergency_reserve)
        critical_pressure = any(snapshot.pressure_state == WorkerPressureState.CRITICAL for snapshot in eligible)
        return hard_ceiling, emergency_reserve, critical_pressure

    def _active_reservation_tokens(
        self,
        *,
        serving_weight_version: str,
        emergency: bool | None = None,
    ) -> int:
        return sum(
            lease.routing_context.reservation_tokens
            for lease in self._leases_by_key.values()
            if lease.routing_context.serving_weight_version == serving_weight_version
            and (emergency is None or lease.emergency == emergency)
        )

    def _reply(
        self,
        *,
        status: BudgetAcquireStatus,
        reason_code: AdmissionReason,
        lease: _LeaseRecord | None = None,
    ) -> CoordinatorAcquireReply:
        ttl_remaining_s = None
        capacity_generation = self._capacity_generation
        if lease is not None:
            ttl_remaining_s = max(0.0, lease.expires_at - self._clock())
            capacity_generation = lease.capacity_generation
        return CoordinatorAcquireReply(
            status=status,
            reason_code=reason_code,
            coordinator_epoch=self._coordinator_epoch,
            capacity_generation=capacity_generation,
            availability_seq=self._availability_seq,
            ttl_remaining_s=ttl_remaining_s,
        )

    def try_acquire(
        self,
        *,
        routing_context: RoutingContext,
        admission_decision_id: str,
        emergency: bool,
    ) -> CoordinatorAcquireReply:
        self._sweep_expired()
        key = self._key(
            routing_context=routing_context,
            admission_decision_id=admission_decision_id,
        )
        if not self._owner_is_active(routing_context.owner_epoch):
            return self._reply(
                status=BudgetAcquireStatus.REJECTED,
                reason_code=AdmissionReason.DEGRADED,
            )
        dispatch_key = (routing_context.owner_epoch, routing_context.dispatch_id)
        if key in self._tombstone_expiry_by_key or dispatch_key in self._tombstone_expiry_by_dispatch:
            return self._reply(
                status=BudgetAcquireStatus.REJECTED,
                reason_code=AdmissionReason.DEGRADED,
            )
        active_dispatch_lease_key = self._lease_key_by_dispatch.get(dispatch_key)
        if active_dispatch_lease_key is not None and active_dispatch_lease_key != key:
            return self._reply(
                status=BudgetAcquireStatus.REJECTED,
                reason_code=AdmissionReason.DEGRADED,
            )
        existing = self._leases_by_key.get(key)
        if existing is not None:
            if existing.routing_context != routing_context or existing.emergency != emergency:
                return self._reply(
                    status=BudgetAcquireStatus.REJECTED,
                    reason_code=AdmissionReason.DEGRADED,
                )
            return self._reply(
                status=BudgetAcquireStatus.ACQUIRED,
                reason_code=AdmissionReason.CAPACITY_AVAILABLE,
                lease=existing,
            )
        if len(self._tombstone_expiry_by_key) >= self._tombstone_limit:
            return self._reply(
                status=BudgetAcquireStatus.BYPASS,
                reason_code=AdmissionReason.DEGRADED,
            )
        capacity_view = self._capacity_view(
            serving_weight_version=routing_context.serving_weight_version,
        )
        if capacity_view is None:
            return self._reply(
                status=BudgetAcquireStatus.BYPASS,
                reason_code=AdmissionReason.DEGRADED,
            )
        hard_ceiling, emergency_reserve, critical_pressure = capacity_view
        if critical_pressure:
            return self._reply(
                status=BudgetAcquireStatus.CAPACITY_EXHAUSTED,
                reason_code=AdmissionReason.PRESSURE_GUARD,
            )
        normal_ceiling = max(0, hard_ceiling - emergency_reserve)
        total_active_tokens = self._active_reservation_tokens(
            serving_weight_version=routing_context.serving_weight_version,
        )
        normal_active_tokens = self._active_reservation_tokens(
            serving_weight_version=routing_context.serving_weight_version,
            emergency=False,
        )
        exceeds_hard_ceiling = total_active_tokens + routing_context.reservation_tokens > hard_ceiling
        exceeds_normal_ceiling = (
            not emergency and normal_active_tokens + routing_context.reservation_tokens > normal_ceiling
        )
        if exceeds_hard_ceiling or exceeds_normal_ceiling:
            return self._reply(
                status=BudgetAcquireStatus.CAPACITY_EXHAUSTED,
                reason_code=AdmissionReason.CAPACITY_EXHAUSTED,
            )
        lease = _LeaseRecord(
            routing_context=routing_context,
            admission_decision_id=admission_decision_id,
            expires_at=self._clock() + self._lease_ttl_s,
            capacity_generation=self._capacity_generation,
            emergency=emergency,
        )
        self._leases_by_key[key] = lease
        self._lease_key_by_dispatch[dispatch_key] = key
        return self._reply(
            status=BudgetAcquireStatus.ACQUIRED,
            reason_code=AdmissionReason.CAPACITY_AVAILABLE,
            lease=lease,
        )

    def reconcile(
        self,
        *,
        routing_context: RoutingContext,
        admission_decision_id: str,
    ) -> CoordinatorAcquireReply:
        self._sweep_expired()
        key = self._key(
            routing_context=routing_context,
            admission_decision_id=admission_decision_id,
        )
        lease = self._leases_by_key.get(key)
        if lease is not None:
            if lease.routing_context != routing_context:
                return self._reply(
                    status=BudgetAcquireStatus.REJECTED,
                    reason_code=AdmissionReason.DEGRADED,
                )
            return self._reply(
                status=BudgetAcquireStatus.ACQUIRED,
                reason_code=AdmissionReason.CAPACITY_AVAILABLE,
                lease=lease,
            )
        if key in self._tombstone_expiry_by_key:
            return self._reply(
                status=BudgetAcquireStatus.REJECTED,
                reason_code=AdmissionReason.DEGRADED,
            )
        return self._reply(
            status=BudgetAcquireStatus.UNKNOWN,
            reason_code=AdmissionReason.DEGRADED,
        )

    def cancel_unknown(
        self,
        *,
        routing_context: RoutingContext,
        admission_decision_id: str,
    ) -> None:
        key = self._key(
            routing_context=routing_context,
            admission_decision_id=admission_decision_id,
        )
        existing = self._leases_by_key.get(key)
        if existing is not None and existing.routing_context != routing_context:
            raise ValueError("cannot cancel a lease with a conflicting routing context")
        released = self._remove_lease(key)
        self._remember_tombstone(key)
        if released:
            self._availability_seq += 1

    def _lease_record(self, *, lease: AdmissionLease) -> _LeaseRecord | None:
        if lease.coordinator_epoch != self._coordinator_epoch:
            return None
        key = (lease.owner_epoch, lease.dispatch_id, lease.admission_decision_id)
        record = self._leases_by_key.get(key)
        if (
            record is None
            or record.routing_context.reservation_tokens != lease.reservation_tokens
            or not self._owner_is_active(lease.owner_epoch)
        ):
            return None
        return record

    def revalidate(self, *, lease: AdmissionLease) -> CoordinatorLeaseValidation:
        self._sweep_expired()
        record = self._lease_record(lease=lease)
        valid = record is not None and (
            record.activated
            or (
                record.capacity_generation == self._capacity_generation
                and self._capacity_view(serving_weight_version=record.routing_context.serving_weight_version)
                is not None
            )
        )
        return CoordinatorLeaseValidation(
            valid=valid,
            coordinator_epoch=self._coordinator_epoch,
            capacity_generation=self._capacity_generation if record is None else record.capacity_generation,
            availability_seq=self._availability_seq,
            ttl_remaining_s=None if record is None else max(0.0, record.expires_at - self._clock()),
        )

    def activate(self, *, lease: AdmissionLease) -> CoordinatorLeaseValidation:
        self._sweep_expired()
        record = self._lease_record(lease=lease)
        if record is None:
            return CoordinatorLeaseValidation(
                valid=False,
                coordinator_epoch=self._coordinator_epoch,
                capacity_generation=self._capacity_generation,
                availability_seq=self._availability_seq,
            )
        if not record.activated and (
            record.capacity_generation != self._capacity_generation
            or self._capacity_view(serving_weight_version=record.routing_context.serving_weight_version) is None
        ):
            self._remove_lease(record.key)
            self._availability_seq += 1
            return CoordinatorLeaseValidation(
                valid=False,
                coordinator_epoch=self._coordinator_epoch,
                capacity_generation=self._capacity_generation,
                availability_seq=self._availability_seq,
            )
        record.activated = True
        record.expires_at = self._clock() + self._lease_ttl_s
        return CoordinatorLeaseValidation(
            valid=True,
            coordinator_epoch=self._coordinator_epoch,
            capacity_generation=record.capacity_generation,
            availability_seq=self._availability_seq,
            ttl_remaining_s=self._lease_ttl_s,
        )

    def renew(self, *, lease: AdmissionLease) -> CoordinatorLeaseValidation:
        self._sweep_expired()
        record = self._lease_record(lease=lease)
        if record is None or not record.activated:
            return CoordinatorLeaseValidation(
                valid=False,
                coordinator_epoch=self._coordinator_epoch,
                capacity_generation=self._capacity_generation,
                availability_seq=self._availability_seq,
            )
        record.expires_at = self._clock() + self._lease_ttl_s
        return CoordinatorLeaseValidation(
            valid=True,
            coordinator_epoch=self._coordinator_epoch,
            capacity_generation=record.capacity_generation,
            availability_seq=self._availability_seq,
            ttl_remaining_s=self._lease_ttl_s,
        )

    def release(
        self,
        *,
        lease: AdmissionLease,
        outcome: LeaseReleaseOutcome,
    ) -> None:
        del outcome
        record = self._lease_record(lease=lease)
        if record is None:
            return
        released = self._remove_lease(record.key)
        if released:
            self._availability_seq += 1

    def availability_seq(self) -> int:
        self._sweep_expired()
        self._observe_snapshot_staleness()
        return self._availability_seq

    def snapshot(self) -> dict[str, object]:
        self._sweep_expired()
        batch = self._snapshot_batch
        return {
            "coordinator_epoch": self._coordinator_epoch,
            "capacity_generation": self._capacity_generation,
            "availability_seq": self._availability_seq,
            "registered_owners": len(self._owner_epoch_by_shard),
            "active_leases": len(self._leases_by_key),
            "in_flight_leases": sum(1 for lease in self._leases_by_key.values() if lease.activated),
            "reserved_tokens": sum(lease.routing_context.reservation_tokens for lease in self._leases_by_key.values()),
            "tombstones": len(self._tombstone_expiry_by_key),
            "snapshot_source_id": None if batch is None else batch.source_id,
            "snapshot_source_open": False if batch is None else batch.source_open,
            "snapshot_complete": False if batch is None else batch.complete,
            "snapshot_workers": 0 if batch is None else len(batch.snapshots),
        }


@ray.remote(num_cpus=0.25, max_concurrency=1, max_restarts=0, max_task_retries=0)
class AdmissionBudgetCoordinator:
    def __init__(
        self,
        *,
        lease_ttl_s: float = 30.0,
        snapshot_ttl_s: float = 5.0,
        safety_headroom_ratio: float = 0.1,
        emergency_reserve_ratio: float = 0.1,
    ) -> None:
        self._core = AdmissionBudgetCoordinatorCore(
            lease_ttl_s=lease_ttl_s,
            snapshot_ttl_s=snapshot_ttl_s,
            safety_headroom_ratio=safety_headroom_ratio,
            emergency_reserve_ratio=emergency_reserve_ratio,
        )

    def register_owner(self, *, shard_id: int, owner_epoch: int) -> None:
        self._core.register_owner(shard_id=shard_id, owner_epoch=owner_epoch)

    def replace_worker_snapshots(self, *, batch: WorkerSnapshotBatch) -> int:
        return self._core.replace_worker_snapshots(batch=batch)

    def fence_worker_snapshot_source(self, *, source_id: str, publisher_epoch: str) -> int:
        return self._core.fence_worker_snapshot_source(
            source_id=source_id,
            publisher_epoch=publisher_epoch,
        )

    def try_acquire(
        self,
        *,
        routing_context: RoutingContext,
        admission_decision_id: str,
        emergency: bool,
    ) -> CoordinatorAcquireReply:
        return self._core.try_acquire(
            routing_context=routing_context,
            admission_decision_id=admission_decision_id,
            emergency=emergency,
        )

    def reconcile(
        self,
        *,
        routing_context: RoutingContext,
        admission_decision_id: str,
    ) -> CoordinatorAcquireReply:
        return self._core.reconcile(
            routing_context=routing_context,
            admission_decision_id=admission_decision_id,
        )

    def cancel_unknown(
        self,
        *,
        routing_context: RoutingContext,
        admission_decision_id: str,
    ) -> None:
        self._core.cancel_unknown(
            routing_context=routing_context,
            admission_decision_id=admission_decision_id,
        )

    def revalidate(self, *, lease: AdmissionLease) -> CoordinatorLeaseValidation:
        return self._core.revalidate(lease=lease)

    def activate(self, *, lease: AdmissionLease) -> CoordinatorLeaseValidation:
        return self._core.activate(lease=lease)

    def renew(self, *, lease: AdmissionLease) -> CoordinatorLeaseValidation:
        return self._core.renew(lease=lease)

    def release(
        self,
        *,
        lease: AdmissionLease,
        outcome: LeaseReleaseOutcome,
    ) -> None:
        self._core.release(lease=lease, outcome=outcome)

    def availability_seq(self) -> int:
        return self._core.availability_seq()

    def health(self) -> dict[str, object]:
        return self._core.snapshot()


def create_admission_budget_coordinator():
    try:
        stale = ray.get_actor(AGENTIC_ADMISSION_BUDGET_COORDINATOR_NAME)
    except ValueError:
        pass
    else:
        ray.kill(stale, no_restart=True)
    return AdmissionBudgetCoordinator.options(
        name=AGENTIC_ADMISSION_BUDGET_COORDINATOR_NAME,
    ).remote()


def shutdown_admission_budget_coordinator() -> None:
    try:
        coordinator = ray.get_actor(AGENTIC_ADMISSION_BUDGET_COORDINATOR_NAME)
    except Exception:
        return
    try:
        ray.kill(coordinator, no_restart=True)
    except Exception:
        return


class RayAdmissionBudgetPort:
    """Translate coordinator RPCs into shard-local admission contracts."""

    def __init__(
        self,
        *,
        coordinator,
        clock: Callable[[], float] = time.monotonic,
        deadline_safety_ratio: float = 0.05,
        rpc_timeout_s: float = 2.0,
    ) -> None:
        if not 0 <= deadline_safety_ratio < 1:
            raise ValueError("deadline_safety_ratio must be in [0, 1)")
        if rpc_timeout_s <= 0:
            raise ValueError("rpc_timeout_s must be positive")
        self._coordinator = coordinator
        self._clock = clock
        self._deadline_safety_ratio = float(deadline_safety_ratio)
        self._rpc_timeout_s = float(rpc_timeout_s)
        self._availability_seq = 0

    async def _await_rpc(self, object_ref):
        return await asyncio.wait_for(
            asyncio.shield(object_ref),
            timeout=self._rpc_timeout_s,
        )

    async def register_owner(self, *, shard_id: int, owner_epoch: int) -> None:
        await self._await_rpc(
            self._coordinator.register_owner.remote(
                shard_id=shard_id,
                owner_epoch=owner_epoch,
            )
        )

    def _local_lease(
        self,
        *,
        reply: CoordinatorAcquireReply,
        routing_context: RoutingContext,
        admission_decision_id: str,
    ) -> AdmissionLease:
        ttl_remaining_s = reply.ttl_remaining_s
        if ttl_remaining_s is None or ttl_remaining_s <= 0:
            raise RuntimeError("Coordinator acquired reply has no usable remaining TTL.")
        safe_ttl_s = ttl_remaining_s * (1.0 - self._deadline_safety_ratio)
        if safe_ttl_s <= 0:
            raise RuntimeError("Coordinator acquired reply TTL is exhausted in transit.")
        return AdmissionLease(
            owner_epoch=routing_context.owner_epoch,
            dispatch_id=routing_context.dispatch_id,
            admission_decision_id=admission_decision_id,
            reservation_tokens=routing_context.reservation_tokens,
            ttl_s=safe_ttl_s,
            expires_at_local_monotonic=self._clock() + safe_ttl_s,
            coordinator_epoch=reply.coordinator_epoch,
            capacity_generation=reply.capacity_generation,
        )

    def _budget_result(
        self,
        *,
        reply: CoordinatorAcquireReply,
        routing_context: RoutingContext,
        admission_decision_id: str,
    ) -> BudgetAcquireResult:
        self._availability_seq = max(self._availability_seq, reply.availability_seq)
        lease = None
        if reply.status == BudgetAcquireStatus.ACQUIRED:
            lease = self._local_lease(
                reply=reply,
                routing_context=routing_context,
                admission_decision_id=admission_decision_id,
            )
        return BudgetAcquireResult(
            status=reply.status,
            reason_code=reply.reason_code,
            lease=lease,
        )

    async def try_acquire(
        self,
        *,
        routing_context: RoutingContext,
        admission_decision_id: str,
        emergency: bool,
    ) -> BudgetAcquireResult:
        try:
            reply = await self._await_rpc(
                self._coordinator.try_acquire.remote(
                    routing_context=routing_context,
                    admission_decision_id=admission_decision_id,
                    emergency=emergency,
                )
            )
        except (
            ray.exceptions.RayActorError,
            ray.exceptions.TaskCancelledError,
            TimeoutError,
        ) as exc:
            raise AdmissionBudgetUnknownCommitOutcome(str(exc)) from exc
        return self._budget_result(
            reply=reply,
            routing_context=routing_context,
            admission_decision_id=admission_decision_id,
        )

    async def reconcile(
        self,
        *,
        routing_context: RoutingContext,
        admission_decision_id: str,
    ) -> BudgetAcquireResult:
        try:
            reply = await self._await_rpc(
                self._coordinator.reconcile.remote(
                    routing_context=routing_context,
                    admission_decision_id=admission_decision_id,
                )
            )
        except ray.exceptions.RayActorError as exc:
            raise AdmissionBudgetUnavailableBeforeCommit(str(exc)) from exc
        except (ray.exceptions.TaskCancelledError, TimeoutError) as exc:
            raise AdmissionBudgetUnknownCommitOutcome(str(exc)) from exc
        return self._budget_result(
            reply=reply,
            routing_context=routing_context,
            admission_decision_id=admission_decision_id,
        )

    async def cancel_unknown(
        self,
        *,
        routing_context: RoutingContext,
        admission_decision_id: str,
    ) -> None:
        try:
            await self._await_rpc(
                self._coordinator.cancel_unknown.remote(
                    routing_context=routing_context,
                    admission_decision_id=admission_decision_id,
                )
            )
        except ray.exceptions.RayActorError as exc:
            raise AdmissionBudgetUnavailableBeforeCommit(str(exc)) from exc
        except (ray.exceptions.TaskCancelledError, TimeoutError) as exc:
            raise AdmissionBudgetUnknownCommitOutcome(str(exc)) from exc

    async def revalidate(self, *, lease: AdmissionLease) -> bool:
        try:
            reply = await self._await_rpc(self._coordinator.revalidate.remote(lease=lease))
        except (ray.exceptions.RayActorError, ray.exceptions.TaskCancelledError, TimeoutError):
            return False
        self._availability_seq = max(self._availability_seq, reply.availability_seq)
        return bool(reply.valid and reply.coordinator_epoch == lease.coordinator_epoch)

    def _renewed_local_lease(
        self,
        *,
        lease: AdmissionLease,
        reply: CoordinatorLeaseValidation,
    ) -> AdmissionLease | None:
        self._availability_seq = max(self._availability_seq, reply.availability_seq)
        if not reply.valid or reply.coordinator_epoch != lease.coordinator_epoch:
            return None
        ttl_remaining_s = reply.ttl_remaining_s
        if ttl_remaining_s is None or ttl_remaining_s <= 0:
            return None
        safe_ttl_s = ttl_remaining_s * (1.0 - self._deadline_safety_ratio)
        return AdmissionLease(
            owner_epoch=lease.owner_epoch,
            dispatch_id=lease.dispatch_id,
            admission_decision_id=lease.admission_decision_id,
            reservation_tokens=lease.reservation_tokens,
            ttl_s=safe_ttl_s,
            expires_at_local_monotonic=self._clock() + safe_ttl_s,
            coordinator_epoch=reply.coordinator_epoch,
            capacity_generation=reply.capacity_generation,
        )

    async def activate(self, *, lease: AdmissionLease) -> AdmissionLease | None:
        try:
            reply = await self._await_rpc(self._coordinator.activate.remote(lease=lease))
        except (ray.exceptions.RayActorError, ray.exceptions.TaskCancelledError, TimeoutError):
            return None
        return self._renewed_local_lease(lease=lease, reply=reply)

    async def renew(self, *, lease: AdmissionLease) -> AdmissionLease | None:
        try:
            reply = await self._await_rpc(self._coordinator.renew.remote(lease=lease))
        except (ray.exceptions.RayActorError, ray.exceptions.TaskCancelledError, TimeoutError) as exc:
            raise AdmissionBudgetUnknownRenewalOutcome(str(exc)) from exc
        return self._renewed_local_lease(lease=lease, reply=reply)

    async def release(
        self,
        *,
        lease: AdmissionLease,
        outcome: LeaseReleaseOutcome,
    ) -> None:
        try:
            await self._await_rpc(
                self._coordinator.release.remote(
                    lease=lease,
                    outcome=outcome,
                )
            )
        except ray.exceptions.RayActorError:
            return

    async def availability_seq(self) -> int:
        availability_seq = await self._await_rpc(self._coordinator.availability_seq.remote())
        self._availability_seq = max(self._availability_seq, int(availability_seq))
        return self._availability_seq
