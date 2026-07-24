# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import secrets
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Protocol

from relax.agentic.session.contracts import (
    AdmissionAction,
    AdmissionDecision,
    AdmissionGrant,
    AdmissionLease,
    AdmissionReason,
    BudgetAcquireResult,
    BudgetAcquireStatus,
    LeaseReleaseOutcome,
    RouteObservation,
    RoutingContext,
)
from relax.agentic.session.state import RequestKind
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class AdmissionBudgetUnavailableBeforeCommit(RuntimeError):
    """The budget service failed before it could commit an acquisition."""


class AdmissionBudgetUnknownCommitOutcome(RuntimeError):
    """The budget service may have committed an acquisition before failing."""


class AdmissionBudgetUnknownRenewalOutcome(RuntimeError):
    """An in-flight lease renewal outcome could not be determined."""


class AdmissionBudgetPort(Protocol):
    """Shard-side budget port.

    Implementations must normalize lease deadlines to the shard's local
    monotonic clock and make ``release`` idempotent by lease identity.
    """

    async def try_acquire(
        self,
        *,
        routing_context: RoutingContext,
        admission_decision_id: str,
        emergency: bool,
    ) -> BudgetAcquireResult: ...

    async def reconcile(
        self,
        *,
        routing_context: RoutingContext,
        admission_decision_id: str,
    ) -> BudgetAcquireResult: ...

    async def cancel_unknown(
        self,
        *,
        routing_context: RoutingContext,
        admission_decision_id: str,
    ) -> None: ...

    async def revalidate(self, *, lease: AdmissionLease) -> bool: ...

    async def activate(self, *, lease: AdmissionLease) -> AdmissionLease | None: ...

    async def renew(self, *, lease: AdmissionLease) -> AdmissionLease | None: ...

    async def release(
        self,
        *,
        lease: AdmissionLease,
        outcome: LeaseReleaseOutcome,
    ) -> None: ...

    async def availability_seq(self) -> int: ...

    async def record_route_observation(
        self,
        *,
        routing_context: RoutingContext,
        observation: RouteObservation,
        completion_tokens: int,
        event_seq: int,
        capacity_generation: int,
    ) -> bool: ...

    async def invalidate_resident_context(
        self,
        *,
        owner_epoch: int,
        engine_session_id: str,
        event_seq: int,
    ) -> bool: ...


class DisabledAdmissionBudgetPort:
    async def try_acquire(
        self,
        *,
        routing_context: RoutingContext,
        admission_decision_id: str,
        emergency: bool,
    ) -> BudgetAcquireResult:
        del routing_context, admission_decision_id, emergency
        return BudgetAcquireResult(
            status=BudgetAcquireStatus.BYPASS,
            reason_code=AdmissionReason.FEATURE_DISABLED,
        )

    async def reconcile(
        self,
        *,
        routing_context: RoutingContext,
        admission_decision_id: str,
    ) -> BudgetAcquireResult:
        del routing_context, admission_decision_id
        return BudgetAcquireResult(
            status=BudgetAcquireStatus.BYPASS,
            reason_code=AdmissionReason.FEATURE_DISABLED,
        )

    async def cancel_unknown(
        self,
        *,
        routing_context: RoutingContext,
        admission_decision_id: str,
    ) -> None:
        del routing_context, admission_decision_id

    async def revalidate(self, *, lease: AdmissionLease) -> bool:
        del lease
        return False

    async def activate(self, *, lease: AdmissionLease) -> AdmissionLease | None:
        del lease
        return None

    async def renew(self, *, lease: AdmissionLease) -> AdmissionLease | None:
        del lease
        return None

    async def release(
        self,
        *,
        lease: AdmissionLease,
        outcome: LeaseReleaseOutcome,
    ) -> None:
        del lease, outcome

    async def availability_seq(self) -> int:
        return 0

    async def record_route_observation(
        self,
        *,
        routing_context: RoutingContext,
        observation: RouteObservation,
        completion_tokens: int,
        event_seq: int,
        capacity_generation: int,
    ) -> bool:
        del routing_context, observation, completion_tokens, event_seq, capacity_generation
        return False

    async def invalidate_resident_context(
        self,
        *,
        owner_epoch: int,
        engine_session_id: str,
        event_seq: int,
    ) -> bool:
        del owner_epoch, engine_session_id, event_seq
        return False


@dataclass
class _AdmissionWaiter:
    routing_context: RoutingContext
    request_kind: RequestKind
    admission_decision_id: str
    enqueued_at: float
    future: asyncio.Future[AdmissionGrant]
    cancelled: bool = False


class AdmissionTicket:
    def __init__(
        self,
        *,
        scheduler: ProgramScheduler,
        waiter: _AdmissionWaiter | None,
        initial_decision: AdmissionDecision,
        grant: AdmissionGrant | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._waiter = waiter
        self.initial_decision = initial_decision
        self._grant = grant
        self._cleanup_task: asyncio.Task[None] | None = None

    async def wait(self) -> AdmissionGrant:
        if self._grant is not None:
            return self._grant
        if self._waiter is None:
            raise RuntimeError("Deferred admission ticket has no waiter.")
        return await self._scheduler.wait(self._waiter)

    async def cancel(self, *, outcome: LeaseReleaseOutcome = LeaseReleaseOutcome.CANCELLED) -> None:
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cancel_once(outcome=outcome))
        cleanup_task = self._cleanup_task
        try:
            await asyncio.shield(cleanup_task)
        except Exception:
            if cleanup_task.done() and self._cleanup_task is cleanup_task:
                self._cleanup_task = None
            raise

    async def _cancel_once(self, *, outcome: LeaseReleaseOutcome) -> None:
        if self._grant is not None:
            if self._grant.lease is not None:
                await self._scheduler.release(lease=self._grant.lease, outcome=outcome)
            return
        if self._waiter is not None:
            await self._scheduler.cancel(self._waiter, outcome=outcome)


class ProgramScheduler:
    def __init__(
        self,
        *,
        budget_port: AdmissionBudgetPort | None = None,
        max_wait_s: float = 30.0,
        protected_max_wait_s: float = 5.0,
        capacity_poll_s: float = 0.1,
        clock: Callable[[], float] = time.monotonic,
        decision_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if max_wait_s <= 0 or protected_max_wait_s <= 0 or capacity_poll_s <= 0:
            raise ValueError("admission wait bounds must be positive")
        self._budget_port = budget_port or DisabledAdmissionBudgetPort()
        self._max_wait_s = float(max_wait_s)
        self._protected_max_wait_s = float(protected_max_wait_s)
        self._capacity_poll_s = float(capacity_poll_s)
        self._capacity_poll_max_s = max(0.5, self._capacity_poll_s)
        self._clock = clock
        self._decision_id_factory = decision_id_factory or (lambda: f"decision-{secrets.token_hex(16)}")
        self._waiters_by_program: dict[str, deque[_AdmissionWaiter]] = {}
        self._ready_programs: deque[str] = deque()
        self._ready_program_keys: set[str] = set()
        self._pump_lock = asyncio.Lock()
        self._release_tasks: dict[tuple[int, str, str], asyncio.Task[None]] = {}
        self._released_lease_keys: set[tuple[int, str, str]] = set()
        self._released_lease_order: deque[tuple[int, str, str]] = deque()
        self._released_lease_limit = 8192
        self._capacity_watch_task: asyncio.Task[None] | None = None
        self._last_availability_seq: int | None = None

    @staticmethod
    def _decision(
        *,
        action: AdmissionAction,
        reason_code: AdmissionReason,
        routing_context: RoutingContext | None,
        admission_decision_id: str,
    ) -> AdmissionDecision:
        return AdmissionDecision(
            action=action,
            reason_code=reason_code,
            reservation_tokens=0 if routing_context is None else routing_context.reservation_tokens,
            admission_decision_id=admission_decision_id,
            owner_epoch=0 if routing_context is None else routing_context.owner_epoch,
        )

    def _append_ready_program(self, program_id: str) -> None:
        if program_id in self._ready_program_keys:
            return
        waiters = self._waiters_by_program.get(program_id)
        if not waiters:
            return
        self._ready_programs.append(program_id)
        self._ready_program_keys.add(program_id)

    def _enqueue(self, waiter: _AdmissionWaiter) -> None:
        program_id = waiter.routing_context.program_id
        self._waiters_by_program.setdefault(program_id, deque()).append(waiter)
        self._append_ready_program(program_id)

    def _has_deferred_waiters(self) -> bool:
        return any(
            not waiter.cancelled and not waiter.future.done()
            for waiters in self._waiters_by_program.values()
            for waiter in waiters
        )

    def _ensure_capacity_watch(self) -> None:
        if self._capacity_watch_task is not None and not self._capacity_watch_task.done():
            return
        self._capacity_watch_task = asyncio.create_task(self._watch_capacity())
        self._capacity_watch_task.add_done_callback(self._observe_background_task)

    def _stop_capacity_watch_if_idle(self) -> None:
        if self._has_deferred_waiters():
            return
        task = self._capacity_watch_task
        self._capacity_watch_task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    def _remove_waiter(self, waiter: _AdmissionWaiter) -> None:
        program_id = waiter.routing_context.program_id
        waiters = self._waiters_by_program.get(program_id)
        if waiters is None:
            return
        try:
            waiters.remove(waiter)
        except ValueError:
            return
        if not waiters:
            self._waiters_by_program.pop(program_id, None)
            self._ready_program_keys.discard(program_id)
            while program_id in self._ready_programs:
                self._ready_programs.remove(program_id)
        self._stop_capacity_watch_if_idle()

    async def submit(
        self,
        *,
        routing_context: RoutingContext | None,
        dispatch_id: str,
        request_kind: RequestKind,
        bypass_reason: AdmissionReason | None = None,
    ) -> AdmissionTicket:
        admission_decision_id = self._decision_id_factory()
        if bypass_reason is not None:
            decision = self._decision(
                action=AdmissionAction.BYPASS,
                reason_code=bypass_reason,
                routing_context=routing_context,
                admission_decision_id=admission_decision_id,
            )
            return AdmissionTicket(
                scheduler=self,
                waiter=None,
                initial_decision=decision,
                grant=AdmissionGrant(decision=decision),
            )
        if routing_context is None:
            decision = self._decision(
                action=AdmissionAction.BYPASS,
                reason_code=AdmissionReason.MISSING_IDENTITY,
                routing_context=None,
                admission_decision_id=admission_decision_id,
            )
            return AdmissionTicket(
                scheduler=self,
                waiter=None,
                initial_decision=decision,
                grant=AdmissionGrant(decision=decision),
            )
        if routing_context.dispatch_id != dispatch_id:
            raise ValueError("routing_context dispatch_id does not match admission dispatch_id")
        waiter = _AdmissionWaiter(
            routing_context=routing_context,
            request_kind=request_kind,
            admission_decision_id=admission_decision_id,
            enqueued_at=self._clock(),
            future=asyncio.get_running_loop().create_future(),
        )
        self._enqueue(waiter)
        try:
            await asyncio.shield(self._pump())
        except asyncio.CancelledError:
            waiter.cancelled = True
            self._remove_waiter(waiter)
            waiter.future.cancel()
            raise
        if waiter.future.done():
            grant = waiter.future.result()
            return AdmissionTicket(
                scheduler=self,
                waiter=None,
                initial_decision=grant.decision,
                grant=grant,
            )
        self._ensure_capacity_watch()
        decision = self._decision(
            action=AdmissionAction.DEFER,
            reason_code=AdmissionReason.CAPACITY_EXHAUSTED,
            routing_context=routing_context,
            admission_decision_id=admission_decision_id,
        )
        return AdmissionTicket(
            scheduler=self,
            waiter=waiter,
            initial_decision=decision,
        )

    async def wait(self, waiter: _AdmissionWaiter) -> AdmissionGrant:
        wait_bound_s = self._protected_max_wait_s if waiter.request_kind == RequestKind.PROTECTED else self._max_wait_s
        timeout_s = max(0.0, wait_bound_s - (self._clock() - waiter.enqueued_at))
        try:
            return await asyncio.wait_for(asyncio.shield(waiter.future), timeout=timeout_s)
        except TimeoutError:
            force_task = asyncio.create_task(self._force_waiter(waiter))
            force_task.add_done_callback(self._observe_background_task)
            try:
                return await asyncio.shield(waiter.future)
            except asyncio.CancelledError:
                waiter.cancelled = True
                self._remove_waiter(waiter)
                waiter.future.cancel()
                raise
        except asyncio.CancelledError:
            waiter.cancelled = True
            self._remove_waiter(waiter)
            waiter.future.cancel()
            raise

    async def _resolve_unknown(self, waiter: _AdmissionWaiter) -> BudgetAcquireResult:
        try:
            result = await self._budget_port.reconcile(
                routing_context=waiter.routing_context,
                admission_decision_id=waiter.admission_decision_id,
            )
        except AdmissionBudgetUnavailableBeforeCommit:
            return BudgetAcquireResult(
                status=BudgetAcquireStatus.BYPASS,
                reason_code=AdmissionReason.DEGRADED,
            )
        if result.status != BudgetAcquireStatus.UNKNOWN:
            return result
        try:
            await self._budget_port.cancel_unknown(
                routing_context=waiter.routing_context,
                admission_decision_id=waiter.admission_decision_id,
            )
        except AdmissionBudgetUnavailableBeforeCommit:
            pass
        return BudgetAcquireResult(
            status=BudgetAcquireStatus.BYPASS,
            reason_code=AdmissionReason.DEGRADED,
        )

    async def _acquire_waiter(self, waiter: _AdmissionWaiter, *, emergency: bool) -> BudgetAcquireResult:
        try:
            result = await self._budget_port.try_acquire(
                routing_context=waiter.routing_context,
                admission_decision_id=waiter.admission_decision_id,
                emergency=emergency,
            )
        except AdmissionBudgetUnavailableBeforeCommit:
            return BudgetAcquireResult(
                status=BudgetAcquireStatus.BYPASS,
                reason_code=AdmissionReason.DEGRADED,
            )
        except AdmissionBudgetUnknownCommitOutcome:
            return await self._resolve_unknown(waiter)
        if result.status == BudgetAcquireStatus.UNKNOWN:
            return await self._resolve_unknown(waiter)
        return result

    def _grant_waiter(self, waiter: _AdmissionWaiter, result: BudgetAcquireResult) -> None:
        if result.status == BudgetAcquireStatus.ACQUIRED:
            if (
                result.lease.dispatch_id != waiter.routing_context.dispatch_id
                or result.lease.owner_epoch != waiter.routing_context.owner_epoch
                or result.lease.admission_decision_id != waiter.admission_decision_id
                or result.lease.reservation_tokens != waiter.routing_context.reservation_tokens
            ):
                raise RuntimeError("Budget port returned a lease for a different admission dispatch.")
            action = AdmissionAction.ADMIT
            lease = result.lease
        elif result.status == BudgetAcquireStatus.BYPASS:
            action = AdmissionAction.BYPASS
            lease = None
        else:
            raise RuntimeError(f"Cannot grant unresolved budget status {result.status.value!r}.")
        decision = self._decision(
            action=action,
            reason_code=result.reason_code,
            routing_context=waiter.routing_context,
            admission_decision_id=waiter.admission_decision_id,
        )
        waiter.future.set_result(AdmissionGrant(decision=decision, lease=lease))

    async def _release_cancelled_result(
        self,
        *,
        waiter: _AdmissionWaiter,
        result: BudgetAcquireResult,
    ) -> None:
        if result.status == BudgetAcquireStatus.ACQUIRED:
            try:
                await self._release_budget(
                    lease=result.lease,
                    outcome=LeaseReleaseOutcome.CANCELLED,
                )
            except Exception:
                logger.exception(
                    "Failed to release lease acquired for cancelled dispatch_id=%s",
                    waiter.routing_context.dispatch_id,
                )

    async def _deliver_waiter_result(
        self,
        *,
        waiter: _AdmissionWaiter,
        result: BudgetAcquireResult,
    ) -> None:
        if waiter.cancelled or waiter.future.done():
            await self._release_cancelled_result(waiter=waiter, result=result)
            return
        try:
            self._grant_waiter(waiter, result)
        except Exception as exc:
            if result.status == BudgetAcquireStatus.ACQUIRED:
                try:
                    await self._release_budget(
                        lease=result.lease,
                        outcome=LeaseReleaseOutcome.FAILED,
                    )
                except Exception:
                    logger.exception(
                        "Failed to release invalid lease for dispatch_id=%s",
                        waiter.routing_context.dispatch_id,
                    )
            if not waiter.future.done():
                waiter.future.set_exception(exc)

    async def _pump(self) -> None:
        async with self._pump_lock:
            while self._ready_programs:
                round_size = len(self._ready_programs)
                completed_in_round = 0
                for _ in range(round_size):
                    program_id = self._ready_programs.popleft()
                    self._ready_program_keys.discard(program_id)
                    waiters = self._waiters_by_program.get(program_id)
                    if not waiters:
                        continue
                    while waiters and (waiters[0].cancelled or waiters[0].future.done()):
                        waiters.popleft()
                    if not waiters:
                        self._waiters_by_program.pop(program_id, None)
                        continue
                    waiter = waiters.popleft()
                    try:
                        result = await self._acquire_waiter(waiter, emergency=False)
                    except Exception as exc:
                        if not waiter.future.done():
                            waiter.future.set_exception(exc)
                        completed_in_round += 1
                    else:
                        if result.status == BudgetAcquireStatus.CAPACITY_EXHAUSTED:
                            waiters.appendleft(waiter)
                        else:
                            await self._deliver_waiter_result(waiter=waiter, result=result)
                            completed_in_round += 1
                    if waiters:
                        self._append_ready_program(program_id)
                    else:
                        self._waiters_by_program.pop(program_id, None)
                if completed_in_round == 0:
                    break
            self._stop_capacity_watch_if_idle()

    async def _force_waiter(self, waiter: _AdmissionWaiter) -> None:
        try:
            async with self._pump_lock:
                if waiter.future.done() or waiter.cancelled:
                    return
                self._remove_waiter(waiter)
                result = await self._acquire_waiter(
                    waiter,
                    emergency=waiter.request_kind == RequestKind.PROTECTED,
                )
                if result.status == BudgetAcquireStatus.CAPACITY_EXHAUSTED:
                    result = BudgetAcquireResult(
                        status=BudgetAcquireStatus.BYPASS,
                        reason_code=AdmissionReason.FAIRNESS_RESERVE,
                    )
                await self._deliver_waiter_result(waiter=waiter, result=result)
        except Exception as exc:
            if not waiter.future.done() and not waiter.cancelled:
                waiter.future.set_exception(exc)
            else:
                logger.exception(
                    "Admission force attempt failed after dispatch cancellation dispatch_id=%s",
                    waiter.routing_context.dispatch_id,
                )

    @staticmethod
    def _observe_background_task(task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Admission background task failed.")

    async def cancel(
        self,
        waiter: _AdmissionWaiter,
        *,
        outcome: LeaseReleaseOutcome,
    ) -> None:
        waiter.cancelled = True
        self._remove_waiter(waiter)
        if not waiter.future.done():
            waiter.future.cancel()
            return
        if waiter.future.cancelled():
            return
        grant = waiter.future.result()
        if grant.lease is not None:
            await self.release(lease=grant.lease, outcome=outcome)

    async def revalidate(self, *, lease: AdmissionLease) -> bool:
        if lease.expires_at_local_monotonic <= self._clock():
            return False
        return await self._budget_port.revalidate(lease=lease)

    async def renew(self, *, lease: AdmissionLease) -> AdmissionLease | None:
        if lease.expires_at_local_monotonic <= self._clock():
            return None
        return await self._budget_port.renew(lease=lease)

    async def activate(self, *, lease: AdmissionLease) -> AdmissionLease | None:
        if lease.expires_at_local_monotonic <= self._clock():
            return None
        return await self._budget_port.activate(lease=lease)

    async def _watch_capacity(self) -> None:
        failure_reported = False
        poll_s = self._capacity_poll_s
        try:
            while self._has_deferred_waiters():
                await asyncio.sleep(poll_s)
                try:
                    availability_seq = await self._budget_port.availability_seq()
                except Exception as exc:
                    if not failure_reported:
                        logger.warning("Admission capacity watch is unavailable: %s", exc)
                        failure_reported = True
                    poll_s = min(self._capacity_poll_max_s, poll_s * 2.0)
                    continue
                failure_reported = False
                if self._last_availability_seq is None or availability_seq != self._last_availability_seq:
                    self._last_availability_seq = availability_seq
                    poll_s = self._capacity_poll_s
                    await self._pump()
                else:
                    poll_s = min(self._capacity_poll_max_s, poll_s * 2.0)
        finally:
            if self._capacity_watch_task is asyncio.current_task():
                self._capacity_watch_task = None

    @staticmethod
    def _lease_key(lease: AdmissionLease) -> tuple[int, str, str]:
        return lease.owner_epoch, lease.dispatch_id, lease.admission_decision_id

    def _remember_released_lease(self, lease_key: tuple[int, str, str]) -> None:
        if lease_key in self._released_lease_keys:
            return
        self._released_lease_keys.add(lease_key)
        self._released_lease_order.append(lease_key)
        if len(self._released_lease_order) > self._released_lease_limit:
            expired_key = self._released_lease_order.popleft()
            self._released_lease_keys.discard(expired_key)

    def _finish_release_task(
        self,
        *,
        lease_key: tuple[int, str, str],
        release_task: asyncio.Task[None],
    ) -> None:
        if self._release_tasks.get(lease_key) is release_task:
            self._release_tasks.pop(lease_key, None)
        if release_task.cancelled() or release_task.exception() is not None:
            return
        self._remember_released_lease(lease_key)

    async def _release_budget(
        self,
        *,
        lease: AdmissionLease,
        outcome: LeaseReleaseOutcome,
    ) -> None:
        lease_key = self._lease_key(lease)
        if lease_key in self._released_lease_keys:
            return
        release_task = self._release_tasks.get(lease_key)
        if release_task is None:
            release_task = asyncio.create_task(
                self._budget_port.release(
                    lease=lease,
                    outcome=outcome,
                )
            )
            self._release_tasks[lease_key] = release_task
            release_task.add_done_callback(
                lambda task, key=lease_key: self._finish_release_task(
                    lease_key=key,
                    release_task=task,
                )
            )
        await asyncio.shield(release_task)
        self._remember_released_lease(lease_key)

    async def release(
        self,
        *,
        lease: AdmissionLease,
        outcome: LeaseReleaseOutcome,
    ) -> None:
        await self._release_budget(lease=lease, outcome=outcome)
        await asyncio.shield(self._pump())

    async def notify_capacity_changed(self) -> None:
        await asyncio.shield(self._pump())

    async def record_route_observation(
        self,
        *,
        routing_context: RoutingContext,
        observation: RouteObservation,
        completion_tokens: int,
        event_seq: int,
        capacity_generation: int,
    ) -> bool:
        return await self._budget_port.record_route_observation(
            routing_context=routing_context,
            observation=observation,
            completion_tokens=completion_tokens,
            event_seq=event_seq,
            capacity_generation=capacity_generation,
        )

    async def invalidate_resident_context(
        self,
        *,
        owner_epoch: int,
        engine_session_id: str,
        event_seq: int,
    ) -> bool:
        return await self._budget_port.invalidate_resident_context(
            owner_epoch=owner_epoch,
            engine_session_id=engine_session_id,
            event_seq=event_seq,
        )

    def snapshot(self) -> dict[str, int]:
        deferred = sum(
            1
            for waiters in self._waiters_by_program.values()
            for waiter in waiters
            if not waiter.cancelled and not waiter.future.done()
        )
        return {
            "deferred_requests": deferred,
            "ready_programs": len(self._ready_program_keys),
        }
