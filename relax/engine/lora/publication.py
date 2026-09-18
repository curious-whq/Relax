# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Single-owner publication and lease ledger for immutable LoRA versions.

The owner must run on one event loop. Synchronous ledger operations never
yield; network operations run in owned, shielded tasks. This module does not
create Ray actors or change the production generation path. An engine adapter
must implement the fencing and terminal-evidence contract before being used
here.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, Sequence
from uuid import uuid4

from relax.engine.lora.journal import ControlJournal
from relax.engine.lora.snapshot import AdapterConflictError, AdapterSnapshot


class AdapterCapacityError(RuntimeError):
    """Pinned, loading or unresolved versions occupy all logical slots."""


class AdapterNotReadyError(RuntimeError):
    """The requested immutable version cannot be served by this engine."""


class AdapterPublicationError(RuntimeError):
    """Publication failed; inspect the ledger for outstanding cleanup."""


class AdapterLeaseError(RuntimeError):
    """A closed or conflicting Session/request identity was reused."""


class VersionState(str, Enum):
    PREPARING = "preparing"
    PUBLISHED = "published"
    RETIRING = "retiring"
    CLEANUP_FAILED = "cleanup_failed"
    ABORTED = "aborted"
    RETIRED = "retired"


@dataclass(frozen=True)
class EngineIdentity:
    engine_id: str
    incarnation: str


@dataclass(frozen=True)
class EngineReceipt:
    engine: EngineIdentity
    operation_id: str
    version_id: str
    digest: str
    pinned: bool = False
    absent: bool = False
    fenced: bool = False


class AdapterEngine(Protocol):
    """Required engine boundary, stronger than raw SGLang load/unload HTTP.

    ``prepare`` verifies artifact bytes and acknowledges all workers loaded and
    protected from eviction. ``retire`` fences this operation before unloading:
    even an earlier timed-out prepare can never recreate it after the receipt.
    Both operations are idempotent by operation ID and use bounded transport
    calls. A missing adapter alone is NOT a successful retirement fence.
    Failure or an ambiguous timeout must raise, preserving the coordinator's
    capacity charge.
    """

    @property
    def identity(self) -> EngineIdentity: ...

    async def prepare(self, snapshot: AdapterSnapshot, operation_id: str) -> EngineReceipt: ...

    async def retire(self, snapshot: AdapterSnapshot, operation_id: str) -> EngineReceipt: ...


@dataclass(frozen=True)
class AdapterBinding:
    version_id: str
    digest: str
    lora_path: str
    publication_epoch: int


@dataclass(frozen=True)
class RequestLease:
    request_id: str
    session_id: str
    engine: EngineIdentity
    operation_id: str
    binding: AdapterBinding

    def generation_fields(self) -> dict[str, str]:
        """Fields that must accompany the actual request to this exact
        engine."""
        return {"lora_path": self.binding.lora_path}


@dataclass(frozen=True)
class RequestTerminal:
    """Trusted engine evidence that this physical attempt has stopped.

    An HTTP disconnect, caller cancellation or an abort-submitted ACK is not
    terminal evidence. A transport retry must either be deduplicated by the
    engine or receive a separate request lease before sending.
    """

    request_id: str
    engine: EngineIdentity
    operation_id: str


@dataclass
class _Version:
    artifact: AdapterSnapshot
    operation_id: str = field(default_factory=lambda: uuid4().hex)
    state: VersionState = VersionState.PREPARING
    binding: AdapterBinding | None = None
    sessions: set[str] = field(default_factory=set)
    requests: set[str] = field(default_factory=set)
    ready: set[str] = field(default_factory=set)
    cleanup_pending: set[str] = field(default_factory=set)
    cancelled: bool = False
    errors: dict[str, str] = field(default_factory=dict)
    publication_seconds: float | None = None
    expected_epoch: int = 0
    publish_task: asyncio.Task[AdapterBinding] | None = None
    retire_task: asyncio.Task[None] | None = None


def _observe_task(task: asyncio.Task) -> None:
    # A caller may stop waiting; the owner still retains and finishes the task.
    if not task.cancelled():
        task.exception()


class AdapterVersionManager:
    """Own a fixed engine cohort, one default pointer and two kinds of leases.

    The production gateway supplies a single-writer journal. Memory-only mode
    is reserved for deterministic protocol tests. Recovery retains the original
    owner and refuses a changed engine incarnation without death evidence.
    """

    def __init__(
        self,
        engines: Sequence[AdapterEngine],
        *,
        capacity: int,
        base_model_digest: str,
        prepare_concurrency: int = 2,
        journal: ControlJournal | None = None,
    ) -> None:
        if capacity < 1 or not engines:
            raise ValueError("positive capacity and at least one target engine are required")
        self._engines = {engine.identity.engine_id: engine for engine in engines}
        if len(self._engines) != len(engines):
            raise ValueError("target engine IDs must be unique")
        self._identities = {key: engine.identity for key, engine in self._engines.items()}
        if any(not identity.engine_id or not identity.incarnation for identity in self._identities.values()):
            raise ValueError("engine IDs and incarnations must be nonempty")
        self._capacity = capacity
        self._base_model_digest = base_model_digest
        self._versions: dict[str, _Version] = {}
        self._sessions: dict[str, AdapterBinding] = {}
        self._closed_sessions: set[str] = set()
        self._requests: dict[str, RequestLease] = {}
        self._finished_requests: set[str] = set()
        self._default: AdapterBinding | None = None
        self._epoch = 0
        self._attempts: dict[str, _Version] = {}
        self._unavailable_engines: set[str] = set()
        if prepare_concurrency < 1:
            raise ValueError("prepare_concurrency must be positive")
        self._prepare_slots = asyncio.Semaphore(prepare_concurrency)
        self.journal = journal
        self._load_recovery: asyncio.Task | None = None
        if journal is not None:
            self._restore()

    def _version_body(self, record: _Version) -> dict:
        return {
            "artifact": {**asdict(record.artifact), "path": str(record.artifact.path)},
            "operation_id": record.operation_id,
            "state": record.state.value,
            "binding": asdict(record.binding) if record.binding else None,
            "ready": sorted(record.ready),
            "cleanup_pending": sorted(record.cleanup_pending),
            "cancelled": record.cancelled,
            "expected_epoch": record.expected_epoch,
            "errors": record.errors,
        }

    def _save_version(self, record: _Version) -> None:
        if self.journal is not None:
            active = record.state not in {VersionState.RETIRED, VersionState.ABORTED}
            body = self._version_body(record)
            rows = [("attempt", record.operation_id, body, active)]
            if self._versions.get(record.artifact.version_id) is record:
                rows.append(("version", record.artifact.version_id, body, active))
            self.journal.put_many(rows)

    def _restore(self) -> None:
        cohort = {key: asdict(value) for key, value in self._identities.items()}
        stored = self.journal.get("meta", "cohort")
        if stored is not None and stored != cohort:
            raise AdapterNotReadyError(
                "engine incarnation changed; old execution death must be proven before recovery"
            )
        self.journal.put("meta", "cohort", cohort)
        default = self.journal.get("meta", "default")
        if default:
            self._default = AdapterBinding(**default)
            self._epoch = self._default.publication_epoch
        for _, value in self.journal.records("version"):
            self._restore_version(value)
        for sid, value in self.journal.records("session"):
            binding = AdapterBinding(**value["binding"])
            self._sessions[sid] = binding
            self._versions[binding.version_id].sessions.add(sid)
        for rid, value in self.journal.records("lease"):
            lease = RequestLease(
                **{**value, "engine": EngineIdentity(**value["engine"]), "binding": AdapterBinding(**value["binding"])}
            )
            self._requests[rid] = lease
            self._versions[lease.binding.version_id].requests.add(rid)

    def _restore_version(self, value: dict) -> _Version:
        artifact = AdapterSnapshot(**{**value["artifact"], "path": Path(value["artifact"]["path"])})
        record = _Version(
            artifact,
            operation_id=value["operation_id"],
            state=VersionState(value["state"]),
            binding=AdapterBinding(**value["binding"]) if value["binding"] else None,
            ready=set(value["ready"]),
            cleanup_pending=set(value["cleanup_pending"]),
            cancelled=value["cancelled"],
            expected_epoch=value["expected_epoch"],
            errors=value["errors"],
        )
        self._versions[artifact.version_id] = record
        self._attempts[record.operation_id] = record
        return record

    async def durable(self) -> None:
        if self.journal is not None:
            await self.journal.barrier()

    async def recover(self) -> None:
        await self.durable()
        held = self.journal.get("meta", "physical_load") if self.journal else None
        if held and held.get("active"):
            record = self._attempts[held["operation_id"]]

            async def recover_load() -> None:
                # Finish observing the previous owner's physical permit before
                # any new operation can start a disk/GPU load on either engine.
                try:
                    await self._engines[held["engine_id"]].prepare(record.artifact, record.operation_id)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # HttpAdapterEngine raises only after physical completion or fence.
                    pass
                self.journal.put("meta", "physical_load", {"active": False})
                await self.durable()

            self._load_recovery = asyncio.create_task(recover_load())
            self._load_recovery.add_done_callback(_observe_task)
        for record in tuple(self._versions.values()):
            if record.state is VersionState.PREPARING:
                record.publish_task = asyncio.create_task(self._publish(record))
                record.publish_task.add_done_callback(_observe_task)
            elif record.state is VersionState.RETIRING:
                record.state = VersionState.CLEANUP_FAILED
        # Recovery reconciles unfinished executions before attempting retirement.

    @property
    def default(self) -> AdapterBinding | None:
        return self._default

    def _occupancy(self) -> int:
        return sum(v.state not in {VersionState.ABORTED, VersionState.RETIRED} for v in self._versions.values())

    def _existing(self, artifact: AdapterSnapshot) -> _Version | None:
        if artifact.base_model_digest != self._base_model_digest:
            raise AdapterConflictError("adapter was exported for a different base model")
        record = self._versions.get(artifact.version_id)
        if record is None and self.journal is not None:
            saved = self.journal.get("version", artifact.version_id)
            if saved:
                record = self._restore_version(saved)
        if record is not None and record.artifact.digest != artifact.digest:
            raise AdapterConflictError(f"adapter version {artifact.version_id!r} already has different content")
        return record

    async def publish(self, artifact: AdapterSnapshot, *, expected_default_epoch: int | None = None) -> AdapterBinding:
        """Retry/join publication without cancelling it when a caller
        disconnects.

        Previously published IDs return their original binding, even after
        retirement; they never roll the default back. Failed IDs can be retried
        with the same content only after fenced cleanup has completed.
        """
        record = self._existing(artifact)
        if record is not None and record.binding is not None:
            return record.binding
        if record is not None and record.publish_task is not None and not record.publish_task.done():
            return await asyncio.shield(record.publish_task)
        if record is not None and record.state is not VersionState.ABORTED:
            if record.publish_task is None:
                raise AdapterPublicationError("version has no publication task")
            return await asyncio.shield(record.publish_task)
        if self._occupancy() >= self._capacity:
            raise AdapterCapacityError(f"adapter capacity {self._capacity} is occupied")
        if record is not None and self.journal is not None:
            self._attempts.pop(record.operation_id, None)
        record = _Version(
            artifact=artifact,
            cleanup_pending=set(self._engines),
            expected_epoch=self._epoch if expected_default_epoch is None else expected_default_epoch,
        )
        self._attempts[record.operation_id] = record
        self._versions[artifact.version_id] = record
        self._save_version(record)
        record.publish_task = asyncio.create_task(self._publish(record), name=f"lora-publish:{artifact.version_id}")
        record.publish_task.add_done_callback(_observe_task)
        return await asyncio.shield(record.publish_task)

    def cancel_publication(self, version_id: str, operation_id: str | None = None) -> bool:
        """Request pre-commit cancellation; never undo an already published
        default."""
        if operation_id is not None and operation_id not in self._attempts and self.journal is not None:
            saved = self.journal.get("attempt", operation_id)
            if saved and saved["artifact"]["version_id"] == version_id:
                return False  # Historical operation cannot cancel a newer retry.
        record = self._versions[version_id] if operation_id is None else self._attempts[operation_id]
        if record.artifact.version_id != version_id:
            raise AdapterConflictError("publication operation/version mismatch")
        if record.state is not VersionState.PREPARING:
            return False
        record.cancelled = True
        self._save_version(record)
        # Install remote fences now; a blocked warmup must not delay cancellation.
        if record.retire_task is None:
            record.retire_task = asyncio.create_task(self._retire(record))
            record.retire_task.add_done_callback(_observe_task)
        return True

    def _check_engine(self, engine_id: str) -> AdapterEngine:
        engine = self._engines[engine_id]
        if engine_id in self._unavailable_engines or engine.identity != self._identities[engine_id]:
            raise AdapterNotReadyError(f"engine incarnation changed: {engine_id}")
        return engine

    def mark_engine_unavailable(self, engine_id: str) -> None:
        """Fence admissions after lost residency/health; recovery needs a new
        owner."""
        if engine_id not in self._engines:
            raise AdapterNotReadyError(f"unknown engine: {engine_id}")
        self._unavailable_engines.add(engine_id)

    def _validate_receipt(self, record: _Version, engine_id: str, receipt: EngineReceipt) -> None:
        self._check_engine(engine_id)
        if (
            receipt.engine != self._identities[engine_id]
            or receipt.operation_id != record.operation_id
            or receipt.version_id != record.artifact.version_id
            or receipt.digest != record.artifact.digest
        ):
            raise AdapterNotReadyError(f"stale or conflicting engine receipt: {engine_id}")

    async def _prepare_one(self, record: _Version, engine_id: str) -> None:
        if self._load_recovery is not None:
            await asyncio.shield(self._load_recovery)
        async with self._prepare_slots:
            engine = self._check_engine(engine_id)
            if self.journal is not None:
                self.journal.put(
                    "meta",
                    "physical_load",
                    {"active": True, "operation_id": record.operation_id, "engine_id": engine_id},
                )
                await self.durable()
            try:
                receipt = await engine.prepare(record.artifact, record.operation_id)
            except asyncio.CancelledError:
                raise  # A stopped observer must not release a remote physical permit.
            except Exception:
                if self.journal is not None:
                    self.journal.put("meta", "physical_load", {"active": False})
                    await self.durable()
                raise
            if self.journal is not None:
                self.journal.put("meta", "physical_load", {"active": False})
                await self.durable()
        self._validate_receipt(record, engine_id, receipt)
        if not receipt.pinned or receipt.absent or receipt.fenced:
            raise AdapterNotReadyError(f"adapter is not loaded and pinned: {engine_id}")
        record.ready.add(engine_id)
        self._save_version(record)

    async def _publish(self, record: _Version) -> AdapterBinding:
        started = time.monotonic()
        await self.durable()  # Persist operation identity before any remote load.
        results = await asyncio.gather(
            *(self._prepare_one(record, key) for key in self._engines), return_exceptions=True
        )
        record.errors = {
            key: str(result) for key, result in zip(self._engines, results) if isinstance(result, BaseException)
        }
        # A fresh operation-scoped remote proof, not comparison with a cached identity.
        for key, engine in self._engines.items():
            try:
                self._check_engine(key)
                if hasattr(engine, "prepared_receipt"):
                    receipt = await engine.prepared_receipt(record.operation_id)
                    self._validate_receipt(record, key, receipt)
                    if not receipt.pinned or receipt.fenced or receipt.absent:
                        raise AdapterNotReadyError("preparation proof is no longer valid")
            except Exception as error:
                record.errors[key] = str(error)
        # CAS ordering: a slow earlier publication cannot overwrite a later commit.
        if record.expected_epoch != self._epoch:
            record.errors["commit"] = "DEFAULT_EPOCH_CONFLICT"
        if record.cancelled or record.errors:
            record.state = VersionState.RETIRING
            if record.retire_task is not None:
                await asyncio.shield(record.retire_task)
                record.state = VersionState.CLEANUP_FAILED if record.cleanup_pending else VersionState.ABORTED
                self._save_version(record)
                await self.durable()
            else:
                await self._retire(record)
            raise AdapterPublicationError(f"publication of {record.artifact.version_id!r} failed or was cancelled")
        self._epoch += 1
        record.binding = AdapterBinding(
            record.artifact.version_id, record.artifact.digest, record.artifact.lora_path, self._epoch
        )
        record.state = VersionState.PUBLISHED
        # Single event-loop CAS commit, without disk/network/GPU waits.
        self._default = record.binding
        record.publication_seconds = time.monotonic() - started
        if self.journal is not None:
            self.journal.put_many(
                [
                    ("version", record.artifact.version_id, self._version_body(record), True),
                    ("attempt", record.operation_id, self._version_body(record), True),
                    ("meta", "default", asdict(record.binding), True),
                ]
            )
        await self.durable()
        return record.binding

    def prospective_binding(self, session_id: str) -> AdapterBinding | None:
        if self.journal is not None and session_id not in self._sessions:
            saved = self.journal.get("session", session_id)
            if saved and saved.get("closed"):
                raise AdapterLeaseError("Session identity is already closed")
        if not session_id or session_id in self._closed_sessions:
            raise AdapterLeaseError("Session identity is empty or already closed")
        return self._sessions.get(session_id, self._default)

    def bind_once(self, session_id: str) -> AdapterBinding:
        if self.journal is not None and session_id not in self._sessions:
            saved = self.journal.get("session", session_id)
            if saved and saved.get("closed"):
                raise AdapterLeaseError("Session identity is already closed")
        if not session_id or session_id in self._closed_sessions:
            raise AdapterLeaseError("Session identity is empty or already closed")
        if session_id in self._sessions:
            return self._sessions[session_id]
        if self._default is None:
            raise AdapterNotReadyError("no adapter version has been published")
        binding = self._default
        self._sessions[session_id] = binding
        self._versions[binding.version_id].sessions.add(session_id)
        if self.journal is not None:
            self.journal.put("session", session_id, {"binding": asdict(binding), "closed": False})
        return binding

    def release_session(self, session_id: str) -> bool:
        """Close once, keeping a tombstone even if bind_once has not arrived
        yet."""
        self._closed_sessions.add(session_id)
        if self.journal is not None:
            self.journal.put("session", session_id, {"closed": True}, active=False)
        binding = self._sessions.pop(session_id, None)
        if binding is None:
            return False
        self._versions[binding.version_id].sessions.remove(session_id)
        return True

    def begin_request(self, session_id: str, request_id: str, engine_id: str) -> RequestLease:
        """Reserve BEFORE sending one physical attempt to this exact engine."""
        if session_id not in self._sessions or session_id in self._closed_sessions:
            raise AdapterLeaseError("Session is unbound or closed")
        if not request_id or request_id in self._finished_requests:
            raise AdapterLeaseError("request identity is empty or already terminal")
        if engine_id not in self._engines:
            raise AdapterNotReadyError(f"engine is outside the publication cohort: {engine_id}")
        self._check_engine(engine_id)
        binding = self._sessions[session_id]
        record = self._versions[binding.version_id]
        if record.state is not VersionState.PUBLISHED or engine_id not in record.ready:
            raise AdapterNotReadyError(f"bound adapter is unavailable on {engine_id}")
        lease = RequestLease(request_id, session_id, self._identities[engine_id], record.operation_id, binding)
        previous = self._requests.get(request_id)
        if previous is not None and previous != lease:
            raise AdapterLeaseError("request identity was reused for a different attempt")
        self._requests[request_id] = lease
        record.requests.add(request_id)
        if self.journal is not None:
            self.journal.put("lease", request_id, asdict(lease))
        return lease

    def finish_request(self, terminal: RequestTerminal, *, persist: bool = True) -> bool:
        """Release once, only on trusted engine terminal evidence."""
        lease = self._requests.get(terminal.request_id)
        if lease is None or lease.engine != terminal.engine or lease.operation_id != terminal.operation_id:
            raise AdapterLeaseError("unknown or stale request terminal")
        if terminal.request_id in self._finished_requests:
            return False
        self._finished_requests.add(terminal.request_id)
        self._versions[lease.binding.version_id].requests.remove(terminal.request_id)
        if self.journal is not None and persist:
            self.journal.put("lease", terminal.request_id, asdict(lease), active=False)
        return True

    async def _retire_one(self, record: _Version, engine_id: str) -> None:
        engine = self._check_engine(engine_id)
        receipt = await engine.retire(record.artifact, record.operation_id)
        self._validate_receipt(record, engine_id, receipt)
        if not receipt.absent or not receipt.fenced:
            raise AdapterNotReadyError(f"retirement did not fence pending loads: {engine_id}")
        record.cleanup_pending.discard(engine_id)
        record.ready.discard(engine_id)

    async def _retire(self, record: _Version) -> None:
        self._save_version(record)
        await self.durable()
        targets = sorted(record.cleanup_pending)
        results = await asyncio.gather(*(self._retire_one(record, key) for key in targets), return_exceptions=True)
        for key, result in zip(targets, results):
            if isinstance(result, BaseException):
                record.errors[key] = str(result)
        if record.cleanup_pending:
            record.state = VersionState.CLEANUP_FAILED
        else:
            record.state = VersionState.RETIRED if record.binding is not None else VersionState.ABORTED
        self._save_version(record)
        await self.durable()

    async def collect(self) -> None:
        """Retire eligible versions, retrying only unconfirmed engine
        retirements."""
        tasks = []
        for record in self._versions.values():
            if record.binding is not None and record.binding == self._default:
                continue
            if record.sessions or record.requests:
                continue
            if record.state in {VersionState.PUBLISHED, VersionState.CLEANUP_FAILED}:
                record.state = VersionState.RETIRING
                record.retire_task = asyncio.create_task(self._retire(record))
                record.retire_task.add_done_callback(_observe_task)
            if record.retire_task is not None and not record.retire_task.done():
                tasks.append(record.retire_task)
        if tasks:
            await asyncio.shield(asyncio.gather(*tasks))
        if self.journal is not None:
            # Historical identities remain in indexed storage, not online scans.
            for key, record in tuple(self._versions.items()):
                if record.state in {VersionState.RETIRED, VersionState.ABORTED} and (
                    record.publish_task is None or record.publish_task.done()
                ):
                    self._versions.pop(key)
                    self._attempts.pop(record.operation_id, None)

    def snapshot(self) -> dict[str, object]:
        """Return machine-readable state without exposing mutable ledger
        entries."""
        return {
            "default_version": self._default.version_id if self._default else None,
            "publication_epoch": self._epoch,
            "capacity": self._capacity,
            "occupied": self._occupancy(),
            "versions": {
                key: {
                    "digest": record.artifact.digest,
                    "operation_id": record.operation_id,
                    "state": record.state.value,
                    "ready_engines": sorted(record.ready),
                    "cleanup_pending": sorted(record.cleanup_pending),
                    "session_refs": len(record.sessions),
                    "request_refs": len(record.requests),
                    "cancelled": record.cancelled,
                    "errors": dict(record.errors),
                    "publication_seconds": record.publication_seconds,
                }
                for key, record in self._versions.items()
            },
        }
