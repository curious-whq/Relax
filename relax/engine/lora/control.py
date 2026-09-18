# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Engine-local fencing and owned request execution, independent of
HTTP/GPU."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol
from uuid import uuid4

from relax.engine.lora.execution import CAPABILITY, ExecutionPhase, ExecutionRecord
from relax.engine.lora.journal import ControlJournal
from relax.engine.lora.publication import EngineIdentity, EngineReceipt
from relax.engine.lora.snapshot import AdapterConflictError, AdapterSnapshot


def body_digest(body: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def aborted_output() -> dict[str, Any]:
    return {
        "text": "",
        "output_ids": [],
        "meta_info": {"finish_reason": {"type": "abort"}, "output_token_logprobs": []},
    }


def owned_task(work: Any) -> asyncio.Task:
    task = asyncio.create_task(work)
    task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    return task


class LocalAdapterBackend(Protocol):
    """Resource owner contract: delivery is separate from scheduler drain
    proof."""

    async def capabilities(self) -> dict[str, Any]: ...

    async def execution_status(self, request_id: str) -> dict[str, Any]: ...

    async def forget_execution(self, request_id: str) -> None: ...

    async def prepared_status(self, operation_id: str) -> bool: ...

    async def physical_barrier(self) -> None: ...

    async def load(self, artifact: AdapterSnapshot, operation_id: str) -> None: ...

    async def unload(self, artifact: AdapterSnapshot, operation_id: str) -> None: ...

    async def generate(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def abort(self, request_id: str) -> None: ...

    async def close_session(self, session_id: str) -> None: ...


class BackendRejectedError(ValueError):
    """The backend proves that no scheduler request was submitted."""


@dataclass
class _Operation:
    artifact: AdapterSnapshot
    fenced: bool = False
    absent: bool = False
    ready: bool = False
    prepare: asyncio.Task | None = None
    retire: asyncio.Task | None = None
    requests: set[str] = field(default_factory=set)


@dataclass
class _Request:
    operation_id: str
    digest: str
    payload: dict[str, Any]
    task: asyncio.Task | None = None
    result: dict[str, Any] = field(default_factory=lambda: {"status": "pending"})
    cancel_requested: bool = False
    submitted: bool = False
    abort_task: asyncio.Task | None = None
    execution: ExecutionRecord | None = None
    completion: asyncio.Future | None = None
    observer: asyncio.Task | None = None
    drained_at: float | None = None


class EngineControl:
    """One HTTP process owns the engine until it exits; no owner takeover."""

    def __init__(
        self,
        backend: LocalAdapterBackend,
        *,
        engine_id: str,
        base_digest: str,
        capacity: int,
        journal: ControlJournal | None = None,
        request_capacity: int = 32,
    ) -> None:
        self.backend = backend
        if journal is not None and not all(
            callable(getattr(backend, method, None))
            for method in (
                "capabilities",
                "execution_status",
                "forget_execution",
                "prepared_status",
                "physical_barrier",
            )
        ):
            raise TypeError("production backend must implement the execution/drain capability interface")
        self.identity = EngineIdentity(engine_id, uuid4().hex)
        self.base_digest = base_digest
        self.capacity = capacity
        self.journal = journal
        self.request_capacity = request_capacity
        self.active_count = 0
        self._warmup_count = 0
        self.owner_id: str | None = None
        self.operations: dict[str, _Operation] = {}
        self.versions: dict[str, str] = {}
        self.digests: dict[str, str] = {}
        self.requests: dict[str, _Request] = {}
        self.aborted: dict[str, str] = {}

    def info(self) -> dict[str, Any]:
        return {
            "protocol": 2,
            "capability": CAPABILITY,
            "engine": asdict(self.identity),
            "base_digest": self.base_digest,
            "capacity": self.capacity,
            "request_capacity": self.request_capacity,
        }

    def claim(self, owner_id: str, incarnation: str) -> dict[str, Any]:
        if not owner_id or incarnation != self.identity.incarnation:
            raise AdapterConflictError("invalid owner or engine incarnation")
        if self.owner_id not in (None, owner_id):
            raise AdapterConflictError("engine belongs to another owner; restart the entire managed cohort")
        self.owner_id = owner_id
        return self.info()

    def authorize(self, owner_id: str, incarnation: str) -> None:
        if self.owner_id is None or self.owner_id != owner_id or incarnation != self.identity.incarnation:
            raise AdapterConflictError("owner or engine incarnation mismatch")

    def _operation(self, artifact: AdapterSnapshot, operation_id: str) -> _Operation:
        if artifact.base_model_digest != self.base_digest:
            raise AdapterConflictError("base-model digest mismatch")
        if not operation_id:
            raise ValueError("operation ID is required")
        existing = self.operations.get(operation_id)
        if existing is None and self.journal is not None:
            saved = self.journal.get("engine-operation", self.identity.incarnation + operation_id)
            if saved is not None:
                from pathlib import Path

                old = AdapterSnapshot(**{**saved["artifact"], "path": Path(saved["artifact"]["path"])})
                existing = _Operation(old, fenced=True, absent=True)
                self.operations[operation_id] = existing
        if existing is not None:
            if existing.artifact != artifact:
                raise AdapterConflictError("operation ID reused with a different artifact")
            return existing
        previous = self.digests.get(artifact.version_id)
        if previous is None and self.journal is not None:
            saved = self.journal.get("engine-version", self.identity.incarnation + artifact.version_id)
            previous = saved["digest"] if saved else None
        if previous is not None and previous != artifact.digest:
            raise AdapterConflictError("version ID content conflict")
        self.digests[artifact.version_id] = artifact.digest
        if self.journal is not None:
            self.journal.put(
                "engine-version", self.identity.incarnation + artifact.version_id, {"digest": artifact.digest}
            )
        operation = _Operation(artifact)
        self.operations[operation_id] = operation
        return operation

    def _receipt(self, operation_id: str) -> EngineReceipt:
        op = self.operations[operation_id]
        return EngineReceipt(
            self.identity,
            operation_id,
            op.artifact.version_id,
            op.artifact.digest,
            pinned=op.ready,
            absent=op.absent,
            fenced=op.fenced,
        )

    async def prepare(self, artifact: AdapterSnapshot, operation_id: str) -> EngineReceipt:
        op = self._operation(artifact, operation_id)
        if op.fenced:
            raise AdapterConflictError("operation has been fenced")
        if op.prepare is None:
            active = self.versions.get(artifact.version_id)
            if active is not None and active != operation_id:
                raise AdapterConflictError("version still belongs to an earlier operation")
            if len(self.versions) >= self.capacity:
                raise AdapterConflictError("engine adapter capacity exhausted")
            self.versions[artifact.version_id] = operation_id
            op.prepare = owned_task(self._load(op, operation_id))
        await asyncio.shield(op.prepare)
        if op.fenced:
            raise AdapterConflictError("operation was fenced while preparing")
        return self._receipt(operation_id)

    async def _load(self, op: _Operation, operation_id: str) -> None:
        await asyncio.to_thread(op.artifact.verify)
        if op.fenced:
            return
        await self.backend.load(op.artifact, operation_id)
        if op.fenced:
            return
        # GPU warmup has exactly the same execution owner and drain rules.
        if hasattr(self.backend, "execution_status"):
            rid = f"warmup-{operation_id}"
            payload = {
                "rid": rid,
                "input_ids": [0],
                "lora_path": op.artifact.lora_path,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            }
            request = self._admit(op, operation_id, payload, kind="warmup")
            await asyncio.shield(request.completion)
            result = request.result
            if result["status"] != "terminal" or result["output"]["meta_info"]["finish_reason"]["type"] == "abort":
                raise RuntimeError("adapter GPU warmup failed")
            await self.forget_async(rid, operation_id)
        op.ready = not op.fenced

    async def retire(self, artifact: AdapterSnapshot, operation_id: str) -> EngineReceipt:
        op = self._operation(artifact, operation_id)
        # The fence is installed before the first await, including retire-before-prepare.
        op.fenced = True
        for rid in tuple(op.requests):
            await self.abort(rid, operation_id)
        if op.retire is None or (op.retire.done() and not op.absent):
            op.retire = owned_task(self._unload(op, operation_id))
        await asyncio.shield(op.retire)
        receipt = self._receipt(operation_id)
        if self.journal is not None and op.absent:
            self.journal.put(
                "engine-operation",
                self.identity.incarnation + operation_id,
                {"artifact": {**asdict(artifact), "path": str(artifact.path)}, "receipt": asdict(receipt)},
                active=False,
            )
            await self.journal.barrier()
            self.operations.pop(operation_id, None)
            self.digests.pop(artifact.version_id, None)
        return receipt

    async def _unload(self, op: _Operation, operation_id: str) -> None:
        if op.prepare is not None:
            await asyncio.gather(op.prepare, return_exceptions=True)
        # No request can be admitted after fenced was set. Unknown requests keep
        # the operation pinned even when their Python task has returned an error.
        if op.requests:
            raise RuntimeError("adapter still has backend requests without terminal evidence")
        await self.backend.unload(op.artifact, operation_id)
        op.ready = False
        op.absent = True
        if self.versions.get(op.artifact.version_id) == operation_id:
            del self.versions[op.artifact.version_id]

    async def generate(self, operation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = payload.get("rid")
        if not isinstance(request_id, str) or not request_id or payload.get("stream", False):
            raise ValueError("a nonempty rid and non-streaming generation are required")
        op = self.operations[operation_id]
        if payload.get("lora_path") != op.artifact.lora_path:
            raise AdapterConflictError("request does not select the bound adapter")
        digest = body_digest(payload)
        if self.journal is not None:
            saved = self.journal.get("engine-request", self.identity.incarnation + request_id)
            if saved is not None:
                if saved["operation_id"] != operation_id or saved["digest"] != digest:
                    raise AdapterConflictError("rid reused with different content")
                return {"status": "expired", "drained": True, "code": "RESULT_EXPIRED"}
        request = self.requests.get(request_id)
        if request is not None:
            if request.digest != digest or request.operation_id != operation_id:
                raise AdapterConflictError("rid reused with different generation content")
        else:
            if op.fenced or not op.ready:
                raise AdapterConflictError("adapter is not ready or has been fenced")
            aborted_operation = self.aborted.get(request_id)
            if aborted_operation is None and self.journal is not None:
                saved = self.journal.get("engine-abort", self.identity.incarnation + request_id)
                aborted_operation = saved["operation_id"] if saved else None
            if aborted_operation is not None and aborted_operation != operation_id:
                raise AdapterConflictError("rid belongs to another operation")
            request = self._admit(op, operation_id, payload, cancelled=aborted_operation is not None)
        await asyncio.shield(request.completion)
        return dict(request.result)

    def _admit(
        self,
        op: _Operation,
        operation_id: str,
        payload: dict[str, Any],
        *,
        kind: str = "generation",
        cancelled: bool = False,
    ) -> _Request:
        rid = payload["rid"]
        if (kind == "warmup" and self._warmup_count >= 1) or (
            kind != "warmup" and self.active_count - self._warmup_count >= self.request_capacity
        ):
            raise BackendRejectedError("ENGINE_GENERATION_CAPACITY_EXHAUSTED")
        self.active_count += 1
        self._warmup_count += int(kind == "warmup")
        request = _Request(operation_id, body_digest(payload), dict(payload), cancel_requested=cancelled)
        request.execution = ExecutionRecord(rid, operation_id, request.digest, kind=kind)
        request.completion = asyncio.get_running_loop().create_future()
        self.requests[rid] = request
        op.requests.add(rid)
        request.task = owned_task(self._generate(rid, request))
        if hasattr(self.backend, "execution_status"):
            request.observer = owned_task(self._observe(rid, request))
        return request

    def _drained(self, rid: str, request: _Request, *, never_submitted: bool = False) -> None:
        if request.execution.drained:
            return
        self.active_count -= 1
        self._warmup_count -= int(request.execution.kind == "warmup")
        request.execution.observe_drain(never_submitted=never_submitted)
        request.result = request.execution.response()
        self.operations[request.operation_id].requests.discard(rid)
        if not request.completion.done():
            request.completion.set_result(request.result)
        if request.task is not None and request.task is not asyncio.current_task() and not request.task.done():
            # Physical drain and producer fence are proven; stop a stranded local
            # delivery coroutine without changing execution ownership.
            request.task.cancel()

    async def _generate(self, request_id: str, request: _Request) -> None:
        execution = request.execution
        try:
            if request.cancel_requested:
                execution.output = aborted_output()
                self._drained(request_id, request, never_submitted=True)
                return
            request.submitted = True
            execution.phase = ExecutionPhase.SUBMITTED
            execution.output = await self.backend.generate(request.payload)
            if not execution.output.get("meta_info", {}).get("finish_reason"):
                execution.output = None
                raise RuntimeError("backend returned no terminal result")
            if not hasattr(self.backend, "execution_status"):
                self._drained(request_id, request)
            else:
                await self.reconcile(request_id, request.operation_id)
        except BackendRejectedError as error:
            execution.delivery_error = str(error)
            if not hasattr(self.backend, "execution_status"):
                self._drained(request_id, request, never_submitted=True)
            else:
                # Even a tokenizer rejection must release its native lease.
                await self.reconcile(request_id, request.operation_id)
        except Exception as error:
            execution.delivery_error = f"{type(error).__name__}: {error}"
            request.result = execution.response()
            if not hasattr(self.backend, "execution_status"):
                request.result = {"status": "unknown", "error": execution.delivery_error}
                request.completion.set_result(request.result)
        finally:
            request.payload = {}

    async def _observe(self, rid: str, request: _Request) -> None:
        while not request.execution.drained:
            try:
                await self.reconcile(rid, request.operation_id)
            except Exception as error:
                request.execution.delivery_error = str(error)
            await asyncio.sleep(0.1)

    async def reconcile(self, rid: str, operation_id: str) -> dict[str, Any]:
        request = self.requests.get(rid)
        if request is not None and request.operation_id != operation_id:
            raise AdapterConflictError("request operation mismatch")
        if (
            request is not None
            and not request.execution.drained
            and request.submitted
            and hasattr(self.backend, "execution_status")
        ):
            proof = await self.backend.execution_status(rid)
            if proof.get("drained"):
                # Scheduler proof can beat the detokenizer result. Allow bounded
                # delivery grace while the independent observer keeps running.
                request.drained_at = request.drained_at or time.monotonic()
                if (
                    request.execution.output is None
                    and not request.task.done()
                    and time.monotonic() - request.drained_at < 1
                ):
                    return self.request_status(rid, operation_id)
                self._drained(rid, request, never_submitted=proof.get("never_submitted", False))
        return self.request_status(rid, operation_id)

    async def operation_status(self, operation_id: str) -> dict[str, Any]:
        if operation_id not in self.operations and self.journal is not None:
            saved = self.journal.get("engine-operation", self.identity.incarnation + operation_id)
            if saved is not None:
                return {"receipt": saved["receipt"], "preparing": False, "error": None}
        op = self.operations[operation_id]
        if op.prepare is not None and op.prepare.done() and hasattr(self.backend, "physical_barrier"):
            await self.backend.physical_barrier()
        if op.ready and hasattr(self.backend, "prepared_status"):
            if not await self.backend.prepared_status(operation_id):
                op.ready = False
        return {
            "receipt": asdict(self._receipt(operation_id)),
            "preparing": op.prepare is not None and not op.prepare.done(),
            "error": str(op.prepare.exception())
            if op.prepare and op.prepare.done() and op.prepare.exception()
            else None,
        }

    async def abort(self, request_id: str, operation_id: str) -> dict[str, Any]:
        previous = self.aborted.get(request_id)
        if previous is not None and previous != operation_id:
            raise AdapterConflictError("abort rid belongs to another operation")
        request = self.requests.get(request_id)
        if request is not None and request.operation_id != operation_id:
            raise AdapterConflictError("abort operation mismatch")
        self.aborted[request_id] = operation_id
        if self.journal is not None:
            self.journal.put(
                "engine-abort", self.identity.incarnation + request_id, {"operation_id": operation_id}, active=False
            )
            await self.journal.barrier()
        if request is not None:
            request.cancel_requested = True
            if request.abort_task is None and request.result["status"] in {"pending", "unknown"}:
                request.abort_task = owned_task(self._abort_until_terminal(request_id, request))
        return self.request_status(request_id, operation_id)

    async def _abort_until_terminal(self, request_id: str, request: _Request) -> None:
        while request.result["status"] in {"pending", "unknown"}:
            if request.submitted:
                try:
                    await self.backend.abort(request_id)
                    await self.reconcile(request_id, request.operation_id)
                except Exception as error:
                    request.execution.delivery_error = str(error)
            await asyncio.sleep(0.05)

    def request_status(self, request_id: str, operation_id: str) -> dict[str, Any]:
        request = self.requests.get(request_id)
        if request is not None:
            if request.operation_id != operation_id:
                raise AdapterConflictError("request operation mismatch")
            return dict(request.result)
        if self.aborted.get(request_id) == operation_id:
            return {"status": "terminal", "output": aborted_output()}
        if self.journal is not None:
            saved = self.journal.get("engine-request", self.identity.incarnation + request_id)
            if saved and saved["operation_id"] == operation_id:
                return {"status": "expired", "drained": True, "code": "RESULT_EXPIRED"}
        return {"status": "absent"}

    def forget(self, request_id: str, operation_id: str) -> None:
        """Drop large terminal output, retaining the immutable request
        tombstone."""
        request = self.requests.get(request_id)
        if request is None or request.operation_id != operation_id:
            raise AdapterConflictError("unknown request or operation mismatch")
        if request.result["status"] not in {"terminal", "terminal_error", "rejected", "expired"}:
            raise AdapterConflictError("cannot forget a request without terminal evidence")
        request.result = {"status": "expired"}

    async def forget_async(self, request_id: str, operation_id: str) -> None:
        request = self.requests.get(request_id)
        if request is None and self.journal is not None:
            saved = self.journal.get("engine-request", self.identity.incarnation + request_id)
            if saved and saved["operation_id"] == operation_id:
                return
        if request is None or request.operation_id != operation_id or not request.execution.drained:
            raise AdapterConflictError("cannot acknowledge an unsettled execution")
        if hasattr(self.backend, "forget_execution"):
            await self.backend.forget_execution(request_id)
        self.forget(request_id, operation_id)
        if self.journal is not None:
            self.journal.put(
                "engine-request",
                self.identity.incarnation + request_id,
                {"operation_id": operation_id, "digest": request.digest},
                active=False,
            )
            await self.journal.barrier()
            self.requests.pop(request_id, None)
            self.aborted.pop(request_id, None)
