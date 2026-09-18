# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Single-owner online publication and generation gateway for dedicated
engines."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import time
from collections import defaultdict
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from relax.engine.lora.admission import request_tokens
from relax.engine.lora.artifact import ModelContract
from relax.engine.lora.control import aborted_output, body_digest, owned_task
from relax.engine.lora.http import HttpAdapterEngine
from relax.engine.lora.journal import ControlJournal
from relax.engine.lora.publication import (
    AdapterCapacityError,
    AdapterLeaseError,
    AdapterNotReadyError,
    AdapterPublicationError,
    AdapterVersionManager,
    RequestLease,
    RequestTerminal,
)
from relax.engine.lora.snapshot import AdapterConflictError, fingerprint_model, snapshot_adapter


@dataclass
class _Dispatch:
    digest: str
    lease: RequestLease
    engine: HttpAdapterEngine
    payload: dict[str, Any]
    started: float
    task: asyncio.Task | None = None
    result: dict[str, Any] | None = None
    settled: bool = False
    forgotten: bool = False
    completion: asyncio.Future | None = None
    reserved_tokens: int = 0


class PublicationGateway:
    def __init__(
        self,
        engines: list[HttpAdapterEngine],
        *,
        capacity: int,
        base_digest: str,
        store: Path,
        journal: ControlJournal | None = None,
        contract: ModelContract | None = None,
        requests_per_engine: int = 32,
        max_payload_bytes: int = 4 * 1024 * 1024,
        artifact_bytes: int = 8 * 1024**3,
        token_budget_per_engine: int = 65536,
        max_request_tokens: int = 8192,
    ) -> None:
        self.engines = {engine.identity.engine_id: engine for engine in engines}
        # Disk loads may occupy one scheduler; stagger engines so publication
        # never deliberately blocks every engine with a simultaneous disk load.
        self.manager = AdapterVersionManager(
            engines, capacity=capacity, base_model_digest=base_digest, prepare_concurrency=1, journal=journal
        )
        self.journal = journal
        self.contract = contract
        if requests_per_engine < 1 or any(engine.request_capacity < requests_per_engine for engine in engines):
            raise ValueError("gateway request budget exceeds a target engine's advertised capacity")
        self.requests_per_engine = requests_per_engine
        self.max_payload_bytes = max_payload_bytes
        self.token_budget_per_engine = token_budget_per_engine
        self.max_request_tokens = max_request_tokens
        if min(token_budget_per_engine, max_request_tokens, artifact_bytes, max_payload_bytes) <= 0:
            raise ValueError("resource budgets must be positive")
        self.engine_tokens = {key: 0 for key in self.engines}
        self.artifact_bytes = artifact_bytes
        self._staging_active = False
        self.pending: dict[str, _Dispatch] = {}
        self.by_session: dict[str, set[str]] = defaultdict(set)
        self.to_forget: dict[str, _Dispatch] = {}
        self.engine_load = {key: 0 for key in self.engines}
        self.affinity: dict[str, str] = {}
        self._collection: asyncio.Task | None = None
        self.close_outbox: Any = None
        self.ray_address: str | None = None
        self.owners: dict[str, dict[str, str]] = {}
        self.owner_sessions: dict[str, set[str]] = defaultdict(set)
        self._owner_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._dead_owners: set[str] = set()
        self.base_digest = base_digest
        self.store = store
        self.requests: dict[str, _Dispatch] = {}
        self.aborted: set[str] = set()
        self.aborted_digests: dict[str, str] = {}
        self.closed: set[str] = set()
        self.completed = 0
        self.failed = 0
        self.aborted_count = 0
        self.latency_sum = 0.0
        self.generated_tokens = 0

    async def publish(
        self, version_id: str, export_dir: str, expected_default_epoch: int | None = None
    ) -> dict[str, Any]:
        # Lightweight admission BEFORE copying/hashing. Logical reservation and
        # artifact disk budgets are independent; only one staging task is admitted.
        if self._staging_active:
            raise AdapterCapacityError("ARTIFACT_STAGING_BUSY")
        if version_id not in self.manager._versions and self.manager._occupancy() >= self.manager._capacity:
            raise AdapterCapacityError("VERSION_CAPACITY_EXHAUSTED")
        self._staging_active = True
        try:
            source = Path(export_dir)
            if self.contract is not None:
                await asyncio.to_thread(self.contract.validate, source)
            size = sum(item.stat().st_size for item in source.iterdir() if item.is_file())
            used = await asyncio.to_thread(
                lambda: sum(item.stat().st_size for item in self.store.glob("*/*") if item.is_file())
            )
            if size > self.artifact_bytes or used + size > self.artifact_bytes:
                raise AdapterCapacityError("ARTIFACT_STORAGE_EXHAUSTED")
            artifact = await asyncio.to_thread(
                snapshot_adapter, source, self.store, version_id=version_id, base_model_digest=self.base_digest
            )
        finally:
            self._staging_active = False
        binding = await self.manager.publish(artifact, expected_default_epoch=expected_default_epoch)
        record = self.manager._versions[version_id]
        return {
            **asdict(binding),
            "operation_id": record.operation_id,
            "state": record.state.value,
            "is_default": self.manager.default == binding,
        }

    async def archive(self, version_id: str) -> dict[str, Any]:
        if self.journal is None:
            raise ValueError("artifact archival requires durable identity metadata")
        if self._staging_active:
            raise AdapterCapacityError("artifact staging is busy")
        self._staging_active = True
        try:
            record = self.manager._versions.get(version_id)
            saved = (
                self.manager._version_body(record)
                if record
                else await asyncio.to_thread(self.journal.get, "version", version_id)
            )
            if not saved or saved["state"] not in {"retired", "aborted"}:
                raise AdapterLeaseError("only fully retired/aborted artifacts can be archived")
            # Keep the immutable identity and operation history after deleting bytes.
            await self.manager.durable()
            path = Path(saved["artifact"]["path"])
            if path.parent.resolve() != self.store.resolve() or path.is_symlink():
                raise ValueError("artifact is outside the managed store")
            if path.exists():
                await asyncio.to_thread(shutil.rmtree, path)
            return {"version_id": version_id, "digest": saved["artifact"]["digest"], "archived": True}
        finally:
            self._staging_active = False

    async def refresh_health(self) -> None:
        for key, engine in self.engines.items():
            try:
                for record in tuple(self.manager._versions.values()):
                    if record.binding is None or record.state.value != "published":
                        continue
                    receipt = await engine.prepared_receipt(record.operation_id)
                    if receipt.engine != self.manager._identities[key] or not receipt.pinned or receipt.fenced:
                        raise AdapterNotReadyError("preparation proof invalidated")
                self.manager._unavailable_engines.discard(key)
            except Exception:
                self.manager.mark_engine_unavailable(key)

    async def recover(self) -> None:
        await self.manager.recover()
        if self.journal is None:
            return
        for key, value in self.journal.records("owner"):
            self.owners[key] = {"actor_id": value["actor_id"], "epoch": value["epoch"]}
            if value.get("dead"):
                self._dead_owners.add(key)
        for sid, value in self.journal.records("session_owner"):
            self.owner_sessions[value["owner"]].add(sid)
        for rid, value in self.journal.records("dispatch"):
            lease_value = value["lease"]
            from relax.engine.lora.publication import AdapterBinding, EngineIdentity

            lease = RequestLease(
                **{
                    **lease_value,
                    "engine": EngineIdentity(**lease_value["engine"]),
                    "binding": AdapterBinding(**lease_value["binding"]),
                }
            )
            dispatch = _Dispatch(
                value["digest"],
                lease,
                self.engines[lease.engine.engine_id],
                {},
                time.monotonic(),
                completion=asyncio.get_running_loop().create_future(),
            )
            self.requests[rid] = self.pending[rid] = dispatch
            self.by_session[lease.session_id].add(rid)
            dispatch.reserved_tokens = value.get("tokens", self.max_request_tokens)
            self.engine_load[lease.engine.engine_id] += 1
            self.engine_tokens[lease.engine.engine_id] += dispatch.reserved_tokens
            closed = self.journal.get("session", lease.session_id)
            if closed and closed.get("closed"):
                self.closed.add(lease.session_id)
        # A crash between lease journal and dispatch journal precedes dispatch's
        # durability barrier, so fence the possible delayed admission explicitly.
        represented = {item.lease.request_id for item in self.pending.values()}
        for lease in tuple(self.manager._requests.values()):
            if lease.request_id not in represented:
                result = await self.engines[lease.engine.engine_id].call(
                    "abort",
                    {
                        "rid": lease.request_id,
                        "operation_id": lease.operation_id,
                    },
                )
                if result.get("status") == "terminal":
                    self.manager.finish_request(RequestTerminal(lease.request_id, lease.engine, lease.operation_id))
        await self.manager.durable()

    def bind(self, session_id: str) -> dict[str, Any]:
        if session_id not in self.manager._sessions and len(self.manager._sessions) >= 10000:
            raise AdapterCapacityError("SESSION_CAPACITY_EXHAUSTED")
        return asdict(self.manager.bind_once(session_id))

    async def bind_owned(
        self,
        session_id: str,
        owner: dict[str, str] | None = None,
        spool_id: str | None = None,
        spool_url: str | None = None,
    ) -> dict[str, Any]:
        if owner is not None:
            if (
                self.close_outbox is None
                or spool_id != self.close_outbox.identity
                or spool_url != self.close_outbox.gateway_url
            ):
                raise ValueError("Session close outbox is not served by this gateway")
            if (
                not isinstance(owner, dict)
                or set(owner) != {"actor_id", "epoch"}
                or any(not isinstance(value, str) for value in owner.values())
            ):
                raise ValueError("invalid Session owner")
            key = owner["actor_id"] + ":" + owner["epoch"]
            async with self._owner_locks[key]:
                return await self._bind_owned(session_id, owner)
        return await self._bind_owned(session_id, owner)

    async def _bind_owned(self, session_id: str, owner: dict[str, str] | None) -> dict[str, Any]:
        if owner is not None:
            if self.journal is None or not self.ray_address:
                raise ValueError("Ray-owned Sessions require durable journal and an explicit Ray State address")
            if set(owner) != {"actor_id", "epoch"} or any(
                not isinstance(value, str) or not value for value in owner.values()
            ):
                raise ValueError("invalid Session owner identity")
            key = owner["actor_id"] + ":" + owner["epoch"]
            saved = await asyncio.to_thread(self.journal.get, "owner", key)
            if key in self._dead_owners or (saved and saved.get("dead")):
                raise AdapterLeaseError("Session owner has been fenced after confirmed death")
            previous = await asyncio.to_thread(self.journal.get, "session_owner", session_id)
            if previous and (previous.get("closed") or previous.get("owner") != key):
                raise AdapterLeaseError("Session belongs to a different owner incarnation")
            self.owners[key] = owner
            self.owner_sessions[key].add(session_id)
            self.journal.put("owner", key, {**owner, "dead": False})
            self.journal.put("session_owner", session_id, {"owner": key})
            await self.manager.durable()
        result = self.bind(session_id)
        await self.manager.durable()
        return result

    async def reconcile_owners(self) -> None:
        if self.close_outbox is not None:
            for path, sid in await asyncio.to_thread(self.close_outbox.pending):
                await self.close(sid)
                await asyncio.to_thread(self.close_outbox.acknowledge, path)
        if not self.ray_address:
            return
        from ray.util.state import get_actor

        for key, owner in tuple(self.owners.items()):
            if key not in self._dead_owners:
                try:
                    actor = await asyncio.to_thread(get_actor, owner["actor_id"], address=self.ray_address, timeout=5)
                except Exception:
                    continue
                if actor is None or actor.actor_id != owner["actor_id"] or actor.state != "DEAD":
                    continue
            async with self._owner_locks[key]:
                self._dead_owners.add(key)
                # Remains active until every dependent close has been journaled.
                self.journal.put("owner", key, {**owner, "dead": True})
                await self.manager.durable()
                for sid in tuple(self.owner_sessions[key]):
                    await self.close(sid)
                self.journal.put("owner", key, {**owner, "dead": True}, active=False)
                await self.manager.durable()
                self.owners.pop(key)
                self.owner_sessions.pop(key, None)

    async def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        session_id = payload.get("session_id")
        rid = payload.get("rid")
        if not isinstance(session_id, str) or not isinstance(rid, str) or not rid or not session_id:
            raise ValueError("session_id and rid are required")
        if "lora_path" in payload or payload.get("stream", False):
            raise ValueError("gateway chooses lora_path; use non-streaming requests")
        if session_id in self.closed:
            raise AdapterLeaseError("Session is closed")
        tokens = request_tokens(payload, max_tokens=self.max_request_tokens)
        encoded = json.dumps(payload, allow_nan=False).encode()
        if len(encoded) > self.max_payload_bytes:
            raise ValueError("PAYLOAD_TOO_LARGE")
        expected = payload.get("expected_adapter")
        prospective = self.manager.prospective_binding(session_id)
        if expected is not None and (prospective is None or expected != asdict(prospective)):
            raise AdapterConflictError("Session adapter binding mismatch")
        digest = body_digest(payload)
        if self.journal is not None:
            history = await asyncio.to_thread(self.journal.get, "result", rid)
            if history is not None:
                if history["digest"] != digest:
                    raise AdapterConflictError("rid reused with different content")
                if history["result"]["status"] == "terminal":
                    return history["result"]["output"]
                raise AdapterLeaseError(history["result"].get("code", "RESULT_EXPIRED"))
        previous = self.requests.get(rid)
        if previous is not None:
            if previous.digest != digest:
                raise AdapterConflictError("rid reused with different content")
            dispatch = previous
        else:
            available = [
                key
                for key in self.engines
                if self.engine_load[key] < self.requests_per_engine
                and self.engine_tokens[key] + tokens <= self.token_budget_per_engine
                and key not in self.manager._unavailable_engines
            ]
            if not available:
                raise AdapterCapacityError("GENERATION_CAPACITY_EXHAUSTED")
            # Unknown previous attempts may not be migrated or duplicated.
            previous_engines = {
                self.pending[key].lease.engine.engine_id for key in self.by_session.get(session_id, ())
            }
            if previous_engines:
                available = [key for key in available if key in previous_engines]
                if not available:
                    raise AdapterNotReadyError("Session has an unsettled physical attempt")
            selected = self.affinity.get(session_id)
            if selected not in available:
                selected = min(available, key=lambda key: (self.engine_load[key], key))
            self.bind(session_id)
            binding = self.manager.bind_once(session_id)
            payload.pop("expected_adapter", None)
            if rid in self.aborted or (self.journal is not None and self.journal.get("abort", rid) is not None):
                if self.aborted_digests.setdefault(rid, digest) != digest:
                    raise AdapterConflictError("aborted rid reused with different content")
                output = aborted_output()
                output["meta_info"]["lora_adapter"] = asdict(binding)
                if self.journal is not None:
                    self.journal.put_result(rid, digest, {"status": "terminal", "output": output})
                await self.manager.durable()
                return output
            self.affinity[session_id] = selected
            engine = self.engines[selected]
            physical_rid = hashlib.sha256((engine.owner_id + rid).encode()).hexdigest()
            lease = self.manager.begin_request(session_id, physical_rid, selected)
            # SGLang session_id is a KV hint, unrelated to binding. The managed
            # endpoint uses full prompts and ordinary per-version radix caching.
            payload.pop("session_id")
            payload.update(lease.generation_fields(), rid=physical_rid)
            dispatch = _Dispatch(
                digest,
                lease,
                engine,
                payload,
                time.monotonic(),
                completion=asyncio.get_running_loop().create_future(),
                reserved_tokens=tokens,
            )
            self.requests[rid] = self.pending[rid] = dispatch
            self.by_session[session_id].add(rid)
            self.engine_load[selected] += 1
            self.engine_tokens[selected] += tokens
            if self.journal is not None:
                self.journal.put("dispatch", rid, {"digest": digest, "lease": asdict(lease), "tokens": tokens})
            dispatch.task = owned_task(self._dispatch(dispatch))
        try:
            waiters = {dispatch.completion}
            if dispatch.task is not None:
                waiters.add(dispatch.task)
            await asyncio.wait(waiters, timeout=30, return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not dispatch.settled:
                await self._reconcile(dispatch)
        if not dispatch.settled:
            raise AdapterNotReadyError("backend terminal state is unknown; version remains pinned")
        if dispatch.result["status"] in {"rejected", "terminal_error"}:
            raise ValueError(dispatch.result["error"])
        if dispatch.result["status"] == "expired":
            raise AdapterLeaseError("Session is closed; request result expired")
        await self.manager.durable()
        return dispatch.result["output"]

    def _settle(self, dispatch: _Dispatch, result: dict[str, Any]) -> None:
        if dispatch.settled:
            return
        if len(json.dumps(result).encode()) > self.max_payload_bytes:
            result = (
                {"status": "terminal_error", "code": "RESULT_TOO_LARGE", "error": "output budget exceeded"}
                if result.get("status") in {"terminal", "terminal_error", "rejected"}
                else result
            )
        dispatch.result = result
        if result.get("status") not in {"terminal", "terminal_error", "rejected"}:
            return
        lease = dispatch.lease
        self.manager.finish_request(RequestTerminal(lease.request_id, lease.engine, lease.operation_id), persist=False)
        dispatch.settled = True
        dispatch.payload = {}
        self.engine_load[lease.engine.engine_id] -= 1
        self.engine_tokens[lease.engine.engine_id] -= dispatch.reserved_tokens
        logical = next((key for key in self.by_session[lease.session_id] if self.pending[key] is dispatch), None)
        if logical is not None:
            self.pending.pop(logical)
            self.by_session[lease.session_id].discard(logical)
            self.to_forget[logical] = dispatch
        if result["status"] in {"rejected", "terminal_error"}:
            self.failed += 1
        else:
            self._record_output(dispatch, result)
        if self.journal is not None and logical is not None:
            self.journal.put_result(
                logical,
                dispatch.digest,
                result,
                terminal_records=[
                    ("lease", lease.request_id, asdict(lease), False),
                    ("dispatch", logical, {"digest": dispatch.digest, "lease": asdict(lease)}, False),
                ],
            )
        if dispatch.completion is not None and not dispatch.completion.done():
            dispatch.completion.set_result(result)

    def _record_output(self, dispatch: _Dispatch, result: dict[str, Any]) -> None:
        lease = dispatch.lease
        output = result["output"]
        metadata = output.setdefault("meta_info", {})
        metadata["lora_adapter"] = asdict(lease.binding)
        metadata["lora_engine"] = asdict(lease.engine)
        self.completed += 1
        self.aborted_count += int(metadata.get("finish_reason", {}).get("type") == "abort")
        self.failed += int((metadata.get("finish_reason", {}).get("status_code") or 0) >= 400)
        self.latency_sum += time.monotonic() - dispatch.started
        self.generated_tokens += int(metadata.get("completion_tokens", len(output.get("output_ids", []))))

    async def _dispatch(self, dispatch: _Dispatch) -> None:
        try:
            await self.manager.durable()
            result = await dispatch.engine.call(
                "generate", {"operation_id": dispatch.lease.operation_id, "payload": dispatch.payload}
            )
            self._settle(dispatch, result)
        except Exception as error:
            if not dispatch.settled:
                self.failed += 1
                dispatch.result = {"status": "unknown", "error": f"{type(error).__name__}: {error}"}

    async def _reconcile(self, dispatch: _Dispatch) -> None:
        if dispatch.settled:
            return
        try:
            body = {"rid": dispatch.lease.request_id, "operation_id": dispatch.lease.operation_id}
            result = await dispatch.engine.call("status", body)
            if result["status"] == "absent":
                # Fence a possibly delayed generate before treating absence as
                # terminal. This is not a retry of physical model generation.
                result = await dispatch.engine.call("abort", body)
            self._settle(dispatch, result)
        except Exception:
            pass  # Unknown is intentionally retained; no TTL-based reclamation.

    async def abort(self, rid: str) -> None:
        if not rid:
            raise ValueError("rid must not be empty")
        self.aborted.add(rid)
        if self.journal is not None:
            self.journal.put("abort", rid, {"cancelled": True}, active=False)
            await self.manager.durable()
        dispatch = self.requests.get(rid)
        if dispatch is not None and not dispatch.settled:
            result = await dispatch.engine.call(
                "abort", {"rid": dispatch.lease.request_id, "operation_id": dispatch.lease.operation_id}
            )
            self._settle(dispatch, result)

    async def close(self, session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("a nonempty Session identity is required")
        self.closed.add(session_id)
        self.manager.release_session(session_id)
        await self.manager.durable()  # ACK transfers durable ownership of close.
        for rid in tuple(self.by_session.get(session_id, ())):
            owned_task(self.abort(rid))
        self.affinity.pop(session_id, None)
        if self.journal is not None:
            previous = await asyncio.to_thread(self.journal.get, "session_owner", session_id)
            if previous and previous.get("owner"):
                self.owner_sessions[previous["owner"]].discard(session_id)
            self.journal.put("session_owner", session_id, {"closed": True}, active=False)
            if not self.by_session.get(session_id):
                self.closed.discard(session_id)
                self.manager._closed_sessions.discard(session_id)
            await self.manager.durable()

    async def tick(self) -> None:
        # Work is proportional to active/unacknowledged requests, not history.
        await asyncio.gather(*(self._reconcile(item) for item in tuple(self.pending.values())))
        for sid in tuple(self.closed):
            for rid in tuple(self.by_session.get(sid, ())):
                try:
                    await self.abort(rid)
                except Exception:
                    pass
        if self._collection is None or self._collection.done():
            self._collection = owned_task(self.manager.collect())
        await asyncio.sleep(0)
        for rid, item in tuple(self.to_forget.items()):
            if self.journal is None and item.lease.session_id not in self.closed:
                continue
            try:
                await self.manager.durable()  # Result ownership transferred before engine compaction.
                await item.engine.call(
                    "forget", {"rid": item.lease.request_id, "operation_id": item.lease.operation_id}
                )
            except Exception:
                continue
            item.forgotten = True
            self.to_forget.pop(rid, None)
            if self.journal is not None:
                self.requests.pop(rid, None)
                self.manager._requests.pop(item.lease.request_id, None)
                self.manager._finished_requests.discard(item.lease.request_id)
                if not self.by_session.get(item.lease.session_id):
                    self.by_session.pop(item.lease.session_id, None)
                    self.closed.discard(item.lease.session_id)
                    self.manager._closed_sessions.discard(item.lease.session_id)
                self.aborted.discard(rid)
                self.aborted_digests.pop(rid, None)
            else:
                item.result = {"status": "expired"}

    def state(self) -> dict[str, Any]:
        return {
            **self.manager.snapshot(),
            "completed_requests": self.completed,
            "failed_requests": self.failed,
            "aborted_requests": self.aborted_count,
            "generated_tokens": self.generated_tokens,
            "latency_seconds_sum": self.latency_sum,
            "unsettled_requests": len(self.pending),
            "generation_capacity": len(self.engines) * self.requests_per_engine,
            "requests_per_engine": self.requests_per_engine,
            "engine_load": dict(self.engine_load),
            "engine_reserved_tokens": dict(self.engine_tokens),
            "token_budget_per_engine": self.token_budget_per_engine,
            "max_request_tokens": self.max_request_tokens,
            "max_payload_bytes": self.max_payload_bytes,
        }


def install_gateway_routes(app: Any, get_gateway: Any) -> None:
    from fastapi import HTTPException

    @app.get("/state")
    async def state() -> dict[str, Any]:
        return get_gateway().state()

    @app.post("/{action}")
    async def operate(action: str, body: dict[str, Any]) -> dict[str, Any]:
        gateway = get_gateway()
        try:
            if action == "publish":
                return await gateway.publish(
                    body["version_id"], body["export_dir"], body.get("expected_default_epoch")
                )
            if action == "archive_artifact":
                return await gateway.archive(body["version_id"])
            if action == "publication_status":
                operation_id = body["operation_id"]
                record = gateway.manager._attempts.get(operation_id)
                value = (
                    gateway.manager._version_body(record)
                    if record
                    else (
                        await asyncio.to_thread(gateway.journal.get, "attempt", operation_id)
                        if gateway.journal
                        else None
                    )
                )
                if value is None:
                    raise KeyError("unknown publication operation")
                return value
            if action == "bind_session":
                return await gateway.bind_owned(
                    body["session_id"], body.get("owner"), body.get("spool_id"), body.get("spool_url")
                )
            if action == "generate":
                return await gateway.generate(body)
            if action == "abort_request":
                await gateway.abort(body["rid"])
            elif action == "close_session":
                await gateway.close(body["session_id"])
            elif action == "collect":
                await gateway.tick()
            elif action == "cancel_publication":
                cancelled = gateway.manager.cancel_publication(body["version_id"], body["operation_id"])
                await gateway.manager.durable()
                return {"cancelled": cancelled, "operation_id": body["operation_id"]}
            else:
                raise HTTPException(404, "unknown publication operation")
            return {"success": True}
        except AdapterCapacityError as error:
            raise HTTPException(
                507, {"code": "CAPACITY_EXHAUSTED", "message": str(error), "retryable": True}
            ) from error
        except (AdapterConflictError, AdapterLeaseError) as error:
            raise HTTPException(
                409, {"code": "IDENTITY_CONFLICT", "message": str(error), "retryable": False}
            ) from error
        except (AdapterNotReadyError, AdapterPublicationError) as error:
            raise HTTPException(
                503,
                {"code": "UNAVAILABLE", "message": str(error), "retryable": True, "definitely_not_submitted": False},
            ) from error
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(400, {"code": "INVALID_REQUEST", "message": str(error), "retryable": False}) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-url", action="append", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--requests-per-engine", type=int, default=32)
    parser.add_argument("--max-lora-rank", type=int, required=True)
    parser.add_argument("--target-module", action="append", required=True)
    parser.add_argument("--artifact-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--token-budget-per-engine", type=int, default=65536)
    parser.add_argument("--max-request-tokens", type=int, default=8192)
    parser.add_argument("--public-url", help="Canonical URL used by Agentic clients and close outbox")
    parser.add_argument("--close-outbox-dir", type=Path)
    parser.add_argument("--ray-state-address", help="Explicit Ray dashboard address for actor-death evidence")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if len(args.engine_url) != 2 or len(set(args.engine_url)) != 2:
        parser.error("provide two distinct --engine-url targets")
    import uvicorn
    from fastapi import FastAPI

    gateway = None

    @asynccontextmanager
    async def lifespan(app: Any):
        nonlocal gateway
        journal = ControlJournal(args.state_dir)
        owner = journal.owner_id
        base_digest = await asyncio.to_thread(fingerprint_model, args.model_path)
        from relax.engine.lora.access import headers
        from relax.engine.lora.artifact import read_object

        contract = ModelContract(
            base_digest,
            read_object(Path(args.model_path) / "config.json"),
            args.max_lora_rank,
            tuple(args.target_module),
        )
        async with AsyncExitStack() as stack:
            engines = []
            for url in args.engine_url:
                client = await stack.enter_async_context(
                    httpx.AsyncClient(
                        timeout=10.0,
                        limits=httpx.Limits(max_connections=8),
                        trust_env=False,
                        headers=headers(control=True),
                    )
                )
                data = await stack.enter_async_context(
                    httpx.AsyncClient(
                        timeout=30.0,
                        limits=httpx.Limits(max_connections=args.requests_per_engine),
                        trust_env=False,
                        headers=headers(control=True),
                    )
                )
                engines.append(HttpAdapterEngine(url, owner, client, data))
            await asyncio.gather(*(engine.connect(base_digest, args.capacity) for engine in engines))
            gateway = PublicationGateway(
                engines,
                capacity=args.capacity,
                base_digest=base_digest,
                store=args.store,
                journal=journal,
                contract=contract,
                requests_per_engine=args.requests_per_engine,
                artifact_bytes=args.artifact_bytes,
                token_budget_per_engine=args.token_budget_per_engine,
                max_request_tokens=args.max_request_tokens,
            )
            gateway.ray_address = args.ray_state_address
            if args.close_outbox_dir:
                from relax.engine.lora.outbox import SessionCloseOutbox

                if not args.public_url:
                    raise ValueError("--close-outbox-dir requires --public-url")
                gateway.close_outbox = SessionCloseOutbox(args.close_outbox_dir, args.public_url)
            await gateway.recover()

            async def reconcile_loop() -> None:
                while True:
                    await gateway.tick()
                    await asyncio.sleep(1.0)

            async def owner_loop() -> None:
                while True:
                    try:
                        await gateway.reconcile_owners()
                        await gateway.refresh_health()
                    except Exception:
                        # Durable intents remain pending; polling failure never
                        # converts a liveness observation into death evidence.
                        pass
                    await asyncio.sleep(1.0)

            poller = owned_task(reconcile_loop())
            owners = owned_task(owner_loop())
            try:
                yield
            finally:
                poller.cancel()
                owners.cancel()
                await asyncio.gather(poller, owners, return_exceptions=True)
                # Keep owner identity and unsettled references for same-cohort recovery.
                await journal.barrier()
                journal.close()

    app = FastAPI(lifespan=lifespan)
    install_gateway_routes(app, lambda: gateway)
    from relax.engine.lora.access import install_access

    install_access(app)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
