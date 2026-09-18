# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU fault injection; fake responses are not GPU/logprob validation."""

import asyncio
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from relax.engine.lora.publication import (
    AdapterCapacityError,
    AdapterLeaseError,
    AdapterNotReadyError,
    AdapterPublicationError,
    AdapterVersionManager,
    EngineIdentity,
    EngineReceipt,
    RequestTerminal,
)
from relax.engine.lora.snapshot import AdapterConflictError, AdapterSnapshot, snapshot_adapter


BASE_DIGEST = "a" * 64


class FakeEngine:
    """Model remote work surviving a lost ACK, with an operation fencing
    ledger."""

    def __init__(self, name: str):
        self.identity = EngineIdentity(name, "boot-1")
        self.started = defaultdict(asyncio.Event)
        self.loaded = defaultdict(asyncio.Event)
        self.gates = {}
        self.fail_load = set()
        self.lose_load_ack = set()
        self.fail_retire = set()
        self.lose_retire_ack_once = set()
        self.bad_receipts = {}
        self.retire_gates = {}
        self.retire_started = defaultdict(asyncio.Event)
        self.prepare_calls = Counter()
        self.retire_calls = Counter()
        self.unloads = Counter()
        self.resident = {}
        self.fenced = set()
        self.remote_loads = {}

    def receipt(self, artifact, operation_id, **kwargs):
        return EngineReceipt(self.identity, operation_id, artifact.version_id, artifact.digest, **kwargs)

    async def _load(self, artifact, operation_id):
        version = artifact.version_id
        if version in self.gates:
            await self.gates[version].wait()
        if operation_id not in self.fenced:
            artifact.verify()
            self.resident[version] = (artifact, operation_id)
        self.loaded[version].set()

    async def prepare(self, artifact, operation_id):
        version = artifact.version_id
        self.prepare_calls[version] += 1
        self.started[version].set()
        if version in self.lose_load_ack:
            self.remote_loads[version] = asyncio.create_task(self._load(artifact, operation_id))
            raise TimeoutError("load ACK lost; remote execution may continue")
        await self._load(artifact, operation_id)
        if version in self.fail_load:
            raise RuntimeError("injected failure after allocation")
        receipt = self.receipt(artifact, operation_id, pinned=True)
        return replace(receipt, **self.bad_receipts.get(version, {}))

    async def retire(self, artifact, operation_id):
        version = artifact.version_id
        self.retire_calls[version] += 1
        self.retire_started[version].set()
        if version in self.retire_gates:
            await self.retire_gates[version].wait()
        if version in self.fail_retire:
            raise TimeoutError("retirement unconfirmed")
        self.fenced.add(operation_id)
        current = self.resident.get(version)
        if current is not None and current[1] == operation_id:
            del self.resident[version]
            self.unloads[version] += 1
        if version in self.lose_retire_ack_once:
            self.lose_retire_ack_once.remove(version)
            raise TimeoutError("retirement completed but ACK lost")
        return self.receipt(artifact, operation_id, absent=True, fenced=True)

    def generate(self, lease):
        payload = lease.generation_fields()
        assert payload == {"lora_path": f"relax_policy@{lease.binding.version_id}"}
        loaded = self.resident.get(lease.binding.version_id)
        if loaded is None or loaded[1] != lease.operation_id or self.identity != lease.engine:
            raise AdapterNotReadyError("requested version is absent; no fallback")
        assert loaded[0].digest == lease.binding.digest
        return lease.binding.version_id


def _artifact(tmp_path: Path, version: str, *, content: str | None = None) -> AdapterSnapshot:
    export = tmp_path / f"export-{version}"
    export.mkdir(exist_ok=True)
    (export / "adapter_config.json").write_text('{"r":4,"lora_alpha":8}')
    (export / "adapter_model.safetensors").write_text(content or version)
    return snapshot_adapter(export, tmp_path / "store", version_id=version, base_model_digest=BASE_DIGEST)


def _manager(*engines, capacity=2):
    return AdapterVersionManager(engines, capacity=capacity, base_model_digest=BASE_DIGEST)


def _terminal(lease):
    return RequestTerminal(lease.request_id, lease.engine, lease.operation_id)


def _run(scenario):
    # No pytest-asyncio dependency needed for the CPU-only control-plane suite.
    asyncio.run(asyncio.wait_for(scenario(), timeout=10))


def test_publication_waits_for_both_engines_and_preserves_session_binding(tmp_path):
    async def scenario():
        first, second = FakeEngine("first"), FakeEngine("second")
        manager = _manager(first, second)
        a = await manager.publish(_artifact(tmp_path, "A"))
        assert manager.bind_once("old") == a
        first_attempt = manager.begin_request("old", "old:0", "first")
        assert first.generate(first_attempt) == "A"
        manager.finish_request(_terminal(first_attempt))
        second.gates["B"] = asyncio.Event()
        publication = asyncio.create_task(manager.publish(_artifact(tmp_path, "B")))
        await first.loaded["B"].wait()
        assert manager.snapshot()["default_version"] == "A"
        assert manager.bind_once("during").version_id == "A"
        # A tool round has no active backend request but still holds the version.
        assert manager.snapshot()["versions"]["A"]["session_refs"] == 2
        second.gates["B"].set()
        b = await publication
        assert manager.bind_once("new") == b
        assert manager.bind_once("old") == a
        # Abort/resume gets another physical attempt; binding remains A on either engine.
        resumed = manager.begin_request("old", "old:1", "second")
        fresh = manager.begin_request("new", "new:0", "first")
        assert second.generate(resumed) == "A"
        assert first.generate(fresh) == "B"
        await manager.collect()
        assert first.unloads["A"] == second.unloads["A"] == 0
        assert manager.snapshot()["publication_epoch"] == 2

    _run(scenario)


def test_publication_failure_waits_for_late_load_before_cleanup(tmp_path):
    async def scenario():
        first, second = FakeEngine("first"), FakeEngine("second")
        manager = _manager(first, second)
        await manager.publish(_artifact(tmp_path, "A"))
        first.fail_load.add("B")
        second.gates["B"] = asyncio.Event()
        publication = asyncio.create_task(manager.publish(_artifact(tmp_path, "B")))
        await second.started["B"].wait()
        assert manager.default.version_id == "A"
        assert manager.snapshot()["occupied"] == 2
        assert not publication.done()
        assert first.retire_calls["B"] == 0
        second.gates["B"].set()
        with pytest.raises(AdapterPublicationError):
            await publication
        assert manager.snapshot()["versions"]["B"]["state"] == "aborted"
        assert manager.snapshot()["occupied"] == 1
        assert first.unloads["B"] == second.unloads["B"] == 1
        assert manager.default.version_id == "A"

    _run(scenario)


def test_publication_timeout_fences_remote_late_load(tmp_path):
    async def scenario():
        first, second = FakeEngine("first"), FakeEngine("second")
        manager = _manager(first, second)
        await manager.publish(_artifact(tmp_path, "A"))
        second.gates["B"] = asyncio.Event()
        second.lose_load_ack.add("B")
        with pytest.raises(AdapterPublicationError):
            await manager.publish(_artifact(tmp_path, "B"))
        assert manager.snapshot()["versions"]["B"]["state"] == "aborted"
        second.gates["B"].set()
        await second.remote_loads["B"]
        assert "B" not in second.resident
        assert "B" not in first.resident
        assert manager.default.version_id == "A"

    _run(scenario)


def test_publication_unconfirmed_cleanup_keeps_capacity_and_retries_only_pending_engine(tmp_path):
    async def scenario():
        first, second = FakeEngine("first"), FakeEngine("second")
        manager = _manager(first, second)
        await manager.publish(_artifact(tmp_path, "A"))
        second.lose_load_ack.add("B")
        second.fail_retire.add("B")
        with pytest.raises(AdapterPublicationError):
            await manager.publish(_artifact(tmp_path, "B"))
        await second.remote_loads["B"]
        assert manager.snapshot()["versions"]["B"]["state"] == "cleanup_failed"
        assert manager.snapshot()["versions"]["B"]["cleanup_pending"] == ["second"]
        with pytest.raises(AdapterCapacityError):
            await manager.publish(_artifact(tmp_path, "C"))
        assert first.retire_calls["B"] == 1
        second.fail_retire.clear()
        await manager.collect()
        assert first.retire_calls["B"] == 1
        assert second.unloads["B"] == 1
        assert manager.snapshot()["occupied"] == 1

    _run(scenario)


def test_publication_duplicate_join_conflict_and_historical_retry(tmp_path):
    async def scenario():
        first, second = FakeEngine("first"), FakeEngine("second")
        manager = _manager(first, second)
        artifact = _artifact(tmp_path, "A")
        second.gates["A"] = asyncio.Event()
        one = asyncio.create_task(manager.publish(artifact))
        await second.started["A"].wait()
        two = asyncio.create_task(manager.publish(artifact))
        with pytest.raises(AdapterConflictError):
            await manager.publish(replace(artifact, digest="b" * 64))
        second.gates["A"].set()
        a, duplicate = await asyncio.gather(one, two)
        assert duplicate == a
        assert first.prepare_calls["A"] == second.prepare_calls["A"] == 1
        await manager.publish(_artifact(tmp_path, "B"))
        await manager.collect()
        assert manager.snapshot()["versions"]["A"]["state"] == "retired"
        assert await manager.publish(artifact) == a
        assert manager.default.version_id == "B"
        with pytest.raises(AdapterConflictError):
            await manager.publish(replace(artifact, digest="c" * 64))

    _run(scenario)


def test_publication_lost_retirement_ack_does_not_unload_twice(tmp_path):
    async def scenario():
        first, second = FakeEngine("first"), FakeEngine("second")
        manager = _manager(first, second)
        await manager.publish(_artifact(tmp_path, "A"))
        await manager.publish(_artifact(tmp_path, "B"))
        second.lose_retire_ack_once.add("A")
        await manager.collect()
        assert manager.snapshot()["versions"]["A"]["state"] == "cleanup_failed"
        assert manager.snapshot()["occupied"] == 2
        assert first.unloads["A"] == second.unloads["A"] == 1
        await manager.collect()
        assert manager.snapshot()["occupied"] == 1
        assert first.retire_calls["A"] == 1
        assert second.retire_calls["A"] == 2
        assert first.unloads["A"] == second.unloads["A"] == 1

    _run(scenario)


def test_publication_failed_retry_has_new_operation_and_rejects_old_ack(tmp_path):
    async def scenario():
        first, second = FakeEngine("first"), FakeEngine("second")
        manager = _manager(first, second)
        artifact = _artifact(tmp_path, "A")
        second.fail_load.add("A")
        with pytest.raises(AdapterPublicationError):
            await manager.publish(artifact)
        original_operation = manager.snapshot()["versions"]["A"]["operation_id"]
        second.fail_load.clear()
        second.bad_receipts["A"] = {"operation_id": original_operation}
        with pytest.raises(AdapterPublicationError):
            await manager.publish(artifact)
        assert manager.default is None
        assert manager.snapshot()["versions"]["A"]["operation_id"] != original_operation
        second.bad_receipts.clear()
        assert (await manager.publish(artifact)).version_id == "A"

    _run(scenario)


def test_publication_caller_cancellation_does_not_cancel_owned_publication(tmp_path):
    async def scenario():
        first, second = FakeEngine("first"), FakeEngine("second")
        manager = _manager(first, second)
        artifact = _artifact(tmp_path, "A")
        second.gates["A"] = asyncio.Event()
        caller = asyncio.create_task(manager.publish(artifact))
        await second.started["A"].wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        second.gates["A"].set()
        assert (await manager.publish(artifact)).version_id == "A"
        assert second.prepare_calls["A"] == 1

    _run(scenario)


def test_publication_explicit_cancellation_cleans_late_ack_and_cannot_undo_commit(tmp_path):
    async def scenario():
        first, second = FakeEngine("first"), FakeEngine("second")
        manager = _manager(first, second)
        await manager.publish(_artifact(tmp_path, "A"))
        second.gates["B"] = asyncio.Event()
        publication = asyncio.create_task(manager.publish(_artifact(tmp_path, "B")))
        await second.started["B"].wait()
        assert manager.cancel_publication("B")
        second.gates["B"].set()
        with pytest.raises(AdapterPublicationError):
            await publication
        assert manager.default.version_id == "A"
        assert not manager.cancel_publication("A")
        assert "B" not in first.resident and "B" not in second.resident

    _run(scenario)


@pytest.mark.parametrize("release_order", ["session_first", "terminal_first"])
def test_publication_capacity_waits_for_session_and_backend_terminal(tmp_path, release_order):
    async def scenario():
        first, second = FakeEngine("first"), FakeEngine("second")
        manager = _manager(first, second)
        await manager.publish(_artifact(tmp_path, "A"))
        manager.bind_once("old")
        lease = manager.begin_request("old", "attempt:0", "first")
        await manager.publish(_artifact(tmp_path, "B"))
        c = _artifact(tmp_path, "C")
        with pytest.raises(AdapterCapacityError):
            await manager.publish(c)
        if release_order == "session_first":
            assert manager.release_session("old")
            assert not manager.release_session("old")
        else:
            assert manager.finish_request(_terminal(lease))
            assert not manager.finish_request(_terminal(lease))
        with pytest.raises(AdapterCapacityError):
            await manager.publish(c)
        assert first.unloads["A"] == second.unloads["A"] == 0
        if release_order == "session_first":
            assert manager.finish_request(_terminal(lease))
        else:
            assert manager.release_session("old")
        # Racing collectors share one retirement task per version.
        await asyncio.gather(manager.collect(), manager.collect())
        assert first.unloads["A"] == second.unloads["A"] == 1
        assert first.retire_calls["A"] == second.retire_calls["A"] == 1
        await manager.publish(c)
        # Late duplicate replies do not subtract a C request or resurrect A.
        assert not manager.finish_request(_terminal(lease))
        assert manager.default.version_id == "C"

    _run(scenario)


def test_publication_terminal_session_and_request_ids_cannot_be_resurrected(tmp_path):
    async def scenario():
        engine = FakeEngine("first")
        manager = _manager(engine)
        with pytest.raises(AdapterNotReadyError):
            manager.bind_once("new")
        manager.release_session("closed-before-bind")
        await manager.publish(_artifact(tmp_path, "A"))
        with pytest.raises(AdapterLeaseError):
            manager.bind_once("closed-before-bind")
        manager.bind_once("old")
        lease = manager.begin_request("old", "attempt", "first")
        assert manager.begin_request("old", "attempt", "first") == lease
        manager.bind_once("other")
        with pytest.raises(AdapterLeaseError):
            manager.begin_request("other", "attempt", "first")
        with pytest.raises(AdapterLeaseError):
            manager.finish_request(replace(_terminal(lease), operation_id="stale"))
        assert manager.snapshot()["versions"]["A"]["request_refs"] == 1
        manager.finish_request(_terminal(lease))
        with pytest.raises(AdapterLeaseError):
            manager.begin_request("old", "attempt", "first")
        manager.release_session("old")
        with pytest.raises(AdapterLeaseError):
            manager.bind_once("old")
        with pytest.raises(AdapterLeaseError):
            manager.begin_request("old", "next-attempt", "first")

    _run(scenario)


@pytest.mark.parametrize("receipt_patch", [{"pinned": False}, {"digest": "c" * 64}, {"version_id": "wrong"}])
def test_publication_rejects_unpinned_or_conflicting_receipt(tmp_path, receipt_patch):
    async def scenario():
        first, second = FakeEngine("first"), FakeEngine("second")
        manager = _manager(first, second)
        second.bad_receipts["A"] = receipt_patch
        with pytest.raises(AdapterPublicationError):
            await manager.publish(_artifact(tmp_path, "A"))
        assert manager.default is None
        assert manager.snapshot()["occupied"] == 0

    _run(scenario)


def test_publication_engine_restart_invalidates_ready_before_commit(tmp_path):
    async def scenario():
        first, second = FakeEngine("first"), FakeEngine("second")
        manager = _manager(first, second)
        second.gates["A"] = asyncio.Event()
        publication = asyncio.create_task(manager.publish(_artifact(tmp_path, "A")))
        await first.loaded["A"].wait()
        first.identity = EngineIdentity("first", "boot-2")
        second.gates["A"].set()
        with pytest.raises(AdapterPublicationError):
            await publication
        assert manager.default is None
        assert manager.snapshot()["versions"]["A"]["state"] == "cleanup_failed"
        assert manager.snapshot()["occupied"] == 1

    _run(scenario)


def test_publication_missing_engine_version_fails_without_fallback(tmp_path):
    async def scenario():
        first, second = FakeEngine("first"), FakeEngine("second")
        manager = _manager(first, second)
        await manager.publish(_artifact(tmp_path, "A"))
        manager.bind_once("old")
        manager.mark_engine_unavailable("first")
        with pytest.raises(AdapterNotReadyError):
            manager.begin_request("old", "attempt", "first")
        with pytest.raises(AdapterNotReadyError):
            manager.begin_request("old", "attempt", "not-a-target")
        lease = manager.begin_request("old", "attempt", "second")
        second.resident.clear()
        with pytest.raises(AdapterNotReadyError):
            second.generate(lease)
        # A failed local call with unknown backend state does not release the lease.
        assert manager.snapshot()["versions"]["A"]["request_refs"] == 1

    _run(scenario)


def test_publication_cancelled_collector_keeps_retirement_and_capacity(tmp_path):
    async def scenario():
        first, second = FakeEngine("first"), FakeEngine("second")
        manager = _manager(first, second)
        await manager.publish(_artifact(tmp_path, "A"))
        await manager.publish(_artifact(tmp_path, "B"))
        second.retire_gates["A"] = asyncio.Event()
        collector = asyncio.create_task(manager.collect())
        await second.retire_started["A"].wait()
        collector.cancel()
        with pytest.raises(asyncio.CancelledError):
            await collector
        assert manager.snapshot()["occupied"] == 2
        assert manager.snapshot()["versions"]["A"]["state"] == "retiring"
        second.retire_gates["A"].set()
        await manager.collect()
        assert manager.snapshot()["occupied"] == 1
        assert first.retire_calls["A"] == second.retire_calls["A"] == 1

    _run(scenario)


def test_publication_rejects_different_base_model_without_loading(tmp_path):
    async def scenario():
        engine = FakeEngine("first")
        manager = _manager(engine)
        artifact = _artifact(tmp_path, "A")
        with pytest.raises(AdapterConflictError, match="base model"):
            await manager.publish(replace(artifact, base_model_digest="b" * 64))
        assert not engine.prepare_calls

    _run(scenario)
