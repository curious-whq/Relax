# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace

import pytest

from relax.agentic.session import service as session_service
from relax.agentic.session.contracts import AgenticIdentity, SessionControlRef
from relax.agentic.session.service import AgenticSessionShard
from relax.agentic.session.sglang_capabilities import resolve_sglang_capability_profile
from relax.agentic.session.sglang_lifecycle import (
    EngineSessionLifecycleState,
    SessionControlResult,
    SessionControlStatus,
    SGLangSessionControlPort,
    SGLangWorkerTarget,
)


def _ready_profile():
    return resolve_sglang_capability_profile(
        router_managed=True,
        use_slime_router=False,
        has_pd_disaggregation=False,
        radix_cache_disabled=False,
        hierarchical_cache_enabled=False,
        lifecycle_ready=True,
    )


def _workers(*urls: str):
    return {
        "workers": [
            {
                "id": f"worker-{index}",
                "url": url,
                "is_healthy": True,
            }
            for index, url in enumerate(urls)
        ]
    }


@pytest.mark.asyncio
async def test_session_control_open_fans_out_and_partial_failure_fails_open() -> None:
    calls: list[tuple[str, dict]] = []

    async def fake_get(url):
        assert url.endswith("/workers")
        return _workers("http://worker-1", "http://worker-2")

    async def fake_post(url, payload, max_retries):
        calls.append((url, payload))
        if url == "http://worker-2/open_session":
            raise RuntimeError("worker unavailable")
        return {}

    port = SGLangSessionControlPort(
        router_ip="router",
        router_port=30000,
        capability_profile=_ready_profile(),
        get_fn=fake_get,
        post_fn=fake_post,
    )

    result = await port.open_session(engine_session_id="engine-1", capacity_of_str_len=8192)

    assert result.status == SessionControlStatus.BYPASS
    assert [target.url for target in result.failed_targets] == ["http://worker-2"]
    assert ("http://worker-1/close_session", {"session_id": "engine-1"}) in calls


@pytest.mark.asyncio
async def test_session_control_close_retries_only_failed_targets() -> None:
    attempts: dict[str, int] = {}

    async def fake_get(url):
        assert url.endswith("/workers")
        return _workers("http://worker-1", "http://worker-2")

    async def fake_post(url, payload, max_retries):
        assert payload == {"session_id": "engine-1"}
        attempts[url] = attempts.get(url, 0) + 1
        if url == "http://worker-2/close_session" and attempts[url] < 3:
            raise RuntimeError("retry")
        return {}

    port = SGLangSessionControlPort(
        router_ip="router",
        router_port=30000,
        capability_profile=_ready_profile(),
        get_fn=fake_get,
        post_fn=fake_post,
        close_attempts=3,
    )

    result = await port.close_session(engine_session_id="engine-1")

    assert result.status == SessionControlStatus.CLOSED
    assert result.attempts == 3
    assert attempts["http://worker-1/close_session"] == 1
    assert attempts["http://worker-2/close_session"] == 3


class _FakeSessionControlPort:
    enabled = True

    def __init__(self, *, block_close: bool = False):
        self.block_close = block_close
        self.close_started = asyncio.Event()
        self.allow_close = asyncio.Event()
        self.close_calls = 0

    async def open_session(self, *, engine_session_id, capacity_of_str_len):
        assert capacity_of_str_len == 4096
        return SessionControlResult(
            status=SessionControlStatus.OPENED,
            targets=(SGLangWorkerTarget(url="http://worker-1"),),
            attempts=1,
        )

    async def close_session(self, *, engine_session_id, known_targets):
        self.close_calls += 1
        self.close_started.set()
        if self.block_close:
            await self.allow_close.wait()
        return SessionControlResult(
            status=SessionControlStatus.CLOSED,
            targets=tuple(known_targets),
            attempts=1,
        )


def _identity(
    *,
    engine_session_id: str,
    parent_engine_session_id: str | None = None,
) -> AgenticIdentity:
    return AgenticIdentity(
        program_id="program-1",
        program_owner_key="owner-1",
        root_session_id="root-1",
        engine_session_id=engine_session_id,
        parent_engine_session_id=parent_engine_session_id,
    )


def _registration_entry(identity: AgenticIdentity, *, nonce: str) -> dict:
    return {
        "session_id": identity.engine_session_id,
        "identity": identity.to_payload(),
        "credential_nonce": nonce,
        "event_id": f"engine_session_open:{identity.engine_session_id}",
        "scope_id": "train",
        "rollout_id": 1,
        "group_id": "group-1",
        "group_generation": 0,
        "gate_reason": None,
        "sampling_params": {"max_new_tokens": 8},
        "session_seed": {"group_index": 0},
    }


def _test_shard(port: _FakeSessionControlPort):
    shard_cls = AgenticSessionShard.__ray_metadata__.modified_class
    shard = object.__new__(shard_cls)
    shard._shard_index = 0
    shard._shard_count = 1
    shard._owner_epoch = 17
    shard._program_records = {}
    shard._program_locks = {}
    shard._registration_lock = asyncio.Lock()
    shard._terminal_program_owner_keys = deque()
    shard._credential_session_ids = {}
    shard._session_records = {}
    shard._session_locks = {}
    shard._close_tombstones = {}
    shard._close_tombstone_order = deque()
    shard._session_control_port = port
    shard.args = SimpleNamespace(rollout_max_context_len=4096)
    return shard


def _control_ref(identity: AgenticIdentity) -> SessionControlRef:
    return SessionControlRef(
        program_owner_key=identity.program_owner_key,
        engine_session_id=identity.engine_session_id,
        owner_epoch=17,
    )


@pytest.mark.asyncio
async def test_shard_enables_physical_session_only_after_open_succeeds() -> None:
    shard = _test_shard(_FakeSessionControlPort())
    identity = _identity(engine_session_id="engine-root")
    await shard.register_sessions_batch(entries=[_registration_entry(identity, nonce="0" * 64)])
    record = shard._session_records[identity.engine_session_id]

    assert record.lifecycle_state == EngineSessionLifecycleState.UNOPENED
    assert await shard._ensure_engine_session_open(session_id=identity.engine_session_id)
    assert record.lifecycle_state == EngineSessionLifecycleState.ACTIVE
    assert record.lifecycle_targets == (SGLangWorkerTarget(url="http://worker-1"),)


@pytest.mark.asyncio
async def test_duplicate_discard_waits_for_one_close_and_reuses_tombstone() -> None:
    port = _FakeSessionControlPort(block_close=True)
    shard = _test_shard(port)
    identity = _identity(engine_session_id="engine-root")
    await shard.register_sessions_batch(entries=[_registration_entry(identity, nonce="1" * 64)])

    first = asyncio.create_task(shard.discard_session(control_ref=_control_ref(identity)))
    await port.close_started.wait()
    program = shard._program_records[identity.program_owner_key]
    assert not program.finalized

    duplicate = asyncio.create_task(shard.discard_session(control_ref=_control_ref(identity)))
    await asyncio.sleep(0)
    assert not duplicate.done()
    port.allow_close.set()

    assert await first is True
    assert await duplicate is True
    assert port.close_calls == 1
    assert program.finalized
    tombstone = shard._close_tombstones[identity.engine_session_id]
    assert tombstone.state == EngineSessionLifecycleState.TERMINATED_OK


@pytest.mark.asyncio
async def test_program_finalizes_only_after_every_sibling_close_finishes() -> None:
    port = _FakeSessionControlPort()
    shard = _test_shard(port)
    root = _identity(engine_session_id="engine-root")
    child = _identity(
        engine_session_id="engine-child",
        parent_engine_session_id=root.engine_session_id,
    )
    await shard.register_sessions_batch(
        entries=[
            _registration_entry(root, nonce="2" * 64),
            _registration_entry(child, nonce="3" * 64),
        ]
    )
    program = shard._program_records[root.program_owner_key]

    assert await shard.discard_session(control_ref=_control_ref(root)) is True
    assert not program.finalized
    assert program.terminal_engine_session_ids == {root.engine_session_id}

    assert await shard.discard_session(control_ref=_control_ref(child)) is True
    assert program.finalized
    assert program.terminal_engine_session_ids == {
        root.engine_session_id,
        child.engine_session_id,
    }


@pytest.mark.asyncio
async def test_hung_runner_reaches_terminal_without_unsafe_close(monkeypatch) -> None:
    monkeypatch.setattr(session_service, "_AGENTIC_RUNNER_CLEANUP_TIMEOUT_S", 0.01)
    port = _FakeSessionControlPort()
    shard = _test_shard(port)
    identity = _identity(engine_session_id="engine-root")
    await shard.register_sessions_batch(entries=[_registration_entry(identity, nonce="4" * 64)])
    release_runner = asyncio.Event()

    async def stubborn_runner():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release_runner.wait()

    runner = asyncio.create_task(stubborn_runner())
    await asyncio.sleep(0)
    record = shard._session_records[identity.engine_session_id]
    record.active_ir_runner_tasks["request-1"] = runner

    assert await shard.discard_session(control_ref=_control_ref(identity)) is True
    tombstone = shard._close_tombstones[identity.engine_session_id]
    assert tombstone.state == EngineSessionLifecycleState.TERMINATED_EXHAUSTED
    assert port.close_calls == 0

    release_runner.set()
    await runner
