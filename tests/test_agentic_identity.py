# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import deque
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from relax.agentic.pipeline.runtime import (
    ManagedCommandAppSpec,
    RuntimeDomain,
    SessionInput,
    execute_managed_session_input,
)
from relax.agentic.session.contracts import (
    AgenticIdentity,
    SessionControlRef,
    SessionRegistrationGrant,
)
from relax.agentic.session.service import (
    AgenticChatAPIService,
    AgenticChatRequestError,
    AgenticSessionShard,
    _credential_for_nonce,
    _credential_route_from_request,
    _shard_index_for_owner_key,
)


def _identity(
    *,
    engine_session_id: str = "engine-root",
    parent_engine_session_id: str | None = None,
    program_id: str = "program-1",
    program_owner_key: str = "owner-1",
    root_session_id: str = "root-1",
) -> AgenticIdentity:
    return AgenticIdentity(
        program_id=program_id,
        program_owner_key=program_owner_key,
        root_session_id=root_session_id,
        engine_session_id=engine_session_id,
        parent_engine_session_id=parent_engine_session_id,
    )


def _test_shard():
    shard_cls = AgenticSessionShard.__ray_metadata__.modified_class
    shard = object.__new__(shard_cls)
    shard._shard_index = 0
    shard._shard_count = 1
    shard._owner_epoch = 71
    shard._program_records = {}
    shard._program_locks = {}
    shard._registration_lock = asyncio.Lock()
    shard._terminal_program_owner_keys = deque()
    shard._credential_session_ids = {}
    shard._session_records = {}
    shard._session_locks = {}
    return shard_cls, shard


def _registration_entry(
    identity: AgenticIdentity,
    *,
    nonce: str,
    event_id: str | None = None,
) -> dict:
    return {
        "session_id": identity.engine_session_id,
        "identity": identity.to_payload(),
        "credential_nonce": nonce,
        "event_id": event_id or f"engine_session_open:{identity.engine_session_id}",
        "scope_id": "train",
        "rollout_id": 3,
        "group_id": "group-1",
        "group_generation": 0,
        "gate_reason": None,
        "sampling_params": {"max_new_tokens": 8},
        "session_seed": {"group_index": 1},
    }


def _request_with_bearer(credential: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"authorization", f"Bearer {credential}".encode("ascii"))],
        }
    )


def test_session_control_ref_rejects_invalid_epoch() -> None:
    with pytest.raises(ValueError, match="owner_epoch"):
        SessionControlRef(
            program_owner_key="owner-1",
            engine_session_id="engine-1",
            owner_epoch=-1,
        )


def test_credential_route_is_stateless_and_does_not_expose_identity() -> None:
    nonce = "a" * 64
    credential = _credential_for_nonce(shard_index=3, owner_epoch=29, credential_nonce=nonce)
    request = _request_with_bearer(credential)

    first = _credential_route_from_request(request=request, shard_count=8)
    second = _credential_route_from_request(request=request, shard_count=8)

    assert first == second == (3, 29, hashlib.sha256(credential.encode("utf-8")).hexdigest())
    assert "engine" not in credential
    assert "program" not in credential


@pytest.mark.parametrize(
    "credential",
    [
        "engine-root",
        "relax-v2.0.1." + "a" * 64,
        "relax-v1.99.1." + "a" * 64,
        "relax-v1.0.1.short",
        "relax-v1.0.1." + "g" * 64,
        "relax-v1.0.0." + "a" * 64,
        "relax-v1.00.1." + "a" * 64,
        "relax-v1.0.01." + "a" * 64,
        "relax-v1." + "1" * 5000 + ".1." + "a" * 64,
        "relax-v1.0." + "1" * 5000 + "." + "a" * 64,
    ],
)
def test_credential_route_rejects_malformed_or_out_of_range_token(credential: str) -> None:
    with pytest.raises(AgenticChatRequestError) as exc_info:
        _credential_route_from_request(request=_request_with_bearer(credential), shard_count=4)
    assert exc_info.value.status_code == 401
    assert credential not in exc_info.value.message


@pytest.mark.asyncio
async def test_registration_is_idempotent_and_stores_only_credential_digest() -> None:
    _shard_cls, shard = _test_shard()
    identity = _identity()
    entry = _registration_entry(identity, nonce="1" * 64)

    first = (await shard.register_sessions_batch(entries=[entry]))[0]
    second = (await shard.register_sessions_batch(entries=[entry]))[0]

    assert first == second
    assert first["credential"] != identity.engine_session_id
    assert first["owner_epoch"] == shard._owner_epoch
    credential_digest = hashlib.sha256(first["credential"].encode("utf-8")).hexdigest()
    assert shard._credential_session_ids == {credential_digest: identity.engine_session_id}
    assert shard._session_records[identity.engine_session_id].credential_digest == credential_digest
    assert first["credential"] not in repr(shard._session_records[identity.engine_session_id])


@pytest.mark.asyncio
async def test_credential_collision_does_not_create_second_program() -> None:
    _shard_cls, shard = _test_shard()
    first = _identity()
    second = _identity(
        engine_session_id="engine-2",
        program_id="program-2",
        program_owner_key="owner-2",
        root_session_id="root-2",
    )

    with pytest.raises(RuntimeError, match="claimed by multiple engine sessions"):
        await shard.register_sessions_batch(
            entries=[
                _registration_entry(first, nonce="d" * 64),
                _registration_entry(second, nonce="d" * 64),
            ]
        )

    assert first.program_owner_key not in shard._program_records
    assert second.program_owner_key not in shard._program_records
    assert shard._session_records == {}


@pytest.mark.asyncio
async def test_concurrent_cross_program_registration_claims_credential_once() -> None:
    _shard_cls, shard = _test_shard()
    first = _identity()
    second = _identity(
        engine_session_id="engine-2",
        program_id="program-2",
        program_owner_key="owner-2",
        root_session_id="root-2",
    )

    results = await asyncio.gather(
        shard.register_sessions_batch(entries=[_registration_entry(first, nonce="f" * 64)]),
        shard.register_sessions_batch(entries=[_registration_entry(second, nonce="f" * 64)]),
        return_exceptions=True,
    )

    assert sum(isinstance(result, list) for result in results) == 1
    assert sum(isinstance(result, RuntimeError) for result in results) == 1
    assert len(shard._session_records) == 1
    assert len(shard._program_records) == 1


@pytest.mark.asyncio
async def test_sibling_sessions_share_program_owner_and_advance_event_sequence() -> None:
    _shard_cls, shard = _test_shard()
    root = _identity()
    sibling = _identity(
        engine_session_id="engine-child",
        parent_engine_session_id=root.engine_session_id,
    )

    sibling_entry = _registration_entry(sibling, nonce="3" * 64)
    root_grant, sibling_grant = await shard.register_sessions_batch(
        entries=[
            _registration_entry(root, nonce="2" * 64),
            sibling_entry,
        ]
    )
    duplicate_grant = (await shard.register_sessions_batch(entries=[sibling_entry]))[0]

    assert root_grant["program_owner_key"] == sibling_grant["program_owner_key"]
    assert root_grant["credential"] != sibling_grant["credential"]
    assert root_grant["event_seq"] < sibling_grant["event_seq"]
    assert sibling_grant["event_seq"] == duplicate_grant["event_seq"]
    program = shard._program_records[root.program_owner_key]
    assert program.engine_session_ids == {"engine-root", "engine-child"}


@pytest.mark.asyncio
async def test_registration_rejects_unknown_or_cross_program_parent() -> None:
    _shard_cls, shard = _test_shard()
    unknown_parent = _identity(
        engine_session_id="engine-child",
        parent_engine_session_id="missing-parent",
    )
    with pytest.raises(RuntimeError, match="before its parent"):
        await shard.register_sessions_batch(entries=[_registration_entry(unknown_parent, nonce="4" * 64)])

    root = _identity()
    await shard.register_sessions_batch(entries=[_registration_entry(root, nonce="5" * 64)])
    cross_program = _identity(
        engine_session_id="engine-cross",
        parent_engine_session_id=root.engine_session_id,
        program_id="program-2",
    )
    with pytest.raises(RuntimeError, match="existing program owner"):
        await shard.register_sessions_batch(entries=[_registration_entry(cross_program, nonce="6" * 64)])


@pytest.mark.asyncio
async def test_stale_owner_epoch_cannot_discard_current_session() -> None:
    _shard_cls, shard = _test_shard()
    identity = _identity()
    grant = (await shard.register_sessions_batch(entries=[_registration_entry(identity, nonce="7" * 64)]))[0]
    stale_ref = SessionControlRef(
        program_owner_key=identity.program_owner_key,
        engine_session_id=identity.engine_session_id,
        owner_epoch=grant["owner_epoch"] + 1,
    )

    assert await shard.discard_session(control_ref=stale_ref) is False
    assert identity.engine_session_id in shard._session_records

    current_ref = SessionControlRef(
        program_owner_key=identity.program_owner_key,
        engine_session_id=identity.engine_session_id,
        owner_epoch=grant["owner_epoch"],
    )
    assert await shard.discard_session(control_ref=current_ref) is True
    assert identity.engine_session_id not in shard._session_records
    assert shard._credential_session_ids == {}
    late_sibling = _identity(
        engine_session_id="engine-late",
        parent_engine_session_id=identity.engine_session_id,
    )
    with pytest.raises(RuntimeError, match="finalized agentic program"):
        await shard.register_sessions_batch(entries=[_registration_entry(late_sibling, nonce="9" * 64)])


@pytest.mark.asyncio
async def test_tampered_credential_does_not_authenticate() -> None:
    _shard_cls, shard = _test_shard()
    identity = _identity()
    grant = (await shard.register_sessions_batch(entries=[_registration_entry(identity, nonce="8" * 64)]))[0]
    _, owner_epoch, credential_digest = _credential_route_from_request(
        request=_request_with_bearer(grant["credential"]),
        shard_count=1,
    )
    tampered = grant["credential"][:-1] + "9"
    _, tampered_epoch, tampered_digest = _credential_route_from_request(
        request=_request_with_bearer(tampered),
        shard_count=1,
    )

    assert (
        shard._authenticated_session_id(
            credential_digest=credential_digest,
            owner_epoch=owner_epoch,
        )
        == identity.engine_session_id
    )
    assert (
        shard._authenticated_session_id(
            credential_digest=tampered_digest,
            owner_epoch=tampered_epoch,
        )
        is None
    )


@pytest.mark.asyncio
async def test_owner_epoch_rotation_invalidates_old_credential() -> None:
    _shard_cls, old_shard = _test_shard()
    identity = _identity()
    old_grant = (await old_shard.register_sessions_batch(entries=[_registration_entry(identity, nonce="e" * 64)]))[0]

    _shard_cls, new_shard = _test_shard()
    new_shard._owner_epoch = old_shard._owner_epoch + 1
    new_grant = (await new_shard.register_sessions_batch(entries=[_registration_entry(identity, nonce="e" * 64)]))[0]
    _, old_epoch, old_digest = _credential_route_from_request(
        request=_request_with_bearer(old_grant["credential"]),
        shard_count=1,
    )
    _, new_epoch, new_digest = _credential_route_from_request(
        request=_request_with_bearer(new_grant["credential"]),
        shard_count=1,
    )

    assert old_grant["credential"] != new_grant["credential"]
    assert (
        new_shard._authenticated_session_id(
            credential_digest=old_digest,
            owner_epoch=old_epoch,
        )
        is None
    )
    assert (
        new_shard._authenticated_session_id(
            credential_digest=new_digest,
            owner_epoch=new_epoch,
        )
        == identity.engine_session_id
    )


def test_siblings_route_to_same_owner_shard() -> None:
    owner_key = "shared-program-owner"
    owner_shard = _shard_index_for_owner_key(owner_key, 16)
    credential = _credential_for_nonce(
        shard_index=owner_shard,
        owner_epoch=41,
        credential_nonce="c" * 64,
    )
    credential_shard, owner_epoch, _digest = _credential_route_from_request(
        request=_request_with_bearer(credential),
        shard_count=16,
    )
    assert credential_shard == owner_shard
    assert owner_epoch == 41


class _FakeLauncherClient:
    def __init__(self) -> None:
        self.env = None

    async def launch(self, *, command, cwd, env):
        del command, cwd
        self.env = dict(env)
        now = time.time()
        return {
            "started_at": now,
            "spawn_returned_at": now,
            "handle": "handle-1",
            "pid": 123,
        }

    async def wait(self, *, handle):
        del handle
        return {
            "exited_at": time.time(),
            "exit_code": 0,
            "stdout_b64": "",
            "stderr_b64": "",
        }


@pytest.mark.asyncio
async def test_managed_session_env_separates_identity_from_credential() -> None:
    launcher = _FakeLauncherClient()
    credential = _credential_for_nonce(shard_index=0, owner_epoch=17, credential_nonce="a" * 64)
    session_input = SessionInput(
        session_id="engine-1",
        api_key=credential,
        rollout_mode="train",
        group_id="group-1",
        input_payload={"messages": []},
        metadata={"request_id": "request-1"},
    )

    await execute_managed_session_input(
        spec=ManagedCommandAppSpec(command="agent", cwd=None),
        session_input=session_input,
        launcher_client=launcher,
    )

    assert launcher.env["RELAX_SESSION_ID"] == "engine-1"
    assert launcher.env["RELAX_API_KEY"] == credential
    assert launcher.env["OPENAI_API_KEY"] == credential
    assert "api_key" not in session_input.to_agent_payload()
    assert credential not in repr(session_input)


@pytest.mark.asyncio
async def test_runtime_registration_propagates_grant_without_exporting_credential() -> None:
    runtime = object.__new__(RuntimeDomain)
    runtime._notified_session_rollouts = {}
    runtime._session_identities = {}
    runtime._session_registration_nonces = {}
    runtime._session_registration_grants = {}
    runtime._rollout_mode = "train"
    envelope = SimpleNamespace(
        session_id="engine-1",
        rollout_id=4,
        seed=SimpleNamespace(
            group_index=2,
            index=0,
            label=None,
            train_metadata=None,
        ),
        metadata={},
        input_payload={"messages": [{"role": "user", "content": "hello"}]},
        request_id="request-1",
    )
    request_plan = SimpleNamespace(
        envelope=envelope,
        request_sampling_params={"max_new_tokens": 8},
    )
    entries = runtime._prepare_request_registration_entries(
        [request_plan],
        scope_id="train",
    )
    identity = runtime._session_identities["engine-1"]
    credential = _credential_for_nonce(
        shard_index=0,
        owner_epoch=23,
        credential_nonce=entries[0]["credential_nonce"],
    )
    grant = SessionRegistrationGrant(
        control_ref=SessionControlRef(
            program_owner_key=identity.program_owner_key,
            engine_session_id=identity.engine_session_id,
            owner_epoch=23,
        ),
        credential=credential,
        event_seq=2,
    )

    class _FakeControlClient:
        async def register_sessions_batch(self, *, entries):
            assert entries[0]["identity"] == identity.to_payload()
            return [grant]

    runtime._ensure_service_client = lambda: _FakeControlClient()

    async def _refresh_session_debug_state(*, client):
        del client

    runtime._refresh_session_debug_state = _refresh_session_debug_state
    await runtime._register_managed_sessions_batch(entries=entries)
    session_input = runtime._build_managed_session_input(envelope=envelope)

    assert runtime._control_ref_for_session(session_id="engine-1") == grant.control_ref
    assert session_input.api_key == credential
    assert credential not in session_input.to_agent_payload()


class _RemoteCall:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = []

    async def remote(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _FakeSessionShardHandle:
    def __init__(self) -> None:
        self.chat_authenticated = _RemoteCall(
            {
                "request_id": "request-1",
                "message": {"role": "assistant", "content": "ok"},
                "logprobs": None,
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
        self.mark_chat_service_response_ready_authenticated = _RemoteCall(True)


@pytest.mark.asyncio
async def test_chat_api_forwards_only_credential_digest_and_does_not_echo_bearer() -> None:
    credential = _credential_for_nonce(shard_index=0, owner_epoch=31, credential_nonce="b" * 64)
    handle = _FakeSessionShardHandle()
    service_cls = AgenticChatAPIService.func_or_class.__wrapped__
    service = object.__new__(service_cls)
    service._shard_handles = [handle]
    service.args = SimpleNamespace(hf_checkpoint="/tmp/model")
    request_body = json.dumps(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
        }
    ).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": request_body, "more_body": False}

    response = await service._chat_completions_impl(
        Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": [
                    (b"authorization", f"Bearer {credential}".encode("ascii")),
                    (b"content-type", b"application/json"),
                ],
            },
            receive=receive,
        )
    )

    response_payload = json.loads(response.body)
    expected_digest = hashlib.sha256(credential.encode("utf-8")).hexdigest()
    assert handle.chat_authenticated.calls[0]["credential_digest"] == expected_digest
    assert handle.chat_authenticated.calls[0]["owner_epoch"] == 31
    assert handle.mark_chat_service_response_ready_authenticated.calls[0]["owner_epoch"] == 31
    assert credential not in repr(handle.chat_authenticated.calls)
    assert credential not in response.body.decode("utf-8")
    assert response_payload["id"].startswith("chatcmpl_request-1_")
