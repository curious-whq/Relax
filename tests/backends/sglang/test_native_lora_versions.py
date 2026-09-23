# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run native version/execution protocols and the actual owned communicator.

Registry, loader and CUDA completion are controlled doubles. These tests are
not live SGLang or GPU acceptance; the separately marked native hooks/pool and
GPU tests remain necessary.
"""

import asyncio
import importlib.util
import os
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.backends.sglang.conftest import load_tokenizer_controls, prepare_version, request_state


@pytest.fixture
def native(native_abort_type, monkeypatch):
    root = os.environ.get("RELAX_SGLANG_SOURCE")
    if not root:
        pytest.skip("requires patched native source: RELAX_SGLANG_SOURCE")
    modules = []
    for name, relative in (
        ("native_versions", "lora/version_control.py"),
        ("native_version_rpc", "managers/communicator.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, Path(root) / "python/sglang/srt" / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    modules[0].TokenizerControlMixin = load_tokenizer_controls(Path(root), *modules, monkeypatch)
    return tuple(modules)


@pytest.fixture
def h(native):
    return Harness(native)


class Registry:
    def __init__(self):
        self.adapters = {}
        self.refs = {}
        self.releases = Counter()
        self.unregisters = Counter()
        self.fail_release = False
        self.release_seen = asyncio.Event()

    async def register(self, ref):
        assert ref.lora_name not in self.adapters
        self.adapters[ref.lora_name] = ref
        self.refs[ref.lora_id] = 0

    async def acquire_exact(self, ref):
        assert self.adapters.get(ref.lora_name) == ref
        self.refs[ref.lora_id] += 1

    async def release(self, uid):
        if self.fail_release:
            raise RuntimeError("unknown release outcome")
        assert self.refs[uid] > 0
        self.refs[uid] -= 1
        self.releases[uid] += 1
        self.release_seen.set()

    async def unregister_exact(self, ref):
        assert self.adapters.get(ref.lora_name) == ref
        self.adapters.pop(ref.lora_name)
        self.unregisters[ref.lora_id] += 1

    async def wait_for_unload(self, uid):
        assert self.refs[uid] == 0
        self.refs.pop(uid)


class Loader:
    def __init__(self):
        self.resources = {}
        self.loads = Counter()
        self.unloads = Counter()
        self.allow_clear = False
        self.cleanup_error = False
        self.clearing = asyncio.Event()
        self.resident = set()
        self.pending_lora_load_events = {}
        self.copy_complete = True

    def load_lora_adapter(self, ref):
        assert ref.lora_id not in self.resources
        self.loads[ref.lora_id] += 1
        self.resources[ref.lora_id] = ref
        return SimpleNamespace(success=True)

    def stage_managed_residency(self, ref, stream):
        self.load_stream = stream
        event = SimpleNamespace(query=lambda: self.copy_complete)
        self.pending_lora_load_events[ref.lora_id] = event
        self.resident.add(ref.lora_id)
        return event, None

    def validate_managed_residency(self, ref):
        if self.resources.get(ref.lora_id) != ref or ref.lora_id not in self.resident:
            raise ValueError("not resident")

    def _validate_managed_batch_ids(self, ids):
        assert all(uid in self.resident for uid in ids)

    def poll_unload_drained_lora(self, ref):
        assert self.resources[ref.lora_id] == ref
        self.clearing.set()
        if self.cleanup_error:
            return SimpleNamespace(success=False, error_message="clear failed")
        if not self.allow_clear:
            return None
        self.resources.pop(ref.lora_id)
        self.unloads[ref.lora_id] += 1
        return SimpleNamespace(success=True)


class Harness:
    def __init__(self, native):
        self.n, self.rpc = native
        rpc = self.rpc
        self.registry, self.loader = Registry(), Loader()
        self.commands = []
        self.replies = []
        self.events = defaultdict(asyncio.Event)
        self.hold_load_ack = False
        self.hold_ready_ack = False
        self.held_ready_ack = None
        self.held_ack = None
        self.kv_closes = Counter()
        self.drop_execution_ack = False
        self.engine = SimpleNamespace(
            lora_execution_owner=("cohort", "boot"),
            lora_execution_limit=100,
            lora_execution_accepting=True,
            ps=SimpleNamespace(
                pp_rank=0, pp_size=1, tp_rank=0, tp_size=1, attn_tp_rank=0, attn_cp_rank=0, attn_dp_rank=0
            ),
            device_module=SimpleNamespace(Event=lambda: SimpleNamespace(record=lambda **kw: None, query=lambda: True)),
            lora_versions={},
            lora_version_names={},
            lora_retirements={},
            lora_sessions={},
            lora_session_closures={},
            running_batch=None,
            last_batch=None,
            result_queue=[],
            waiting_queue=[],
            chunked_req=None,
            _pending_chunked_abort_req=None,
            ipc_channels=SimpleNamespace(send_to_tokenizer=SimpleNamespace(send_output=self.receive_wire)),
            tp_worker=SimpleNamespace(model_runner=SimpleNamespace(lora_manager=self.loader)),
            abort_request=lambda command, **kw: None,
            _close_lora_session=lambda sid: self.kv_closes.update([sid]),
            lora_loads={},
            forward_stream=object(),
        )
        channels = [
            rpc.FanOutCommunicator(
                self.send_command,
                1,
                mode="owned",
                correlation_key=lambda message: message.control_key,
                response_rank=lambda message: message.sender_rank,
            )
            for _ in range(2)
        ]
        self.mutate, self.fence = channels
        self.control = self.n.LoRAVersionControl(
            self.registry,
            self.send_execution,
            "cohort",
            "boot",
            100,
            mutate=self.mutate,
            fence=self.fence,
        )

    async def health(self, generate, timeout):
        return await self.n.TokenizerControlMixin.health_lora_publication(
            SimpleNamespace(lora_version_control=self.control), generate, timeout
        )

    async def memory(self, *args, **kwargs):
        return await self.n.TokenizerControlMixin._change_lora_memory(
            SimpleNamespace(lora_version_control=self.control), *args, **kwargs
        )

    async def finish_clear(self, retire):
        self.loader.allow_clear = True
        self.drive()
        return await asyncio.wait_for(retire, timeout=1)

    async def load_version(self):
        version = self.version()
        await self.load(*version)
        return version

    async def load(self, *version):
        return await load_fixture(self.control, version)

    def version(self, uid="uid-A", name="A"):
        return (
            self.n.NativeLoadIdentity("boot", uid),
            "digest-" + name,
            SimpleNamespace(lora_id=uid, lora_name=name, lora_path="/fixture/" + name, pinned=True),
        )

    def command(self, version, action, control_id="test-control"):
        identity, digest, ref = version
        return self.n.LoRAVersionCommand(identity, control_id, action, ref, digest)

    def send_command(self, command):
        self.commands.append(command)
        reply = self.n.control_version(self.engine, command)
        self.events[command.action].set()
        if reply is not None:
            if self.hold_load_ack and command.action == "load":
                self.held_ack = reply
            elif self.hold_ready_ack and command.action == "ready":
                self.held_ready_ack = reply
            else:
                self.receive(reply)
        if command.action in ("load", "cleanup"):
            self.drive()

    def send_execution(self, message):
        if isinstance(message, self.n.LoRASessionClose):
            reply = self.n.control_session_close(self.engine, message)
            self.events["session_close"].set()
            if reply is not None:
                self.receive(reply)
            self.drive()
        else:
            self.n.control_execution(self.engine, message)

    def receive_wire(self, reply, *_):
        if isinstance(reply, self.n.LoRAVersionReply) and reply.action == "load" and self.hold_load_ack:
            self.held_ack = reply
        else:
            self.receive(reply)

    def receive(self, reply):
        self.replies.append(reply)
        if isinstance(reply, self.n.LoRAVersionReply):
            (self.fence if reply.action == "fence" else self.mutate).handle_recv(reply)
        elif isinstance(reply, self.n.LoRASessionReply):
            self.control.observe_session(reply)
        elif not self.drop_execution_ack:
            self.control.observe(reply)

    def drive(self):
        self.n.poll_version_loads(self.engine)
        self.n.poll_version_retirements(self.engine)
        self.n.poll_session_closures(self.engine)

    async def acquire(self, version, rid="request", submit=True, sid="session"):
        identity, _, ref = version
        # Execution/retirement unit fixtures assume a previously verified READY
        # version. Dedicated prepare tests below run the actual readiness protocol.
        if not self.control.versions[identity.native_lora_id].fenced:
            self.control.versions[identity.native_lora_id].state = "READY"
            self.engine.lora_versions[identity.native_lora_id].state = "READY"
        attempt = self.n.LoRAExecutionIdentity("cohort", *identity.instance_key, "owner", sid, rid)
        await self.control.acquire(attempt, ref, request_state())
        if submit:
            self.control.begin_submit(attempt)
        return attempt

    def admit(self, attempt):
        req = SimpleNamespace(
            lora_engine_boot_id=attempt.engine_boot_id,
            lora_request_kind=attempt.kind,
            lora_session_id=attempt.session_id,
            rid=attempt.rid,
            lora_id=attempt.native_lora_id,
            http_worker_ipc=None,
        )
        self.n.accept_execution(self.engine, req)
        return req


async def load_fixture(control, version):
    """Isolate load/retire tests; prepare tests exercise the full work task."""
    record = control._version(*version)
    if record.fenced:
        raise ValueError("VERSION_FENCED")
    if record.state not in ("LOADED", "READY"):
        await control._load(record)
    return record.state


async def wait(event):
    await asyncio.wait_for(event.wait(), timeout=1)


async def start_until(coroutine, event):
    task = asyncio.create_task(coroutine)
    await wait(event)
    return task


async def cancel_waiter(waiter):
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter


async def test_native_version_load_is_idempotent_and_not_ready_without_warmup(h):
    version = h.version()
    assert await h.load(*version) == "LOADED"
    assert await h.load(*version) == "LOADED"
    assert h.loader.loads == {"uid-A": 1}
    assert h.control.versions["uid-A"].registered
    with pytest.raises(ValueError, match="INSTANCE_CONTENT_CONFLICT"):
        await h.load(version[0], "changed", version[2])
    assert len(h.commands) == 1
    attempt = h.n.LoRAExecutionIdentity("cohort", *version[0].instance_key, "owner", "session", "business")
    with pytest.raises(ValueError, match="ADAPTER_NOT_READY"):
        await h.control.acquire(attempt, version[2], request_state())
    with pytest.raises(ValueError, match="ADAPTER_NOT_READY"):
        h.admit(attempt)
    assert h.registry.refs == {"uid-A": 0}


async def test_native_version_fence_overtakes_late_load_ack_without_cancelling_owned_load(h):
    version = h.version()
    h.hold_load_ack = True
    load_waiter = await start_until(h.load(*version), h.events["load"])
    await cancel_waiter(load_waiter)
    retire = await start_until(h.control.retire_version(*version), h.events["fence"])
    assert h.engine.lora_versions["uid-A"].fenced
    assert not h.events["cleanup"].is_set()
    assert h.loader.resources and not h.registry.adapters
    h.receive(h.held_ack)
    await wait(h.loader.clearing)
    assert not h.registry.adapters and not retire.done()
    assert await h.finish_clear(retire) == "ABSENT"
    assert h.loader.unloads == {"uid-A": 1}


async def test_native_version_retire_closes_accepted_but_not_submitted_request(h):
    version = await h.load_version()
    attempt = await h.acquire(version, submit=False)
    retire = await start_until(h.control.retire_version(*version), h.loader.clearing)
    assert h.registry.releases == {"uid-A": 1}
    with pytest.raises(ValueError, match="ATTEMPT_NOT_ADMITTED"):
        h.control.begin_submit(attempt)
    assert await h.finish_clear(retire) == "ABSENT"


async def test_native_version_retire_waits_for_submit_in_transit(h):
    version = await h.load_version()
    attempt = await h.acquire(version)
    retire = await start_until(h.control.retire_version(*version), h.events["fence"])
    assert h.registry.refs["uid-A"] == 1 and not h.events["cleanup"].is_set()
    with pytest.raises(ValueError, match="ADAPTER_NOT_READY"):
        h.admit(attempt)
    # The scheduler's error delivery triggers exact cancel, which now
    # installs the missing-attempt fence before reporting never-submitted.
    h.control.observe(
        h.n.LoRAExecutionReply(attempt.rid, attempt.engine_boot_id, "REQUEST_FINISHED", error="VERSION_FENCED")
    )
    await wait(h.loader.clearing)
    assert await h.finish_clear(retire) == "ABSENT"
    assert h.registry.releases == {"uid-A": 1}


async def test_native_version_waits_for_gpu_then_clear_while_other_version_runs(h):
    a, b = h.version(), h.version("uid-B", "B")
    await h.load(*a)
    await h.load(*b)
    ar = h.admit(await h.acquire(a, "A-request"))
    h.n.finish_native_request(h.engine, ar)
    br = h.admit(await h.acquire(b, "B-request", sid="session-B"))
    done = False
    h.engine.lora_versions["uid-A"].last_use_event = SimpleNamespace(query=lambda: done)
    h.engine.running_batch = SimpleNamespace(reqs=[br])
    retire = await start_until(h.control.retire_version(*a), h.events["fence"])
    h.drive()
    assert not h.loader.clearing.is_set()
    done = True
    h.drive()
    await wait(h.loader.clearing)
    assert h.engine.lora_versions["uid-A"].state == "CLEARING"
    assert "uid-A" in h.loader.resources and not retire.done()
    assert br in h.engine.running_batch.reqs and h.registry.refs["uid-B"] == 1
    assert await h.finish_clear(retire) == "ABSENT"
    assert h.loader.unloads == {"uid-A": 1} and "uid-B" in h.loader.resources
    assert h.registry.unregisters == {"uid-A": 1}


async def test_native_old_operation_cannot_unload_same_name_new_instance(h):
    first, second = h.version(), h.version("uid-A2", "A")
    await h.load(*first)
    h.loader.allow_clear = True
    await h.control.retire_version(*first)
    await h.load(*second)
    for action in ("fence", "cleanup", "status", "load"):
        reply = h.n.control_version(h.engine, h.command(first, action, "late-" + action))
        assert reply.actual_unload_count == 1
    h.drive()
    assert h.loader.unloads == {"uid-A": 1}
    assert h.loader.resources == {"uid-A2": second[2]}
    assert h.engine.lora_version_names == {"A": second[0].instance_key}
    assert h.registry.adapters == {"A": second[2]}


async def test_native_cleanup_retry_does_not_unregister_or_release_counter_twice(h):
    version = await h.load_version()
    h.loader.cleanup_error = True
    with pytest.raises(RuntimeError, match="clear failed"):
        await h.control.retire_version(*version)
    assert h.control.versions["uid-A"].state == "CLEANUP_PENDING"
    assert h.loader.resources and h.registry.unregisters == {"uid-A": 1}
    h.loader.cleanup_error = False
    h.loader.allow_clear = True
    assert await h.control.retire_version(*version) == "ABSENT"
    assert h.registry.unregisters == h.loader.unloads == {"uid-A": 1}


async def test_native_release_unknown_blocks_version_cleanup(h):
    version = await h.load_version()
    await h.acquire(version, submit=False)
    h.registry.fail_release = True
    with pytest.raises(RuntimeError, match="unknown release outcome"):
        await h.control.retire_version(*version)
    assert not h.events["cleanup"].is_set()
    assert h.registry.refs["uid-A"] == 1 and h.loader.resources


async def test_native_version_identity_conflict_and_reused_uid_have_no_side_effects(h):
    version = await h.load_version()
    command = h.command(version, "cleanup")
    assert h.n.control_version(h.engine, command).error == "VERSION_FENCE_REQUIRED"
    conflict = replace(command, identity=replace(version[0], native_lora_id="changed"))
    assert h.n.control_version(h.engine, conflict).error == "ADAPTER_IDENTITY_MISMATCH"
    reused = replace(h.command(version, "load"), digest="different-digest")
    assert h.n.control_version(h.engine, reused).error == "INSTANCE_CONTENT_CONFLICT"
    assert h.loader.loads == {"uid-A": 1} and h.loader.unloads == {}


async def test_native_version_retire_during_registry_registration_waits_and_cleans_once(h):
    version = h.version()
    entered, proceed = asyncio.Event(), asyncio.Event()
    original = h.registry.register

    async def register(ref):
        entered.set()
        await proceed.wait()
        await original(ref)

    h.registry.register = register
    load = await start_until(h.load(*version), entered)
    retire = await start_until(h.control.retire_version(*version), h.events["fence"])
    assert not h.events["cleanup"].is_set()
    proceed.set()
    assert await load == "RETIRING"
    await wait(h.loader.clearing)
    assert await h.finish_clear(retire) == "ABSENT"
    assert h.registry.unregisters == {"uid-A": 1}
    assert not h.registry.refs and not h.registry.adapters
    assert h.loader.unloads == {"uid-A": 1}


def session_identity(h, sid="session", owner="owner"):
    return h.n.LoRASessionIdentity("cohort", "boot", owner, sid)


@pytest.mark.parametrize("in_transit", [False, True])
async def test_native_session_close_fences_delayed_submit_and_closes_kv_once(h, in_transit):
    version = await h.load_version()
    if in_transit:
        attempt = await h.acquire(version)
        h.drop_execution_ack = True  # Session proof must suffice without the individual ACK.
    else:
        attempt = h.n.LoRAExecutionIdentity("cohort", *version[0].instance_key, "owner", "session", "late")
    identity = session_identity(h)
    assert await h.control.close_session(identity) == "SESSION_DRAINED"
    assert await h.control.close_session(identity) == "SESSION_DRAINED"
    with pytest.raises(ValueError, match="SESSION_CLOSED"):
        await h.acquire(version)
    with pytest.raises(ValueError, match="SESSION_CLOSED"):
        h.admit(attempt)
    assert attempt.rid not in h.control.requests
    assert h.kv_closes == {"session": 1}
    assert h.registry.refs == {"uid-A": 0}
    assert h.registry.releases["uid-A"] == int(in_transit)


async def test_native_session_close_waits_for_gpu_use_and_preserves_other_session(h):
    version = await h.load_version()
    attempt = await h.acquire(version)
    h.admit(attempt)
    done = False
    h.engine.device_module.Event = lambda: SimpleNamespace(query=lambda: done, record=lambda **kw: None)
    other = replace(attempt, session_id="other", rid="other-rid")
    await h.control.acquire(other, version[2], request_state())
    h.control.begin_submit(other)
    other_execution = h.admit(other)
    h.engine.running_batch = SimpleNamespace(reqs=[other_execution])
    waiter = await start_until(h.control.close_session(session_identity(h)), h.events["session_close"])
    command = h.control.sessions["session"].command
    reply = h.n.session_reply(command, "SESSION_DRAINED")
    assert not h.control.observe_session(replace(reply, control_id="old"))
    assert not h.control.observe_session(replace(reply, identity=replace(reply.identity, engine_boot_id="old")))
    assert not h.kv_closes and not waiter.done()
    assert h.registry.refs == {"uid-A": 1}  # Native Req finished; Session GPU fence is still pending.
    await cancel_waiter(waiter)
    done = True
    h.drive()
    assert await h.control.close_session(session_identity(h)) == "SESSION_DRAINED"
    assert h.registry.refs == {"uid-A": 1}
    assert other_execution in h.engine.running_batch.reqs
    assert h.kv_closes == {"session": 1}
    assert not h.control.sessions["other"].closing
    assert not h.control.observe_session(reply)
    assert h.registry.releases == {"uid-A": 1}


async def test_native_session_close_pending_acquire_cannot_release_before_increment(h):
    version = await h.load_version()
    entered, proceed = asyncio.Event(), asyncio.Event()
    original = h.registry.acquire_exact

    async def acquire(ref):
        entered.set()
        await proceed.wait()
        await original(ref)

    h.registry.acquire_exact = acquire
    waiter = await start_until(h.acquire(version), entered)
    close = await start_until(h.control.close_session(session_identity(h)), h.events["session_close"])
    assert not close.done() and not h.registry.releases
    proceed.set()
    with pytest.raises(ValueError, match="ATTEMPT_CANCELLED"):
        await waiter
    assert await close == "SESSION_DRAINED"
    assert h.registry.refs == {"uid-A": 0}
    assert h.registry.releases == {"uid-A": 1}


async def test_native_session_partial_kv_failure_never_turns_retry_noop_into_completion(h):
    path = Path(os.environ["RELAX_SGLANG_SOURCE"]) / "python/sglang/srt/mem_cache/unified_cache/session_ref_tracker.py"
    spec = importlib.util.spec_from_file_location("native_session_tracker", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    component = SimpleNamespace(refs=1, calls=0, reset_session_state=lambda: None)

    def release(sid):
        component.calls += 1
        raise RuntimeError("component close failed")

    component.release_session = release
    tracker = module.UnifiedSessionRefTracker(
        components=(component,),
        tree_core=None,
        enable_session_radix_cache=True,
    )
    tracker.open_radix_session("session")
    h.engine._close_lora_session = tracker.release_radix_session
    version = await h.load_version()
    attempt = await h.acquire(version)
    h.drop_execution_ack = True
    with pytest.raises(RuntimeError, match="SESSION_KV_CLOSE_UNKNOWN"):
        await h.control.close_session(session_identity(h))
    assert h.control.sessions["session"].closing
    assert not h.control.requests[attempt.rid].drained
    assert h.registry.refs == {"uid-A": 1}
    # Reproduce the actual native hazard: a tombstone makes the second call
    # return normally even though the component still owns its protection.
    assert tracker.release_radix_session("session") == 0
    assert component.refs == 1 and component.calls == 1
    with pytest.raises(RuntimeError, match="SESSION_KV_CLOSE_UNKNOWN"):
        await h.control.close_session(session_identity(h))
    assert not h.control.sessions["session"].closed
    assert not h.engine.lora_sessions["session"].closed
    assert component.calls == 1
    assert h.registry.refs == {"uid-A": 1}


async def test_native_session_close_rejects_foreign_boot_and_other_owner_same_kv_id(h):
    identity = session_identity(h)
    assert await h.control.close_session(identity) == "SESSION_DRAINED"
    for wrong, error in (
        (replace(identity, engine_boot_id="old"), "ENGINE_EPOCH_MISMATCH"),
        (replace(identity, owner_epoch="other"), "SESSION_ID_CONFLICT"),
    ):
        with pytest.raises(ValueError, match=error):
            await h.control.close_session(wrong)
        reply = h.n.control_session_close(h.engine, h.n.LoRASessionClose(wrong, "wrong"))
        assert reply.state == "UNKNOWN" and reply.error == error
    assert h.kv_closes == {"session": 1}


async def test_native_session_close_full_quota_preserves_known_close_and_rejects_unknown(h):
    h.control.max_records = h.engine.lora_execution_limit = 3
    version = await h.load_version()
    await h.acquire(version)
    # Version + Session + attempt fill quota. Existing close needs no new slot.
    assert await h.control.close_session(session_identity(h)) == "SESSION_DRAINED"
    with pytest.raises(ValueError, match="LIFECYCLE_CAPACITY_EXCEEDED"):
        await h.control.close_session(session_identity(h, sid="unknown"))
    assert not h.control.accepting
    # Worker stores no attempt tombstones; exhaust its independent close quota.
    h.engine.lora_execution_limit = 2
    reply = h.n.control_session_close(h.engine, h.n.LoRASessionClose(session_identity(h, sid="unknown"), "full"))
    assert reply.error == "LIFECYCLE_CAPACITY_EXCEEDED"
    assert not h.engine.lora_execution_accepting
    assert "unknown" not in h.engine.lora_sessions
    assert h.kv_closes == {"session": 1}


async def test_native_session_binding_cannot_change_adapter_between_attempts(h):
    a, b = h.version(), h.version("uid-B", "B")
    await h.load(*a)
    await h.load(*b)
    first = await h.acquire(a)
    h.admit(first)
    with pytest.raises(ValueError, match="SESSION_ADAPTER_CONFLICT"):
        await h.acquire(b, rid="next")
    # The trusted tokenizer enforces binding; workers do not mirror that table.
    assert h.registry.refs == {"uid-A": 1, "uid-B": 0}


async def warmup_execution(h, identity, ref, *, complete=True, resident=True, forward=True, terminal=True):
    await h.control.acquire(identity, ref, request_state())
    h.control.begin_submit(identity)
    execution = h.admit(identity)
    if forward:
        event = SimpleNamespace(complete=complete, record=lambda **kw: None)
        event.query = lambda: event.complete
        h.engine.device_module = SimpleNamespace(Event=lambda: event)
        h.engine.forward_stream = object()
        batch = h.n.begin_execution_batch(h.engine, SimpleNamespace(reqs=[execution]))
        h.n.finish_execution_batch(h.engine, batch)
    if resident:
        h.loader.resident.add(ref.lora_id)
    elif ref is not None:
        h.loader.resident.discard(ref.lora_id)  # Fault after prepare's residency copy.
    if terminal:
        h.n.finish_native_request(h.engine, execution)
    h.control.delivered(identity)
    h.events["warmup_output"].set()
    return execution


async def test_native_prepare_waits_for_gpu_drain_and_shields_shared_warmup(h):
    version = h.version()

    async def warmup(identity, ref):
        await warmup_execution(h, identity, ref, complete=False)

    waiter = await start_until(prepare_version(h.control, *version, warmup), h.events["session_close"])
    record = h.control.versions["uid-A"]
    assert record.state == "WARMING" and not h.events["ready"].is_set()
    execution = h.engine.lora_versions["uid-A"]
    assert execution.warmed and not execution.last_use_event.query()
    assert h.registry.refs == {"uid-A": 0}  # Instance event still prevents retirement.
    await cancel_waiter(waiter)
    assert not record.work_task.done()
    execution.last_use_event.complete = True
    h.drive()
    assert await prepare_version(h.control, *version, warmup) == "READY"
    assert h.engine.lora_versions["uid-A"].state == "READY"
    assert h.registry.refs == {"uid-A": 0}
    assert await prepare_version(h.control, *version, warmup) == "READY"
    assert h.loader.loads == {"uid-A": 1}
    assert len(h.kv_closes) == 1
    assert sum(c.action == "ready" for c in h.commands) == 1
    assert h.admit(await h.acquire(version)).lora_id == "uid-A"


@pytest.mark.parametrize("missing", ["forward", "residency", "execution", "output"])
async def test_native_prepare_rejects_incomplete_warmup_evidence(h, missing):
    h.loader.allow_clear = True
    version = h.version()

    async def warmup(identity, ref):
        if missing == "execution":
            return
        await warmup_execution(h, identity, ref, resident=missing != "residency", forward=missing != "forward")
        if missing == "output":
            raise RuntimeError("warmup output failed")

    with pytest.raises(RuntimeError):
        await prepare_version(h.control, *version, warmup)
    assert h.control.versions["uid-A"].state == "ABSENT"
    assert h.engine.lora_versions["uid-A"].state == "ABSENT"
    assert not h.registry.refs
    h.loader.allow_clear = True
    assert await h.control.retire_version(*version) == "ABSENT"
    assert h.loader.unloads == {"uid-A": 1}


async def test_native_retire_overtakes_ready_ack_and_prepare_cannot_restore_ready(h):
    version = h.version()
    h.hold_ready_ack = True
    h.loader.allow_clear = True

    async def warmup(identity, ref):
        await warmup_execution(h, identity, ref)

    prepare = await start_until(prepare_version(h.control, *version, warmup), h.events["ready"])
    assert h.engine.lora_versions["uid-A"].state == "READY"
    retire = await start_until(h.control.retire_version(*version), h.events["fence"])
    assert not h.events["cleanup"].is_set()  # Owned ready still has mutation channel.
    h.receive(h.held_ready_ack)
    with pytest.raises(ValueError, match="VERSION_FENCED"):
        await prepare
    assert await retire == "ABSENT"
    assert h.control.versions["uid-A"].state == "ABSENT"
    assert h.loader.unloads == {"uid-A": 1}


async def test_native_retire_before_warmup_acquire_prevents_late_generation(h):
    version = h.version()
    entered, proceed = asyncio.Event(), asyncio.Event()
    h.loader.allow_clear = True

    async def warmup(identity, ref):
        entered.set()
        await proceed.wait()
        await warmup_execution(h, identity, ref)

    prepare = await start_until(prepare_version(h.control, *version, warmup), entered)
    retire = await start_until(h.control.retire_version(*version), h.events["fence"])
    assert not retire.done()
    proceed.set()
    with pytest.raises((ValueError, RuntimeError), match="VERSION_FENCED|ATTEMPT_ALREADY_EXISTS"):
        await prepare
    assert await retire == "ABSENT"
    assert not h.events["ready"].is_set()
    assert h.control.versions["uid-A"].state == "ABSENT"
    assert h.registry.releases == {} and h.loader.unloads == {"uid-A": 1}


@pytest.mark.parametrize("proof", ["other-adapter", "base-probe"])
async def test_native_ready_rejects_foreign_warmup_proof(h, proof):
    version = await h.load_version()
    if proof == "other-adapter":
        other = h.version("uid-B", "B")
        await h.load(*other)
        attempt = h.n.LoRAExecutionIdentity(
            "cohort", *other[0].instance_key, "native-prepare", "session", "warmup", kind="warmup"
        )
        h.control.versions["uid-B"].warmup = attempt
        await warmup_execution(h, attempt, other[2])
        h.drive()
    else:

        async def generate(identity, ref):
            await warmup_execution(h, identity, ref, resident=False)

        await h.health(generate, 1)
    reply = h.n.control_version(h.engine, h.command(version, "ready"))
    assert reply.error == "WARMUP_DRAIN_UNCONFIRMED"
    assert h.control.versions["uid-A"].state == h.engine.lora_versions["uid-A"].state == "LOADED"


async def test_native_retire_during_warmup_forward_waits_for_event_and_late_abort(h):
    version = h.version()
    aborted = asyncio.Event()
    h.loader.allow_clear = True
    h.engine.abort_request = lambda command, **kw: aborted.set()

    async def warmup(identity, ref):
        req = await warmup_execution(h, identity, ref, complete=False, terminal=False)
        # The real output has not ended yet; keep this request in the native queue.
        h.engine.waiting_queue = [req]
        await aborted.wait()
        h.engine.waiting_queue.clear()
        raise RuntimeError("warmup aborted by retirement")

    prepare = await start_until(prepare_version(h.control, *version, warmup), h.events["warmup_output"])
    record = h.control.versions["uid-A"]
    execution = h.engine.lora_versions["uid-A"]
    retire = await start_until(h.control.retire_version(*version), aborted)
    await wait(h.events["fence"])
    assert not execution.last_use_event.query() and not h.loader.clearing.is_set()
    assert "uid-A" in h.loader.resources
    execution.last_use_event.complete = True
    h.drive()
    assert await retire == "ABSENT"
    with pytest.raises(RuntimeError, match="warmup aborted"):
        await prepare
    assert record.state == "ABSENT" and "warmup aborted" in record.error
    assert h.loader.unloads == {"uid-A": 1}
    assert h.registry.releases == {"uid-A": 1}
    assert not h.events["ready"].is_set()


async def test_native_health_ignores_other_requests_and_shields_shared_probe(h):
    version = await h.load_version()
    business = await h.acquire(version)
    h.admit(business)
    proceed = asyncio.Event()

    async def generate(identity, ref):
        await warmup_execution(h, identity, ref, resident=False, complete=False)
        await proceed.wait()

    first = await start_until(h.health(generate, 1), h.events["warmup_output"])
    identity = h.control.probe_identity
    second = asyncio.create_task(h.health(generate, 1))
    await cancel_waiter(first)
    h.n.finish_native_request(h.engine, SimpleNamespace(rid=business.rid, lora_request_kind="business"))
    h.drive()  # Unrelated business terminal cannot prove probe GPU completion.
    assert h.control.requests[business.rid].used_gpu is False
    assert not second.done()
    assert h.control.probe_identity == identity and len(h.control.requests) == 1
    proceed.set()
    await wait(h.events["session_close"])
    h.engine.lora_sessions[identity.session_id].completion_event.complete = True
    h.drive()
    await second
    assert h.registry.releases == {"uid-A": 1}  # No base/probe reference release.
    with pytest.raises(ValueError, match="SESSION_CLOSED"):
        h.admit(identity)
    assert not h.control.observe(h.n.LoRAExecutionReply(identity.rid, "boot", "REQUEST_FINISHED"))


async def test_native_health_deadline_retains_owned_cleanup_and_prevents_probe_overlap(h):
    cancelled = asyncio.Event()

    async def generate(identity, ref):
        await warmup_execution(h, identity, ref, resident=False, complete=False)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    first = await start_until(h.health(generate, 0.01), cancelled)
    await wait(h.events["session_close"])
    identity = h.control.probe_identity
    execution = h.engine.lora_sessions[identity.session_id]
    assert execution.closing and not execution.closed
    assert not first.done() and not h.control.probe_task.done()
    second = asyncio.create_task(h.health(generate, 1))
    await asyncio.sleep(0)  # Run the second waiter; it must reuse the owned operation.
    assert h.control.probe_identity == identity and not h.control.requests
    execution.completion_event.complete = True
    h.drive()
    for waiter in (first, second):
        with pytest.raises(asyncio.TimeoutError):
            await waiter
    assert identity.rid not in h.control.requests
    assert h.registry.refs == h.registry.releases == {}


async def test_native_session_close_reclaims_attempts_and_preserves_other_sessions(h):
    version = await h.load_version()
    old = await h.acquire(version, sid="old", rid="old-rid", submit=False)
    await h.control.close_session(h.n.LoRASessionIdentity(*old.session_key))
    assert not h.control.requests and not h.control.sessions["old"].attempts
    # Compact identity history remains budgeted; it must still reject RID reuse.
    with pytest.raises(ValueError, match="ATTEMPT_ALREADY_EXISTS"):
        await h.acquire(version, sid="new", rid="old-rid", submit=False)

    current = await h.acquire(version, sid="current", rid="current-rid", submit=False)
    other = await h.acquire(version, sid="other", rid="other-rid", submit=False)

    await h.control.close_session(h.n.LoRASessionIdentity(*current.session_key))
    assert current.rid not in h.control.requests
    assert other.rid in h.control.requests and h.registry.refs["uid-A"] == 1
    assert h.control.sessions["current"].closed
    assert not h.control.sessions["current"].attempts
    assert h.engine.lora_sessions["current"].completion_event is None
    h.control.cancel(other)
    await h.control._record(other).completion.wait()


@pytest.mark.parametrize("overlap", [False, True])
async def test_native_residency_copy_must_finish_before_load_ack_and_warmup(h, overlap):
    version = h.version()
    h.loader.copy_complete = False
    warmed = []
    if overlap:
        h.engine.enable_lora_overlap_loading = True
        h.engine.lora_overlap_loader = SimpleNamespace(load_stream=object())

    async def warmup(identity, ref):
        warmed.append(identity)
        await warmup_execution(h, identity, ref)

    prepare = await start_until(prepare_version(h.control, *version, warmup), h.events["load"])
    assert not prepare.done() and not warmed
    assert h.engine.lora_versions["uid-A"].state == "RESIDENCY_PENDING"
    assert not h.registry.adapters
    if overlap:
        assert h.loader.load_stream is h.engine.lora_overlap_loader.load_stream
        h.loader.pending_lora_load_events.clear()
        h.drive()
        assert not prepare.done() and not warmed  # Shared-map absence is not completion.
    h.loader.copy_complete = True
    h.drive()
    assert await prepare == "READY"
    assert len(warmed) == 1 and not h.engine.lora_loads
    assert not h.loader.pending_lora_load_events


async def test_native_cancel_during_residency_copy_cannot_clear_or_start_warmup(h):
    version = h.version()
    h.loader.copy_complete = False
    h.loader.allow_clear = True
    warmed = []

    async def warmup(*args):
        warmed.append(args)

    prepare = await start_until(prepare_version(h.control, *version, warmup), h.events["load"])
    retire = await start_until(h.control.retire_version(*version), h.events["fence"])
    h.drive()
    assert not retire.done() and not h.loader.unloads and not warmed
    h.loader.copy_complete = True
    h.drive()
    with pytest.raises(RuntimeError, match="VERSION_FENCED"):
        await prepare
    assert await retire == "ABSENT"
    assert h.loader.unloads == {"uid-A": 1} and not warmed


async def test_native_unknown_copy_completion_keeps_version_capacity(h):
    version = h.version()
    h.loader.copy_complete = False
    load = await start_until(h.load(*version), h.events["load"])
    record = h.engine.lora_versions["uid-A"]

    def fail_query():
        raise RuntimeError("CUDA completion unknown")

    record.load_event.query = fail_query
    h.drive()
    with pytest.raises(RuntimeError, match="CUDA completion unknown"):
        await load
    with pytest.raises(RuntimeError, match="RESIDENCY_COPY_UNKNOWN"):
        await h.control.retire_version(*version)
    assert record.state == "CLEANUP_PENDING" and not h.loader.unloads
    assert h.control.versions["uid-A"].state == "CLEANUP_PENDING"


@pytest.mark.parametrize("ranks", [2, 4])
async def test_native_load_and_retire_require_every_execution_rank(native, ranks):
    n, rpc = native
    workers = [Harness(native) for _ in range(ranks)]
    boots = tuple(f"worker-{rank}" for rank in range(ranks))
    registry = Registry()
    observed = []

    def receive(reply):
        observed.append(reply)
        if owner.valid_worker(reply):
            (fence if reply.action == "fence" else mutate).handle_recv(reply)

    def send(command):
        for worker in workers:
            reply = n.control_version(worker.engine, command)
            if reply is not None:
                n.send_managed_reply(worker.engine, reply)
            n.poll_version_loads(worker.engine)
            n.poll_version_retirements(worker.engine)

    mutate, fence = [
        rpc.FanOutCommunicator(
            send,
            ranks,
            mode="owned",
            correlation_key=lambda message: message.control_key,
            response_rank=lambda message: message.sender_rank,
        )
        for _ in range(2)
    ]
    owner = n.LoRAVersionControl(
        registry, lambda _: None, "cohort", "boot", 100, mutate=mutate, fence=fence, worker_boots=boots
    )
    for rank, worker in enumerate(workers):
        worker.engine.ps = SimpleNamespace(tp_rank=rank, tp_size=ranks)
        worker.engine._lora_scheduler_boot_id = boots[rank]
        worker.engine.ipc_channels.send_to_tokenizer.send_output = receive
        worker.loader.allow_clear = rank != ranks - 1
    workers[-1].loader.copy_complete = False
    version = workers[0].version()
    load = asyncio.create_task(load_fixture(owner, version))
    for _ in range(20):
        await asyncio.sleep(0)
    assert not load.done() and not registry.adapters
    assert len(observed) == ranks - 1
    # A duplicate leader ACK cannot satisfy the missing worker.
    receive(observed[0])
    receive(replace(observed[0], sender_rank=ranks - 1, worker_boot_id="old-boot"))
    await asyncio.sleep(0)
    assert not load.done()
    workers[-1].loader.copy_complete = True
    n.poll_version_loads(workers[-1].engine)
    assert await load == "LOADED"
    late = workers[-1].engine
    event = SimpleNamespace(complete=False)
    event.query = lambda: event.complete
    late.lora_versions["uid-A"].last_use_event = event
    # A later PP microbatch can still own A after the leader has finished.
    late.mbs = [SimpleNamespace(reqs=[SimpleNamespace(lora_id="uid-A")])]
    retire = asyncio.create_task(owner.retire_version(*version))
    await workers[0].loader.clearing.wait()
    assert not workers[-1].loader.clearing.is_set() and not retire.done()
    event.complete = True
    n.poll_version_retirements(late)
    assert not workers[-1].loader.clearing.is_set()  # Req ownership remains.
    late.mbs = [None]
    event.complete = False
    n.poll_version_retirements(late)
    assert not workers[-1].loader.clearing.is_set()  # GPU use remains.
    event.complete = True
    n.poll_version_retirements(late)
    await workers[-1].loader.clearing.wait()
    for _ in range(4):
        await asyncio.sleep(0)
    assert not retire.done()
    assert owner.versions["uid-A"].state != "ABSENT"
    assert all(worker.loader.unloads["uid-A"] == 1 for worker in workers[:-1])
    assert workers[-1].loader.unloads["uid-A"] == 0
    workers[-1].loader.allow_clear = True
    n.poll_version_retirements(workers[-1].engine)
    assert await retire == "ABSENT"
    assert await owner.retire_version(*version) == "ABSENT"
    assert all(worker.loader.unloads["uid-A"] == 1 for worker in workers)
    assert registry.unregisters["uid-A"] == 1
    assert owner.versions["uid-A"].actual_unload_count == 1


def memory_channel(harness):
    """Real owned IPC aggregation with a CPU memory-saver stand-in."""
    n, rpc = harness.n, harness.rpc
    engine = harness.engine
    engine.server_args = SimpleNamespace(enable_memory_saver=True, enable_weights_cpu_backup=True)
    engine.weight_updater = SimpleNamespace(offload_tags=set())
    engine.is_fully_idle = lambda: not n.native_requests(engine)
    calls = []

    def apply(action, tags):
        assert engine._lora_memory_transition
        calls.append((action, tuple(tags)))
        if action == "suspend":
            engine.weight_updater.offload_tags.update(tags)
        else:
            engine.weight_updater.offload_tags.difference_update(tags)

    engine._apply_lora_memory = apply
    engine._lora_memory_consensus = lambda value: [value]

    def dispatch(command):
        channel.handle_recv(n.control_memory(engine, command))

    channel = rpc.FanOutCommunicator(
        dispatch,
        1,
        mode="owned",
        correlation_key=lambda message: message.control_key,
        response_rank=lambda message: message.sender_rank,
    )
    original = harness.control.mutate

    async def dispatch_any(command):
        return await (channel(command) if isinstance(command, n.LoRAMemoryCommand) else original(command))

    harness.control.mutate = dispatch_any
    return calls


async def test_memory_handoff_waits_for_execution_and_survives_waiter_cancel(h):
    calls = memory_channel(h)
    version = await h.load_version()
    attempt = await h.acquire(version)
    h.admit(attempt)
    waiter = asyncio.create_task(h.memory("suspend"))
    await asyncio.sleep(0)
    assert not calls and h.registry.refs["uid-A"] == 1
    with pytest.raises(ValueError, match="ENGINE_SUSPENDED"):
        await h.control.acquire(replace(attempt, rid="late"), version[2], request_state())
    await cancel_waiter(waiter)
    assert not h.control.memory_task.done()
    # Native proof, not HTTP completion, releases the execution reference.
    h.control.observe(h.n.LoRAExecutionReply(attempt.rid, attempt.engine_boot_id, "REQUEST_FINISHED", used_gpu=True))
    await h.control.memory_task
    assert h.control.memory_state == "SUSPENDED"
    assert h.registry.adapters["A"] == version[2]
    assert h.registry.unregisters["uid-A"] == 0
    assert h.loader.unloads["uid-A"] == 0
    assert len(calls) == 3
    await h.memory("suspend")
    assert len(calls) == 3
    await h.memory("resume", ["weights"])
    assert h.control.memory_state == "SUSPENDED"
    await h.memory("resume", ["weights"])
    assert len(calls) == 4
    await h.memory("resume", ["kv_cache", "cuda_graph"])
    assert h.control.memory_state == "RESIDENT"
    assert h.control.resident.is_set()
    assert h.control.versions["uid-A"].identity == version[0]
    await h.acquire(version, rid="after-resume")
    h.control.cancel(replace(attempt, rid="after-resume"))
    h.control.observe(h.n.LoRAExecutionReply("after-resume", attempt.engine_boot_id, "REQUEST_FINISHED"))
    await h.control._record(replace(attempt, rid="after-resume")).completion.wait()


async def test_session_close_is_fenced_but_keeps_reference_until_memory_resume(h):
    memory_channel(h)
    await h.load_version()
    await h.memory("suspend")
    identity = h.n.LoRASessionIdentity("cohort", "boot", "owner", "session")
    close = asyncio.create_task(h.control.close_session(identity))
    await asyncio.sleep(0)
    assert h.control.sessions["session"].closing
    assert not h.events["session_close"].is_set()
    assert not close.done()
    await h.memory("resume")
    assert await close == "SESSION_DRAINED"
    assert h.kv_closes["session"] == 1
    assert h.registry.unregisters["uid-A"] == 0


@pytest.mark.parametrize("failure", ["rank-error", "wrong-boot", "missing-rank", "wrong-tags"])
async def test_memory_partial_failure_never_reopens_admission(native, failure):
    n, _ = native
    control = n.LoRAVersionControl(Registry(), lambda _: None, "cohort", "boot", 100, worker_boots=("w0", "w1"))

    async def memory(command):
        replies = [
            n.LoRAMemoryReply(
                "cohort",
                "boot",
                command.control_id,
                command.action,
                command.tags,
                command.tags,
                sender_rank=rank,
                worker_boot_id=f"w{rank}",
            )
            for rank in (0, 1)
        ]
        if failure == "rank-error":
            replies[1] = replace(replies[1], error="partial device release")
        elif failure == "wrong-boot":
            replies[1] = replace(replies[1], worker_boot_id="restarted")
        elif failure == "missing-rank":
            replies.pop()
        else:
            replies[1] = replace(replies[1], offloaded_tags=())
        return replies

    control.mutate = memory
    with pytest.raises(RuntimeError):
        await n.TokenizerControlMixin._change_lora_memory(SimpleNamespace(lora_version_control=control), "suspend")
    assert control.memory_state == "UNKNOWN"
    assert not control.resident.is_set()
    with pytest.raises(ValueError, match="MEMORY_STATE_UNKNOWN"):
        await n.TokenizerControlMixin._change_lora_memory(SimpleNamespace(lora_version_control=control), "resume")


async def test_old_memory_http_and_ipc_replay_cannot_suspend_a_restored_engine(h):
    calls = memory_channel(h)
    sent = []
    original = h.control.mutate

    async def memory(command):
        sent.append(command)
        return await original(command)

    h.control.mutate = memory
    await h.memory("suspend", sequence=1)
    await h.memory("suspend", sequence=1)
    assert len(sent) == 1
    await h.memory("resume", sequence=2)
    before = list(calls)
    with pytest.raises(ValueError, match="STALE_MEMORY_OPERATION"):
        await h.memory("suspend", sequence=1)
    reply = h.n.control_memory(h.engine, sent[0])
    assert reply.error == "STALE_MEMORY_OPERATION"
    assert calls == before and not h.engine.weight_updater.offload_tags
    assert h.engine.lora_memory_state == h.control.memory_state == "RESIDENT"
    assert h.engine.lora_memory_reply.sequence == 2
    assert h.n.control_memory(h.engine, sent[1]).error is None
    assert calls == before


@pytest.mark.parametrize("failure_phase", ["preflight", "kv_cache", "weights"])
def test_memory_worker_error_is_agreed_before_following_collective_phase(native, failure_phase):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    async def scenario():
        n, _ = native
        harnesses = [Harness(native), Harness(native)]
        applied = [memory_channel(h) for h in harnesses]
        barrier = Barrier(2, timeout=2)
        votes = [None, None]

        def consensus(rank, value):
            votes[rank] = value
            barrier.wait()
            result = list(votes)
            barrier.wait()
            return result

        for rank, h in enumerate(harnesses):
            h.engine._lora_memory_consensus = lambda value, rank=rank: consensus(rank, value)
        failing = harnesses[1].engine
        if failure_phase == "preflight":
            failing.is_fully_idle = lambda: False
        else:
            apply = failing._apply_lora_memory

            def fail(action, tags):
                if failure_phase in tags:
                    raise RuntimeError("injected rank-local memory error")
                apply(action, tags)

            failing._apply_lora_memory = fail
        command = n.LoRAMemoryCommand(
            "cohort", "boot", "memory-1", "suspend", ("cuda_graph", "kv_cache", "weights"), sequence=1
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda h: n.control_memory(h.engine, command), harnesses))
        assert all(result.error for result in results)
        assert all(not any("cuda_graph" in tags for _, tags in calls) for calls in applied)
        if failure_phase == "preflight":
            assert applied == [[], []]
        else:
            assert all(h.engine.lora_memory_state == "UNKNOWN" for h in harnesses)
        assert all(result.sequence == 1 for result in results)

    asyncio.run(scenario())


async def test_attention_dp_prepare_warms_every_group_before_ready(native):
    n, _ = native
    registry, calls, commands = Registry(), [], []
    boots = ("worker-0", "worker-1")
    last_group = asyncio.Event()
    reached_last = asyncio.Event()

    async def mutate(command):
        commands.append(command)
        if command.action == "ready":
            assert calls == [0, 1]
            assert all(record.drained for record in owner.requests.values())
        return [
            n.LoRAVersionReply(
                command.identity,
                command.control_id,
                command.action,
                "LOADED" if command.action == "load" else "READY",
                sender_rank=rank,
                worker_boot_id=boot,
            )
            for rank, boot in enumerate(boots)
        ]

    def send(message):
        if isinstance(message, n.LoRASessionClose):
            for rank, boot in enumerate(boots):
                owner.observe_session(
                    n.LoRASessionReply(
                        message.identity,
                        message.control_id,
                        "SESSION_DRAINED",
                        sender_rank=rank,
                        worker_boot_id=boot,
                    )
                )

    owner = n.LoRAVersionControl(
        registry,
        send,
        "cohort",
        "boot",
        100,
        mutate=mutate,
        worker_boots=boots,
        tp_size=2,
        dp_size=2,
    )
    version = Harness(native).version()

    async def warmup(attempt, ref):
        calls.append(attempt.dp_rank)
        await owner.acquire(attempt, ref, request_state())
        owner.begin_submit(attempt)
        if attempt.dp_rank == 1:
            reached_last.set()
            await last_group.wait()
        owner.delivered(attempt)
        owner.observe(
            n.LoRAExecutionReply(
                attempt.rid,
                attempt.engine_boot_id,
                "REQUEST_FINISHED",
                used_gpu=True,
                sender_rank=attempt.dp_rank,
                worker_boot_id=boots[attempt.dp_rank],
            )
        )

    prepare = await start_until(prepare_version(owner, *version, warmup), reached_last)
    assert not prepare.done()
    assert [command.action for command in commands] == ["load"]
    last_group.set()
    assert await prepare == "READY"
    assert await prepare_version(owner, *version, warmup) == "READY"
    assert calls == [0, 1] and registry.refs == {"uid-A": 0}
    assert registry.releases == {"uid-A": 2}


async def test_native_memory_rejects_gpu_tail_after_logical_request_completion(h):
    calls = memory_channel(h)
    await h.load_version()
    event = SimpleNamespace(query=lambda: False)
    h.engine.lora_versions["uid-A"].last_use_event = event
    command = h.n.LoRAMemoryCommand("cohort", "boot", "suspend", "suspend", ("weights",))
    result = h.n.control_memory(h.engine, command)
    assert result.error == "MEMORY_DRAIN_UNCONFIRMED" and not calls
    assert h.loader.unloads == {}


async def test_native_http_retire_before_artifact_validation_fences_late_prepare(h, tmp_path):
    manager = h.n.TokenizerControlMixin()
    manager.lora_version_control = h.control
    manager._lora_artifact_root = tmp_path
    payload = dict(cohort_id="cohort", engine_boot_id="boot", native_lora_id="new-instance", digest="a" * 64)
    before = len(h.commands)
    assert manager.lora_publication_status(payload)["state"] == "UNKNOWN"
    assert len(h.commands) == before and not h.control.versions
    retired = await manager.retire_lora_publication(payload)
    assert retired["state"] == "ABSENT" and retired["actual_unload_count"] == 0
    record = h.control.versions["new-instance"]
    version = record.identity, record.digest, record.ref
    assert h.n.control_version(h.engine, h.command(version, "load")).error == "VERSION_FENCED"
    # A delayed target-side verification cannot reopen this identity, even if
    # its path is no longer accessible. Fenced prepare must do no file access.
    late = await manager.prepare_lora_publication({**payload, "path": str(tmp_path / "not-present")})
    assert late["state"] == "ABSENT" and not h.loader.loads
    before = len(h.commands)
    for _ in range(4):
        assert manager.lora_publication_status(payload)["state"] == "ABSENT"
    assert len(h.commands) == before
    with pytest.raises(ValueError, match="INSTANCE_CONTENT_CONFLICT"):
        manager.lora_publication_status({**payload, "digest": "b" * 64})


async def test_native_http_prepare_shares_one_task_and_returns_actual_instance(h, tmp_path):
    manager = h.n.TokenizerControlMixin()
    manager.lora_version_control = h.control
    manager._lora_artifact_root = tmp_path
    entered, proceed = asyncio.Event(), asyncio.Event()

    async def generate(identity, ref):
        entered.set()
        await proceed.wait()
        await warmup_execution(h, identity, ref)

    manager._generate_lora_control_token = generate
    payload = dict(
        cohort_id="cohort", engine_boot_id="boot", native_lora_id="new-instance", digest="a" * 64, path=str(tmp_path)
    )
    await manager.prepare_lora_publication(payload)
    await entered.wait()
    record = h.control.versions["new-instance"]
    task = record.work_task
    for _ in range(4):
        assert (await manager.prepare_lora_publication(payload))["state"] == "WARMING"
        assert record.work_task is task
    proceed.set()
    assert await task == "READY"
    result = manager.lora_publication_status(payload)
    assert result["native_lora_id"] == "new-instance" and result["digest"] == payload["digest"]
    assert result["resident"] and result["pinned"]
    assert h.loader.loads == {"new-instance": 1}
