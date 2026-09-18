# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the real engine/gateway HTTP protocol with only GPU work faked."""

import asyncio
from collections import Counter, defaultdict
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import FastAPI

from relax.engine.lora.control import BackendRejectedError, EngineControl, aborted_output
from relax.engine.lora.gateway import PublicationGateway, install_gateway_routes
from relax.engine.lora.http import HttpAdapterEngine, install_engine_routes
from relax.engine.lora.publication import AdapterCapacityError, AdapterLeaseError, AdapterPublicationError
from relax.engine.lora.snapshot import AdapterConflictError, snapshot_adapter


BASE = "a" * 64


def run(work):
    asyncio.run(asyncio.wait_for(work(), 10))


def artifact(tmp_path, name):
    source = tmp_path / f"export-{name}"
    source.mkdir(exist_ok=True)
    (source / "adapter_config.json").write_text('{"r":4}')
    (source / "adapter_model.safetensors").write_text(name)
    return snapshot_adapter(source, tmp_path / "store", version_id=name, base_model_digest=BASE)


class Backend:
    def __init__(self):
        self.resident = {}
        self.loads = Counter()
        self.unloads = Counter()
        self.started = defaultdict(asyncio.Event)
        self.load_gate = {}
        self.fail = set()
        self.generation_gate = None
        self.generation_started = asyncio.Event()
        self.payloads = []
        self.cancelled = set()
        self.reject = False
        self.unknown = False

    async def load(self, snapshot, op):
        self.loads[snapshot.version_id] += 1
        self.started[snapshot.version_id].set()
        if snapshot.version_id in self.load_gate:
            await self.load_gate[snapshot.version_id].wait()
        self.resident[op] = snapshot
        if snapshot.version_id in self.fail:
            raise RuntimeError("partial allocation")

    async def unload(self, snapshot, op):
        if self.resident.pop(op, None) is not None:
            self.unloads[snapshot.version_id] += 1

    async def generate(self, payload):
        self.payloads.append(dict(payload))
        self.generation_started.set()
        if self.generation_gate is not None:
            await self.generation_gate.wait()
        if self.reject:
            raise BackendRejectedError("invalid prompt, never submitted")
        if self.unknown:
            raise RuntimeError("transport failed after scheduler submission")
        if payload["rid"] in self.cancelled:
            return aborted_output()
        return {"text": payload["lora_path"], "output_ids": [42], "meta_info": {"finish_reason": {"type": "stop"}}}

    async def abort(self, rid):
        self.cancelled.add(rid)
        # Deliberately no terminal ACK: physical completion uses the test gate.

    async def close_session(self, session):
        pass


class LoseReply(httpx.AsyncBaseTransport):
    def __init__(self, app):
        self.inner = httpx.ASGITransport(app=app)
        self.lose_once = set()

    async def handle_async_request(self, request):
        response = await self.inner.handle_async_request(request)
        action = request.url.path.rsplit("/", 1)[-1]
        if action in self.lose_once:
            self.lose_once.remove(action)
            raise httpx.ReadError("injected lost ACK", request=request)
        return response


@asynccontextmanager
async def cohort(tmp_path):
    backends = [Backend(), Backend()]
    controls = [EngineControl(item, engine_id=str(i), base_digest=BASE, capacity=2) for i, item in enumerate(backends)]
    transports = []
    for control in controls:
        app = FastAPI()
        install_engine_routes(app, control)
        transports.append(LoseReply(app))
    async with httpx.AsyncClient(
        mounts={f"http://engine-{i}": transport for i, transport in enumerate(transports)}
    ) as client:
        engines = [HttpAdapterEngine(f"http://engine-{i}", "owner", client) for i in range(2)]
        await asyncio.gather(*(engine.connect(BASE, 2) for engine in engines))
        gateway = PublicationGateway(engines, capacity=2, base_digest=BASE, store=tmp_path / "store")
        yield gateway, backends, controls, transports


def request(session="old", rid="r1"):
    return {"session_id": session, "rid": rid, "input_ids": [1, 2], "sampling_params": {"max_new_tokens": 1}}


def test_http_publication_binding_capacity_and_result_expiry(tmp_path):
    async def scenario():
        async with cohort(tmp_path) as (gateway, backends, controls, _):
            a, b, c = [artifact(tmp_path, name) for name in "ABC"]
            await gateway.manager.publish(a)
            assert (await gateway.generate(request()))["text"] == a.lora_path
            backends[1].load_gate["B"] = asyncio.Event()
            publishing = asyncio.create_task(gateway.manager.publish(b))
            await backends[1].started["B"].wait()
            assert gateway.bind("during")["version_id"] == "A"
            assert gateway.state()["default_version"] == "A"
            backends[1].load_gate["B"].set()
            await publishing
            assert (await gateway.generate(request("new", "r2")))["text"] == b.lora_path
            assert (await gateway.generate(request("old", "r3")))["text"] == a.lora_path
            with pytest.raises(AdapterCapacityError):
                await gateway.manager.publish(c)
            await gateway.close("old")
            await gateway.close("during")
            await gateway.tick()
            if gateway._collection is not None:
                await gateway._collection
            assert all(backend.unloads["A"] == 1 for backend in backends)
            assert gateway.requests["r1"].result == {"status": "expired"}
            assert any(req.result == {"status": "expired"} for ctl in controls for req in ctl.requests.values())
            with pytest.raises(AdapterLeaseError):
                await gateway.generate(request())
            await gateway.manager.publish(c)
            await gateway.tick()
            if gateway._collection is not None:
                await gateway._collection
            assert all(backend.unloads["A"] == 1 for backend in backends)

    run(scenario)


def test_http_failed_load_rolls_back_and_duplicate_publish_is_idempotent(tmp_path):
    async def scenario():
        async with cohort(tmp_path) as (gateway, backends, _, _):
            a, b = [artifact(tmp_path, name) for name in "AB"]
            await gateway.manager.publish(a)
            await gateway.manager.publish(a)
            assert all(backend.loads["A"] == 1 for backend in backends)
            backends[1].fail.add("B")
            with pytest.raises(AdapterPublicationError):
                await gateway.manager.publish(b)
            assert gateway.state()["default_version"] == "A"
            assert gateway.state()["occupied"] == 1
            assert all(backend.unloads["B"] == 1 for backend in backends)
            backends[1].fail.clear()
            await gateway.manager.publish(b)
            assert gateway.bind("fresh")["version_id"] == "B"

    run(scenario)


def test_engine_retire_fences_late_prepare_and_waits_for_allocation(tmp_path):
    async def scenario():
        backend = Backend()
        control = EngineControl(backend, engine_id="one", base_digest=BASE, capacity=2)
        a = artifact(tmp_path, "A")
        backend.load_gate["A"] = asyncio.Event()
        preparing = asyncio.create_task(control.prepare(a, "op"))
        await backend.started["A"].wait()
        retiring = asyncio.create_task(control.retire(a, "op"))
        await asyncio.sleep(0)
        assert not retiring.done()
        backend.load_gate["A"].set()
        with pytest.raises(AdapterConflictError):
            await preparing
        receipt = await retiring
        assert receipt.fenced and receipt.absent
        with pytest.raises(AdapterConflictError):
            await control.prepare(a, "op")
        await control.retire(a, "op")
        assert backend.unloads["A"] == 1
        await control.retire(a, "never-loaded")
        with pytest.raises(AdapterConflictError):
            await control.prepare(a, "never-loaded")

    run(scenario)


def test_http_cancel_and_lost_reply_do_not_release_inflight_version(tmp_path):
    async def scenario():
        async with cohort(tmp_path) as (gateway, backends, _, transports):
            await gateway.manager.publish(artifact(tmp_path, "A"))
            for backend in backends:
                backend.generation_gate = asyncio.Event()
            generating = asyncio.create_task(gateway.generate(request()))
            while not gateway.requests:
                await asyncio.sleep(0)
            dispatch = gateway.requests["r1"]
            index = int(dispatch.engine.identity.engine_id)
            await backends[index].generation_started.wait()
            transports[index].lose_once.add("generate")
            generating.cancel()
            with pytest.raises(asyncio.CancelledError):
                await generating
            await gateway.manager.publish(artifact(tmp_path, "B"))
            await gateway.close("old")
            assert gateway.state()["versions"]["A"]["request_refs"] == 1
            assert all(backend.unloads["A"] == 0 for backend in backends)
            with pytest.raises(AdapterCapacityError):
                await gateway.manager.publish(artifact(tmp_path, "C"))
            backends[index].generation_gate.set()
            await dispatch.task
            await gateway.tick()
            if gateway._collection is not None:
                await gateway._collection
            assert gateway.state()["versions"]["A"]["state"] == "retired"
            assert len(backends[index].payloads) == 1

    run(scenario)


def test_late_pending_ack_cannot_overwrite_completion(tmp_path):
    async def scenario():
        async with cohort(tmp_path) as (gateway, _, _, _):
            await gateway.manager.publish(artifact(tmp_path, "A"))
            output = await gateway.generate(request())
            dispatch = gateway.requests["r1"]
            gateway._settle(dispatch, {"status": "pending"})
            gateway._settle(dispatch, {"status": "unknown"})
            assert (await gateway.generate(request())) == output
            assert gateway.completed == 1
            with pytest.raises(AdapterConflictError):
                await gateway.generate({**request(), "input_ids": [99]})

    run(scenario)


@pytest.mark.parametrize("unknown", [False, True])
def test_rejected_versus_unknown_request_references(tmp_path, unknown):
    async def scenario():
        async with cohort(tmp_path) as (gateway, backends, _, _):
            await gateway.manager.publish(artifact(tmp_path, "A"))
            for backend in backends:
                backend.reject = not unknown
                backend.unknown = unknown
            with pytest.raises((ValueError, RuntimeError)):
                await gateway.generate(request())
            refs = gateway.state()["versions"]["A"]["request_refs"]
            assert refs == int(unknown)

    run(scenario)


def test_abort_before_generate_and_close_before_bind(tmp_path):
    async def scenario():
        async with cohort(tmp_path) as (gateway, backends, _, _):
            await gateway.manager.publish(artifact(tmp_path, "A"))
            await gateway.abort("r1")
            assert (await gateway.generate(request()))["meta_info"]["finish_reason"]["type"] == "abort"
            assert not any(backend.payloads for backend in backends)
            with pytest.raises(AdapterConflictError):
                await gateway.generate({**request(), "input_ids": [9]})
            await gateway.close("unbound")
            with pytest.raises(AdapterLeaseError):
                gateway.bind("unbound")

    run(scenario)


def test_http_owner_and_boot_fencing(tmp_path):
    async def scenario():
        async with cohort(tmp_path) as (gateway, _, _, _):
            engine = gateway.engines["0"]
            for fields in ({"owner_id": "different"}, {"incarnation": "old-boot"}):
                response = await engine.client.post(
                    f"{engine.url}/relax/lora/claim",
                    json={
                        "owner_id": engine.owner_id,
                        "incarnation": engine.identity.incarnation,
                        **fields,
                    },
                )
                assert response.status_code == 409

    run(scenario)


def test_gateway_http_capacity_status_and_real_export_path(tmp_path):
    async def scenario():
        async with cohort(tmp_path) as (gateway, _, _, _):
            for name in "ABC":
                artifact(tmp_path, name)
            app = FastAPI()
            install_gateway_routes(app, lambda: gateway)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
                for name in "AB":
                    response = await client.post(
                        "/publish", json={"version_id": name, "export_dir": str(tmp_path / f"export-{name}")}
                    )
                    assert response.status_code == 200
                    if name == "A":
                        assert (await client.post("/generate", json=request())).status_code == 200
                response = await client.post(
                    "/publish", json={"version_id": "C", "export_dir": str(tmp_path / "export-C")}
                )
                assert response.status_code == 507
                (tmp_path / "export-A" / "adapter_model.safetensors").write_text("conflict")
                response = await client.post(
                    "/publish", json={"version_id": "A", "export_dir": str(tmp_path / "export-A")}
                )
                assert response.status_code == 409

    run(scenario)


@pytest.mark.parametrize("lost_action", ["prepare", "retire"])
def test_http_lost_publication_receipt_keeps_capacity_until_fenced(tmp_path, lost_action):
    async def scenario():
        async with cohort(tmp_path) as (gateway, backends, _, transports):
            await gateway.manager.publish(artifact(tmp_path, "A"))
            gateway.bind("old")
            transports[0].lose_once.add(lost_action)
            if lost_action == "prepare":
                await gateway.manager.publish(artifact(tmp_path, "B"))
                assert gateway.state()["default_version"] == "B"
                assert all(backend.loads["B"] == 1 for backend in backends)
                return
            backends[1].fail.add("B")
            with pytest.raises(AdapterPublicationError):
                await gateway.manager.publish(artifact(tmp_path, "B"))
            assert gateway.state()["default_version"] == "A"
            assert gateway.state()["occupied"] == (2 if lost_action == "retire" else 1)
            await gateway.tick()
            if gateway._collection is not None:
                await gateway._collection
            assert gateway.state()["occupied"] == 1
            assert all(backend.unloads["B"] == 1 for backend in backends)

    run(scenario)


def test_engine_abort_fences_generate_that_has_not_arrived(tmp_path):
    async def scenario():
        backend = Backend()
        control = EngineControl(backend, engine_id="one", base_digest=BASE, capacity=2)
        snapshot = artifact(tmp_path, "A")
        await control.prepare(snapshot, "op")
        result = await control.abort("physical-rid", "op")
        assert result["status"] == "terminal"
        body = {"rid": "physical-rid", "input_ids": [1], "lora_path": snapshot.lora_path}
        result = await control.generate("op", body)
        assert result["output"]["meta_info"]["finish_reason"]["type"] == "abort"
        assert not backend.payloads and not control.operations["op"].requests
        with pytest.raises(AdapterConflictError):
            await control.generate("op", {**body, "input_ids": [2]})

    run(scenario)
