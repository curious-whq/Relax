# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
import struct
from pathlib import Path

import pytest
from test_control_http import BASE, Backend, artifact, cohort, request, run

from relax.engine.lora.artifact import ModelContract, canonical, provenance
from relax.engine.lora.control import EngineControl
from relax.engine.lora.execution import SchedulerExecutions
from relax.engine.lora.gateway import PublicationGateway
from relax.engine.lora.journal import ControlJournal
from relax.engine.lora.outbox import SessionCloseOutbox


def tensor_export(root: Path) -> ModelContract:
    root.mkdir()
    contract = ModelContract(
        BASE,
        {
            "model_type": "llama",
            "hidden_size": 4,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "intermediate_size": 8,
        },
        4,
        ("q_proj",),
    )
    config = {"r": 2, "lora_alpha": 4, "target_modules": ["q_proj"], "peft_type": "LORA", "task_type": "CAUSAL_LM"}
    (root / "adapter_config.json").write_bytes(canonical(config))
    index = {}
    offset = 0
    for name, shape in contract._shapes(2, ["q_proj"]).items():
        size = shape[0] * shape[1] * 2
        index[name] = {"dtype": "F16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    header = canonical(index)
    (root / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(offset))
    (root / "producer_manifest.json").write_bytes(
        canonical(provenance(contract, root, producer="test", export_step=1))
    )
    return contract


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_rslora", True),
        ("use_dora", True),
        ("rank_pattern", {"q_proj": 3}),
        ("alpha_pattern", {"q_proj": 7}),
        ("modules_to_save", ["lm_head"]),
        ("init_lora_weights", "pissa"),
        ("target_modules", "all-linear"),
    ],
)
def test_contract_rejects_unsupported_semantics_before_loading(tmp_path, field, value):
    contract = tensor_export(tmp_path / "adapter")
    contract.validate(tmp_path / "adapter")
    path = tmp_path / "adapter" / "adapter_config.json"
    config = json.loads(path.read_bytes())
    config[field] = value
    path.write_bytes(canonical(config))
    with pytest.raises(ValueError):
        contract.validate(path.parent)


def test_contract_rejects_shape_dtype_and_producer_base_conflicts(tmp_path):
    root = tmp_path / "adapter"
    contract = tensor_export(root)
    weights = root / "adapter_model.safetensors"
    data = weights.read_bytes()
    header_size = struct.unpack("<Q", data[:8])[0]
    index = json.loads(data[8 : 8 + header_size])
    next(iter(index.values()))["shape"] = [1, 8]
    encoded = canonical(index)
    weights.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data[8 + header_size :])
    with pytest.raises(ValueError, match="shape/dtype"):
        contract.validate(root)
    weights.write_bytes(data)
    wrong = ModelContract("b" * 64, contract.config, 4, ("q_proj",))
    with pytest.raises(ValueError, match="provenance"):
        wrong.validate(root)


def test_scheduler_drain_and_watermark_prevent_late_admission():
    ledger = SchedulerExecutions()
    rid = "a" * 32 + f"{1:016x}"
    assert ledger.admit(rid, "op")
    ledger.terminal(rid)
    assert not ledger.status(rid)["drained"]
    assert not ledger.fence_version("op")["drained"]
    ledger.drain_boundary()
    assert ledger.fence_version("op")["drained"]
    ledger.acknowledge(rid)
    assert not ledger.active
    assert not ledger.admit(rid, "another-op")
    assert ledger.watermarks["a" * 32].through == 1
    rid2 = "a" * 32 + f"{2:016x}"
    ledger.cancel_fence(rid2)
    assert ledger.status(rid2)["never_submitted"]
    assert not ledger.admit(rid2, "new-op")


class ObservedBackend(Backend):
    def __init__(self):
        super().__init__()
        self.drained = set()
        self.mode = "normal"

    async def generate(self, payload):
        rid = payload["rid"]
        if rid.startswith("warmup-") or self.mode == "normal":
            result = await super().generate(payload)
            self.drained.add(rid)
            return result
        self.generation_started.set()
        if self.mode == "lost":
            raise RuntimeError("Python delivery failed after physical submission")
        await asyncio.Event().wait()

    async def execution_status(self, rid):
        return {"drained": rid in self.drained}

    async def forget_execution(self, rid):
        assert rid in self.drained


def test_unknown_recovers_from_independent_late_scheduler_proof(tmp_path):
    async def scenario():
        backend = ObservedBackend()
        control = EngineControl(backend, engine_id="one", base_digest=BASE, capacity=2)
        a = artifact(tmp_path, "A")
        await control.prepare(a, "op")
        backend.mode = "lost"
        generating = asyncio.create_task(
            control.generate("op", {"rid": "r", "input_ids": [1], "lora_path": a.lora_path})
        )
        await backend.generation_started.wait()
        await asyncio.sleep(0.01)
        assert control.operations["op"].requests == {"r"}
        assert not generating.done()
        backend.drained.add("r")
        result = await asyncio.wait_for(generating, 2)
        assert result["status"] == "terminal_error" and result["code"] == "RESULT_LOST"
        assert not control.operations["op"].requests
        await control.retire(a, "op")
        assert backend.unloads["A"] == 1

    run(scenario)


def test_gateway_reconciles_even_when_generate_transport_never_returns(tmp_path):
    async def scenario():
        async with cohort(tmp_path) as (gateway, _, _, _):
            a = artifact(tmp_path, "A")
            await gateway.manager.publish(a)
            engine = gateway.engines["0"]
            original = engine.call
            delivered = asyncio.Event()

            async def lost_connection(action, body):
                result = await original(action, body)
                if action == "generate":
                    delivered.set()
                    await asyncio.Event().wait()
                return result

            engine.call = lost_connection
            generating = asyncio.create_task(gateway.generate(request()))
            await delivered.wait()
            assert not generating.done()
            await gateway.tick()
            assert (await asyncio.wait_for(generating, 1))["text"] == a.lora_path
            assert gateway.state()["unsettled_requests"] == 0
            gateway.requests["r1"].task.cancel()

    run(scenario)


def test_gateway_restart_retains_bindings_owner_and_unknown_references(tmp_path):
    async def scenario():
        async with cohort(tmp_path) as (initial, backends, _, _):
            journal = ControlJournal(tmp_path / "journal")
            owner = journal.owner_id
            gateway = PublicationGateway(
                list(initial.engines.values()), capacity=2, base_digest=BASE, store=tmp_path / "store", journal=journal
            )
            await gateway.manager.publish(artifact(tmp_path, "A"))
            await gateway.bind_owned("old")
            assert (await gateway.generate(request()))["text"].endswith("@A")
            await gateway.manager.publish(artifact(tmp_path, "B"))
            await journal.barrier()
            journal.close()
            journal = ControlJournal(tmp_path / "journal")
            assert journal.owner_id == owner
            recovered = PublicationGateway(
                list(initial.engines.values()), capacity=2, base_digest=BASE, store=tmp_path / "store", journal=journal
            )
            await recovered.recover()
            assert recovered.bind("old")["version_id"] == "A"
            assert recovered.bind("new")["version_id"] == "B"
            assert (await recovered.generate(request()))["text"].endswith("@A")
            assert sum(len(item.payloads) for item in backends) == 1
            await recovered.close("old")
            await recovered.manager.collect()
            await journal.barrier()
            journal.close()

    run(scenario)


def test_outbox_preserves_responsibility_across_producer_exit(tmp_path):
    path = SessionCloseOutbox(tmp_path, "http://gateway").enqueue("session")
    reader = SessionCloseOutbox(tmp_path, "http://gateway")
    assert reader.pending() == [(path, "session")]
    assert SessionCloseOutbox(tmp_path, "http://other").pending() == []
    reader.acknowledge(path)
    assert reader.pending() == []


def test_acknowledgement_ranges_stay_bounded_behind_one_unknown_request():
    from relax.engine.lora.execution import SequenceWatermark

    watermark = SequenceWatermark()
    for sequence in range(2, 10002):
        watermark.acknowledge(sequence)
    assert watermark.through == 0
    assert watermark.gaps == [(2, 10001)]
    watermark.acknowledge(1)
    assert watermark.through == 10001 and watermark.gaps == []


def test_terminal_journal_records_commit_together(tmp_path):
    async def scenario():
        journal = ControlJournal(tmp_path)
        journal.put("lease", "physical", {"x": 1})
        journal.put("dispatch", "logical", {"x": 1})
        await journal.barrier()
        transactions = []
        journal._db.set_trace_callback(transactions.append)
        journal.put_result(
            "logical",
            "digest",
            {"status": "terminal", "output": {}},
            terminal_records=[
                ("lease", "physical", {"x": 1}, False),
                ("dispatch", "logical", {"x": 1}, False),
            ],
        )
        await journal.barrier()
        assert sum(command.startswith("BEGIN") for command in transactions) == 1
        assert sum(command == "COMMIT" for command in transactions) == 1
        journal.close()
        recovered = ControlJournal(tmp_path)
        assert not recovered.records("lease") and not recovered.records("dispatch")
        assert recovered.get("result", "logical")["result"]["status"] == "terminal"
        recovered.close()

    run(scenario)


def test_result_quota_expires_output_without_reexecuting_identity(tmp_path):
    async def scenario():
        journal = ControlJournal(tmp_path, result_bytes=200)
        journal.put_result("r", "digest", {"status": "terminal", "output": {"text": "x" * 1000}})
        await journal.barrier()
        assert journal.get("result", "r") == {
            "digest": "digest",
            "result": {"status": "expired", "code": "RESULT_EXPIRED"},
        }
        journal.close()

    run(scenario)


def test_physical_prepare_permit_survives_http_timeout(tmp_path):
    import httpx

    async def scenario():
        async with cohort(tmp_path) as (gateway, backends, _, transports):
            gate = asyncio.Event()
            backends[0].load_gate["A"] = gate
            transport = transports[0]
            original = transport.handle_async_request
            owned = []

            async def lose_before_reply(request):
                if request.url.path.endswith("/prepare"):
                    task = asyncio.create_task(original(request))
                    owned.append(task)
                    await backends[0].started["A"].wait()
                    raise httpx.ReadTimeout("injected while physical load continues", request=request)
                return await original(request)

            transport.handle_async_request = lose_before_reply
            publishing = asyncio.create_task(gateway.manager.publish(artifact(tmp_path, "A")))
            await backends[0].started["A"].wait()
            await asyncio.sleep(0.05)
            assert not backends[1].started["A"].is_set()
            gate.set()
            await publishing
            await asyncio.gather(*owned)
            assert all(backend.loads["A"] == 1 for backend in backends)

    run(scenario)


def test_late_cancel_cannot_cancel_retry_operation(tmp_path):
    from relax.engine.lora.publication import AdapterPublicationError

    async def scenario():
        async with cohort(tmp_path) as (gateway, backends, _, _):
            a = artifact(tmp_path, "A")
            backends[1].fail.add("A")
            with pytest.raises(AdapterPublicationError):
                await gateway.manager.publish(a)
            old = gateway.manager.snapshot()["versions"]["A"]["operation_id"]
            backends[1].fail.clear()
            backends[1].load_gate["A"] = asyncio.Event()
            publishing = asyncio.create_task(gateway.manager.publish(a))
            while gateway.manager.snapshot()["versions"]["A"]["operation_id"] == old:
                await asyncio.sleep(0)
            assert gateway.manager.cancel_publication("A", old) is False
            backends[1].load_gate["A"].set()
            assert (await publishing).version_id == "A"

    run(scenario)


def test_invalid_generation_does_not_bind_session(tmp_path):
    async def scenario():
        async with cohort(tmp_path) as (gateway, _, _, _):
            await gateway.manager.publish(artifact(tmp_path, "A"))
            with pytest.raises(ValueError):
                await gateway.generate({**request(), "sampling_params": {"max_new_tokens": -1}})
            assert gateway.state()["versions"]["A"]["session_refs"] == 0

    run(scenario)


def test_dead_owner_cleanup_resumes_after_crash(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    state = ModuleType("ray.util.state")
    state.get_actor = lambda actor_id, **kw: SimpleNamespace(actor_id=actor_id, state="DEAD")
    monkeypatch.setitem(sys.modules, "ray.util.state", state)

    async def scenario():
        async with cohort(tmp_path) as (initial, _, _, _):
            journal = ControlJournal(tmp_path / "journal")
            gateway = PublicationGateway(
                list(initial.engines.values()), capacity=2, base_digest=BASE, store=tmp_path / "store", journal=journal
            )
            gateway.ray_address = "http://ray-state"
            gateway.close_outbox = SessionCloseOutbox(tmp_path / "outbox", "http://gateway")
            await gateway.manager.publish(artifact(tmp_path, "A"))
            owner = {"actor_id": "actor", "epoch": "boot"}
            for sid in ("one", "two"):
                await gateway.bind_owned(sid, owner, gateway.close_outbox.identity, gateway.close_outbox.gateway_url)
            close = gateway.close

            async def crash(sid):
                await close(sid)
                raise RuntimeError("simulated crash after the first durable close")

            gateway.close = crash
            with pytest.raises(RuntimeError, match="simulated crash"):
                await gateway.reconcile_owners()
            await journal.barrier()
            journal.close()
            journal = ControlJournal(tmp_path / "journal")
            restored = PublicationGateway(
                list(initial.engines.values()), capacity=2, base_digest=BASE, store=tmp_path / "store", journal=journal
            )
            restored.ray_address = "http://ray-state"
            restored.close_outbox = gateway.close_outbox
            await restored.recover()
            assert len(restored.manager._sessions) == 1
            await restored.reconcile_owners()
            assert not restored.manager._sessions
            from relax.engine.lora.publication import AdapterLeaseError

            with pytest.raises(AdapterLeaseError):
                await restored.bind_owned(
                    "late", owner, restored.close_outbox.identity, restored.close_outbox.gateway_url
                )
            journal.close()

    run(scenario)


def test_producer_seal_owns_bytes_after_source_changes(tmp_path, monkeypatch):
    from relax.engine.lora import cli

    source = tmp_path / "export"
    contract = tensor_export(source)
    model = tmp_path / "base"
    model.mkdir()
    (model / "config.json").write_bytes(canonical(contract.config))
    monkeypatch.setattr(cli, "producer_base_digest", lambda path: BASE)
    sealed = cli.seal_export(source, model, tmp_path / "sealed", "version", producer="test-job", export_step=42)
    expected = (sealed.path / "adapter_model.safetensors").read_bytes()
    (source / "adapter_model.safetensors").write_bytes(b"a later export replaced the source")
    sealed.verify()
    contract.validate(sealed.path)
    assert (sealed.path / "adapter_model.safetensors").read_bytes() == expected
    assert json.loads((sealed.path / "producer_manifest.json").read_bytes())["export_step"] == 42
