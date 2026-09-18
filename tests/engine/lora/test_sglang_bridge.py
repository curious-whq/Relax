# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check the version-pinned tokenizer bridge without importing GPU
libraries."""

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from relax.engine.lora.control import BackendRejectedError
from relax.engine.lora.sglang_server import SGLangLocalBackend, remove_managed_adapter, validate_server_args


class GenerateInput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TokenizedInput(GenerateInput):
    pass


class Aborted(ValueError):
    pass


@pytest.fixture
def stub_sglang(monkeypatch):
    io = ModuleType("sglang.srt.managers.io_struct")
    io.GenerateReqInput = GenerateInput
    io.TokenizedGenerateReqInput = TokenizedInput
    io.UnloadLoRAAdapterReqInput = GenerateInput
    tm = ModuleType("sglang.srt.managers.tokenizer_manager")
    tm.RequestAbortedError = Aborted
    monkeypatch.setitem(sys.modules, io.__name__, io)
    monkeypatch.setitem(sys.modules, tm.__name__, tm)
    protocol = ModuleType("relax.engine.lora.sglang_protocol")
    protocol.SchedulerChannel = lambda manager: SimpleNamespace(call=AsyncMock(return_value={"drained": False}))
    monkeypatch.setitem(sys.modules, protocol.__name__, protocol)


@pytest.mark.parametrize("mode", ["reject", "abort_before_send", "unknown", "terminal_abort", "complete"])
def test_bridge_preserves_submission_and_terminal_evidence(stub_sglang, mode):
    async def scenario():
        ref = SimpleNamespace(lora_id="relax-version-op")
        registry = SimpleNamespace(
            get_all_adapters=lambda: {"relax_policy@A": ref},
            release=AsyncMock(),
            acquire=AsyncMock(return_value=ref.lora_id),
        )
        manager = SimpleNamespace(
            lora_registry=registry,
            _validate_and_resolve_lora=AsyncMock(),
            _dispatch_to_scheduler=Mock(),
            _handle_abort_finish_reason=AsyncMock(),
        )
        release = registry.release

        async def generate(obj, request):
            await manager._validate_and_resolve_lora(obj)
            assert obj.lora_id == ref.lora_id
            if mode == "reject":
                raise ValueError("bad tokens")
            if mode == "abort_before_send":
                raise Aborted("cancelled during tokenization")
            manager._dispatch_to_scheduler(TokenizedInput(rid=obj.rid))
            if mode == "unknown":
                raise RuntimeError("lost scheduler communication")
            reason = {"type": "abort", "status_code": 503} if mode == "terminal_abort" else {"type": "stop"}
            output = {"output_ids": [], "meta_info": {"finish_reason": reason}}
            if mode == "terminal_abort":
                assert await manager._handle_abort_finish_reason(output, SimpleNamespace(obj=obj), False) == output
            await registry.release(obj.lora_id)
            yield output

        manager.generate_request = generate
        backend = SGLangLocalBackend(lambda: manager)
        body = {"rid": "r", "input_ids": [1], "lora_path": "relax_policy@A"}
        if mode == "reject":
            with pytest.raises(BackendRejectedError, match="bad tokens"):
                await backend.generate(body)
        elif mode == "unknown":
            with pytest.raises(RuntimeError):
                await backend.generate(body)
            assert len(backend._requests) == 1
            assert not (await backend.execution_status("r"))["drained"]
        else:
            result = await backend.generate(body)
            assert result["meta_info"]["finish_reason"]["type"] == ("stop" if mode == "complete" else "abort")
        release.assert_not_awaited()
        assert len(backend._requests) == 1
        backend._channel.call.return_value = {"drained": True}
        assert (await backend.execution_status("r"))["drained"]
        release.assert_awaited_once_with(ref.lora_id)
        await backend.forget_execution("r")
        assert backend._requests == {}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "body",
    [
        {"input_ids": [[1], [2]]},
        {"text": ["one", "two"]},
        {"input_ids": [1], "sampling_params": {"n": 2}},
        {"input_ids": [1], "text": "both"},
        {"input_ids": [1], "unknown_field": True},
    ],
)
def test_bridge_rejects_batch_and_unknown_fields_before_dispatch(stub_sglang, body):
    async def scenario():
        manager = SimpleNamespace(generate_request=Mock())
        backend = SGLangLocalBackend(lambda: manager)
        with pytest.raises(BackendRejectedError):
            await backend.generate({"rid": "r", "lora_path": "relax_policy@A", **body})
        manager.generate_request.assert_not_called()

    asyncio.run(scenario())


def test_partial_worker_load_cleanup_is_idempotent():
    ref = SimpleNamespace(lora_id="relax-version-op", pinned=True)
    slots = {ref.lora_id: 1}
    event = Mock()
    manager = SimpleNamespace(
        pending_lora_load_events={ref.lora_id: event},
        memory_pool=SimpleNamespace(remove_lora=lambda uid: slots.pop(uid, None)),
        _notify_lora_slots_updated=Mock(),
        configs={ref.lora_id: {}},
        loras={ref.lora_id: object()},
        lora_refs={},
        num_pinned_loras=0,
        create_lora_update_result=lambda **kwargs: kwargs,
    )
    assert remove_managed_adapter(manager, ref) == {"success": True}
    assert remove_managed_adapter(manager, ref) == {"success": True}
    assert not manager.configs and not manager.loras and not slots
    assert manager.num_pinned_loras == 0
    event.synchronize.assert_called_once()
    manager._notify_lora_slots_updated.assert_called_once_with({1})


@pytest.mark.parametrize("unsupported", ["grpc_port", "smg_grpc_mode", "encoder_only", "enable_lora_overlap_loading"])
def test_wrapper_rejects_unsupported_transports_and_loading(tmp_path, unsupported):
    args = SimpleNamespace(
        disable_overlap_schedule=True,
        enable_lora=True,
        max_loras_per_batch=3,
        max_loaded_loras=3,
        model_path=str(tmp_path),
    )
    validate_server_args(args, 2)
    setattr(args, unsupported, True)
    with pytest.raises(ValueError, match=unsupported):
        validate_server_args(args, 2)


def test_cancelled_dispatched_warmup_prevents_unload(stub_sglang):
    async def scenario():
        submitted = asyncio.Event()
        ref = SimpleNamespace(lora_id="relax-version-op", lora_name="relax_policy@A")
        manager = SimpleNamespace(
            lora_registry=SimpleNamespace(
                get_all_adapters=lambda: {ref.lora_name: ref},
                release=AsyncMock(),
                acquire=AsyncMock(return_value=ref.lora_id),
            ),
            _validate_and_resolve_lora=AsyncMock(),
            _dispatch_to_scheduler=Mock(),
            _handle_abort_finish_reason=AsyncMock(),
        )

        async def generate(obj, request):
            await manager._validate_and_resolve_lora(obj)
            manager._dispatch_to_scheduler(TokenizedInput(rid=obj.rid))
            submitted.set()
            await asyncio.Event().wait()
            yield {}

        manager.generate_request = generate
        backend = SGLangLocalBackend(lambda: manager)
        backend._refs["op"] = ref
        task = asyncio.create_task(
            backend.generate({"rid": "warmup-op", "input_ids": [0], "lora_path": ref.lora_name})
        )
        await submitted.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(backend._requests) == 1
        with pytest.raises(RuntimeError, match="not drained"):
            await backend.unload(None, "op")
        assert "op" in backend._refs

    asyncio.run(scenario())
