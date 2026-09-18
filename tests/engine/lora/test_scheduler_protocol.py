# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise scheduler hooks directly; only the optional SGLang IPC types are
stubbed."""

import ast
import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


class Message:
    def __init_subclass__(cls, **kwargs):
        pass

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Dispatcher:
    def __init__(self, handlers=()):
        self.handlers = dict(handlers)

    def __iadd__(self, other):
        self.handlers.update(other.handlers)
        return self

    def __call__(self, message):
        return self.handlers[type(message)](message)


def protocol():
    source = Path(__file__).parents[3] / "relax/engine/lora/sglang_protocol.py"
    module = ast.parse(source.read_text())
    module.body = [
        node
        for node in module.body
        if not (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("sglang"))
    ]
    scope = {
        "AbortReq": type("Abort", (Message,), {}),
        "BaseReq": Message,
        "LoadLoRAAdapterReqInput": Message,
        "UnloadLoRAAdapterReqInput": Message,
        "hook_custom_types": lambda *types: None,
        "TypeBasedDispatcher": Dispatcher,
    }
    exec(compile(module, str(source), "exec"), scope)
    return SimpleNamespace(**scope)


def scheduler_type(api):
    @dataclass(frozen=True)
    class Channels:
        send_to_tokenizer: object
        send_to_detokenizer: object

    class Scheduler:
        def __init__(self):
            self.server_args = SimpleNamespace(disable_overlap_schedule=True)
            self.sent = []
            self.queue = []
            self.running = False
            sender = SimpleNamespace(send_output=lambda output, *args, **kwargs: self.sent.append(output))
            self.ipc_channels = Channels(sender, sender)
            self.output_streamer = SimpleNamespace(send_to_detokenizer=sender)
            self._request_dispatcher = Dispatcher()

        def handle_generate_request(self, request):
            self.queue.append(request.rid)

        def process_input_requests(self, requests):
            for request in requests:
                self.handle_generate_request(request)

        def process_batch_result(self):
            output = SimpleNamespace(rids=list(self.queue), finished_reasons=[{"type": "stop"}] * len(self.queue))
            self.output_streamer.send_to_detokenizer.send_output(output)
            # Observing the result before this boundary is insufficient.
            assert all(not self.relax_executions.status(rid)["drained"] for rid in self.queue)
            self.queue.clear()

        def abort_request(self, request):
            if request.rid in self.queue and not self.running:
                self.queue.remove(request.rid)
                self.ipc_channels.send_to_tokenizer.send_output(api.AbortReq(rid=request.rid), request)

    api.install_scheduler_contract(Scheduler)
    return Scheduler


def test_duplicate_scheduler_delivery_cannot_forge_terminal_proof():
    api = protocol()
    scheduler = scheduler_type(api)()
    rid = "a" * 32 + f"{1:016x}"
    request = SimpleNamespace(rid=rid, lora_id="relax-version-op")
    scheduler.process_input_requests([request, request])
    assert scheduler.queue == [rid]
    assert not scheduler.sent
    assert not scheduler.relax_executions.status(rid)["drained"]
    scheduler.process_batch_result()
    assert scheduler.relax_executions.status(rid)["drained"]


@pytest.mark.parametrize("running", [False, True])
def test_cancel_proof_waits_for_actual_scheduler_boundary(running):
    api = protocol()
    scheduler = scheduler_type(api)()
    rid = "a" * 32 + f"{1:016x}"
    scheduler.process_input_requests([SimpleNamespace(rid=rid, lora_id="relax-version-op")])
    scheduler.running = running
    api.command_handler(scheduler, api.ExecutionCommand(rid="command", action="cancel", execution_id=rid))
    assert scheduler.relax_executions.status(rid)["drained"] == (not running)
    if running:
        scheduler.process_batch_result()
        assert scheduler.relax_executions.status(rid)["drained"]


def test_correlated_reply_cannot_complete_a_different_command():
    api = protocol()

    async def scenario():
        messages = []
        manager = SimpleNamespace(
            _result_dispatcher=Dispatcher(),
            auto_create_handle_loop=lambda: None,
            _dispatch_to_scheduler=messages.append,
        )
        channel = api.SchedulerChannel(manager)
        with pytest.raises(asyncio.TimeoutError):
            await channel.call("status", execution_id="one", timeout=0.001)
        late = api.ExecutionReply(rid=messages[0].rid, scheduler_epoch="boot", body='{"drained":true}')
        next_call = asyncio.create_task(channel.call("load_status", operation_id="op"))
        await asyncio.sleep(0)
        channel.receive(late)
        assert not next_call.done()
        channel.receive(api.ExecutionReply(rid=messages[1].rid, scheduler_epoch="boot", body='{"success":true}'))
        assert await next_call == {"success": True}

    asyncio.run(scenario())
