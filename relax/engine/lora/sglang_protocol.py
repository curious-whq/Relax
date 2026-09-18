# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Explicit scheduler drain protocol for the pinned SGLang 0.5.17 backend.

Imported only by the managed SGLang processes. This interface is deliberately
versioned; the HTTP control plane checks capabilities before claiming an
engine.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any
from uuid import uuid4

from sglang.srt.managers.io_struct import (
    AbortReq,
    BaseReq,
    LoadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqInput,
    hook_custom_types,
)
from sglang.utils import TypeBasedDispatcher

from relax.engine.lora.execution import CAPABILITY, SchedulerExecutions


class ExecutionCommand(BaseReq, kw_only=True):
    action: str
    execution_id: str = ""
    operation_id: str = ""
    artifact_path: str = ""
    lora_name: str = ""


class ExecutionReply(BaseReq, kw_only=True):
    scheduler_epoch: str
    body: str


hook_custom_types(ExecutionCommand, ExecutionReply)


class ObservedSender:
    def __init__(self, sender: Any, ledger: SchedulerExecutions) -> None:
        self.sender = sender
        self.ledger = ledger

    def send_output(self, output: Any, *args: Any, **kwargs: Any) -> Any:
        # Observe facts before transport delivery. Drain is acknowledged only
        # after the enclosing scheduler operation returns at a safe boundary.
        if isinstance(output, AbortReq):
            self.ledger.terminal(output.rid)
        else:
            for rid, reason in zip(
                getattr(output, "rids", None) or (), getattr(output, "finished_reasons", None) or ()
            ):
                if reason is not None:
                    self.ledger.terminal(rid)
        return self.sender.send_output(output, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.sender, name)


def install_scheduler_contract(scheduler_class: Any) -> None:
    original_init = scheduler_class.__init__
    original_admit = scheduler_class.handle_generate_request
    original_process = scheduler_class.process_batch_result
    original_inputs = scheduler_class.process_input_requests

    def initialize(scheduler: Any, *args: Any, **kwargs: Any) -> None:
        scheduler.relax_executions = SchedulerExecutions()
        scheduler.relax_epoch = uuid4().hex
        scheduler.relax_load_results = {}
        original_init(scheduler, *args, **kwargs)
        if not scheduler.server_args.disable_overlap_schedule:
            raise RuntimeError("execution-drain v2 requires non-overlap scheduling")
        ledger = scheduler.relax_executions
        channels = scheduler.ipc_channels
        scheduler.ipc_channels = replace(
            channels,
            send_to_tokenizer=ObservedSender(channels.send_to_tokenizer, ledger),
            send_to_detokenizer=ObservedSender(channels.send_to_detokenizer, ledger),
        )
        scheduler.output_streamer.send_to_detokenizer = scheduler.ipc_channels.send_to_detokenizer
        scheduler._request_dispatcher += TypeBasedDispatcher(
            [(ExecutionCommand, lambda command: command_handler(scheduler, command))]
        )

    def admit(scheduler: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        uid = request.lora_id
        if isinstance(uid, str) and uid.startswith("relax-version-"):
            if request.rid in scheduler.relax_executions.active:
                return None  # Duplicate delivery cannot terminate the original work.
            if not scheduler.relax_executions.admit(request.rid, uid):
                scheduler.relax_executions.cancel_fence(request.rid)
                scheduler.ipc_channels.send_to_tokenizer.send_output(AbortReq(rid=request.rid), request)
                return None
        return original_admit(scheduler, request, *args, **kwargs)

    def process(scheduler: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_process(scheduler, *args, **kwargs)
        # Non-overlap TP=1 ordinary text forward has completed its result copy.
        # An exception does NOT create a drain proof.
        scheduler.relax_executions.drain_boundary()
        return result

    def inputs(scheduler: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_inputs(scheduler, *args, **kwargs)
        scheduler.relax_executions.drain_boundary()
        return result

    scheduler_class.__init__ = initialize
    scheduler_class.handle_generate_request = admit
    scheduler_class.process_batch_result = process
    scheduler_class.process_input_requests = inputs


def command_handler(scheduler: Any, command: ExecutionCommand) -> ExecutionReply:
    ledger = scheduler.relax_executions
    if command.action == "capabilities":
        result = {"capability": CAPABILITY, "non_overlap": True}
    elif command.action == "load":
        if command.operation_id in ledger.fenced_operations:
            result = {"success": False, "error": "operation fenced"}
        elif command.operation_id in scheduler.relax_load_results:
            result = scheduler.relax_load_results[command.operation_id]
        else:
            try:
                reply = scheduler.load_lora_adapter(
                    LoadLoRAAdapterReqInput(
                        lora_name=command.lora_name,
                        lora_path=command.artifact_path,
                        lora_id=command.operation_id,
                        pinned=True,
                    )
                )
                result = {"success": bool(reply.success), "message": str(reply.message)}
            except Exception as error:
                result = {"success": False, "message": str(error)}
            scheduler.relax_load_results[command.operation_id] = result
    elif command.action == "load_status":
        result = scheduler.relax_load_results.get(command.operation_id, {"success": False, "message": "not loaded"})
    elif command.action == "unload":
        proof = ledger.fence_version(command.operation_id)
        if not proof["drained"]:
            result = {"success": False, "message": "execution is not drained"}
        else:
            reply = scheduler.unload_lora_adapter(
                UnloadLoRAAdapterReqInput(
                    lora_name=command.lora_name,
                    lora_id=command.operation_id,
                )
            )
            result = {"success": bool(reply.success), "message": str(reply.message)}
    elif command.action == "forget_version":
        if ledger.by_operation.get(command.operation_id):
            result = {"error": "version is not drained"}
        else:
            # Single tokenizer sender: all previously sent work precedes this
            # command in FIFO IPC. The producer retains a durable operation fence.
            ledger.fenced_operations.discard(command.operation_id)
            ledger.by_operation.pop(command.operation_id, None)
            scheduler.relax_load_results.pop(command.operation_id, None)
            result = {"forgotten": True}
    elif command.action == "residency":
        manager = scheduler.tp_worker.model_runner.lora_manager
        ref = manager.lora_refs.get(command.operation_id)
        result = {
            "resident": ref is not None
            and ref.pinned
            and command.operation_id in manager.memory_pool.uid_to_buffer_id
            and command.operation_id not in ledger.fenced_operations
        }
    elif command.action == "status":
        result = ledger.status(command.execution_id)
    elif command.action == "cancel":
        ledger.cancel_fence(command.execution_id)
        scheduler.abort_request(AbortReq(rid=command.execution_id))
        ledger.drain_boundary()
        result = ledger.status(command.execution_id)
    elif command.action == "fence_version":
        result = ledger.fence_version(command.operation_id)
    elif command.action == "acknowledge":
        ledger.acknowledge(command.execution_id)
        result = {"acknowledged": True}
    else:
        result = {"error": "unknown scheduler execution command"}
    return ExecutionReply(rid=command.rid, scheduler_epoch=scheduler.relax_epoch, body=json.dumps(result))


class SchedulerChannel:
    """Correlated control messages; a late RPC reply cannot satisfy another
    call."""

    def __init__(self, manager: Any) -> None:
        self.manager = manager
        self.waiters: dict[str, asyncio.Future] = {}
        self.epoch: str | None = None
        manager._result_dispatcher += TypeBasedDispatcher([(ExecutionReply, self.receive)])

    def receive(self, response: ExecutionReply) -> None:
        waiter = self.waiters.get(response.rid)
        if waiter is not None and not waiter.done():
            waiter.set_result(response)

    async def call(
        self,
        action: str,
        *,
        execution_id: str = "",
        operation_id: str = "",
        timeout: float = 5,
        artifact_path: str = "",
        lora_name: str = "",
    ) -> dict[str, Any]:
        rid = uuid4().hex
        waiter = asyncio.get_running_loop().create_future()
        self.waiters[rid] = waiter
        try:
            self.manager.auto_create_handle_loop()
            self.manager._dispatch_to_scheduler(
                ExecutionCommand(
                    rid=rid,
                    action=action,
                    execution_id=execution_id,
                    operation_id=operation_id,
                    artifact_path=artifact_path,
                    lora_name=lora_name,
                )
            )
            response = await asyncio.wait_for(asyncio.shield(waiter), timeout)
            if self.epoch is not None and self.epoch != response.scheduler_epoch:
                raise RuntimeError("scheduler incarnation changed; old drain proofs are invalid")
            self.epoch = response.scheduler_epoch
            result = json.loads(response.body)
            if "error" in result:
                raise RuntimeError(result["error"])
            return result
        finally:
            self.waiters.pop(rid, None)
