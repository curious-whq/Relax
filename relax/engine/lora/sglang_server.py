# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Launch a dedicated SGLang 0.5.17 engine with fenced LoRA endpoints."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from relax.engine.lora.artifact import ModelContract, read_object
from relax.engine.lora.control import BackendRejectedError, EngineControl, aborted_output
from relax.engine.lora.execution import CAPABILITY
from relax.engine.lora.http import install_engine_routes
from relax.engine.lora.snapshot import AdapterSnapshot, fingerprint_model


def remove_managed_adapter(manager: Any, ref: Any) -> Any:
    """Idempotent cleanup including a failed load's partial GPU/CPU
    allocation."""
    uid = ref.lora_id
    pending = getattr(manager, "pending_lora_load_events", {})
    event = pending.get(uid)
    if event is not None:
        event.synchronize()
        pending.pop(uid, None)
    slot = manager.memory_pool.remove_lora(uid)
    if slot is not None:
        manager._notify_lora_slots_updated({slot})
    registered = manager.lora_refs.pop(uid, None)
    manager.configs.pop(uid, None)
    manager.loras.pop(uid, None)
    if registered is not None:
        manager.num_pinned_loras -= int(registered.pinned)
    return manager.create_lora_update_result(success=True)


def run_managed_scheduler(*args: Any, **kwargs: Any) -> Any:
    from sglang.srt.lora.lora_manager import LoRAManager
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process

    from relax.engine.lora.sglang_protocol import install_scheduler_contract

    install_scheduler_contract(Scheduler)

    original = LoRAManager._unload_lora_adapter

    def unload(manager: Any, ref: Any) -> Any:
        if ref.lora_id.startswith("relax-version-"):
            return remove_managed_adapter(manager, ref)
        return original(manager, ref)

    LoRAManager._unload_lora_adapter = unload
    return run_scheduler_process(*args, **kwargs)


class SGLangLocalBackend:
    def __init__(self, manager_getter: Any, contract: ModelContract | None = None) -> None:
        self._get_manager = manager_getter
        self._refs: dict[str, Any] = {}
        self._requests: dict[str, dict[str, Any]] = {}
        self.contract = contract
        self._epoch = uuid4().hex
        self._sequence = 0
        self._by_external: dict[str, str] = {}
        self._channel: Any = None
        self._release: Any = None
        self._hooks_installed = False

    def _install_request_hooks(self, manager: Any) -> None:
        """Own managed LoRA references and preserve the scheduler dispatch
        boundary.

        Managed executions acquire a native registry reference before
        tokenization and release it only after independent scheduler drain
        proof. Stock result delivery must not release this reference early or
        twice.
        """
        if self._hooks_installed:
            return
        from sglang.srt.managers.io_struct import TokenizedGenerateReqInput

        original_resolve = manager._validate_and_resolve_lora
        original_release = manager.lora_registry.release
        self._release = original_release
        original_dispatch = manager._dispatch_to_scheduler
        original_finish = manager._handle_abort_finish_reason

        async def resolve(obj: Any) -> None:
            if isinstance(obj.lora_path, str) and obj.lora_path.startswith("relax_policy@"):
                ref = manager.lora_registry.get_all_adapters().get(obj.lora_path)
                if ref is None or not ref.lora_id.startswith("relax-version-"):
                    raise ValueError("bound immutable adapter is missing")
                obj.lora_id = ref.lora_id
            else:
                await original_resolve(obj)

        async def release(uid: Any) -> None:
            if isinstance(uid, str) and uid.startswith("relax-version-"):
                return
            await original_release(uid)

        def dispatch(obj: Any) -> Any:
            evidence = self._requests.get(getattr(obj, "rid", None))
            if evidence is not None and isinstance(obj, TokenizedGenerateReqInput):
                if evidence["fenced"]:
                    raise BackendRejectedError("execution fenced before scheduler dispatch")
                # Mark before calling transport: a failed send is uncertain.
                evidence["dispatched"] = True
            return original_dispatch(obj)

        async def finish(out: dict[str, Any], state: Any, is_stream: bool) -> Any:
            if state.obj.rid in self._requests:
                # Scheduler already supplied a terminal output. Preserve it even
                # for abort status 400/500/503, which stock nonstream APIs raise.
                if out.get("meta_info", {}).get("finish_reason", {}).get("type") == "abort":
                    return out
            return await original_finish(out, state, is_stream)

        manager._validate_and_resolve_lora = resolve
        manager.lora_registry.release = release
        manager._dispatch_to_scheduler = dispatch
        manager._handle_abort_finish_reason = finish
        from relax.engine.lora.sglang_protocol import SchedulerChannel

        self._channel = SchedulerChannel(manager)
        self._hooks_installed = True

    async def capabilities(self) -> dict[str, Any]:
        self._install_request_hooks(self._get_manager())
        result = await self._channel.call("capabilities")
        if result.get("capability") != CAPABILITY:
            raise RuntimeError("missing scheduler execution/drain contract")
        return result

    async def prepared_status(self, operation_id: str) -> bool:
        ref = self._refs.get(operation_id)
        if ref is None:
            return False
        result = await self._channel.call("residency", operation_id=ref.lora_id)
        return bool(result.get("resident"))

    async def physical_barrier(self) -> None:
        # The sole tokenizer's FIFO channel places this response after all
        # previously submitted synchronous load/unload handlers have returned.
        await self.capabilities()

    async def load(self, artifact: AdapterSnapshot, operation_id: str) -> None:
        from sglang.srt.lora.lora_registry import LoRARef

        if self.contract is None:
            raise RuntimeError("an explicit adapter semantic contract is required")
        await asyncio.to_thread(self.contract.validate, artifact.path)
        await self.capabilities()
        manager = self._get_manager()
        manager.auto_create_handle_loop()
        ref = LoRARef(
            lora_id=f"relax-version-{operation_id}",
            lora_name=artifact.lora_path,
            lora_path=str(artifact.path),
            pinned=True,
        )
        # Retain the ID before communicating: even an unsuccessful load may have
        # allocated worker resources without registering its public name.
        self._refs[operation_id] = ref
        async with manager.lora_update_lock:
            try:
                result = await self._channel.call(
                    "load", operation_id=ref.lora_id, artifact_path=ref.lora_path, lora_name=ref.lora_name
                )
            except asyncio.TimeoutError:
                # A correlated query queued after load proves that the scheduler
                # finished its physical load even if the original ACK was lost.
                while True:
                    try:
                        result = await self._channel.call("load_status", operation_id=ref.lora_id)
                        break
                    except asyncio.TimeoutError:
                        await asyncio.sleep(0.1)
            if not result.get("success"):
                raise RuntimeError(f"SGLang adapter load failed: {result}")
            await manager.lora_registry.register(ref)
            # No automatic disk backfill: a missing immutable version fails closed.

    async def unload(self, artifact: AdapterSnapshot, operation_id: str) -> None:
        ref = self._refs.get(operation_id)
        if ref is None:
            return  # prepare was fenced before load was submitted.
        proof = await self._channel.call("fence_version", operation_id=ref.lora_id)
        if not proof.get("fenced") or not proof.get("drained"):
            raise RuntimeError("scheduler has not drained all adapter executions")
        if any(item["uid"] == ref.lora_id and not item["released"] for item in self._requests.values()):
            raise RuntimeError("adapter still has native execution references")
        manager = self._get_manager()
        async with manager.lora_update_lock:
            registered = manager.lora_registry.get_all_adapters().get(ref.lora_name)
            if registered is not None:
                if registered.lora_id != ref.lora_id:
                    raise RuntimeError("adapter registration belongs to another operation")
                await manager.lora_registry.unregister(ref.lora_name)
                await manager.lora_registry.wait_for_unload(ref.lora_id)
            result = await self._channel.call("unload", operation_id=ref.lora_id, lora_name=ref.lora_name)
            if not result.get("success"):
                raise RuntimeError(f"SGLang adapter cleanup failed: {result}")
            await self._channel.call("forget_version", operation_id=ref.lora_id)
            self._refs.pop(operation_id, None)

    async def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        from sglang.srt.managers.io_struct import GenerateReqInput
        from sglang.srt.managers.tokenizer_manager import RequestAbortedError

        manager = self._get_manager()
        allowed = {
            "rid",
            "input_ids",
            "text",
            "sampling_params",
            "return_logprob",
            "logprob_start_len",
            "top_logprobs_num",
            "token_ids_logprob",
            "lora_path",
            "stream",
        }
        if set(payload) - allowed:
            raise BackendRejectedError(f"unsupported immutable generation fields: {sorted(set(payload) - allowed)}")
        params = payload.get("sampling_params", {})
        ids = payload.get("input_ids")
        text = payload.get("text")
        if (
            not isinstance(params, dict)
            or params.get("n", 1) != 1
            or (ids is not None and (not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids)))
            or (text is not None and not isinstance(text, str))
            or (ids is None) == (text is None)
            or payload.get("stream", False)
        ):
            raise BackendRejectedError("exactly one prompt and sampling_params.n=1 are required")
        self._install_request_hooks(manager)
        external = payload["rid"]
        if external in self._by_external:
            raise BackendRejectedError("duplicate backend admission")
        self._sequence += 1
        rid = f"{self._epoch}{self._sequence:016x}"
        evidence = {"dispatched": False, "fenced": False, "released": False, "uid": None, "done": False}
        self._requests[rid] = evidence
        self._by_external[external] = rid
        output = None
        try:
            evidence["uid"] = await manager.lora_registry.acquire(payload["lora_path"])
            async for item in manager.generate_request(GenerateReqInput(**{**payload, "rid": rid}), None):
                output = item
            if not isinstance(output, dict) or not output.get("meta_info", {}).get("finish_reason"):
                raise RuntimeError("expected one scheduler terminal generation result")
            return output
        except Exception as error:
            if not evidence["dispatched"]:
                evidence["fenced"] = True
                if isinstance(error, RequestAbortedError):
                    return aborted_output()
                raise BackendRejectedError(str(error)) from error
            raise
        finally:
            evidence["done"] = True
            if evidence.get("acknowledged"):
                self._requests.pop(rid, None)
                self._by_external.pop(external, None)

    async def execution_status(self, request_id: str) -> dict[str, Any]:
        rid = self._by_external.get(request_id)
        if rid is None:
            return {"drained": True, "never_submitted": True}
        evidence = self._requests[rid]
        if evidence["done"] and not evidence["dispatched"]:
            evidence["fenced"] = True
        if evidence["uid"] is None and not evidence["done"]:
            return {"drained": False}
        action = "cancel" if evidence["fenced"] else "status"
        proof = await self._channel.call(action, execution_id=rid)
        if proof.get("drained"):
            # Prevent any suspended tokenization coroutine from dispatching late.
            evidence["fenced"] = True
            if not evidence["released"]:
                # Concurrent status callers must not double-decrement the native
                # counter. A single owned release task survives caller cancellation.
                task = evidence.get("release_task")
                if task is None:
                    from relax.engine.lora.control import owned_task

                    async def release() -> None:
                        if evidence["uid"] is not None:
                            await self._release(evidence["uid"])
                        evidence["released"] = True

                    task = evidence["release_task"] = owned_task(release())
                await asyncio.shield(task)
        return proof

    async def abort(self, request_id: str) -> None:
        rid = self._by_external.get(request_id)
        if rid is None:
            return  # EngineControl's pre-admission tombstone handles this race.
        evidence = self._requests[rid]
        evidence["fenced"] = True
        self._get_manager().abort_request(rid)
        await self._channel.call("cancel", execution_id=rid)

    async def forget_execution(self, request_id: str) -> None:
        rid = self._by_external.get(request_id)
        if rid is None:
            return
        if not self._requests[rid]["released"]:
            raise RuntimeError("execution has not released its native reference")
        await self._channel.call("acknowledge", execution_id=rid)
        self._requests[rid]["acknowledged"] = True
        # Keep the producer fence until its coroutine has exited; no late send
        # can bypass the scheduler's compacted acknowledgement watermark.
        if self._requests[rid]["done"]:
            del self._requests[rid]
            del self._by_external[request_id]

    async def close_session(self, session_id: str) -> None:
        # Managed Session references live in the gateway. Ordinary version-keyed
        # KV entries age out without a global flush.
        return None


def validate_server_args(args: Any, capacity: int) -> None:
    if not getattr(args, "disable_overlap_schedule", False):
        raise ValueError("immutable LoRA requires --disable-overlap-schedule for GPU drain proofs")
    if capacity < 1 or not args.enable_lora:
        raise ValueError("positive --version-capacity and --enable-lora are required")
    for name in ("tp_size", "dp_size", "pp_size", "nnodes", "tokenizer_worker_num"):
        if getattr(args, name, 1) != 1:
            raise ValueError(f"immutable LoRA currently requires {name}=1")
    for name in (
        "speculative_algorithm",
        "enable_hierarchical_cache",
        "enable_lora_overlap_loading",
        "use_ray",
        "grpc_port",
        "grpc_mode",
        "smg_grpc_mode",
        "encoder_only",
    ):
        if getattr(args, name, None):
            raise ValueError(f"immutable LoRA does not support {name}")
    if os.environ.get("SGLANG_RUST_SERVER", "0").lower() not in {"0", "false", ""}:
        raise ValueError("immutable LoRA requires the Python HTTP server")
    if getattr(args, "disaggregation_mode", "null") not in (None, "null"):
        raise ValueError("immutable LoRA requires a non-disaggregated engine")
    if getattr(args, "lora_paths", None) or getattr(args, "load_format", "auto") == "dummy":
        raise ValueError("start with real base weights and no preloaded adapters")
    if args.max_loras_per_batch < capacity + 1 or (args.max_loaded_loras or 0) < capacity + 1:
        raise ValueError("max-loras-per-batch and max-loaded-loras must be at least version-capacity + 1")
    if not Path(args.model_path).is_dir():
        raise ValueError("immutable LoRA requires a local, frozen base-model directory")
    tokenizer_path = getattr(args, "tokenizer_path", None)
    if tokenizer_path and Path(tokenizer_path).resolve() != Path(args.model_path).resolve():
        raise ValueError("the tokenizer must belong to the fingerprinted base artifact")
    if getattr(args, "trust_remote_code", False) or getattr(args, "json_model_override_args", "{}") not in (
        None,
        "{}",
    ):
        raise ValueError("custom model code/config overrides are outside the immutable base contract")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--engine-id", required=True)
    parser.add_argument("--version-capacity", type=int, default=2)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--request-capacity", type=int, default=32)
    local, remaining = parser.parse_known_args()
    import sglang
    from sglang.srt.entrypoints import http_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    if sglang.__version__.split("+")[0] != "0.5.17":
        raise RuntimeError("immutable LoRA engine integration is pinned to SGLang 0.5.17")
    args = prepare_server_args(remaining)
    validate_server_args(args, local.version_capacity)
    base_digest = fingerprint_model(args.model_path)
    contract = ModelContract(
        base_digest,
        read_object(Path(args.model_path) / "config.json"),
        args.max_lora_rank or 0,
        tuple(args.lora_target_modules or ()),
    )
    if getattr(args, "quantization", None):
        raise ValueError("contract v1 requires an unquantized base")
    backend = SGLangLocalBackend(lambda: http_server._global_state.tokenizer_manager, contract)
    from relax.engine.lora.journal import ControlJournal

    journal = ControlJournal(local.state_dir)
    control = EngineControl(
        backend,
        engine_id=local.engine_id,
        base_digest=base_digest,
        capacity=local.version_capacity,
        journal=journal,
        request_capacity=local.request_capacity,
    )
    install_engine_routes(http_server.app, control)
    from relax.engine.lora.access import install_access

    install_access(http_server.app, engine=True)

    @http_server.app.middleware("http")
    async def protect_managed_engine(request: Any, call_next: Any) -> Any:
        from fastapi.responses import JSONResponse

        path = request.url.path
        allowed_reads = {"/health", "/model_info", "/server_info", "/get_model_info", "/get_server_info", "/metrics"}
        if path.startswith("/relax/lora/"):
            from sglang.srt.managers.tokenizer_manager import ServerStatus

            manager = http_server._global_state.tokenizer_manager
            if manager.server_status != ServerStatus.Up:
                return JSONResponse({"error": "engine startup warmup is incomplete"}, status_code=503)
        # Before claim, permit native warmup generation. After claim all model
        # requests and all mutations must go through the versioned protocol.
        warmup = control.owner_id is None and path in {"/generate", "/v1/chat/completions"}
        if not path.startswith("/relax/lora/") and path not in allowed_reads and not warmup:
            return JSONResponse(
                {"error": "engine is exclusively managed by immutable LoRA publication"}, status_code=409
            )
        return await call_next(request)

    try:
        http_server.launch_server(args, run_scheduler_process_func=run_managed_scheduler)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
        journal.close()


if __name__ == "__main__":
    main()
