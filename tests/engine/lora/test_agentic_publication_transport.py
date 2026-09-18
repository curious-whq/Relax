# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise actual adapter methods without importing Ray/PyTorch compiler
code."""

import ast
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from relax.engine.lora.outbox import SessionCloseOutbox


def adapter_class():
    source = Path(__file__).parents[3] / "relax/agentic/pipeline/runtime.py"
    node = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "SGLangBackendAdapter"
    )
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, node], type_ignores=[])
    namespace = {
        "asyncio": asyncio,
        "httpx": httpx,
        "time": time,
        "Any": Any,
        "BackendGenerateResult": SimpleNamespace,
        "_sanitize_output_tokens": lambda ids, *args: ids,
        "_is_context_length_error": lambda error: False,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["SGLangBackendAdapter"]


def test_agentic_pending_poll_reuses_rid_and_close_failure_keeps_outbox(tmp_path):
    async def scenario():
        adapter = object.__new__(adapter_class())
        bodies = []
        binding = {"version_id": "A", "digest": "a" * 64, "lora_path": "relax_policy@A", "publication_epoch": 1}

        def data(request):
            bodies.append(json.loads(request.content))
            if len(bodies) == 1:
                return httpx.Response(503, json={"detail": {"code": "UNAVAILABLE"}})
            return httpx.Response(200, json={"output_ids": [4], "meta_info": {"finish_reason": {"type": "stop"}}})

        def control(request):
            if request.url.path == "/bind_session":
                assert json.loads(request.content)["spool_id"] == adapter._publication_outbox.identity
                return httpx.Response(200, json=binding)
            raise httpx.ConnectError("gateway unreachable", request=request)

        async with (
            httpx.AsyncClient(base_url="http://gateway", transport=httpx.MockTransport(data)) as data_client,
            httpx.AsyncClient(base_url="http://gateway", transport=httpx.MockTransport(control)) as control_client,
        ):
            adapter._args = SimpleNamespace(
                use_rollout_routing_replay=False, sglang_router_policy="round_robin", slime_router_sticky=False
            )
            adapter._session_lifecycle = False
            adapter._publication_data = data_client
            adapter._publication_client = control_client
            adapter._publication_owner = {"actor_id": "actor", "epoch": "epoch"}
            adapter._publication_outbox = SessionCloseOutbox(tmp_path, "http://gateway")
            adapter.tokenizer = None
            adapter.compiler = SimpleNamespace(processor=None)
            assert await adapter.bind_adapter_session("s") == binding
            result = await adapter.generate(
                input_ids=[1],
                sampling_params={"max_new_tokens": 1},
                session_id="s",
                request_id="same-rid",
                adapter_binding=binding,
            )
            assert result.new_tokens == [4]
            assert bodies[0] == bodies[1] and bodies[0]["rid"] == "same-rid"
            await adapter.close_adapter_session("s")
            assert [sid for _, sid in adapter._publication_outbox.pending()] == ["s"]

    asyncio.run(scenario())
