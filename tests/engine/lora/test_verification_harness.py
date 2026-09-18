# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Test experiment orchestration and failure detection, not GPU correctness."""

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from relax.engine.lora import verify as experiment
from relax.engine.lora.control import EngineControl, aborted_output
from relax.engine.lora.http import install_engine_routes
from relax.engine.lora.snapshot import fingerprint_model
from tests.engine.lora.test_control_http import Backend


@pytest.mark.parametrize("contaminate_cache", [False, True])
def test_experiment_checks_outputs_and_reports_progress(tmp_path, monkeypatch, contaminate_cache):
    monkeypatch.setenv("RELAX_LORA_CONTROL_TOKEN", "cpu-harness-credential")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"CPU harness only")
    fixtures = tmp_path / "fixtures"
    for version in "AB":
        source = fixtures / version
        source.mkdir(parents=True)
        (source / "adapter_config.json").write_text('{"r":8}')
        (source / "adapter_model.safetensors").write_text(version)

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *a, **kw: SimpleNamespace(encode=lambda *a, **kw: [1, 2])
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    class NumericalBackend(Backend):
        async def load(self, snapshot, op):
            await asyncio.sleep(0.03)
            await super().load(snapshot, op)

        async def generate(self, payload):
            self.payloads.append(payload)
            if payload.get("sampling_params", {}).get("max_new_tokens", 1) > 1:
                while payload["rid"] not in self.cancelled:
                    await asyncio.sleep(0.005)
                return aborted_output()
            await asyncio.sleep(0.005)
            version = payload["lora_path"].rsplit("@", 1)[-1]
            if contaminate_cache and version == "B" and self.loads["B"] > 1:
                version = "A"
            score = -1.0 if version == "A" else -3.0
            return {
                "output_ids": [1],
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "cached_tokens": 1,
                    "output_token_ids_logprobs": [[[score, token] for token in payload["token_ids_logprob"]]],
                },
            }

    transports = {}
    for index in range(2):
        app = FastAPI()
        control = EngineControl(
            NumericalBackend(), engine_id=str(index), base_digest=fingerprint_model(model), capacity=2
        )
        install_engine_routes(app, control)

        @app.get("/server_info")
        async def info():
            return {"version": "CPU test double"}

        transports[f"http://engine-{index}"] = httpx.ASGITransport(app=app)
    original_client = httpx.AsyncClient
    monkeypatch.setattr(experiment.httpx, "AsyncClient", lambda **kwargs: original_client(mounts=transports, **kwargs))
    args = SimpleNamespace(
        engine_url=list(transports),
        model_path=model,
        fixtures=fixtures,
        store=tmp_path / "store",
        atol=0.001,
        rtol=0,
        concurrency=2,
        max_progress_gap=1,
    )
    report = {}
    if contaminate_cache:
        with pytest.raises(AssertionError, match="logprob mismatch"):
            asyncio.run(asyncio.wait_for(experiment.verify(args, report), 10))
    else:
        asyncio.run(asyncio.wait_for(experiment.verify(args, report), 10))
        assert report["completed_after_test_gate"] > 0
        assert all(report["checks"].values())
        assert report["final_state"]["default_version"] == "C"
        assert report["final_state"]["versions"]["A"]["state"] == "retired"
