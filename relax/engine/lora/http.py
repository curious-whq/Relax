# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Versioned engine HTTP protocol; every write is scoped to owner and boot
ID."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from relax.engine.lora.control import EngineControl
from relax.engine.lora.execution import CAPABILITY
from relax.engine.lora.publication import EngineIdentity, EngineReceipt
from relax.engine.lora.snapshot import AdapterConflictError, AdapterSnapshot


def artifact_body(artifact: AdapterSnapshot) -> dict[str, Any]:
    return {**asdict(artifact), "path": str(artifact.path)}


def parse_artifact(body: dict[str, Any]) -> AdapterSnapshot:
    return AdapterSnapshot(**{**body, "path": Path(body["path"])})


def install_engine_routes(app: Any, control: EngineControl) -> None:
    from fastapi import HTTPException

    @app.get("/relax/lora/info")
    async def info() -> dict[str, Any]:
        result = control.info()
        if hasattr(control.backend, "capabilities"):
            result.update(await control.backend.capabilities())
        return result

    @app.post("/relax/lora/{action}")
    async def operate(action: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            if action == "claim":
                return control.claim(body["owner_id"], body["incarnation"])
            control.authorize(body["owner_id"], body["incarnation"])
            if action in {"prepare", "retire"}:
                artifact = parse_artifact(body["artifact"])
                operation = getattr(control, action)
                return asdict(await operation(artifact, body["operation_id"]))
            if action == "operation_status":
                return await control.operation_status(body["operation_id"])
            if action == "generate":
                return await control.generate(body["operation_id"], body["payload"])
            if action == "abort":
                return await control.abort(body["rid"], body["operation_id"])
            if action == "status":
                return await control.reconcile(body["rid"], body["operation_id"])
            if action == "forget":
                await control.forget_async(body["rid"], body["operation_id"])
                return {"success": True}
            if action == "close_session":
                await control.backend.close_session(body["session_id"])
                return {"success": True}
            raise HTTPException(404, "unknown versioned engine operation")
        except AdapterConflictError as error:
            raise HTTPException(409, str(error)) from error
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(400, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error


class HttpAdapterEngine:
    def __init__(
        self, url: str, owner_id: str, client: httpx.AsyncClient, data_client: httpx.AsyncClient | None = None
    ) -> None:
        self.url = url.rstrip("/")
        self.owner_id = owner_id
        self.client = client
        self.data_client = data_client or client
        self._identity: EngineIdentity | None = None
        self.request_capacity = 0
        self._status_slots = asyncio.Semaphore(4)

    @property
    def identity(self) -> EngineIdentity:
        if self._identity is None:
            raise RuntimeError("engine must be connected before creating the version manager")
        return self._identity

    async def connect(self, base_digest: str, capacity: int) -> None:
        response = await self.client.get(f"{self.url}/relax/lora/info")
        response.raise_for_status()
        info = response.json()
        if (
            info["protocol"] != 2
            or info.get("capability") != CAPABILITY
            or info["base_digest"] != base_digest
            or info["capacity"] < capacity
        ):
            raise AdapterConflictError("engine protocol/base/capacity does not match publication configuration")
        self._identity = EngineIdentity(**info["engine"])
        self.request_capacity = info["request_capacity"]
        await self.call("claim", {})

    async def call(self, action: str, body: dict[str, Any]) -> dict[str, Any]:
        if action in {"status", "operation_status"}:
            async with self._status_slots:
                return await self._call(action, body)
        return await self._call(action, body)

    async def _call(self, action: str, body: dict[str, Any]) -> dict[str, Any]:
        # No generic HTTP fallback or blind generation retries. Engine endpoints
        # themselves deduplicate operations; the gateway reconciles lost replies.
        client = self.data_client if action == "generate" else self.client
        response = await client.post(
            f"{self.url}/relax/lora/{action}",
            json={**body, "owner_id": self.owner_id, "incarnation": self.identity.incarnation},
            timeout=httpx.Timeout(30.0, connect=5.0, pool=5.0),
        )
        response.raise_for_status()
        return response.json()

    async def _update(self, action: str, artifact: AdapterSnapshot, operation_id: str) -> EngineReceipt:
        result = await self.call(action, {"artifact": artifact_body(artifact), "operation_id": operation_id})
        return EngineReceipt(**{**result, "engine": EngineIdentity(**result["engine"])})

    async def prepare(self, snapshot: AdapterSnapshot, operation_id: str) -> EngineReceipt:
        try:
            return await self._update("prepare", snapshot, operation_id)
        except httpx.HTTPError:
            # Keep the coordinator's physical-load permit while the engine-owned
            # task runs. Transport expiry does not end the physical operation.
            while True:
                try:
                    state = await self.call("operation_status", {"operation_id": operation_id})
                    if not state["preparing"]:
                        if state["error"]:
                            raise RuntimeError(state["error"])
                        return self._receipt(state["receipt"])
                except httpx.HTTPStatusError as error:
                    if error.response.status_code == 400:
                        # Unknown operation may still have a delayed prepare. A
                        # successful retire installs the fence before permit release.
                        try:
                            await self.retire(snapshot, operation_id)
                        except httpx.HTTPError:
                            await asyncio.sleep(0.2)
                            continue
                        raise RuntimeError("prepare was not confirmed and has been fenced") from error
                except httpx.TransportError:
                    pass
                await asyncio.sleep(0.2)

    async def retire(self, snapshot: AdapterSnapshot, operation_id: str) -> EngineReceipt:
        return await self._update("retire", snapshot, operation_id)

    @staticmethod
    def _receipt(result: dict[str, Any]) -> EngineReceipt:
        return EngineReceipt(**{**result, "engine": EngineIdentity(**result["engine"])})

    async def prepared_receipt(self, operation_id: str) -> EngineReceipt:
        result = await self.call("operation_status", {"operation_id": operation_id})
        return self._receipt(result["receipt"])
