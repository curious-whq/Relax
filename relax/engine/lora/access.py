# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bearer credentials authorize callers; owner/incarnation fields only fence
state."""

from __future__ import annotations

import secrets
from typing import Any

from relax.utils.env import Envs


def headers(*, control: bool = False) -> dict[str, str]:
    token = Envs.RELAX_LORA_CONTROL_TOKEN if control else Envs.RELAX_LORA_DATA_TOKEN
    if not token:
        raise ValueError("immutable LoRA requires an explicit bearer credential")
    return {"Authorization": f"Bearer {token}"}


def install_access(app: Any, *, engine: bool = False, max_body_bytes: int = 4 * 1024 * 1024) -> None:
    from fastapi.responses import JSONResponse

    control = Envs.RELAX_LORA_CONTROL_TOKEN
    data = Envs.RELAX_LORA_DATA_TOKEN
    if not control or (not engine and (not data or secrets.compare_digest(control, data))):
        raise ValueError("configure distinct control/data credentials before starting the gateway")
    data_actions = {"/generate", "/bind_session", "/abort_request", "/close_session", "/state"}

    @app.middleware("http")
    async def authenticate(request: Any, call_next: Any) -> Any:
        if engine and not request.url.path.startswith("/relax/lora/"):
            return await call_next(request)
        supplied = request.headers.get("Authorization", "")
        admin = secrets.compare_digest(supplied, f"Bearer {control}")
        user = bool(data) and secrets.compare_digest(supplied, f"Bearer {data}")
        if not admin and not (not engine and user and request.url.path in data_actions):
            return JSONResponse({"code": "UNAUTHORIZED", "retryable": False}, status_code=401)
        # Content-Length is only an early check. A receive wrapper also bounds
        # chunked bodies before JSON parsing/serialization on the event loop.
        consumed = 0
        receive = request._receive

        async def bounded_receive() -> dict:
            nonlocal consumed
            message = await receive()
            consumed += len(message.get("body", b""))
            if consumed > max_body_bytes:
                from starlette.exceptions import HTTPException

                raise HTTPException(413, "PAYLOAD_TOO_LARGE")
            return message

        request._receive = bounded_receive
        return await call_next(request)
