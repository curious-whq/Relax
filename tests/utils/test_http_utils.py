# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio

import httpx
import pytest

from relax.utils.http_utils import _post


class _StubClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def post(self, url, json=None, headers=None):
        self.calls += 1
        return self._responses.pop(0)


def _response(status_code: int, *, body: str | dict, headers=None):
    request = httpx.Request("POST", "http://test/post")
    if isinstance(body, dict):
        return httpx.Response(status_code, request=request, json=body, headers=headers)
    return httpx.Response(status_code, request=request, text=body, headers=headers)


def test_post_does_not_retry_non_retryable_400():
    client = _StubClient(
        [
            _response(
                400,
                body={"error": {"message": "Requested token count exceeds the model's maximum context length."}},
            )
        ]
    )

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_post(client, "http://test/post", {}, max_retries=5))

    assert client.calls == 1


def test_post_retries_retryable_503_then_succeeds():
    client = _StubClient(
        [
            _response(503, body={"error": {"message": "No available workers"}}),
            _response(200, body={"ok": True}),
        ]
    )

    result = asyncio.run(_post(client, "http://test/post", {}, max_retries=5))

    assert result == {"ok": True}
    assert client.calls == 2


def test_post_can_return_response_headers_without_changing_default_contract():
    client = _StubClient(
        [
            _response(
                200,
                body={"ok": True},
                headers={"X-Relax-Selected-Worker-Id": "worker-1"},
            )
        ]
    )

    result = asyncio.run(
        _post(
            client,
            "http://test/post",
            {},
            return_response_headers=True,
        )
    )

    assert result.body == {"ok": True}
    assert result.headers["x-relax-selected-worker-id"] == "worker-1"
