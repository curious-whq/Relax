# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bounded data-path admission before creating Session or execution
references."""

from __future__ import annotations

from typing import Any


def request_tokens(payload: dict[str, Any], *, max_tokens: int = 8192) -> int:
    allowed = {
        "session_id",
        "rid",
        "expected_adapter",
        "input_ids",
        "sampling_params",
        "return_logprob",
        "logprob_start_len",
        "top_logprobs_num",
        "token_ids_logprob",
    }
    if set(payload) - allowed:
        raise ValueError("UNSUPPORTED_GENERATION_FIELDS")
    ids = payload.get("input_ids")
    params = payload.get("sampling_params", {})
    if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("a single nonempty input_ids prompt is required")
    if not isinstance(params, dict) or params.get("n", 1) != 1:
        raise ValueError("sampling_params.n must be 1")
    output = params.get("max_new_tokens")
    if type(output) is not int or not 1 <= output <= max_tokens:
        raise ValueError("explicit bounded max_new_tokens is required")
    if len(ids) + output > max_tokens:
        raise ValueError("REQUEST_TOKEN_BUDGET_EXCEEDED")
    top = payload.get("top_logprobs_num", 0)
    selected = payload.get("token_ids_logprob", [])
    if type(top) is not int or not 0 <= top <= 32 or not isinstance(selected, list) or len(selected) > 256:
        raise ValueError("LOGPROB_OUTPUT_BUDGET_EXCEEDED")
    return len(ids) + output
