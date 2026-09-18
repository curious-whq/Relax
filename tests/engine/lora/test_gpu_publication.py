# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in real GPU acceptance test; requires two fresh managed engines."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from relax.engine.lora.verify import compare, routed_session, scores


def test_numerical_comparison_rejects_wrong_adapter_and_nonfinite_scores():
    assert compare({1: -1.001}, {1: -1.0}, atol=0.002, rtol=0) < 0.002
    with pytest.raises(AssertionError):
        compare({1: -1.1}, {1: -1.0}, atol=0.002, rtol=0)
    with pytest.raises(AssertionError):
        compare({2: -1.0}, {1: -1.0}, atol=0.002, rtol=0)
    with pytest.raises(AssertionError):
        scores({"meta_info": {"output_token_ids_logprobs": [[[float("nan"), 1]]]}})
    assert routed_session("old", 0) != routed_session("old", 1)


@pytest.mark.skipif(
    not os.environ.get("RELAX_LORA_GPU_CONFIG"),
    reason="requires two fresh GPU SGLang engines, LoRA fixtures and fixed tolerances via RELAX_LORA_GPU_CONFIG",
)
def test_real_gpu_publication_logprobs_cache_and_capacity():
    config = json.loads(Path(os.environ["RELAX_LORA_GPU_CONFIG"]).read_text())
    command = [sys.executable, "-m", "relax.engine.lora.verify"]
    for key, value in config.items():
        for item in value if isinstance(value, list) else [value]:
            command.extend(["--" + key.replace("_", "-"), str(item)])
    subprocess.run(command, check=True, timeout=float(config.get("timeout_seconds", 900)) + 30)
    assert json.loads(Path(config["report"]).read_text())["status"] == "passed"
