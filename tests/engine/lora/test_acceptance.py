# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.engine.lora.acceptance.assertions import audit_evidence, compare, scores, worker_observations


def test_numerical_comparison_uses_declared_asymmetric_tolerance():
    assert compare({1: -1.001}, {1: -1.0}, atol=0.002, rtol=0) < 0.002
    with pytest.raises(AssertionError):
        compare({1: -1.1}, {1: -1.0}, atol=0.002, rtol=0)
    with pytest.raises(AssertionError):
        compare({2: -1.0}, {1: -1.0}, atol=0.002, rtol=0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_numerical_comparison_rejects_nonfinite_in_either_baseline(value):
    with pytest.raises(AssertionError):
        compare({1: value}, {1: value}, atol=0.001, rtol=0.001)
    with pytest.raises(AssertionError):
        scores({"meta_info": {"output_token_ids_logprobs": [[[value, 1]]]}})


@pytest.mark.parametrize("fault", [None, "token_fork", "decode_score", "missing", "nonfinite"])
def test_mixed_decode_checks_every_position_and_rejects_token_forks(fault):
    from copy import deepcopy

    from tests.engine.lora.acceptance.assertions import compare_decode

    reference = {"output_ids": [4, 5], "meta_info": {"output_token_ids_logprobs": [[[-1.0, 7]], [[-2.0, 7]]]}}
    actual = deepcopy(reference)
    if fault == "token_fork":
        actual["output_ids"][1] = 6
    elif fault == "decode_score":
        actual["meta_info"]["output_token_ids_logprobs"][1][0][0] = -3.0
    elif fault == "missing":
        actual["meta_info"]["output_token_ids_logprobs"].pop()
    elif fault == "nonfinite":
        actual["meta_info"]["output_token_ids_logprobs"][1][0][0] = float("nan")
    if fault:
        with pytest.raises(AssertionError):
            compare_decode(actual, reference, atol=1e-3, rtol=1e-3)
    else:
        assert compare_decode(actual, reference, atol=1e-3, rtol=1e-3) == [0, 0]


@pytest.mark.parametrize("cached_tokens", [0, 1, None])
def test_decode_reference_replays_full_sequence_in_fresh_cache_namespace(cached_tokens):
    import httpx

    from tests.engine.lora.acceptance.support import reference_decode

    requests = []
    output = {"output_ids": [4, 5], "meta_info": {"cached_tokens": cached_tokens}}

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["input_ids"] == [1, 2, 3] and body["logprob_start_len"] == -1
        assert body["sampling_params"] == {"temperature": 0, "max_new_tokens": 2, "ignore_eos": True}
        return httpx.Response(200, json=output)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            ctx = SimpleNamespace(
                client=client,
                report={},
                diagnostic_ids=[7],
                config={"baselines": {"A": {"url": "http://reference", "lora_path": "A", "cache_enabled": True}}},
            )
            for _ in range(2):
                if cached_tokens != 0:
                    with pytest.raises(AssertionError, match="diagnostic prefix"):
                        await reference_decode(ctx, "A", [1, 2, 3], tokens=2)
                else:
                    assert await reference_decode(ctx, "A", [1, 2, 3], tokens=2) == output
            assert requests[0]["extra_key"] != requests[1]["extra_key"]
            assert all(item["response"] == output for item in ctx.report["references"])

    asyncio.run(run())


@pytest.mark.parametrize(
    "fault", [None, "score", "different_tokens", "misaligned_scores", "warm_start", "missing_span", "rollout_tokens"]
)
def test_session_replay_preserves_evidence_and_rejects_invalid_references(fault):
    import httpx

    from tests.engine.lora.acceptance.support import score_export

    sample = SimpleNamespace(
        tokens=[10, 20, 30, 40, 50],
        rollout_tokens=[10, 20, 30, 40, 50],
        response_length=3,
        loss_mask=[1, 0, 1],
        rollout_log_probs=[-0.5, 0, -0.7],
        metadata={
            "lora_adapter": {"version_id": "A", "digest": "digest"},
            "lora_attempts": [
                {"adapter_version_id": "A", "adapter_digest": "digest", "token_start": start, "token_end": end}
                for start, end in ((2, 2), (2, 3), (4, 5))
            ],
        },
    )
    if fault == "missing_span":
        sample.metadata["lora_attempts"].pop()
    elif fault == "rollout_tokens":
        sample.rollout_tokens = [10, 99, 30, 40, 50]
    expected = -1.0214 if fault == "score" else -0.5
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        start = (2, 4)[len(requests) % 2]
        requests.append(payload)
        assert payload["input_ids"] == sample.tokens[:start]
        assert payload["logprob_start_len"] == -1
        return httpx.Response(
            200,
            json={
                "output_ids": [99 if fault == "different_tokens" else sample.tokens[start]],
                "meta_info": {
                    "output_token_logprobs": [
                        [
                            expected if start == 2 else -0.7,
                            99 if fault == "misaligned_scores" else sample.tokens[start],
                        ]
                    ],
                    "cached_tokens": (1 if fault == "warm_start" else 0) if start == 2 else 3,
                },
            },
        )

    report = {"samples": [], "numerical": []}

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            ctx = SimpleNamespace(
                client=client,
                report=report,
                atol=1e-3,
                rtol=1e-3,
                config={"baselines": {"A": {"url": "http://baseline", "lora_path": "A", "cache_enabled": True}}},
            )
            await score_export(ctx, SimpleNamespace(samples=[sample]), "A")

    if fault:
        with pytest.raises(AssertionError, match="Session sample 0, adapter A"):
            asyncio.run(run())
    else:
        asyncio.run(run())
    assert report["samples"][0]["tokens"] == sample.tokens
    evidence = report["numerical"][0]
    assert evidence["status"] == ("FAIL" if fault else "PASS")
    assert evidence["actual"] == {2: -0.5, 4: -0.7}  # Exclude inter-turn observations.
    if fault in ("missing_span", "rollout_tokens"):
        assert not requests
        return
    assert "response" in evidence["baseline_attempts"][0][1]
    if fault not in (None, "score"):
        assert len(requests) == 1  # Never score past a token fork or retry a contaminated baseline.
        return
    assert evidence["baseline_repeats"] == [{2: expected, 4: -0.7}] * 2
    assert len(requests) == 4
    assert requests[0]["extra_key"] == requests[1]["extra_key"]
    assert requests[2]["extra_key"] == requests[3]["extra_key"] != requests[0]["extra_key"]
    assert evidence["baseline_attempts"][0][0]["status"] == "NO_OUTPUT"
    assert evidence["baseline_attempts"][0][-1]["response"]["meta_info"]["cached_tokens"] == 3
    assert evidence["max_error"] == pytest.approx(abs(expected + 0.5))
    if fault == "score":
        worst = evidence["worst_positions"][0]
        assert worst["position"] == 2 and worst["token_id"] == 30
        assert worst["baseline_repeat"] == expected
        assert worst["allowed_error"] == pytest.approx(1e-3 + 1e-3 * abs(expected))


def test_partial_evidence_cannot_claim_full_acceptance():
    report = {"checks": {name: {"status": "PASS"} for name in ("old_session_logprobs", "slot_reuse")}}
    assert {"cache_a_to_b", "continuous_traffic", "slot_reuse_negative_control"} <= set(audit_evidence(report))


@pytest.mark.parametrize("fault", [None, "nonfinite", "missing", "positive", "restore"])
def test_stale_slot_control_requires_valid_negative_and_correct_restoration(tmp_path, monkeypatch, fault):
    from tests.engine.lora.acceptance import scenarios, support

    calls = []
    expected = {7: -1.0, 8: -2.0}

    async def reference(*args):
        return expected

    async def diagnostic(ctx, bound, ids, **kwargs):
        calls.append(kwargs["extra_key"])
        index = len(calls) - 1
        values = dict(expected)
        if index == 1 or (fault == "positive" and index == 0) or (fault == "restore" and index == 2):
            values[7] = -3.0
        if index == 1 and fault == "nonfinite":
            values[7] = float("nan")
        if index == 1 and fault == "missing":
            values.pop(8)
        injected = json.loads(ctx.test_control.read_text()).get("stale_slot")
        assert bool(injected) == (index == 1)
        if injected:
            (ctx.root / (kwargs["rid"] + ".stale-slot.json")).write_text(json.dumps(injected))
        return {
            "meta_info": {"output_token_ids_logprobs": [[[value, key] for key, value in values.items()]]},
            "verification_completion": {"execution_workers": [0]},
        }

    monkeypatch.setattr(support, "cold_diagnostic", reference)
    monkeypatch.setattr(support, "diagnostic", diagnostic)
    ctx = SimpleNamespace(
        report={"checks": {}},
        config={"timeout_seconds": 1},
        root=tmp_path,
        test_control=tmp_path / "control.json",
        versions={"B": "B"},
        prompts=["prefix"],
        tokenizer=SimpleNamespace(encode=lambda *a, **k: [1, 2]),
        atol=1e-3,
        rtol=1e-3,
    )
    old = {"engine": "E1", "native_lora_id": "old"}
    current = {"engine": "E1", "native_lora_id": "new", "execution_workers": [0]}
    support.configure_test(ctx)
    if fault:
        with pytest.raises(AssertionError):
            asyncio.run(scenarios.stale_slot_negative_control(ctx, [old], [current]))
        assert ctx.report["checks"]["slot_reuse_negative_control"]["status"] == "INCOMPLETE"
    else:
        asyncio.run(scenarios.stale_slot_negative_control(ctx, [old], [current]))
        assert ctx.report["checks"]["slot_reuse_negative_control"]["status"] == "PASS"
        assert len(calls) == 3
    assert len(calls) == len(set(calls))


def test_worker_resources_cannot_be_proven_by_only_the_leader():
    observation = {"worker_boots": ["rank0", "rank1"], "workers": {"0": {"slot": 1}}}
    with pytest.raises(AssertionError, match="incomplete"):
        worker_observations({"observation": observation})
    observation["workers"]["1"] = {"slot": 2}
    assert set(worker_observations({"observation": observation})) == {"0", "1"}
    observation["worker_boots"] = ["same", "same"]
    with pytest.raises(AssertionError, match="incomplete"):
        worker_observations({"observation": observation})


def test_performance_counts_completions_and_detects_stalled_individual_engine():
    from tests.engine.lora.acceptance.performance import compare_windows, summarize_window

    requests = [
        {"accepted": 1.1, "completed": 1.5, "first_token_at": 1.2},
        {"accepted": 1.6, "completed": 2.1, "first_token_at": 1.8},
    ]
    progress = [
        {"at": 0.9, "engine": "E1", "tokens": 100},
        {"at": 1.2, "engine": "E1", "tokens": 3},
        {"at": 1.8, "engine": "E1", "tokens": 2},
        {"at": 2, "engine": "E2", "tokens": 100},
    ]
    metrics = summarize_window(requests, progress, ("E1", "E2"), 1, 2)
    assert metrics["token_throughput"] == {"all": 5, "E1": 5, "E2": 0}
    assert metrics["accepted"] == 2 and metrics["throughput"] == 1
    assert metrics["p95"] == pytest.approx(0.5)
    assert metrics["max_progress_gaps"]["E2"] == 1
    with pytest.raises(AssertionError, match="pre-registered"):
        compare_windows(metrics, metrics, {"max_progress_gap": 0.9})


@pytest.mark.parametrize("completed,error", [(None, None), (2, "timeout")])
def test_performance_cannot_hide_censored_or_failed_requests(completed, error):
    from tests.engine.lora.acceptance.performance import summarize_window

    with pytest.raises(AssertionError, match="failed, censored"):
        summarize_window([{"accepted": 1, "completed": completed, "error": error}], [], ("E1",), 0, 2)


def test_diagnostic_binding_covers_each_engine_dp_group_and_closes_extras(monkeypatch):
    from tests.engine.lora.acceptance import support

    routes = [(0, "e1"), (0, "e2"), (0, "e1"), (1, "e1"), (1, "e2")]
    candidates = [
        {"dp_rank": dp, "dp_size": 2, "engine": {"engine_id": engine}, "session_id": str(index)}
        for index, (dp, engine) in enumerate(routes)
    ]
    close = AsyncMock()
    monkeypatch.setattr(support, "request", AsyncMock(return_value={"serving_engines": ["e1", "e2"]}))
    monkeypatch.setattr(support, "native_bind", AsyncMock(side_effect=candidates))
    monkeypatch.setattr(support, "close_native", close)
    ctx = SimpleNamespace(config={})
    result = asyncio.run(support.native_bindings(ctx))
    assert {(bound["dp_rank"], bound["engine"]["engine_id"]) for bound in result} == set(routes)
    assert len(result) == 4
    close.assert_awaited_once_with(ctx, candidates[2])


@pytest.mark.parametrize("busy_uuid,allowed", [("GPU-other", True), ("GPU-selected-3", False)])
def test_auto_deployment_gpu_selection_preserves_other_jobs(monkeypatch, busy_uuid, allowed):
    from tests.engine.lora.acceptance import processes

    def query(command, **kwargs):
        if "--query-compute-apps=gpu_uuid,pid" in command:
            return f"{busy_uuid}, 123\n"
        return "0, GPU-other, card, 48000, driver\n2, GPU-selected-2, card, 48000, driver\n3, GPU-selected-3, card, 48000, driver\n"

    monkeypatch.setattr(processes.subprocess, "check_output", query)
    if allowed:
        assert [gpu["uuid"] for gpu in processes.select_gpus(["2", "3"])] == ["GPU-selected-2", "GPU-selected-3"]
    else:
        with pytest.raises(RuntimeError, match="No existing process was stopped"):
            processes.select_gpus(["2", "3"])


def test_auto_deployment_failed_task_reaps_only_owned_process(tmp_path):
    from tests.engine.lora.acceptance.processes import owned_process

    records = []
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with pytest.raises(RuntimeError, match="injected"):
            with owned_process(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                dict(os.environ),
                tmp_path / "owned.log",
                records,
            ) as owned:
                raise RuntimeError("injected")
        assert owned.poll() is not None
        assert unrelated.poll() is None
        assert records[0]["cleanup"] == "SIGNALLED_AND_LEADER_REAPED"
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=10)


@pytest.mark.parametrize(
    "center,spread,verdict", [(0, 0.005, "PASS"), (0.2, 0.005, "FAIL"), (0.05, 0.5, "INCONCLUSIVE")]
)
def test_overhead_target_uses_block_uncertainty_and_checks_each_engine(center, spread, verdict):
    from tests.engine.lora.acceptance.overhead import analyze

    def blocks_for(engine_only=False):
        blocks = []
        for index in range(12):
            cost = center + spread * (1 if index % 2 else -1)
            windows = []
            for mode in ("ordinary", "managed", "managed", "ordinary"):
                factor = 1 - cost / 100 if mode == "managed" else 1
                rates = {"all": 200 if engine_only else 200 * factor, "E1": 100 * factor, "E2": 100}
                if engine_only:
                    rates["E2"] = 200 - rates["E1"]
                else:
                    rates["E2"] *= factor
                windows.append({"mode": mode, "metrics": {"token_throughput": rates}})
            blocks.append({"windows": windows})
        return blocks

    result = analyze(blocks_for(), 0.05)
    assert result["verdict"] == verdict
    assert result["metrics"]["all"]["overhead_percent"] == pytest.approx(center, abs=0.002)
    if verdict == "FAIL":
        assert analyze(blocks_for(engine_only=True), 0.05)["verdict"] == "FAIL"
    if verdict == "PASS":
        assert result["metrics"]["all"]["overhead_ci_percent"][1] < 0.05
        # Identical blocks cannot establish sub-per-mille measurement precision.
        assert analyze([blocks_for()[0]] * 12, 0.05)["verdict"] == "INCONCLUSIVE"
    blocks = blocks_for()
    blocks[-1]["windows"].pop()
    with pytest.raises(ValueError, match="complete"):
        analyze(blocks, 0.05)
    with pytest.raises(ValueError, match="six"):
        analyze(blocks[:5], 0.05)


def test_diagnostic_trace_separates_overlapping_gpu_and_nested_cpu_time(tmp_path):
    import gzip

    from tests.engine.lora.acceptance.overhead_diagnostics import summarize_samples, summarize_trace

    path = tmp_path / "profile.trace.json.gz"
    events = [
        {"cat": "kernel", "ph": "X", "name": "lora_kernel", "ts": 0, "dur": 100, "args": {"device": 0}},
        {"cat": "kernel", "ph": "X", "name": "dense_kernel", "ts": 50, "dur": 100, "args": {"device": 0}},
        {"cat": "gpu_memcpy", "ph": "X", "name": "copy", "ts": 200, "dur": 10, "args": {"device": 0}},
        {"cat": "cpu_op", "ph": "X", "name": "parent", "ts": 0, "dur": 200},
        {"cat": "cpu_op", "ph": "X", "name": "child", "ts": 50, "dur": 100},
    ]
    with gzip.open(path, "wt") as sink:
        json.dump({"traceEvents": events}, sink)
    report = summarize_trace(path)
    assert report["status"] == "COMPLETE"
    gpu = report["gpu_activity"]["0"]
    assert gpu["union_busy_ms"] == pytest.approx(0.16)
    assert gpu["max_internal_gap_ms"] == pytest.approx(0.05)
    assert gpu["busy_fraction_within_span"] == pytest.approx(160 / 210)
    assert report["top_events"]["cpu_op"][0]["inclusive_total_ms"] == pytest.approx(0.2)
    path = tmp_path / "cpu-only.json"
    path.write_text(json.dumps({"traceEvents": events[3:]}))
    assert summarize_trace(path)["status"] == "INCOMPLETE"
    samples = tmp_path / "samples.json"
    samples.write_text(
        json.dumps(
            {
                "shared": {"frames": [{"name": "outer"}, {"name": "bind", "file": "control.py", "line": 20}]},
                "profiles": [{"type": "sampled", "samples": [[0, 1], [0, 1], [0]]}],
            }
        )
    )
    report = summarize_samples(samples)
    assert report["top_leaf_frames"][0] == {"function": "bind", "file": "control.py", "line": 20, "samples": 2}


def test_diagnostic_request_breakdown_never_subtracts_client_and_server_clocks():
    from tests.engine.lora.acceptance.overhead_diagnostics import request_breakdown

    raw = {
        "requests": [
            {
                "rid": "r",
                "engine": "e",
                "accepted": 1,
                "headers_at": 1.1,
                "first_token_at": 2,
                "last_token_at": 4,
                "completed": 4.5,
                "tokens": 3,
                "server_timing": {
                    "request_received_ts": 1000,
                    "api_server_dispatch_finish_ts": 1000.25,
                    "forward_entry_time": 2000,
                    "prefill_finished_time": 2000.5,
                    "queue_time": 0.125,
                },
            }
        ],
        "progress": [{"rid": "r", "at": 2}, {"rid": "r", "at": 4}],
    }
    report = request_breakdown(raw, 0, 2)
    row = report["requests"][0]
    assert row["server_api_dispatch_s"] == 0.25 and row["server_queue_s"] == 0.125
    assert row["server_prefill_span_s"] == 0.5 and row["client_mean_decode_s_per_token"] == 1
    assert row["client_response_tail_s"] == 0.5 and "server_after_prefill_s" not in row
    assert row["chunk_gaps_seconds"]["p50"] == 2


@pytest.mark.parametrize("failure", [None, "start", "stop"])
def test_diagnostics_stop_attempted_profiles_and_preserve_partial_evidence(tmp_path, monkeypatch, failure):
    import httpx

    from tests.engine.lora.acceptance import overhead_diagnostics as diagnostic

    starts, stops, directories = [], [], {}
    real_sleep = asyncio.sleep

    async def immediate(seconds):
        await real_sleep(0)

    async def handler(request):
        host = request.url.host
        if request.url.path == "/start_profile":
            starts.append(host)
            directories[host] = Path(json.loads(request.content)["output_dir"])
            if failure == "start":
                raise httpx.ReadTimeout("start may already have taken effect", request=request)
        elif request.url.path == "/stop_profile":
            stops.append(host)
            (directories[host] / "gpu.trace.json").write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {"ph": "X", "cat": "kernel", "name": "kernel", "ts": 0, "dur": 10},
                        ]
                    }
                )
            )
            if failure == "stop":
                raise httpx.ReadTimeout("trace exists but stop ACK unknown", request=request)
        return httpx.Response(200, json={})

    async def bind_times(ctx):
        return {"status": "COMPLETE"}

    async def ordinary(ctx, bindings):
        return bindings

    async def traffic(ctx, bindings, **options):
        assert options["observe"] is False and options["diagnostic"] is True
        await options["on_ready"]()
        ctx.report.setdefault("traffic_raw", []).append({"requests": [], "progress": []})
        return {"start": 0, "end": 2}

    monkeypatch.setattr(diagnostic, "binding_timings", bind_times)
    monkeypatch.setattr(diagnostic.shutil, "which", lambda name: None)
    monkeypatch.setattr(diagnostic.asyncio, "sleep", immediate)
    monkeypatch.setattr(diagnostic.performance, "ordinary_bindings", ordinary)
    monkeypatch.setattr(diagnostic.performance, "traffic_window", traffic)
    monkeypatch.delenv("SGLANG_PROFILE_V2", raising=False)
    config = {"output": str(tmp_path), "overhead_profile": {"concurrency": 2}}

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            ctx = SimpleNamespace(config=config, client=client, report={})
            report = await diagnostic.collect(
                ctx, [{"engine": {"endpoint": "http://engine0"}}, {"engine": {"endpoint": "http://engine1"}}]
            )
            assert ctx.config is config
            assert report["status"] == "INCOMPLETE"  # Missing py-spy cannot become a clean bill of health.
            assert not report["included_in_overhead_ci"]
            gpu = [phase for phase in report["phases"] if phase["kind"] == "gpu"]
            assert len(gpu) == 2
            assert all(phase["status"] == ("INCOMPLETE" if failure else "COMPLETE") for phase in gpu)
            assert len(starts) == len(stops) == 4  # Even start timeouts require a stop attempt.
            assert len(list((tmp_path / "diagnostics").glob("*/requests.json"))) == 4
            assert (tmp_path / "diagnostics/report.json").is_file()

    asyncio.run(run())


@pytest.mark.parametrize("fail_bind", [False, True])
def test_binding_diagnostic_retains_cleanup_ownership_on_failed_rpc(fail_bind):
    from tests.engine.lora.acceptance.overhead_diagnostics import binding_timings

    calls = []

    async def remote(action, payload):
        calls.append((action, payload["session_id"]))
        if action == "close":
            return {"accepted": True}
        if fail_bind:
            raise TimeoutError("unknown bind outcome")
        return {"binding": "A", "native_lora_id": "instance-A"}

    ctx = SimpleNamespace(
        native_owner="owner", native_sessions=[], manager=SimpleNamespace(lora_control=SimpleNamespace(remote=remote))
    )
    result = asyncio.run(binding_timings(ctx))
    assert result["status"] == ("INCOMPLETE" if fail_bind else "COMPLETE")
    assert len(ctx.native_sessions) == (1 if fail_bind else 12)
    for bound in ctx.native_sessions:
        actions = [action for action, sid in calls if sid == bound["session_id"]]
        assert actions == (["bind", "close"] if fail_bind else ["bind", "bind", "close"])
    assert all(row["close"] == "ACCEPTED" for row in result["raw"])
