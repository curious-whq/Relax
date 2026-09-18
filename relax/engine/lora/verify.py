# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real two-engine logprob, cache, capacity and publication-overlap
experiment."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from relax.engine.lora.gateway import PublicationGateway
from relax.engine.lora.http import HttpAdapterEngine
from relax.engine.lora.publication import AdapterCapacityError
from relax.engine.lora.snapshot import fingerprint_model, snapshot_adapter


def routed_session(prefix: str, engine_index: int, count: int = 2) -> str:
    for index in range(10000):
        session = f"{prefix}-{index}"
        if int(hashlib.sha256(session.encode()).hexdigest(), 16) % count == engine_index:
            return session
    raise RuntimeError("could not construct a Session for the requested engine")


class GatedEngine(HttpAdapterEngine):
    """One real engine with a test-only gate after B's physical prepare."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.b_ready = asyncio.Event()
        self.release_b = asyncio.Event()
        self.hold_b = False

    async def prepare(self, snapshot: Any, operation_id: str) -> Any:
        receipt = await super().prepare(snapshot, operation_id)
        if self.hold_b and snapshot.version_id == "B":
            self.b_ready.set()
            await self.release_b.wait()
        return receipt


def scores(output: dict[str, Any]) -> dict[int, float]:
    entries = output["meta_info"]["output_token_ids_logprobs"][0]
    result = {int(entry[1]): float(entry[0]) for entry in entries}
    if not result or not all(math.isfinite(value) for value in result.values()):
        raise AssertionError("missing or nonfinite fixed-token logprobs")
    return result


def compare(actual: dict[int, float], expected: dict[int, float], *, atol: float, rtol: float) -> float:
    if actual.keys() != expected.keys():
        raise AssertionError("scored token IDs changed")
    error = max(abs(actual[key] - expected[key]) for key in actual)
    if not all(math.isclose(actual[key], expected[key], abs_tol=atol, rel_tol=rtol) for key in actual):
        raise AssertionError(f"logprob mismatch, maximum absolute error {error}")
    return error


async def verify(args: Any, report: dict[str, Any]) -> None:
    from transformers import AutoTokenizer

    if args.atol < 0 or args.rtol < 0 or args.concurrency < 2 or args.max_progress_gap <= 0:
        raise ValueError("nonnegative tolerances, concurrency >= 2, and positive max-progress-gap are required")
    base = await asyncio.to_thread(fingerprint_model, args.model_path)
    owner = uuid4().hex
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    ids = tokenizer.encode("The capital of France is", add_special_tokens=True)
    candidate_ids = sorted(set(tokenizer.encode(" Paris London Berlin Beijing yes no", add_special_tokens=False)))
    payload = {
        "input_ids": ids,
        "return_logprob": True,
        "token_ids_logprob": candidate_ids,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
    }
    report.update(base_model_digest=base, token_ids=ids, candidate_ids=candidate_ids, events=[], checks={})
    from relax.engine.lora.access import headers

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(120, connect=10), trust_env=False, headers=headers(control=True)
    ) as client:
        engines = [GatedEngine(url, owner, client) for url in args.engine_url]
        await asyncio.gather(*(engine.connect(base, 2) for engine in engines))
        if len({engine.identity.engine_id for engine in engines}) != 2:
            raise ValueError("experiment requires two distinct engine identities")
        report["engines"] = [engine.identity.__dict__ for engine in engines]
        report["server_configuration"] = []
        for engine in engines:
            response = await client.get(f"{engine.url}/server_info")
            response.raise_for_status()
            info = response.json()
            report["server_configuration"].append(
                {
                    key: info.get(key)
                    for key in (
                        "version",
                        "dtype",
                        "tp_size",
                        "dp_size",
                        "max_loras_per_batch",
                        "max_loaded_loras",
                        "lora_backend",
                        "max_lora_rank",
                        "attention_backend",
                        "mem_fraction_static",
                        "gpu_id_step",
                    )
                }
            )
        baselines: dict[str, dict[str, dict[int, float]]] = {}
        artifacts = {
            name: snapshot_adapter(args.fixtures / name, args.store, version_id=name, base_model_digest=base)
            for name in ("A", "B")
        }
        report["adapter_digests"] = {name: artifact.digest for name, artifact in artifacts.items()}
        # Each baseline runs with exactly one adapter on its engine. New operation
        # IDs give fresh KV namespaces without issuing any flush-cache operation.
        for engine in engines:
            by_version = baselines[engine.identity.engine_id] = {}
            for name, artifact in artifacts.items():
                operation_id = uuid4().hex
                await engine.prepare(artifact, operation_id)
                try:
                    repeats = []
                    for _ in range(3):
                        baseline_rid = uuid4().hex
                        result = await engine.call(
                            "generate",
                            {
                                "operation_id": operation_id,
                                "payload": {**payload, "rid": baseline_rid, "lora_path": artifact.lora_path},
                            },
                        )
                        if result["status"] != "terminal":
                            raise AssertionError(f"baseline request did not finish: {result}")
                        repeats.append(scores(result["output"]))
                        await engine.call("forget", {"rid": baseline_rid, "operation_id": operation_id})
                    for repeat in repeats[1:]:
                        compare(repeat, repeats[0], atol=args.atol, rtol=args.rtol)
                    by_version[name] = repeats[0]
                finally:
                    await engine.retire(artifact, operation_id)
            separation = max(abs(by_version["A"][key] - by_version["B"][key]) for key in candidate_ids)
            tolerance = max(args.atol + args.rtol * abs(value) for value in by_version["B"].values())
            if separation <= 10 * tolerance:
                raise AssertionError("fixtures are not numerically distinguishable at the predeclared tolerance")
        report["baselines"] = baselines
        gateway = PublicationGateway(engines, capacity=2, base_digest=base, store=args.store)
        await gateway.manager.publish(artifacts["A"])
        old_sessions = [routed_session(f"old-{index}", index % 2) for index in range(args.concurrency)]
        for index, session in enumerate(old_sessions):
            gateway.bind(session)
            gateway.affinity[session] = engines[index % 2].identity.engine_id
        latencies: list[float] = []
        completions: list[float] = []
        failures: list[str] = []
        cache_hits: list[int] = []
        cache_checks: list[dict[str, Any]] = []
        stop = asyncio.Event()

        async def generate(session: str, *, checked: bool = True) -> dict[str, Any]:
            started = time.monotonic()
            output = await gateway.generate({**payload, "rid": uuid4().hex, "session_id": session})
            completed = time.monotonic()
            latencies.append(completed - started)
            completions.append(completed)
            cache_hits.append(int(output["meta_info"].get("cached_tokens", 0)))
            if checked:
                engine_id = output["meta_info"]["lora_engine"]["engine_id"]
                name = gateway.bind(session)["version_id"]
                compare(scores(output), baselines[engine_id][name], atol=args.atol, rtol=args.rtol)
            return output

        async def traffic(session: str) -> None:
            while not stop.is_set():
                try:
                    await generate(session)
                except Exception as error:
                    failures.append(f"{type(error).__name__}: {error}")
                    stop.set()

        async def monitor() -> None:
            while not stop.is_set():
                report["events"].append({"at": time.monotonic(), **gateway.state()})
                await asyncio.sleep(0.05)

        workers = [asyncio.create_task(traffic(session)) for session in old_sessions]
        observer = asyncio.create_task(monitor())
        try:
            await asyncio.sleep(0.5)
            engines[0].hold_b = True
            publish_started = time.monotonic()
            publishing = asyncio.create_task(gateway.manager.publish(artifacts["B"]))
            ready = asyncio.create_task(engines[0].b_ready.wait())
            done, _ = await asyncio.wait((publishing, ready), return_when=asyncio.FIRST_COMPLETED)
            if publishing in done:
                ready.cancel()
                await publishing  # Surface failure without waiting forever on the gate.
                raise AssertionError("B was committed before the partial-readiness gate")
            during = routed_session("during-publication", 1)
            assert gateway.bind(during)["version_id"] == "A"
            await generate(during)
            await gateway.close(during)
            report["checks"]["partial_readiness_uses_A"] = True
            gate_released = time.monotonic()
            engines[0].release_b.set()
            await publishing
            publish_finished = time.monotonic()
            for index, old in enumerate(old_sessions):
                new = routed_session(f"new-{index}", index % 2)
                gateway.bind(new)
                gateway.affinity[new] = gateway.affinity[old]
                # A-warm -> B and B-warm -> A; compare actual numeric output.
                await generate(old)
                b_after_a = await generate(new)
                await generate(new)
                a_after_b = await generate(old)
                if b_after_a["meta_info"]["lora_engine"] != a_after_b["meta_info"]["lora_engine"]:
                    raise AssertionError("cache comparison moved between engines")
                cache_checks.append(
                    {
                        "engine": b_after_a["meta_info"]["lora_engine"]["engine_id"],
                        "B_after_A_cached_tokens": b_after_a["meta_info"].get("cached_tokens", 0),
                        "A_after_B_cached_tokens": a_after_b["meta_info"].get("cached_tokens", 0),
                    }
                )
                await gateway.close(new)
            # Abort a real, longer backend attempt without closing the A Session;
            # its subsequent numerical request must still match the A baseline.
            rid = uuid4().hex
            aborting = asyncio.create_task(
                gateway.generate(
                    {
                        **payload,
                        "rid": rid,
                        "session_id": old_sessions[0],
                        "sampling_params": {"max_new_tokens": 512, "ignore_eos": True, "temperature": 0},
                    }
                )
            )
            await asyncio.sleep(0.05)
            await gateway.abort(rid)
            aborted = await aborting
            if aborted["meta_info"]["finish_reason"]["type"] != "abort":
                raise AssertionError("abort experiment finished before cancellation; no abort/resume evidence")
            await generate(old_sessions[0])
            report["checks"]["abort_resume_A"] = True
            c = snapshot_adapter(args.fixtures / "A", args.store, version_id="C", base_model_digest=base)
            try:
                await gateway.manager.publish(c)
            except AdapterCapacityError:
                report["checks"]["capacity_with_live_A"] = True
            else:
                raise AssertionError("C published while A still had live Sessions")
            await asyncio.sleep(0.5)
        finally:
            engines[0].release_b.set()
            stop.set()
            _, pending = await asyncio.wait([*workers, observer], timeout=5)
            if pending:
                failures.append("traffic did not drain within five seconds; experiment failed")
                for task in pending:
                    task.cancel()
            await asyncio.gather(*workers, observer, return_exceptions=True)
            report["traffic"] = {"completions": completions, "latency_seconds": latencies, "failures": failures}
            for session in old_sessions:
                try:
                    await asyncio.wait_for(gateway.close(session), timeout=10)
                except Exception as error:
                    failures.append(f"Session cleanup unconfirmed: {type(error).__name__}: {error}")
        if failures:
            raise AssertionError(f"generation failures: {failures}")
        overlap = [value for value in completions if publish_started <= value <= publish_finished]
        report["publication_seconds"] = publish_finished - publish_started
        report["publication_after_test_gate_seconds"] = publish_finished - gate_released
        report["completed_during_publication"] = len(overlap)
        report["completed_after_test_gate"] = sum(gate_released <= value <= publish_finished for value in completions)
        window = [publish_started, *sorted(overlap), publish_finished]
        report["maximum_publication_progress_gap"] = max(right - left for left, right in zip(window, window[1:]))
        report["failures"] = failures
        report["cache_hits"] = cache_hits
        report["cache_checks"] = cache_checks
        ordered = sorted(latencies)
        report["request_latency_seconds"] = {
            "p50": ordered[len(ordered) // 2],
            "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
            "p99": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))],
        }
        if (
            not overlap
            or not report["completed_after_test_gate"]
            or report["maximum_publication_progress_gap"] > args.max_progress_gap
        ):
            raise AssertionError("insufficient generation progress during publication")
        if not cache_checks or not all(check["A_after_B_cached_tokens"] > 0 for check in cache_checks):
            raise AssertionError("no cache reuse observed; cache isolation evidence is incomplete")
        await gateway.tick()
        await gateway.manager.collect()
        await gateway.manager.publish(c)
        report["checks"].update(numerical_binding=True, bidirectional_cache=True, capacity_after_release=True)
        report["final_state"] = gateway.state()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-url", action="append", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--max-progress-gap", type=float, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--topology", required=True, help="GPU model, count, engine placement and shared-GPU details")
    parser.add_argument("--timeout-seconds", type=float, default=900)
    args = parser.parse_args()
    if len(args.engine_url) != 2:
        parser.error("two --engine-url targets are required")
    report = {
        "status": "running",
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    try:
        asyncio.run(asyncio.wait_for(verify(args, report), args.timeout_seconds))
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    else:
        report["status"] = "passed"
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
