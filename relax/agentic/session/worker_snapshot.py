# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import inspect
import math
import secrets
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

import httpx
import ray

from relax.agentic.session.contracts import (
    WorkerPressureState,
    WorkerSnapshot,
    WorkerSnapshotBatch,
)
from relax.agentic.session.sglang_capabilities import (
    SGLangCacheKind,
    SGLangCapabilityProfile,
    SGLangRouterKind,
    SGLangTopology,
)
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

AGENTIC_WORKER_SNAPSHOT_PUBLISHER_NAME = "agentic_worker_snapshot_publisher"
WORKER_SNAPSHOT_SOURCE_ID = "sglang-native-router"
WORKER_SNAPSHOT_INTERVAL_S = 1.0
WORKER_SNAPSHOT_TTL_S = 5.0
_WORKER_SNAPSHOT_CALL_TIMEOUT_S = 2.0
_WORKER_SNAPSHOT_RPC_TIMEOUT_S = 2.0
_WORKER_SNAPSHOT_FANOUT_CONCURRENCY = 16
_PRESSURE_TOKEN_USAGE_THRESHOLD = 0.9


class WorkerSnapshotSchemaError(ValueError):
    pass


@dataclass(frozen=True, kw_only=True)
class WorkerRegistryEntry:
    worker_id: str
    engine_epoch: str
    url: str
    healthy: bool


@dataclass(frozen=True, kw_only=True)
class WorkerSnapshotSample:
    batch: WorkerSnapshotBatch
    consensus_weight_version: str | None


GetFn = Callable[[str], Awaitable[Any]]


def worker_snapshot_capability_enabled(profile: SGLangCapabilityProfile) -> bool:
    return bool(
        profile.router_kind == SGLangRouterKind.NATIVE
        and profile.topology == SGLangTopology.REGULAR
        and profile.cache_kind == SGLangCacheKind.DEVICE_RADIX
        and profile.supports_worker_registry
        and profile.session_wire_enabled
    )


def _require_non_empty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerSnapshotSchemaError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_non_negative_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise WorkerSnapshotSchemaError(f"{field_name} must be a non-negative integer")
    return value


def _require_non_negative_number(value: Any, *, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
        raise WorkerSnapshotSchemaError(f"{field_name} must be a finite non-negative number")
    return float(value)


def parse_worker_registry(payload: Any) -> tuple[WorkerRegistryEntry, ...]:
    if not isinstance(payload, dict) or not isinstance(payload.get("workers"), list):
        raise WorkerSnapshotSchemaError("router worker registry must contain a workers list")
    entries: list[WorkerRegistryEntry] = []
    seen_worker_ids: set[str] = set()
    seen_urls: set[str] = set()
    for index, worker in enumerate(payload["workers"]):
        if not isinstance(worker, dict):
            raise WorkerSnapshotSchemaError(f"workers[{index}] must be an object")
        worker_id = _require_non_empty_string(worker.get("id"), field_name=f"workers[{index}].id")
        url = _require_non_empty_string(worker.get("url"), field_name=f"workers[{index}].url").rstrip("/")
        parsed_url = urlsplit(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise WorkerSnapshotSchemaError(f"workers[{index}].url must be an HTTP URL")
        healthy = worker.get("is_healthy")
        if not isinstance(healthy, bool):
            raise WorkerSnapshotSchemaError(f"workers[{index}].is_healthy must be a boolean")
        if worker_id in seen_worker_ids or url in seen_urls:
            raise WorkerSnapshotSchemaError("router worker registry contains duplicate workers")
        seen_worker_ids.add(worker_id)
        seen_urls.add(url)
        entries.append(
            WorkerRegistryEntry(
                worker_id=worker_id,
                engine_epoch=worker_id,
                url=url,
                healthy=healthy,
            )
        )
    return tuple(entries)


def parse_worker_loads(payload: Any) -> tuple[int, WorkerPressureState]:
    if not isinstance(payload, dict) or not isinstance(payload.get("loads"), list):
        raise WorkerSnapshotSchemaError("worker loads response must contain a loads list")
    loads = payload["loads"]
    dp_rank_count = _require_non_negative_int(payload.get("dp_rank_count"), field_name="dp_rank_count")
    if dp_rank_count <= 0 or dp_rank_count != len(loads):
        raise WorkerSnapshotSchemaError("worker loads must cover every DP rank")

    capacity = 0
    critical = False
    seen_dp_ranks: set[int] = set()
    for index, load in enumerate(loads):
        if not isinstance(load, dict):
            raise WorkerSnapshotSchemaError(f"loads[{index}] must be an object")
        dp_rank = _require_non_negative_int(load.get("dp_rank"), field_name=f"loads[{index}].dp_rank")
        if dp_rank in seen_dp_ranks:
            raise WorkerSnapshotSchemaError("worker loads contain duplicate DP ranks")
        seen_dp_ranks.add(dp_rank)
        max_total_num_tokens = _require_non_negative_int(
            load.get("max_total_num_tokens"),
            field_name=f"loads[{index}].max_total_num_tokens",
        )
        if max_total_num_tokens <= 0:
            raise WorkerSnapshotSchemaError("max_total_num_tokens must be positive")
        token_usage = _require_non_negative_number(
            load.get("token_usage"),
            field_name=f"loads[{index}].token_usage",
        )
        num_waiting_reqs = _require_non_negative_int(
            load.get("num_waiting_reqs"),
            field_name=f"loads[{index}].num_waiting_reqs",
        )
        num_running_reqs = _require_non_negative_int(
            load.get("num_running_reqs"),
            field_name=f"loads[{index}].num_running_reqs",
        )
        max_running_requests = _require_non_negative_int(
            load.get("max_running_requests"),
            field_name=f"loads[{index}].max_running_requests",
        )
        if max_running_requests <= 0:
            raise WorkerSnapshotSchemaError("max_running_requests must be positive")
        queues = load.get("queues")
        if not isinstance(queues, dict):
            raise WorkerSnapshotSchemaError(f"loads[{index}].queues must be an object")
        retracted = _require_non_negative_int(
            queues.get("retracted"),
            field_name=f"loads[{index}].queues.retracted",
        )
        capacity += max_total_num_tokens
        critical = critical or (
            token_usage >= _PRESSURE_TOKEN_USAGE_THRESHOLD
            or num_waiting_reqs > 0
            or num_running_reqs >= max_running_requests
            or retracted > 0
        )
    if seen_dp_ranks != set(range(dp_rank_count)):
        raise WorkerSnapshotSchemaError("worker loads must contain contiguous DP ranks")
    return (
        capacity,
        WorkerPressureState.CRITICAL if critical else WorkerPressureState.NORMAL,
    )


def parse_weight_version(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise WorkerSnapshotSchemaError("weight version response must be an object")
    version = _require_non_empty_string(payload.get("weight_version"), field_name="weight_version")
    if version.lower() == "default":
        raise WorkerSnapshotSchemaError("default is not a serving weight version")
    return version


async def _await_result(value: Any, *, timeout_s: float) -> Any:
    if inspect.isawaitable(value):
        return await asyncio.wait_for(asyncio.shield(value), timeout=timeout_s)
    return value


async def _invoke_remote(target: Any, method_name: str, *, timeout_s: float, **kwargs: Any) -> Any:
    method = getattr(target, method_name)
    remote = getattr(method, "remote", None)
    value = remote(**kwargs) if remote is not None else method(**kwargs)
    return await _await_result(value, timeout_s=timeout_s)


class WorkerSnapshotPublisherCore:
    def __init__(
        self,
        *,
        router_ip: str,
        router_port: int,
        coordinator: Any,
        shards: list[Any],
        get_fn: GetFn,
        publisher_epoch: str | None = None,
        source_id: str = WORKER_SNAPSHOT_SOURCE_ID,
        call_timeout_s: float = _WORKER_SNAPSHOT_CALL_TIMEOUT_S,
        rpc_timeout_s: float = _WORKER_SNAPSHOT_RPC_TIMEOUT_S,
    ) -> None:
        if not router_ip or int(router_port) <= 0:
            raise ValueError("a valid SGLang router address is required")
        self._router_url = f"http://{router_ip}:{int(router_port)}"
        self._coordinator = coordinator
        self._shards = list(shards)
        self._get = get_fn
        self._publisher_epoch = publisher_epoch or f"snapshot-{secrets.token_hex(16)}"
        self._source_id = source_id
        self._call_timeout_s = float(call_timeout_s)
        self._rpc_timeout_s = float(rpc_timeout_s)
        self._batch_seq = 0
        self._coordinator_epoch: str | None = None
        self._fanout_semaphore = asyncio.Semaphore(_WORKER_SNAPSHOT_FANOUT_CONCURRENCY)

    @property
    def publisher_epoch(self) -> str:
        return self._publisher_epoch

    async def fence_source(self) -> None:
        capacity_generation = _require_non_negative_int(
            await _invoke_remote(
                self._coordinator,
                "fence_worker_snapshot_source",
                timeout_s=self._rpc_timeout_s,
                source_id=self._source_id,
                publisher_epoch=self._publisher_epoch,
            ),
            field_name="capacity_generation",
        )
        health = await _invoke_remote(
            self._coordinator,
            "health",
            timeout_s=self._rpc_timeout_s,
        )
        if not isinstance(health, dict):
            raise RuntimeError("admission coordinator returned invalid health metadata")
        self._coordinator_epoch = _require_non_empty_string(
            health.get("coordinator_epoch"),
            field_name="coordinator_epoch",
        )
        results = await asyncio.gather(
            *(
                _invoke_remote(
                    shard,
                    "fence_worker_snapshot_source",
                    timeout_s=self._rpc_timeout_s,
                    coordinator_epoch=self._coordinator_epoch,
                    capacity_generation=capacity_generation,
                    source_id=self._source_id,
                    publisher_epoch=self._publisher_epoch,
                )
                for shard in self._shards
            ),
            return_exceptions=True,
        )
        failed = sum(isinstance(result, BaseException) or result is not True for result in results)
        if failed:
            logger.warning("Failed to fence worker snapshot source on %d session shards", failed)

    async def _get_json(self, url: str) -> Any:
        return await asyncio.wait_for(self._get(url), timeout=self._call_timeout_s)

    async def _sample_worker(self, worker: WorkerRegistryEntry) -> WorkerSnapshot:
        async with self._fanout_semaphore:
            loads_payload, version_payload = await asyncio.gather(
                self._get_json(f"{worker.url}/v1/loads"),
                self._get_json(f"{worker.url}/get_weight_version"),
            )
        capacity, pressure_state = parse_worker_loads(loads_payload)
        version = parse_weight_version(version_payload)
        return WorkerSnapshot(
            worker_id=worker.worker_id,
            engine_epoch=worker.engine_epoch,
            serving_weight_version=version,
            safe_execution_capacity_tokens=capacity,
            pressure_state=pressure_state,
        )

    async def _publish(self, sample: WorkerSnapshotSample) -> None:
        if self._coordinator_epoch is None:
            raise RuntimeError("worker snapshot source must be fenced before publishing")
        capacity_generation = _require_non_negative_int(
            await _invoke_remote(
                self._coordinator,
                "replace_worker_snapshots",
                timeout_s=self._rpc_timeout_s,
                batch=sample.batch,
            ),
            field_name="capacity_generation",
        )
        results = await asyncio.gather(
            *(
                _invoke_remote(
                    shard,
                    "accept_worker_snapshot_version",
                    timeout_s=self._rpc_timeout_s,
                    coordinator_epoch=self._coordinator_epoch,
                    capacity_generation=capacity_generation,
                    source_id=sample.batch.source_id,
                    publisher_epoch=sample.batch.publisher_epoch,
                    batch_seq=sample.batch.batch_seq,
                    serving_weight_version=sample.consensus_weight_version,
                    source_open=sample.batch.source_open,
                    complete=sample.batch.complete,
                )
                for shard in self._shards
            ),
            return_exceptions=True,
        )
        failed = sum(isinstance(result, BaseException) or result is not True for result in results)
        if failed:
            logger.warning("Failed to publish serving weight version to %d session shards", failed)

    async def sample_once(self) -> WorkerSnapshotSample:
        self._batch_seq += 1
        try:
            registry_payload = await self._get_json(f"{self._router_url}/workers")
            registry = parse_worker_registry(registry_payload)
        except Exception:
            sample = WorkerSnapshotSample(
                batch=WorkerSnapshotBatch(
                    source_id=self._source_id,
                    publisher_epoch=self._publisher_epoch,
                    batch_seq=self._batch_seq,
                    source_open=False,
                    complete=False,
                    snapshots=(),
                ),
                consensus_weight_version=None,
            )
            await self._publish(sample)
            return sample

        healthy_workers = tuple(worker for worker in registry if worker.healthy)
        results = await asyncio.gather(
            *(self._sample_worker(worker) for worker in healthy_workers),
            return_exceptions=True,
        )
        snapshots = tuple(result for result in results if isinstance(result, WorkerSnapshot))
        versions = {snapshot.serving_weight_version for snapshot in snapshots}
        complete = bool(healthy_workers) and len(snapshots) == len(healthy_workers) and len(versions) == 1
        consensus_weight_version = next(iter(versions)) if complete else None
        sample = WorkerSnapshotSample(
            batch=WorkerSnapshotBatch(
                source_id=self._source_id,
                publisher_epoch=self._publisher_epoch,
                batch_seq=self._batch_seq,
                source_open=True,
                complete=complete,
                snapshots=snapshots,
            ),
            consensus_weight_version=consensus_weight_version,
        )
        await self._publish(sample)
        return sample


@ray.remote(num_cpus=0.25, max_concurrency=2, max_restarts=0, max_task_retries=0)
class WorkerSnapshotPublisher:
    def __init__(
        self,
        *,
        router_ip: str,
        router_port: int,
        coordinator: Any,
        shards: list[Any],
    ) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(None))

        async def get_json(url: str) -> Any:
            response = await self._client.get(url)
            response.raise_for_status()
            return response.json()

        self._core = WorkerSnapshotPublisherCore(
            router_ip=router_ip,
            router_port=router_port,
            coordinator=coordinator,
            shards=shards,
            get_fn=get_json,
        )
        self._stopped = asyncio.Event()
        self._run_task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        while not self._stopped.is_set():
            try:
                await self._core.fence_source()
                break
            except Exception as exc:
                logger.warning("Failed to fence worker snapshot publisher source: %s", exc)
                try:
                    await asyncio.wait_for(self._stopped.wait(), timeout=WORKER_SNAPSHOT_INTERVAL_S)
                except asyncio.TimeoutError:
                    pass
        while not self._stopped.is_set():
            try:
                await self._core.sample_once()
            except Exception as exc:
                logger.warning("Worker snapshot sampling failed; admission will fail open: %s", exc)
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=WORKER_SNAPSHOT_INTERVAL_S)
            except asyncio.TimeoutError:
                pass
        await self._client.aclose()

    async def start(self) -> None:
        if self._run_task is None:
            self._run_task = asyncio.create_task(self._run())

    def health(self) -> dict[str, Any]:
        return {
            "ok": not self._stopped.is_set(),
            "publisher_epoch": self._core.publisher_epoch,
        }

    async def stop(self) -> None:
        self._stopped.set()
        if self._run_task is not None:
            await self._run_task


def create_worker_snapshot_publisher(
    *,
    router_ip: str,
    router_port: int,
    coordinator: Any,
    shards: list[Any],
):
    shutdown_worker_snapshot_publisher()
    publisher = WorkerSnapshotPublisher.options(
        name=AGENTIC_WORKER_SNAPSHOT_PUBLISHER_NAME,
    ).remote(
        router_ip=router_ip,
        router_port=router_port,
        coordinator=coordinator,
        shards=shards,
    )
    ray.get(publisher.health.remote(), timeout=10)
    ray.get(publisher.start.remote(), timeout=10)
    return publisher


def shutdown_worker_snapshot_publisher() -> None:
    try:
        publisher = ray.get_actor(AGENTIC_WORKER_SNAPSHOT_PUBLISHER_NAME)
    except Exception:
        return
    try:
        ray.get(publisher.stop.remote(), timeout=2)
    except Exception:
        pass
    try:
        ray.kill(publisher, no_restart=True)
    except Exception:
        return
