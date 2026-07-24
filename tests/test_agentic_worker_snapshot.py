# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from relax.agentic.pipeline import runtime
from relax.agentic.pipeline.runtime import (
    BackendServingWeightVersionMismatchError,
    SGLangBackendAdapter,
)
from relax.agentic.session import service as session_service
from relax.agentic.session.contracts import WorkerPressureState
from relax.agentic.session.service import AgenticSessionShard
from relax.agentic.session.sglang_capabilities import (
    resolve_sglang_capability_profile,
    unavailable_sglang_capability_profile,
)
from relax.agentic.session.worker_snapshot import (
    WORKER_SNAPSHOT_TTL_S,
    WorkerSnapshotPublisherCore,
    WorkerSnapshotSchemaError,
    parse_weight_version,
    parse_worker_loads,
    parse_worker_registry,
    worker_snapshot_capability_enabled,
)


def _registry(*, version_two_workers: bool = True):
    workers = [
        {
            "id": "registration-1",
            "url": "http://worker-1/",
            "is_healthy": True,
        }
    ]
    if version_two_workers:
        workers.append(
            {
                "id": "registration-2",
                "url": "http://worker-2",
                "is_healthy": True,
            }
        )
    return {"workers": workers}


def _loads(
    *,
    token_usage: float = 0.2,
    waiting: int = 0,
    retracted: int = 0,
    running: int = 2,
    max_running: int = 8,
):
    return {
        "dp_rank_count": 2,
        "loads": [
            {
                "dp_rank": 0,
                "max_total_num_tokens": 1000,
                "token_usage": token_usage,
                "num_waiting_reqs": waiting,
                "num_running_reqs": running,
                "max_running_requests": max_running,
                "queues": {"retracted": retracted},
            },
            {
                "dp_rank": 1,
                "max_total_num_tokens": 1500,
                "token_usage": 0.3,
                "num_waiting_reqs": 0,
                "num_running_reqs": 3,
                "max_running_requests": 8,
                "queues": {"retracted": 0},
            },
        ],
    }


class _Coordinator:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.generation = 0
        self.batches = []

    async def fence_worker_snapshot_source(self, **kwargs):
        self.events.append("coordinator:fence")
        self.generation += 1
        return self.generation

    async def health(self):
        return {"coordinator_epoch": "coordinator-1"}

    async def replace_worker_snapshots(self, *, batch):
        self.events.append("coordinator:replace")
        self.generation += 1
        self.batches.append(batch)
        return self.generation


class _Shard:
    def __init__(self, events: list[str], name: str) -> None:
        self.events = events
        self.name = name
        self.fences = []
        self.updates = []

    async def fence_worker_snapshot_source(self, **kwargs):
        self.events.append(f"{self.name}:fence")
        self.fences.append(kwargs)
        return True

    async def accept_worker_snapshot_version(self, **kwargs):
        self.events.append(f"{self.name}:update")
        self.updates.append(kwargs)
        return True


def test_worker_registry_requires_registration_incarnation() -> None:
    entries = parse_worker_registry(_registry(version_two_workers=False))

    assert entries[0].worker_id == "registration-1"
    assert entries[0].engine_epoch == "registration-1"
    assert entries[0].url == "http://worker-1"
    with pytest.raises(WorkerSnapshotSchemaError, match=r"workers\[0\]\.id"):
        parse_worker_registry(
            {
                "workers": [
                    {
                        "url": "http://worker-1",
                        "is_healthy": True,
                    }
                ]
            }
        )


@pytest.mark.parametrize(
    ("overrides", "expected_pressure"),
    [
        ({}, WorkerPressureState.NORMAL),
        ({"token_usage": 0.9}, WorkerPressureState.CRITICAL),
        ({"waiting": 1}, WorkerPressureState.CRITICAL),
        ({"retracted": 1}, WorkerPressureState.CRITICAL),
        ({"running": 8, "max_running": 8}, WorkerPressureState.CRITICAL),
    ],
)
def test_worker_loads_sum_dp_capacity_and_compute_pressure(overrides, expected_pressure) -> None:
    capacity, pressure = parse_worker_loads(_loads(**overrides))

    assert capacity == 2500
    assert pressure == expected_pressure


def test_worker_loads_require_complete_dp_schema() -> None:
    payload = _loads()
    del payload["loads"][0]["queues"]

    with pytest.raises(WorkerSnapshotSchemaError, match="queues"):
        parse_worker_loads(payload)


def test_weight_version_rejects_default_and_missing() -> None:
    assert parse_weight_version({"weight_version": "weight-7"}) == "weight-7"
    with pytest.raises(WorkerSnapshotSchemaError, match="default"):
        parse_weight_version({"weight_version": "default"})
    with pytest.raises(WorkerSnapshotSchemaError, match="weight_version"):
        parse_weight_version({})


def test_worker_snapshot_capability_is_native_regular_radix_only() -> None:
    enabled = resolve_sglang_capability_profile(
        router_managed=True,
        use_slime_router=False,
        has_pd_disaggregation=False,
        radix_cache_disabled=False,
        hierarchical_cache_enabled=False,
        lifecycle_ready=True,
    )

    assert worker_snapshot_capability_enabled(enabled) is True
    assert worker_snapshot_capability_enabled(unavailable_sglang_capability_profile()) is False


def _ready_profile():
    return resolve_sglang_capability_profile(
        router_managed=True,
        use_slime_router=False,
        has_pd_disaggregation=False,
        radix_cache_disabled=False,
        hierarchical_cache_enabled=False,
        lifecycle_ready=True,
    )


def _launcher_config():
    return SimpleNamespace(
        sglang_server_concurrency=8,
        rollout_num_gpus=2,
        rollout_num_gpus_per_engine=1,
        sglang_router_ip="router",
        sglang_router_port=30000,
    )


def _mock_shard_launcher(monkeypatch):
    coordinator = SimpleNamespace(
        register_owner=SimpleNamespace(remote=lambda **kwargs: kwargs),
    )
    shard = SimpleNamespace(
        name="shard-0",
        register_admission_budget_owner=SimpleNamespace(remote=lambda: None),
    )

    class _ShardOptions:
        def remote(self, config, **kwargs):
            del config, kwargs
            return shard

    def missing_actor(name):
        del name
        raise ValueError("missing")

    monkeypatch.setattr(session_service, "_DEFAULT_SESSION_SHARD_COUNT", 1)
    monkeypatch.setattr(session_service, "_STALE_SESSION_SHARD_CLEANUP_LIMIT", 1)
    monkeypatch.setattr(session_service.ray, "get_actor", missing_actor)
    monkeypatch.setattr(session_service.ray, "get", lambda refs, timeout: None)
    monkeypatch.setattr(session_service, "create_admission_budget_coordinator", lambda: coordinator)
    monkeypatch.setattr(session_service.AgenticSessionShard, "options", lambda **kwargs: _ShardOptions())
    return coordinator, shard


def test_shard_launcher_starts_publisher_after_coordinator_and_shards(monkeypatch) -> None:
    coordinator, shard = _mock_shard_launcher(monkeypatch)
    publisher_calls = []
    monkeypatch.setattr(session_service, "shutdown_worker_snapshot_publisher", lambda: None)
    monkeypatch.setattr(
        session_service,
        "create_worker_snapshot_publisher",
        lambda **kwargs: publisher_calls.append(kwargs),
    )

    handles = session_service.create_agentic_session_shards(
        _launcher_config(),
        sglang_capability_profile=_ready_profile(),
    )

    assert handles == [shard]
    assert publisher_calls == [
        {
            "router_ip": "router",
            "router_port": 30000,
            "coordinator": coordinator,
            "shards": [shard],
        }
    ]


def test_shard_launcher_fails_open_when_publisher_startup_fails(monkeypatch) -> None:
    _, shard = _mock_shard_launcher(monkeypatch)
    shutdown_calls = []
    monkeypatch.setattr(
        session_service,
        "shutdown_worker_snapshot_publisher",
        lambda: shutdown_calls.append(True),
    )
    monkeypatch.setattr(
        session_service,
        "create_worker_snapshot_publisher",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("publisher unavailable")),
    )

    handles = session_service.create_agentic_session_shards(
        _launcher_config(),
        sglang_capability_profile=_ready_profile(),
    )

    assert handles == [shard]
    assert len(shutdown_calls) == 2


@pytest.mark.asyncio
async def test_publisher_submits_coordinator_before_consensus_to_shards() -> None:
    events: list[str] = []
    coordinator = _Coordinator(events)
    shards = [_Shard(events, "shard-0"), _Shard(events, "shard-1")]
    responses = {
        "http://router:30000/workers": _registry(),
        "http://worker-1/v1/loads": _loads(),
        "http://worker-2/v1/loads": _loads(token_usage=0.9),
        "http://worker-1/get_weight_version": {"weight_version": "weight-7"},
        "http://worker-2/get_weight_version": {"weight_version": "weight-7"},
    }

    async def get_json(url):
        return responses[url]

    publisher = WorkerSnapshotPublisherCore(
        router_ip="router",
        router_port=30000,
        coordinator=coordinator,
        shards=shards,
        get_fn=get_json,
        publisher_epoch="publisher-1",
    )
    await publisher.fence_source()
    events.clear()

    sample = await publisher.sample_once()

    assert sample.batch.complete is True
    assert sample.consensus_weight_version == "weight-7"
    assert [snapshot.safe_execution_capacity_tokens for snapshot in sample.batch.snapshots] == [2500, 2500]
    assert sample.batch.snapshots[1].pressure_state == WorkerPressureState.CRITICAL
    assert events[0] == "coordinator:replace"
    assert set(events[1:]) == {"shard-0:update", "shard-1:update"}
    assert all(update["serving_weight_version"] == "weight-7" for shard in shards for update in shard.updates)


@pytest.mark.asyncio
async def test_publisher_partial_worker_failure_commits_incomplete_batch() -> None:
    events: list[str] = []
    coordinator = _Coordinator(events)
    shard = _Shard(events, "shard-0")

    async def get_json(url):
        if url.endswith("/workers"):
            return _registry()
        if url.startswith("http://worker-2"):
            raise RuntimeError("worker unavailable")
        if url.endswith("/v1/loads"):
            return _loads()
        return {"weight_version": "weight-7"}

    publisher = WorkerSnapshotPublisherCore(
        router_ip="router",
        router_port=30000,
        coordinator=coordinator,
        shards=[shard],
        get_fn=get_json,
        publisher_epoch="publisher-1",
    )
    await publisher.fence_source()
    events.clear()

    sample = await publisher.sample_once()

    assert sample.batch.source_open is True
    assert sample.batch.complete is False
    assert sample.consensus_weight_version is None
    assert len(sample.batch.snapshots) == 1
    assert events == ["coordinator:replace", "shard-0:update"]
    assert shard.updates[-1]["complete"] is False


@pytest.mark.asyncio
async def test_publisher_mixed_versions_and_bad_registry_fail_open() -> None:
    events: list[str] = []
    coordinator = _Coordinator(events)
    shard = _Shard(events, "shard-0")
    registry_is_valid = True

    async def get_json(url):
        if url.endswith("/workers"):
            return _registry() if registry_is_valid else {"urls": ["http://worker-1"]}
        if url.endswith("/v1/loads"):
            return _loads()
        version = "weight-1" if "worker-1" in url else "weight-2"
        return {"weight_version": version}

    publisher = WorkerSnapshotPublisherCore(
        router_ip="router",
        router_port=30000,
        coordinator=coordinator,
        shards=[shard],
        get_fn=get_json,
        publisher_epoch="publisher-1",
    )
    await publisher.fence_source()

    mixed = await publisher.sample_once()
    registry_is_valid = False
    closed = await publisher.sample_once()

    assert mixed.batch.source_open is True
    assert mixed.batch.complete is False
    assert closed.batch.source_open is False
    assert closed.batch.complete is False
    assert closed.batch.batch_seq == mixed.batch.batch_seq + 1


class _Scheduler:
    def __init__(self) -> None:
        self.notifications = 0

    async def notify_capacity_changed(self) -> None:
        self.notifications += 1


@pytest.mark.asyncio
async def test_shard_snapshot_fencing_rejects_old_epoch_and_sequence() -> None:
    shard_cls = AgenticSessionShard.__ray_metadata__.modified_class
    shard = object.__new__(shard_cls)
    shard._program_scheduler = _Scheduler()
    shard._snapshot_coordinator_epoch = None
    shard._retired_snapshot_coordinator_epochs = set()
    shard._snapshot_capacity_generation = -1
    shard._snapshot_source_id = None
    shard._snapshot_publisher_epoch = None
    shard._snapshot_batch_seq = -1
    shard._serving_weight_version = None
    shard._snapshot_received_at = None

    assert await shard.fence_worker_snapshot_source(
        coordinator_epoch="coordinator-1",
        capacity_generation=1,
        source_id="router",
        publisher_epoch="publisher-1",
    )
    assert await shard.accept_worker_snapshot_version(
        coordinator_epoch="coordinator-1",
        capacity_generation=2,
        source_id="router",
        publisher_epoch="publisher-1",
        batch_seq=1,
        serving_weight_version="weight-1",
        source_open=True,
        complete=True,
    )
    assert shard._active_serving_weight_version() == "weight-1"
    assert not await shard.accept_worker_snapshot_version(
        coordinator_epoch="coordinator-1",
        capacity_generation=2,
        source_id="router",
        publisher_epoch="publisher-1",
        batch_seq=1,
        serving_weight_version="weight-old",
        source_open=True,
        complete=True,
    )
    assert await shard.fence_worker_snapshot_source(
        coordinator_epoch="coordinator-1",
        capacity_generation=3,
        source_id="router",
        publisher_epoch="publisher-2",
    )
    assert not await shard.accept_worker_snapshot_version(
        coordinator_epoch="coordinator-1",
        capacity_generation=2,
        source_id="router",
        publisher_epoch="publisher-1",
        batch_seq=2,
        serving_weight_version="weight-old",
        source_open=True,
        complete=True,
    )
    assert shard._active_serving_weight_version() is None


def test_shard_snapshot_version_expires_locally() -> None:
    shard_cls = AgenticSessionShard.__ray_metadata__.modified_class
    shard = object.__new__(shard_cls)
    shard._serving_weight_version = "weight-1"
    shard._snapshot_received_at = time.monotonic() - WORKER_SNAPSHOT_TTL_S - 0.1

    assert shard._active_serving_weight_version() is None


def _backend_adapter() -> SGLangBackendAdapter:
    adapter = object.__new__(SGLangBackendAdapter)
    adapter._resolved_router_ip = "router"
    adapter._resolved_router_port = 30000
    adapter._use_rollout_routing_replay = False
    adapter._router_policy = "random"
    adapter._slime_router_sticky = False
    adapter._capability_profile = unavailable_sglang_capability_profile()
    adapter.tokenizer = object()
    adapter.compiler = SimpleNamespace(processor=None)
    return adapter


@pytest.mark.asyncio
async def test_backend_rejects_known_serving_weight_version_mismatch(monkeypatch) -> None:
    async def post_generate(*args, **kwargs):
        return {
            "output_ids": [1],
            "meta_info": {
                "weight_version": "weight-2",
                "finish_reason": {"type": "stop"},
            },
        }

    monkeypatch.setattr(runtime, "post", post_generate)
    monkeypatch.setattr(
        runtime,
        "_extract_output_tokens_and_log_probs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mismatched output must not be decoded")),
    )

    with pytest.raises(BackendServingWeightVersionMismatchError, match="expected=weight-1"):
        await _backend_adapter().generate(
            input_ids=[1],
            sampling_params={},
            session_id=None,
            expected_serving_weight_version="weight-1",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("actual_version", "expected_version"),
    [
        (None, "weight-1"),
        ("weight-2", None),
        ("weight-1", "weight-1"),
    ],
)
async def test_backend_allows_matching_missing_observation_or_bypass(
    monkeypatch,
    actual_version,
    expected_version,
) -> None:
    async def post_generate(*args, **kwargs):
        return {
            "output_ids": [1],
            "meta_info": {
                "weight_version": actual_version,
                "finish_reason": {"type": "stop"},
            },
        }

    monkeypatch.setattr(runtime, "post", post_generate)
    monkeypatch.setattr(runtime, "_extract_output_tokens_and_log_probs", lambda *args, **kwargs: ([1], [0.0]))
    result = await _backend_adapter().generate(
        input_ids=[1],
        sampling_params={},
        session_id=None,
        expected_serving_weight_version=expected_version,
    )

    assert result.new_tokens == [1]
