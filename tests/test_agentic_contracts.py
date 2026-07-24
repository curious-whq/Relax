# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
from dataclasses import fields
from types import SimpleNamespace

import pytest

from relax.agentic.pipeline import runtime
from relax.agentic.pipeline.runtime import SGLangBackendAdapter
from relax.agentic.session.contracts import (
    AdmissionAction,
    AdmissionDecision,
    AdmissionGrant,
    AdmissionLease,
    AdmissionReason,
    AgenticIdentity,
    BudgetAcquireResult,
    BudgetAcquireStatus,
    RouteObservation,
    RoutingContext,
    WorkerPressureState,
    WorkerSnapshot,
    WorkerSnapshotBatch,
)
from relax.agentic.session.service import AgenticSessionShard, _SessionRecord
from relax.agentic.session.state import InflightRequest, RequestKind


def _identity(**overrides) -> AgenticIdentity:
    values = {
        "program_id": "program-1",
        "program_owner_key": "owner-1",
        "root_session_id": "root-1",
        "engine_session_id": "engine-1",
        "parent_engine_session_id": None,
    }
    values.update(overrides)
    return AgenticIdentity(**values)


def _routing_context(**overrides) -> RoutingContext:
    values = {
        "request_id": "request-1",
        "dispatch_id": "dispatch-1",
        "owner_epoch": 1,
        "program_id": "program-1",
        "root_session_id": "root-1",
        "engine_session_id": "engine-1",
        "parent_engine_session_id": None,
        "attempt_id": "attempt-1",
        "context_version_id": "context-1",
        "serving_weight_version": "weight-1",
        "prompt_tokens": 32,
        "expected_decode_tokens": 16,
        "priority": 0,
        "affinity_key": "engine-1",
    }
    values.update(overrides)
    return RoutingContext(**values)


def _backend_adapter(*, router_policy: str = "consistent_hashing", slime_router_sticky: bool = False):
    adapter = object.__new__(SGLangBackendAdapter)
    adapter._resolved_router_ip = "router.test"
    adapter._resolved_router_port = 30000
    adapter._use_rollout_routing_replay = False
    adapter._router_policy = router_policy
    adapter._slime_router_sticky = slime_router_sticky
    adapter.tokenizer = object()
    adapter.compiler = SimpleNamespace(processor=None)
    return adapter


def test_agentic_identity_rejects_missing_internal_id() -> None:
    with pytest.raises(ValueError, match="engine_session_id"):
        _identity(engine_session_id="")


def test_routing_context_contains_no_replay_payload() -> None:
    contract_fields = {item.name for item in fields(RoutingContext)}
    assert "input_ids" not in contract_fields
    assert "messages" not in contract_fields
    assert "tool_output" not in contract_fields


@pytest.mark.parametrize("field_name", ["prompt_tokens", "expected_decode_tokens"])
def test_routing_context_rejects_negative_token_counts(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        _routing_context(**{field_name: -1})


def test_admission_decision_tracks_estimate_without_consuming_budget() -> None:
    decision = AdmissionDecision(
        action=AdmissionAction.DEFER,
        reason_code=AdmissionReason.CAPACITY_EXHAUSTED,
        reservation_tokens=48,
        admission_decision_id="decision-1",
        owner_epoch=1,
    )
    assert decision.reservation_tokens == 48

    with pytest.raises(ValueError, match="reservation_tokens"):
        AdmissionDecision(
            action=AdmissionAction.ADMIT,
            reason_code=AdmissionReason.CAPACITY_AVAILABLE,
            reservation_tokens=-1,
            admission_decision_id="decision-2",
            owner_epoch=1,
        )


def test_admission_decision_allows_explicit_bypass() -> None:
    decision = AdmissionDecision(
        action=AdmissionAction.BYPASS,
        reason_code=AdmissionReason.FEATURE_DISABLED,
        reservation_tokens=0,
        admission_decision_id="decision-1",
        owner_epoch=0,
    )
    assert decision.action == AdmissionAction.BYPASS


def test_admission_lease_and_grant_require_matching_fencing_fields() -> None:
    lease = AdmissionLease(
        owner_epoch=7,
        dispatch_id="dispatch-1",
        admission_decision_id="decision-1",
        reservation_tokens=48,
        ttl_s=60.0,
        expires_at_local_monotonic=100.0,
    )
    decision = AdmissionDecision(
        action=AdmissionAction.ADMIT,
        reason_code=AdmissionReason.CAPACITY_AVAILABLE,
        reservation_tokens=48,
        admission_decision_id="decision-1",
        owner_epoch=7,
    )

    assert AdmissionGrant(decision=decision, lease=lease).lease == lease
    with pytest.raises(ValueError, match="does not match"):
        AdmissionGrant(
            decision=decision,
            lease=AdmissionLease(
                owner_epoch=7,
                dispatch_id="dispatch-1",
                admission_decision_id="decision-other",
                reservation_tokens=48,
                ttl_s=60.0,
                expires_at_local_monotonic=100.0,
            ),
        )


@pytest.mark.parametrize("owner_epoch", [0, -1, False])
def test_admission_lease_requires_positive_owner_epoch(owner_epoch: int) -> None:
    with pytest.raises(ValueError, match="owner_epoch"):
        AdmissionLease(
            owner_epoch=owner_epoch,
            dispatch_id="dispatch-1",
            admission_decision_id="decision-1",
            reservation_tokens=1,
            ttl_s=60.0,
            expires_at_local_monotonic=100.0,
        )


def test_budget_acquire_result_only_carries_lease_when_acquired() -> None:
    with pytest.raises(ValueError, match="requires a lease"):
        BudgetAcquireResult(
            status=BudgetAcquireStatus.ACQUIRED,
            reason_code=AdmissionReason.CAPACITY_AVAILABLE,
        )


def test_worker_snapshot_batch_requires_unique_workers() -> None:
    first = WorkerSnapshot(
        worker_id="worker-1",
        engine_epoch="epoch-1",
        serving_weight_version="weight-1",
        safe_execution_capacity_tokens=100,
    )
    with pytest.raises(ValueError, match="duplicate workers"):
        WorkerSnapshotBatch(
            source_id="router-1",
            publisher_epoch="publisher-epoch-1",
            batch_seq=1,
            source_open=True,
            complete=True,
            snapshots=(
                first,
                WorkerSnapshot(
                    worker_id="worker-1",
                    engine_epoch="epoch-2",
                    serving_weight_version="weight-1",
                    safe_execution_capacity_tokens=100,
                    pressure_state=WorkerPressureState.CRITICAL,
                ),
            ),
        )


def test_worker_snapshot_batch_contains_no_remote_freshness_timestamp() -> None:
    batch_fields = {item.name for item in fields(WorkerSnapshotBatch)}
    snapshot_fields = {item.name for item in fields(WorkerSnapshot)}
    assert "received_at" not in batch_fields
    assert "monotonic_timestamp" not in batch_fields
    assert "received_at" not in snapshot_fields
    assert "monotonic_timestamp" not in snapshot_fields


@pytest.mark.parametrize("owner_epoch", [-1, False])
def test_admission_decision_rejects_invalid_owner_epoch(owner_epoch: int) -> None:
    with pytest.raises(ValueError, match="owner_epoch"):
        AdmissionDecision(
            action=AdmissionAction.BYPASS,
            reason_code=AdmissionReason.DEGRADED,
            reservation_tokens=0,
            admission_decision_id="decision-1",
            owner_epoch=owner_epoch,
        )


@pytest.mark.parametrize(
    ("selected_worker_id", "selected_engine_epoch", "expected"),
    [
        ("worker-1", "engine-epoch-1", True),
        ("worker-1", None, False),
        (None, "engine-epoch-1", False),
        (None, None, False),
    ],
)
def test_route_observation_tracks_incomplete_worker_receipt(
    selected_worker_id: str | None,
    selected_engine_epoch: str | None,
    expected: bool,
) -> None:
    observation = RouteObservation(
        request_id="request-1",
        dispatch_id="dispatch-1",
        owner_epoch=1,
        route_decision_id=None,
        selected_worker_id=selected_worker_id,
        selected_engine_epoch=selected_engine_epoch,
        serving_weight_version="weight-1",
        actual_cached_tokens=8,
        prompt_tokens=32,
    )
    assert observation.has_complete_worker_receipt is expected


def test_route_observation_does_not_reject_provider_cache_anomaly() -> None:
    observation = RouteObservation(
        request_id="request-1",
        dispatch_id="dispatch-1",
        owner_epoch=1,
        route_decision_id=None,
        selected_worker_id=None,
        selected_engine_epoch=None,
        serving_weight_version=None,
        actual_cached_tokens=33,
        prompt_tokens=32,
    )
    assert observation.actual_cached_tokens == 33


@pytest.mark.asyncio
async def test_sglang_adapter_preserves_complete_replay_and_engine_meta(monkeypatch) -> None:
    calls = []

    async def fake_post(url, payload, headers=None):
        calls.append((url, payload, headers))
        return {
            "output_ids": [41],
            "meta_info": {
                "cached_tokens": 24,
                "prompt_tokens": 32,
                "weight_version": "weight-1",
                "finish_reason": {"type": "stop"},
            },
        }

    monkeypatch.setattr(runtime, "post", fake_post)
    monkeypatch.setattr(
        runtime,
        "_extract_output_tokens_and_log_probs",
        lambda *args, **kwargs: ([41], [-0.25]),
    )
    adapter = _backend_adapter()
    input_ids = [11, 12, 13, 14]
    sampling_params = {"max_new_tokens": 8}
    image_data = ["image-1"]
    audio_data = ["audio-1"]
    video_data = ["video-1"]

    result = await adapter.generate(
        input_ids=input_ids,
        sampling_params=sampling_params,
        session_id="logical-session-1",
        request_id="request-1",
        image_data=image_data,
        audio_data=audio_data,
        video_data=video_data,
    )

    _, payload, headers = calls[0]
    assert payload["input_ids"] == input_ids
    assert payload["input_ids"] is not input_ids
    assert payload["sampling_params"] == sampling_params
    assert payload["sampling_params"] is not sampling_params
    assert payload["image_data"] == image_data
    assert payload["image_data"] is not image_data
    assert payload["audio_data"] == audio_data
    assert payload["audio_data"] is not audio_data
    assert payload["video_data"] == video_data
    assert payload["video_data"] is not video_data
    assert "session_id" not in payload
    assert headers == {"X-SMG-Routing-Key": "logical-session-1"}
    assert result.meta_info["cached_tokens"] == 24
    assert result.meta_info["prompt_tokens"] == 32
    assert result.meta_info["weight_version"] == "weight-1"


@pytest.mark.parametrize(
    ("router_policy", "slime_router_sticky", "session_id", "expected_headers"),
    [
        ("consistent_hashing", False, "logical-session-1", {"X-SMG-Routing-Key": "logical-session-1"}),
        ("round_robin", True, "logical-session-1", {"X-SMG-Routing-Key": "logical-session-1"}),
        ("round_robin", False, "logical-session-1", None),
        ("consistent_hashing", False, None, None),
    ],
)
@pytest.mark.asyncio
async def test_sglang_adapter_soft_affinity_header_matrix(
    monkeypatch,
    router_policy: str,
    slime_router_sticky: bool,
    session_id: str | None,
    expected_headers: dict[str, str] | None,
) -> None:
    calls = []

    async def fake_post(url, payload, headers=None):
        calls.append((url, payload, headers))
        return {
            "output_ids": [],
            "meta_info": {
                "finish_reason": {"type": "stop"},
            },
        }

    monkeypatch.setattr(runtime, "post", fake_post)
    monkeypatch.setattr(
        runtime,
        "_extract_output_tokens_and_log_probs",
        lambda *args, **kwargs: ([], []),
    )
    adapter = _backend_adapter(
        router_policy=router_policy,
        slime_router_sticky=slime_router_sticky,
    )

    result = await adapter.generate(
        input_ids=[11, 12],
        sampling_params={"max_new_tokens": 1},
        session_id=session_id,
        request_id="request-1",
    )

    assert calls[0][2] == expected_headers
    assert result.meta_info == {"finish_reason": {"type": "stop"}}


@pytest.mark.asyncio
async def test_session_shard_builds_backend_request_from_complete_replay() -> None:
    shard_cls = AgenticSessionShard.__ray_metadata__.modified_class
    shard = object.__new__(shard_cls)
    request = InflightRequest(
        request_id="request-1",
        parent_state_hash="state-1",
        rollout_id=1,
        kind=RequestKind.FRESH,
        abort_count=0,
        sampling_params={"max_new_tokens": 8},
        history_rollout_token_prefix=[11, 12],
        history_backend_image_data=["image-1"],
        history_backend_audio_data=["audio-1"],
        history_backend_video_data=["video-1"],
        pending_rollout_token_delta=[13, 14],
        runner_epoch=1,
    )
    record = _SessionRecord(scope_id="train")
    record.irs_by_id[request.request_id] = request
    record.active_ir_runner_tasks[request.request_id] = asyncio.current_task()
    shard._session_records = {"logical-session-1": record}
    shard._session_locks = {"logical-session-1": asyncio.Lock()}
    shard._sglang_request_semaphore = None
    shard._sglang_request_limiter = None
    calls = []

    async def fake_generate(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("stop after request capture")

    shard.backend = SimpleNamespace(generate=fake_generate)

    await shard._run_ir(
        session_id="logical-session-1",
        ir_id=request.request_id,
        runner_epoch=1,
    )

    assert calls[0]["input_ids"] == [11, 12, 13, 14]
    assert calls[0]["image_data"] == ["image-1"]
    assert calls[0]["audio_data"] == ["audio-1"]
    assert calls[0]["video_data"] == ["video-1"]


def test_session_shard_accumulates_engine_meta_without_route_receipt() -> None:
    request = InflightRequest(
        request_id="request-1",
        parent_state_hash="state-1",
        rollout_id=1,
        kind=RequestKind.FRESH,
        abort_count=0,
    )

    AgenticSessionShard._accumulate_request_meta(
        request,
        meta_info={
            "cached_tokens": 24,
            "prompt_tokens": 32,
            "completion_tokens": 8,
            "weight_version": 7,
        },
    )

    assert request.pending_prefix_cache_delta == {
        "cached_tokens": 24,
        "total_prompt_tokens": 32,
    }
    assert request.pending_spec_delta["completion_token_num"] == 8
    assert request.pending_weight_version_delta == ["7"]


def test_session_shard_accepts_missing_optional_engine_meta() -> None:
    request = InflightRequest(
        request_id="request-1",
        parent_state_hash="state-1",
        rollout_id=1,
        kind=RequestKind.FRESH,
        abort_count=0,
    )

    AgenticSessionShard._accumulate_request_meta(request, meta_info={})

    assert request.pending_prefix_cache_delta == {
        "cached_tokens": 0,
        "total_prompt_tokens": 0,
    }
    assert request.pending_weight_version_delta == []
