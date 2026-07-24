# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Iterable

from relax.agentic.session.sglang_capabilities import SGLangCapabilityProfile
from relax.utils.http_utils import get, post


_DEFAULT_FANOUT_CONCURRENCY = 16
_DEFAULT_CALL_TIMEOUT_S = 5.0
_DEFAULT_CLOSE_ATTEMPTS = 3
_DEFAULT_CLOSE_DEADLINE_S = 15.0


class EngineSessionLifecycleState(str, Enum):
    BYPASS = "bypass"
    UNOPENED = "unopened"
    OPENING = "opening"
    ACTIVE = "active"
    TERMINATING = "terminating"
    CLOSE_IN_FLIGHT = "close_in_flight"
    RETRY_WAIT = "retry_wait"
    TERMINATED_OK = "terminated_ok"
    TERMINATED_EXHAUSTED = "terminated_exhausted"


class SessionControlStatus(str, Enum):
    BYPASS = "bypass"
    OPENED = "opened"
    CLOSED = "closed"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True, kw_only=True)
class SGLangWorkerTarget:
    url: str
    worker_id: str | None = None
    engine_epoch: str | None = None
    healthy: bool = True


@dataclass(frozen=True, kw_only=True)
class SessionControlResult:
    status: SessionControlStatus
    targets: tuple[SGLangWorkerTarget, ...] = ()
    failed_targets: tuple[SGLangWorkerTarget, ...] = ()
    attempts: int = 0

    @property
    def succeeded(self) -> bool:
        return self.status in {
            SessionControlStatus.BYPASS,
            SessionControlStatus.OPENED,
            SessionControlStatus.CLOSED,
        }


GetFn = Callable[[str], Awaitable[Any]]
PostFn = Callable[..., Awaitable[Any]]


def _target_key(target: SGLangWorkerTarget) -> tuple[str, str | None]:
    return target.url, target.engine_epoch


def _deduplicate_targets(targets: Iterable[SGLangWorkerTarget]) -> tuple[SGLangWorkerTarget, ...]:
    deduplicated: dict[tuple[str, str | None], SGLangWorkerTarget] = {}
    for target in targets:
        if target.url:
            deduplicated[_target_key(target)] = target
    return tuple(sorted(deduplicated.values(), key=lambda target: (target.url, target.engine_epoch or "")))


def _is_duplicate_open_error(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    response_text = getattr(response, "text", "")
    text = f"{exc} {response_text}".lower()
    return "already exist" in text or "same id is still open" in text


class SGLangSessionControlPort:
    def __init__(
        self,
        *,
        router_ip: str | None,
        router_port: int | None,
        capability_profile: SGLangCapabilityProfile,
        get_fn: GetFn | None = None,
        post_fn: PostFn | None = None,
        fanout_concurrency: int = _DEFAULT_FANOUT_CONCURRENCY,
        call_timeout_s: float = _DEFAULT_CALL_TIMEOUT_S,
        close_attempts: int = _DEFAULT_CLOSE_ATTEMPTS,
        close_deadline_s: float = _DEFAULT_CLOSE_DEADLINE_S,
    ) -> None:
        self._router_ip = router_ip
        self._router_port = None if router_port is None else int(router_port)
        self._capability_profile = capability_profile
        self._get = get_fn or get
        self._post = post_fn or post
        self._fanout_concurrency = max(1, int(fanout_concurrency))
        self._call_timeout_s = max(0.01, float(call_timeout_s))
        self._close_attempts = max(1, int(close_attempts))
        self._close_deadline_s = max(self._call_timeout_s, float(close_deadline_s))

    @property
    def enabled(self) -> bool:
        return bool(self._capability_profile.session_wire_enabled and self._router_ip and self._router_port)

    async def worker_targets(self, *, include_unhealthy: bool) -> tuple[SGLangWorkerTarget, ...]:
        if not self.enabled:
            return ()
        base_url = f"http://{self._router_ip}:{self._router_port}"
        first_error: BaseException | None = None
        try:
            response = await asyncio.wait_for(
                self._get(f"{base_url}/workers"),
                timeout=self._call_timeout_s,
            )
            workers = response.get("workers", []) if isinstance(response, dict) else []
            targets = []
            for worker in workers:
                if not isinstance(worker, dict) or not isinstance(worker.get("url"), str):
                    continue
                healthy = bool(worker.get("is_healthy", False))
                if include_unhealthy or healthy:
                    registration_id = str(worker["id"]) if worker.get("id") is not None else None
                    targets.append(
                        SGLangWorkerTarget(
                            url=worker["url"],
                            worker_id=registration_id,
                            engine_epoch=(
                                str(worker["engine_epoch"])
                                if worker.get("engine_epoch") is not None
                                else registration_id
                            ),
                            healthy=healthy,
                        )
                    )
            return _deduplicate_targets(targets)
        except Exception as exc:
            first_error = exc
        try:
            response = await asyncio.wait_for(
                self._get(f"{base_url}/list_workers"),
                timeout=self._call_timeout_s,
            )
            urls = response.get("urls", []) if isinstance(response, dict) else []
            return _deduplicate_targets(SGLangWorkerTarget(url=url) for url in urls if isinstance(url, str) and url)
        except Exception as exc:
            if first_error is not None:
                raise RuntimeError("Failed to query the SGLang worker registry") from exc
            raise

    async def _post_target(
        self,
        *,
        target: SGLangWorkerTarget,
        path: str,
        payload: dict[str, Any],
        duplicate_open_is_success: bool = False,
    ) -> bool:
        try:
            await asyncio.wait_for(
                self._post(f"{target.url}{path}", payload, max_retries=1),
                timeout=self._call_timeout_s,
            )
            return True
        except Exception as exc:
            return bool(duplicate_open_is_success and _is_duplicate_open_error(exc))

    async def _fanout(
        self,
        *,
        targets: Iterable[SGLangWorkerTarget],
        path: str,
        payload: dict[str, Any],
        duplicate_open_is_success: bool = False,
    ) -> tuple[tuple[SGLangWorkerTarget, ...], tuple[SGLangWorkerTarget, ...]]:
        ordered_targets = _deduplicate_targets(targets)
        semaphore = asyncio.Semaphore(self._fanout_concurrency)

        async def invoke(target: SGLangWorkerTarget) -> tuple[SGLangWorkerTarget, bool]:
            async with semaphore:
                succeeded = await self._post_target(
                    target=target,
                    path=path,
                    payload=payload,
                    duplicate_open_is_success=duplicate_open_is_success,
                )
                return target, succeeded

        results = await asyncio.gather(*(invoke(target) for target in ordered_targets))
        succeeded = tuple(target for target, ok in results if ok)
        failed = tuple(target for target, ok in results if not ok)
        return succeeded, failed

    async def open_session(
        self,
        *,
        engine_session_id: str,
        capacity_of_str_len: int,
    ) -> SessionControlResult:
        if not self.enabled:
            return SessionControlResult(status=SessionControlStatus.BYPASS)
        try:
            targets = await self.worker_targets(include_unhealthy=False)
        except Exception:
            return SessionControlResult(status=SessionControlStatus.BYPASS)
        if not targets:
            return SessionControlResult(status=SessionControlStatus.BYPASS)
        succeeded, failed = await self._fanout(
            targets=targets,
            path="/open_session",
            payload={
                "capacity_of_str_len": max(1, int(capacity_of_str_len)),
                "session_id": engine_session_id,
                "streaming": False,
            },
            duplicate_open_is_success=True,
        )
        if failed:
            # A partially opened session is never used for generation. Close
            # the known-open targets as compensation; terminal cleanup will
            # fan out again if the outcome was unknown.
            await self._fanout(
                targets=succeeded,
                path="/close_session",
                payload={"session_id": engine_session_id},
            )
            return SessionControlResult(
                status=SessionControlStatus.BYPASS,
                targets=targets,
                failed_targets=failed,
                attempts=1,
            )
        return SessionControlResult(
            status=SessionControlStatus.OPENED,
            targets=targets,
            attempts=1,
        )

    async def abort_requests(
        self,
        *,
        request_ids: Iterable[str],
        known_targets: Iterable[SGLangWorkerTarget] = (),
    ) -> bool:
        request_ids = tuple(sorted({request_id for request_id in request_ids if request_id}))
        if not request_ids:
            return True
        try:
            registry_targets = await self.worker_targets(include_unhealthy=True)
        except Exception:
            registry_targets = ()
        targets = _deduplicate_targets((*known_targets, *registry_targets))
        if not targets:
            return False
        semaphore = asyncio.Semaphore(self._fanout_concurrency)

        async def invoke(target: SGLangWorkerTarget, request_id: str) -> bool:
            async with semaphore:
                return await self._post_target(
                    target=target,
                    path="/abort_request",
                    payload={"rid": request_id},
                )

        results = await asyncio.gather(
            *(invoke(target, request_id) for request_id in request_ids for target in targets)
        )
        return bool(results) and all(results)

    async def close_session(
        self,
        *,
        engine_session_id: str,
        known_targets: Iterable[SGLangWorkerTarget] = (),
    ) -> SessionControlResult:
        known_targets = tuple(known_targets)
        if not self.enabled and not known_targets:
            return SessionControlResult(status=SessionControlStatus.BYPASS)
        try:
            registry_targets = await self.worker_targets(include_unhealthy=True)
        except Exception:
            registry_targets = ()
        targets = _deduplicate_targets((*known_targets, *registry_targets))
        if not targets:
            return SessionControlResult(status=SessionControlStatus.EXHAUSTED)
        pending = targets
        attempts = 0
        deadline = time.monotonic() + self._close_deadline_s
        while pending and attempts < self._close_attempts and time.monotonic() < deadline:
            attempts += 1
            _, pending = await self._fanout(
                targets=pending,
                path="/close_session",
                payload={"session_id": engine_session_id},
            )
        status = SessionControlStatus.CLOSED if not pending else SessionControlStatus.EXHAUSTED
        return SessionControlResult(
            status=status,
            targets=targets,
            failed_targets=pending,
            attempts=attempts,
        )
