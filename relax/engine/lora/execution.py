# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Physical execution facts, independent of transports and result delivery."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


CAPABILITY = "relax.sglang.execution-drain.v2"


class ExecutionPhase(str, Enum):
    ADMITTED = "admitted"
    SUBMITTED = "submitted"
    DRAINED = "drained"
    SETTLED = "settled"


@dataclass
class ExecutionRecord:
    request_id: str
    operation_id: str
    digest: str
    kind: str = "generation"
    phase: ExecutionPhase = ExecutionPhase.ADMITTED
    cancel_requested: bool = False
    delivery_error: str | None = None
    output: dict[str, Any] | None = None
    error_code: str | None = None

    @property
    def drained(self) -> bool:
        return self.phase in {ExecutionPhase.DRAINED, ExecutionPhase.SETTLED}

    def observe_drain(self, *, never_submitted: bool = False) -> None:
        if self.drained:
            return
        self.phase = ExecutionPhase.DRAINED
        if self.output is None:
            self.error_code = "NOT_SUBMITTED" if never_submitted else "RESULT_LOST"

    def response(self) -> dict[str, Any]:
        if not self.drained:
            return {"status": "pending", "phase": self.phase.value, "observation_error": self.delivery_error}
        if self.output is not None:
            return {"status": "terminal", "drained": True, "output": self.output}
        return {
            "status": "rejected" if self.error_code == "NOT_SUBMITTED" else "terminal_error",
            "drained": True,
            "code": self.error_code,
            "error": self.delivery_error or "result is unavailable",
            "retryable": False,
            "definitely_not_submitted": self.error_code == "NOT_SUBMITTED",
        }


@dataclass
class _SchedulerExecution:
    operation_id: str
    terminal: bool = False
    drained: bool = False


@dataclass
class SequenceWatermark:
    """Acknowledged contiguous sequence plus gaps bounded by unsettled work.

    Sequence numbers are allocated by one TokenizerManager before dispatch.
    Acknowledgement is allowed only after that producer fenced all late sends.
    """

    through: int = 0
    gaps: list[tuple[int, int]] = field(default_factory=list)

    def contains(self, sequence: int) -> bool:
        return sequence <= self.through or any(first <= sequence <= last for first, last in self.gaps)

    def acknowledge(self, sequence: int) -> None:
        if sequence <= self.through:
            return
        intervals = sorted([*self.gaps, (sequence, sequence)])
        merged: list[tuple[int, int]] = []
        for first, last in intervals:
            if merged and first <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(last, merged[-1][1]))
            else:
                merged.append((first, last))
        if merged and merged[0][0] == self.through + 1:
            self.through = merged.pop(0)[1]
        self.gaps = merged


class SchedulerExecutions:
    """Used inside the non-overlap scheduler, not inferred from tokenizer
    state."""

    def __init__(self) -> None:
        self.active: dict[str, _SchedulerExecution] = {}
        self.by_operation: dict[str, set[str]] = {}
        self.fenced_operations: set[str] = set()
        self.cancelled: set[str] = set()
        self.watermarks: dict[str, SequenceWatermark] = {}
        self.pending_terminals: set[str] = set()

    @staticmethod
    def sequence(request_id: str) -> tuple[str, int]:
        if len(request_id) != 48:
            raise ValueError("managed scheduler IDs must contain an epoch and a sequence")
        return request_id[:32], int(request_id[32:], 16)

    def expired(self, request_id: str) -> bool:
        epoch, sequence = self.sequence(request_id)
        return self.watermarks.get(epoch, SequenceWatermark()).contains(sequence)

    def admit(self, request_id: str, operation_id: str) -> bool:
        if self.expired(request_id) or request_id in self.cancelled or operation_id in self.fenced_operations:
            return False
        if request_id in self.active:
            return False
        self.active[request_id] = _SchedulerExecution(operation_id)
        self.by_operation.setdefault(operation_id, set()).add(request_id)
        return True

    def terminal(self, request_id: str) -> None:
        if request_id in self.active:
            self.active[request_id].terminal = True
            self.pending_terminals.add(request_id)

    def drain_boundary(self) -> None:
        """Call after process_batch_result/queued abort returns, never on
        send."""
        for request_id in self.pending_terminals:
            execution = self.active[request_id]
            execution.drained = True
            self.by_operation[execution.operation_id].discard(request_id)
        self.pending_terminals.clear()

    def status(self, request_id: str) -> dict[str, Any]:
        execution = self.active.get(request_id)
        if execution is not None:
            return {"accepted": True, "drained": execution.drained, "terminal": execution.terminal}
        if self.expired(request_id) or request_id in self.cancelled:
            return {"accepted": False, "drained": True, "never_submitted": True}
        return {"accepted": False, "drained": False}

    def cancel_fence(self, request_id: str) -> None:
        self.cancelled.add(request_id)

    def fence_version(self, operation_id: str) -> dict[str, Any]:
        self.fenced_operations.add(operation_id)
        return {"fenced": True, "drained": not self.by_operation.get(operation_id)}

    def acknowledge(self, request_id: str) -> None:
        if not self.status(request_id)["drained"]:
            raise ValueError("cannot acknowledge an execution without a drain proof")
        epoch, sequence = self.sequence(request_id)
        self.watermarks.setdefault(epoch, SequenceWatermark()).acknowledge(sequence)
        self.active.pop(request_id, None)
        self.cancelled.discard(request_id)
