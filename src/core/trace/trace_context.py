"""Minimal trace context placeholder for ingestion/query stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class TraceContext:
    """Minimal trace context; will be extended in later observability phases."""

    trace_type: str = "generic"
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    stages: list[dict[str, Any]] = field(default_factory=list)

    def record_stage(
        self,
        stage: str,
        method: str | None = None,
        details: dict[str, Any] | None = None,
        elapsed_ms: float | None = None,
    ) -> None:
        stage_name = str(stage).strip()
        if not stage_name:
            raise ValueError("TraceContext.record_stage stage cannot be empty.")

        payload: dict[str, Any] = {"stage": stage_name}
        if method:
            payload["method"] = str(method).strip()
        if details is not None:
            payload["details"] = dict(details)
        if elapsed_ms is not None:
            payload["elapsed_ms"] = float(elapsed_ms)
        self.stages.append(payload)

    def stage_timer(self) -> float:
        """Return start marker for elapsed time measurement."""
        return perf_counter()

    @staticmethod
    def stage_elapsed_ms(start_marker: float) -> float:
        return (perf_counter() - start_marker) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at.isoformat(),
            "stages": list(self.stages),
        }
