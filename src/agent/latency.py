from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional


@dataclass
class LatencyTracker:
    """Collects per-request stage timings in milliseconds."""

    started_at: float = field(default_factory=time.perf_counter)
    _stages_ms: Dict[str, float] = field(default_factory=dict)
    _meta: Dict[str, Any] = field(default_factory=dict)

    @contextmanager
    def measure(self, stage_name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._stages_ms[stage_name] = self._stages_ms.get(stage_name, 0.0) + (
                (time.perf_counter() - start) * 1000.0
            )

    def record_ms(self, stage_name: str, value_ms: Optional[float]) -> None:
        if value_ms is None:
            return
        self._stages_ms[stage_name] = self._stages_ms.get(stage_name, 0.0) + max(
            float(value_ms), 0.0
        )

    def set_meta(self, key: str, value: Any) -> None:
        self._meta[key] = value

    def stages_ms(self) -> Dict[str, float]:
        payload = {
            key: round(value, 2)
            for key, value in self._stages_ms.items()
        }
        payload["request_total_ms"] = round(
            (time.perf_counter() - self.started_at) * 1000.0, 2
        )
        return payload

    def to_payload(self) -> Dict[str, Any]:
        return {
            "stages_ms": self.stages_ms(),
            "meta": dict(self._meta),
        }
