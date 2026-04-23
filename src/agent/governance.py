from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

from src.core.config import resolve_path, settings


VALID_AGENT_MODES = {"react", "hybrid", "planned"}


@dataclass(frozen=True)
class AgentExecutionPolicy:
    mode: str = "hybrid"
    enable_structured_skill_result: bool = False
    enable_planner: bool = False
    enable_react_fallback: bool = True

    @classmethod
    def from_settings(cls) -> "AgentExecutionPolicy":
        raw_mode = str(getattr(settings, "AGENT_MODE", "hybrid")).strip().lower()
        mode = raw_mode if raw_mode in VALID_AGENT_MODES else "hybrid"
        return cls(
            mode=mode,
            enable_structured_skill_result=bool(
                getattr(settings, "ENABLE_STRUCTURED_SKILL_RESULT", False)
            ),
            enable_planner=bool(getattr(settings, "ENABLE_PLANNER", False)),
            enable_react_fallback=bool(
                getattr(settings, "ENABLE_REACT_FALLBACK", True)
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def choose_path(self, route: Optional[str]) -> str:
        if self.mode == "react":
            return "react"
        if route == "direct_task":
            return "direct"
        if route == "planned_task":
            return "planned"
        if route == "fallback_react":
            if self.enable_react_fallback:
                return "react"
            return "planned" if self.enable_planner or self.mode == "planned" else "direct"
        if self.mode == "planned":
            return "planned"
        if self.enable_react_fallback:
            return "react"
        return "direct"


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    category: str
    query: str
    expected_route: str
    expected_tools: List[str]
    expected_answer_structure: str
    acceptable_latency_ms: int


def load_phase0_benchmark_cases(
    path: Optional[str] = None,
) -> List[BenchmarkCase]:
    benchmark_path = resolve_path(path or settings.PHASE0_BENCHMARK_PATH)
    with open(benchmark_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [BenchmarkCase(**item) for item in payload]


def evaluate_router_benchmark(
    router: Any,
    cases: Iterable[BenchmarkCase],
) -> Dict[str, Any]:
    evaluated = 0
    mismatches = 0
    by_category: Dict[str, Dict[str, int]] = {}
    mismatch_items: List[Dict[str, Any]] = []

    for case in cases:
        decision = router.route(case.query)
        actual_route = getattr(decision, "route", None)
        category_stats = by_category.setdefault(
            case.category,
            {"total": 0, "mismatch": 0},
        )
        category_stats["total"] += 1
        evaluated += 1

        if actual_route != case.expected_route:
            mismatches += 1
            category_stats["mismatch"] += 1
            mismatch_items.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "query": case.query,
                    "expected_route": case.expected_route,
                    "actual_route": actual_route,
                }
            )

    return {
        "evaluated_cases": evaluated,
        "matched_cases": evaluated - mismatches,
        "mismatched_cases": mismatches,
        "route_mismatch_rate": round((mismatches / evaluated), 4) if evaluated else 0.0,
        "by_category": by_category,
        "mismatches": mismatch_items,
    }


@dataclass(frozen=True)
class RequestObservation:
    route: str
    request_total_ms: float
    agent_mode: str
    execution_path: str
    fallback_used: bool = False
    output_schema_parse_success: bool = True
    route_expected: Optional[str] = None


class GovernanceMetricsRegistry:
    """In-memory metrics store for phase-0 governance baselines."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._observations: List[RequestObservation] = []

    def record(self, observation: RequestObservation) -> None:
        with self._lock:
            self._observations.append(observation)

    def clear(self) -> None:
        with self._lock:
            self._observations.clear()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            observations = list(self._observations)

        total = len(observations)
        latency_values = [obs.request_total_ms for obs in observations]
        fallback_count = sum(1 for obs in observations if obs.fallback_used)
        parse_success_count = sum(
            1 for obs in observations if obs.output_schema_parse_success
        )
        route_evaluated = [obs for obs in observations if obs.route_expected]
        route_mismatch_count = sum(
            1 for obs in route_evaluated if obs.route != obs.route_expected
        )

        by_mode: Dict[str, int] = {}
        by_route: Dict[str, int] = {}
        for obs in observations:
            by_mode[obs.agent_mode] = by_mode.get(obs.agent_mode, 0) + 1
            by_route[obs.route] = by_route.get(obs.route, 0) + 1

        return {
            "total_requests": total,
            "latency_ms": {
                "p50": _percentile(latency_values, 50),
                "p90": _percentile(latency_values, 90),
                "max": round(max(latency_values), 2) if latency_values else 0.0,
            },
            "fallback_rate": round((fallback_count / total), 4) if total else 0.0,
            "output_schema_parse_rate": (
                round((parse_success_count / total), 4) if total else 0.0
            ),
            "route_mismatch_rate": (
                round((route_mismatch_count / len(route_evaluated)), 4)
                if route_evaluated
                else 0.0
            ),
            "route_mismatch_evaluated": len(route_evaluated),
            "by_mode": by_mode,
            "by_route": by_route,
        }


def _percentile(values: List[float], percentile: int) -> float:
    if not values:
        return 0.0

    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 2)

    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return round(ordered[lower], 2)

    interpolated = ordered[lower] + (ordered[upper] - ordered[lower]) * (
        rank - lower
    )
    return round(interpolated, 2)
