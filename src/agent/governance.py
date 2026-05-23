"""Agent 执行策略与治理指标模块，负责从任务画像推断执行模式并记录路由/延迟基线。
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from src.core.config import resolve_path, settings

if TYPE_CHECKING:
    from src.agent.models.task_profile import TaskProfile
    from src.agent.models.execution_context import ExecutionContext
    from src.agent.models.execution_decision import ExecutionDecision


VALID_AGENT_MODES = {"auto", "react", "hybrid", "planned"}


@dataclass(frozen=True)
class AgentExecutionPolicy:
    """Agent 执行策略，根据配置和任务画像选择 direct、planned 或 react。"""
    mode: str = "hybrid"
    enable_structured_skill_result: bool = False
    enable_planner: bool = False
    enable_react_fallback: bool = True

    @property
    def effective_mode(self) -> str:
        """返回策略在当前配置下的实际执行模式。"""
        return "auto" if self.mode == "hybrid" else self.mode

    @classmethod
    def from_settings(cls) -> "AgentExecutionPolicy":
        """从全局配置构造 Agent 执行策略。"""
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
        """将当前对象转换为 dict 相关的兼容结构。"""
        payload = asdict(self)
        payload["effective_mode"] = self.effective_mode
        return payload

    def _legacy_path_from_route(self, route: Optional[str]) -> str:
        """把旧 route 字符串映射为 direct、planned 或 react 执行路径。"""
        if self.effective_mode == "react":
            return "react"
        if route == "direct_task":
            return "direct"
        if route == "planned_task":
            return "planned"
        if route == "fallback_react":
            if self.enable_react_fallback:
                return "react"
            return "planned" if self.enable_planner or self.effective_mode == "planned" else "direct"
        if self.effective_mode == "planned":
            return "planned"
        if self.enable_react_fallback:
            return "react"
        return "direct"

    @staticmethod
    def _legacy_task_type_for_route(route: Optional[str]) -> str:
        """为旧 route 推断兼容 task_type。"""
        if route == "direct_task":
            return "smalltalk"
        if route == "planned_task":
            return "observation_recommendation"
        return "open_domain_reasoning"

    def choose_path(self, route: Optional[str]) -> str:
        """旧版 route 字符串兼容包装。

        旧接口仍返回 direct/planned/react 字符串，但主调用方应迁移到
        decide(profile, context) -> ExecutionDecision。
        当前保留用途：老测试、外部调用方、compatibility adapter。
        """
        if route is None:
            return self._legacy_path_from_route(route)

        from src.agent.models.task_profile import TaskProfile

        profile = TaskProfile.from_legacy_route(
            route=route,
            task_type=self._legacy_task_type_for_route(route),
            confidence=0.0,
        )
        return self.decide(profile).legacy_execution_path

    def decide(
        self,
        profile: "TaskProfile",
        context: "Optional[ExecutionContext]" = None,
    ) -> "ExecutionDecision":
        """策略主输出：基于 TaskProfile/ExecutionContext 推断 ExecutionDecision。

        规则（优先级从高到低）：
        1. effective_mode==react（全局覆盖）-> react
        2. complexity==low + tool_need==none -> direct
        3. tool_need==single + openness!=high -> direct
        4. tool_need==multi 或 complexity==high（且 openness!=high）-> planned
        5. openness==high -> react
        6. 兜底：直接使用 legacy route 映射补全 ExecutionDecision，不反向依赖 choose_path()。

        当前阶段：decide() 已是 Policy 层主决策接口。
        choose_path() 仅保留为旧字符串接口兼容包装。
        历史 ENABLE_EXECUTION_DECISION 配置位已退场，不再切换此逻辑。
        """
        from src.agent.models.execution_decision import ExecutionDecision

        legacy_path = self._legacy_path_from_route(profile.legacy_route)

        if self.effective_mode == "react":
            return ExecutionDecision(
                mode="react",
                reason="global_mode_override_react",
                fallback_modes=[],
                legacy_execution_path=legacy_path,
            )

        if profile.legacy_route == "planned_task" and profile.openness != "high":
            return ExecutionDecision(
                mode="planned",
                reason="preserve_legacy_planned_route",
                fallback_modes=["react"] if self.enable_react_fallback else [],
                legacy_execution_path=legacy_path,
            )

        complexity = profile.complexity
        openness = profile.openness
        tool_need = profile.tool_need

        if complexity == "low" and tool_need == "none":
            return ExecutionDecision(
                mode="direct",
                reason="low_complexity_no_tools",
                fallback_modes=["react"] if self.enable_react_fallback else [],
                legacy_execution_path=legacy_path,
            )

        if tool_need == "single" and openness != "high":
            return ExecutionDecision(
                mode="direct",
                reason="single_tool_low_openness",
                fallback_modes=["react"] if self.enable_react_fallback else [],
                legacy_execution_path=legacy_path,
            )

        if tool_need == "multi" or (complexity == "high" and openness != "high"):
            return ExecutionDecision(
                mode="planned",
                reason="multi_tool_or_high_complexity",
                fallback_modes=["react"] if self.enable_react_fallback else [],
                legacy_execution_path=legacy_path,
            )

        if openness == "high":
            return ExecutionDecision(
                mode="react",
                reason="high_openness_react",
                fallback_modes=[],
                legacy_execution_path=legacy_path,
            )

        # 兜底：委托 choose_path
        return ExecutionDecision(
            mode=legacy_path,
            reason="fallback_to_legacy_route_mapping",
            fallback_modes=[],
            legacy_execution_path=legacy_path,
        )


@dataclass(frozen=True)
class BenchmarkCase:
    """路由基准评测用例结构。"""
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
    """加载 Phase0 路由基准评测用例。"""
    benchmark_path = resolve_path(path or settings.PHASE0_BENCHMARK_PATH)
    with open(benchmark_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [BenchmarkCase(**item) for item in payload]


def evaluate_router_benchmark(
    router: Any,
    cases: Iterable[BenchmarkCase],
    *,
    allow_legacy_route: bool = False,
) -> Dict[str, Any]:
    """执行路由基准评测并统计不匹配用例。"""
    evaluated = 0
    mismatches = 0
    by_category: Dict[str, Dict[str, int]] = {}
    mismatch_items: List[Dict[str, Any]] = []

    for case in cases:
        if hasattr(router, "profile"):
            actual_route = getattr(router.profile(case.query), "legacy_route", None)
        elif allow_legacy_route and hasattr(router, "route"):
            # Explicit legacy fixture path for older benchmark adapters.
            decision = router.route(case.query)
            actual_route = getattr(decision, "route", None)
        else:
            raise ValueError("router benchmark requires profile()")
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
    """治理指标中的单次请求观测记录。"""
    route: str
    request_total_ms: float
    agent_mode: str
    execution_path: str
    fallback_used: bool = False
    output_schema_parse_success: bool = True
    route_expected: Optional[str] = None


class GovernanceMetricsRegistry:
    """用于治理基线统计的内存指标仓库。"""

    def __init__(self) -> None:
        """初始化 GovernanceMetricsRegistry 的依赖、配置和内部状态。"""
        self._lock = threading.Lock()
        self._observations: List[RequestObservation] = []

    def record(self, observation: RequestObservation) -> None:
        """记录一次请求观测。"""
        with self._lock:
            self._observations.append(observation)

    def clear(self) -> None:
        """清空已记录的请求观测。"""
        with self._lock:
            self._observations.clear()

    def snapshot(self) -> Dict[str, Any]:
        """返回当前累计指标或证据的快照。"""
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
    """计算延迟列表的指定百分位值。"""
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
