from __future__ import annotations

from typing import Any, List, Optional

from src.agent.models.execution_plan import ExecutionPlan, PlanStep
from src.agent.models.workflow_graph import WorkflowGraph
from src.core.config import settings


class Planner:
    """
    Planner for planned tasks.

    当前优先输出 WorkflowGraph（plan_graph），ExecutionPlan 仅保留兼容表示。
    """

    def __init__(self, llm: Optional[Any] = None) -> None:
        self._llm = llm

    def plan(
        self,
        *,
        query: str,
        route_decision: Any,
        chat_history: str = "",
        user_profile: str = "",
    ) -> ExecutionPlan:
        """Deprecated compatibility entry returning ExecutionPlan.

        新 planned 主路径应优先使用 `plan_graph()` 获取 WorkflowGraph；
        本方法仅为旧调用方/旧序列化输出提供兼容表示。
        """
        graph = self.plan_graph(
            query=query,
            route_decision=route_decision,
            chat_history=chat_history,
            user_profile=user_profile,
        )
        return ExecutionPlan.from_workflow_graph(
            graph,
            task_type=getattr(route_decision, "task_type", None),
        )

    def plan_graph(
        self,
        *,
        query: str,
        route_decision: Any,
        chat_history: str = "",
        user_profile: str = "",
    ) -> WorkflowGraph:
        """Primary planned-path entry returning WorkflowGraph.

        初版复用现有模板/通用规划逻辑，但对外直接返回 WorkflowGraph，
        使 graph 成为 planned 路径的优先计划表达。
        """
        plan = self._resolve_plan(
            query=query,
            route_decision=route_decision,
            chat_history=chat_history,
            user_profile=user_profile,
        )
        return WorkflowGraph.from_execution_plan(plan)

    def _resolve_plan(
        self,
        *,
        query: str,
        route_decision: Any,
        chat_history: str = "",
        user_profile: str = "",
    ) -> ExecutionPlan:
        task_type = getattr(route_decision, "task_type", "observation_recommendation")
        output_schema = getattr(route_decision, "expected_output_schema", "generic_answer_v1")
        matched_skills = list(getattr(route_decision, "matched_skills", []) or [])

        plan = self._build_template_plan(
            query=query,
            task_type=task_type,
            output_schema=output_schema,
            matched_skills=matched_skills,
        )
        if plan.steps:
            return plan

        return self._build_generic_plan(
            query=query,
            task_type=task_type,
            output_schema=output_schema,
            matched_skills=matched_skills,
            chat_history=chat_history,
            user_profile=user_profile,
        )

    def _build_template_plan(
        self,
        *,
        query: str,
        task_type: str,
        output_schema: str,
        matched_skills: List[str],
    ) -> ExecutionPlan:
        skill_set = set(matched_skills)
        steps: List[PlanStep] = []
        rationale = ""

        if task_type == "observation_recommendation":
            rationale = "观测推荐通常需要先获取环境条件，再生成目标与时段建议。"
            if "weather-lookup" in skill_set or "天气" in query or not skill_set:
                steps.append(
                    PlanStep(
                        id="weather_context",
                        kind="tool",
                        title="查询天气条件",
                        description="获取当前位置或目标地点的天气与云量信息",
                        skill="weather-lookup",
                        retry_policy=1,
                        timeout_ms=8000,
                    )
                )
            steps.append(
                PlanStep(
                    id="observation_plan",
                    kind="tool",
                    title="生成观测计划",
                    description="基于地点和时间生成可执行的观测建议",
                    skill="observation-planner",
                    retry_policy=1,
                    timeout_ms=12000,
                )
            )
        elif task_type == "celestial_event_analysis":
            rationale = "天象分析以天象事件检索为核心，必要时补充观测条件。"
            steps.append(
                PlanStep(
                    id="event_forecast",
                    kind="tool",
                    title="查询天象事件",
                    description="获取指定时段内的主要天象与事件信息",
                    skill="celestial-events-forecast",
                    retry_policy=1,
                    timeout_ms=12000,
                )
            )
            if "weather-lookup" in skill_set or "天气" in query:
                steps.append(
                    PlanStep(
                        id="event_weather",
                        kind="tool",
                        title="补充天气条件",
                        description="补充该地点的观测天气信息",
                        skill="weather-lookup",
                        required=False,
                        retry_policy=1,
                        timeout_ms=8000,
                    )
                )
        elif task_type == "deep_sky_guidance":
            rationale = "深空指导需要目标观测建议，可选补充天气判断。"
            steps.append(
                PlanStep(
                    id="deep_sky_guide",
                    kind="tool",
                    title="生成深空观测指导",
                    description="输出目标可见性、设备建议与观测提示",
                    skill="deep-sky-observing-guide",
                    retry_policy=1,
                    timeout_ms=15000,
                )
            )
            if "weather-lookup" in skill_set or any(token in query for token in ("今晚", "天气", "云量")):
                steps.append(
                    PlanStep(
                        id="deep_sky_weather",
                        kind="tool",
                        title="补充天气条件",
                        description="补充当前地点的天空条件判断",
                        skill="weather-lookup",
                        required=False,
                        retry_policy=1,
                        timeout_ms=8000,
                    )
                )
        elif task_type == "astrophotography_advice":
            rationale = "摄影建议通常可并行获取拍摄参数与天气条件。"
            steps.extend(
                [
                    PlanStep(
                        id="photo_settings",
                        kind="tool",
                        title="计算摄影参数",
                        description="根据目标和器材估算拍摄参数",
                        skill="astrophotography-calculator",
                        parallel_group="imaging_context",
                        retry_policy=1,
                        timeout_ms=15000,
                    ),
                    PlanStep(
                        id="photo_weather",
                        kind="tool",
                        title="查询摄影天气",
                        description="判断云量、透明度等是否适合拍摄",
                        skill="weather-lookup",
                        parallel_group="imaging_context",
                        required=False,
                        retry_policy=1,
                        timeout_ms=8000,
                    ),
                ]
            )

        return ExecutionPlan(
            task_type=task_type,
            output_schema=output_schema,
            steps=steps,
            planner_type="template",
            rationale=rationale,
            planner_version=str(getattr(settings, "PLANNER_VERSION", "planner_v2")),
            schema_version=str(getattr(settings, "SCHEMA_VERSION", "schema_v2")),
            budget_policy_version=str(
                getattr(settings, "BUDGET_POLICY_VERSION", "budget_v1")
            ),
        )

    def _build_generic_plan(
        self,
        *,
        query: str,
        task_type: str,
        output_schema: str,
        matched_skills: List[str],
        chat_history: str,
        user_profile: str,
    ) -> ExecutionPlan:
        steps: List[PlanStep] = []
        for index, skill_name in enumerate(matched_skills, start=1):
            steps.append(
                PlanStep(
                    id=f"tool_{index}",
                    kind="tool",
                    title=f"执行 {skill_name}",
                    description=f"调用 {skill_name} 获取回答所需信息",
                    skill=skill_name,
                    parallel_group="generic_parallel" if len(matched_skills) > 1 else None,
                    retry_policy=1,
                    timeout_ms=10000,
                )
            )

        planner_type = "template"
        rationale = "按已识别技能生成通用执行计划。"
        if not steps and self._llm is not None:
            planner_type = "llm_fallback"
            rationale = (
                "当前未命中专用模板，但保留 LLM Planner 扩展位。"
                f" chat_history={bool(chat_history)}, user_profile={bool(user_profile)}, query={query[:80]}"
            )

        return ExecutionPlan(
            task_type=task_type,
            output_schema=output_schema,
            steps=steps,
            planner_type=planner_type,
            rationale=rationale,
            planner_version=str(getattr(settings, "PLANNER_VERSION", "planner_v2")),
            schema_version=str(getattr(settings, "SCHEMA_VERSION", "schema_v2")),
            budget_policy_version=str(
                getattr(settings, "BUDGET_POLICY_VERSION", "budget_v1")
            ),
        )
