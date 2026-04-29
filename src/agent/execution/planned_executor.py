"""PlannedExecutor — 计划任务执行器（Phase 4 引入，Phase 9 为主路径）。

抽取自 TaskOrchestrator._run_planned_task()，逻辑完全等价。
Phase 9 起：循环依赖已消除（改用 SkillParamBuilder），WorkflowExecutor 为唯一执行引擎。
"""
from __future__ import annotations

from typing import Any, Optional

from src.agent.executor import EventCallback
from src.agent.execution.workflow_executor import WorkflowExecutor
from src.agent.models.execution_event import ExecutionEvent
from src.agent.models.execution_plan import ExecutionPlan
from src.agent.models.final_response import FinalResponse
from src.agent.models.execution_trace_entry import ExecutionTraceEntry
from src.agent.models.workflow_graph import WorkflowGraph
from src.agent.planner import Planner
from src.agent.policies.budget_policy import RequestBudgetTracker
from src.agent.policies.fallback_policy import FallbackPolicy
from src.agent.request_router import RouteDecision
from src.agent.skill_param_builder import SkillParamBuilder
from src.core.config import settings


class PlannedExecutor:
    """封装 planned_task 主链路：plan -> execute(DAG) -> synthesize。

    主路径使用 WorkflowGraph + WorkflowExecutor；
    ExecutionPlan 仅作为兼容输入/输出视图保留。
    """

    def __init__(
        self,
        skill_manager: Any,
        llm: Any,
        synthesizer: Any,
        planner: Optional[Planner] = None,
        fallback_policy: Optional[FallbackPolicy] = None,
        workflow_executor: Optional[WorkflowExecutor] = None,
    ) -> None:
        self._skill_manager = skill_manager
        self._llm = llm
        self._synthesizer = synthesizer
        self._planner = planner or Planner(llm=llm)
        self._fallback_policy = fallback_policy or FallbackPolicy()
        self._workflow_executor = workflow_executor or WorkflowExecutor(skill_manager=skill_manager)
        self._param_builder = SkillParamBuilder(skill_manager)

    async def run(
        self,
        decision: RouteDecision,
        query: str,
        *,
        chat_history: str = "",
        user_profile: str = "",
        execution_plan: Optional[ExecutionPlan] = None,
        event_callback: Optional[EventCallback] = None,
        budget_tracker: Optional[RequestBudgetTracker] = None,
    ) -> FinalResponse:
        budget_tracker = budget_tracker or RequestBudgetTracker()

        plan, graph = self._resolve_plan_and_graph(
            decision,
            query,
            chat_history=chat_history,
            user_profile=user_profile,
            execution_plan=execution_plan,
        )
        if not graph.nodes:
            raise ValueError(f"no planned-task steps resolved for {decision.task_type}")

        outcome = await self._workflow_executor.execute(
            graph,
            plan,
            query=query,
            param_builder=self._param_builder.build,
            event_callback=event_callback,
            budget_tracker=budget_tracker,
        )

        fallback_decision = self._fallback_policy.decide_for_execution(
            outcome=outcome,
            plan=plan,
        )

        response = self._synthesizer.synthesize(
            query=query,
            task_type=decision.task_type,
            output_schema=decision.expected_output_schema,
            skill_results=outcome.skill_results,
            chat_history=chat_history,
            user_profile=user_profile,
            route=decision.route,
            execution_plan=plan.to_dict(),
            execution_trace=[step.to_dict() for step in outcome.step_results],
            route_decision=decision.to_meta(),
            fallback_path=[fallback_decision.to_dict()] if fallback_decision else [],
            budget_usage=budget_tracker.snapshot() if budget_tracker else None,
            versions={
                "router_policy_version": str(
                    getattr(settings, "ROUTER_POLICY_VERSION", "router_v1")
                ),
                "planner_version": str(getattr(settings, "PLANNER_VERSION", "planner_v2")),
                "schema_version": str(getattr(settings, "SCHEMA_VERSION", "schema_v2")),
                "synth_prompt_version": str(
                    getattr(settings, "SYNTH_PROMPT_VERSION", "synth_prompt_v2")
                ),
                "fallback_policy_version": self._fallback_policy.version,
                "budget_policy_version": (
                    budget_tracker.budget.policy_version if budget_tracker else "budget_v1"
                ),
            },
        )
        response.route = decision.route
        response.task_type = decision.task_type
        response.execution_plan = plan.to_dict()
        response.execution_trace = [step.to_dict() for step in outcome.step_results]
        response.route_decision = decision.to_meta()
        response.fallback_path = [fallback_decision.to_dict()] if fallback_decision else []
        response.budget_usage = budget_tracker.snapshot() if budget_tracker else None
        response.versions = {
            "router_policy_version": str(
                getattr(settings, "ROUTER_POLICY_VERSION", "router_v1")
            ),
            "planner_version": str(getattr(settings, "PLANNER_VERSION", "planner_v2")),
            "schema_version": str(getattr(settings, "SCHEMA_VERSION", "schema_v2")),
            "synth_prompt_version": str(
                getattr(settings, "SYNTH_PROMPT_VERSION", "synth_prompt_v2")
            ),
            "fallback_policy_version": self._fallback_policy.version,
            "budget_policy_version": (
                budget_tracker.budget.policy_version if budget_tracker else "budget_v1"
            ),
        }
        response.execution_events = self._build_execution_events(
            response=response,
            fallback_decision=fallback_decision,
        )
        return response

    def preview_plan(
        self,
        decision: RouteDecision,
        query: str,
        *,
        chat_history: str = "",
        user_profile: str = "",
        execution_plan: Optional[ExecutionPlan] = None,
    ) -> ExecutionPlan:
        """为展示层提供兼容计划视图，不触发实际执行。"""
        plan, _ = self._resolve_plan_and_graph(
            decision,
            query,
            chat_history=chat_history,
            user_profile=user_profile,
            execution_plan=execution_plan,
        )
        return plan

    def _resolve_plan_and_graph(
        self,
        decision: RouteDecision,
        query: str,
        *,
        chat_history: str = "",
        user_profile: str = "",
        execution_plan: Optional[ExecutionPlan] = None,
    ) -> tuple[ExecutionPlan, WorkflowGraph]:
        """优先使用原生 WorkflowGraph；ExecutionPlan 仅保留兼容层。"""
        if execution_plan is not None:
            return execution_plan, WorkflowGraph.from_execution_plan(execution_plan)

        use_graph_planning = bool(getattr(settings, "ENABLE_WORKFLOW_GRAPH", True))
        if use_graph_planning and hasattr(self._planner, "plan_graph"):
            try:
                graph = self._planner.plan_graph(
                    query=query,
                    route_decision=decision,
                    chat_history=chat_history,
                    user_profile=user_profile,
                )
                if graph.nodes:
                    plan = ExecutionPlan.from_workflow_graph(
                        graph,
                        task_type=decision.task_type,
                    )
                    return plan, graph
            except Exception:
                pass

        plan = self._planner.plan(
            query=query,
            route_decision=decision,
            chat_history=chat_history,
            user_profile=user_profile,
        )
        return plan, WorkflowGraph.from_execution_plan(plan)

    def _build_execution_events(
        self,
        *,
        response: FinalResponse,
        fallback_decision: Optional[Any],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if response.execution_plan:
            events.append(
                ExecutionEvent(
                    type="plan_created",
                    payload={"plan": dict(response.execution_plan)},
                    source="planned",
                ).to_dict()
            )

        for trace in response.execution_trace:
            for event in ExecutionTraceEntry.from_dict(trace).to_execution_events(
                source="planned"
            ):
                events.append(event.to_dict())

        if fallback_decision is not None:
            events.append(
                ExecutionEvent(
                    type="fallback_triggered",
                    payload=fallback_decision.to_dict(),
                    source="planned",
                ).to_dict()
            )

        events.append(
            ExecutionEvent(
                type="answer_ready",
                payload={
                    "answer": response.answer,
                    "summary": response.summary,
                    "route": response.route,
                    "task_type": response.task_type,
                },
                source="planned",
            ).to_dict()
        )
        return events
