"""PlannedExecutor — 计划任务执行器（Phase 4 引入，Phase 9 为主路径）。

抽取自 TaskOrchestrator._run_planned_task()，逻辑完全等价。
Phase 9 起：循环依赖已消除（改用 SkillParamBuilder），WorkflowExecutor 为唯一执行引擎。
"""
from __future__ import annotations

from typing import Any, Optional

from src.agent.executor import EventCallback
from src.agent.execution.workflow_executor import WorkflowExecutor
from src.agent.models.execution_plan import ExecutionPlan
from src.agent.models.final_response import FinalResponse
from src.agent.models.workflow_graph import WorkflowGraph
from src.agent.planner import Planner
from src.agent.policies.budget_policy import RequestBudgetTracker
from src.agent.policies.fallback_policy import FallbackPolicy
from src.agent.request_router import RouteDecision
from src.agent.skill_param_builder import SkillParamBuilder
from src.core.config import settings


class PlannedExecutor:
    """封装 planned_task 主链路：plan -> execute(DAG) -> synthesize。

    ExecutionPlan 统一转换为 WorkflowGraph 后由 WorkflowExecutor 执行。
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

        plan = execution_plan or self._planner.plan(
            query=query,
            route_decision=decision,
            chat_history=chat_history,
            user_profile=user_profile,
        )
        if not plan.steps:
            raise ValueError(f"no planned-task steps resolved for {decision.task_type}")

        graph = WorkflowGraph.from_execution_plan(plan)
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

        return self._synthesizer.synthesize(
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
