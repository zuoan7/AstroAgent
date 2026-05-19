"""WorkflowExecutor — 基于 WorkflowGraph 的拓扑执行器（Phase 6 引入，Phase 9 为唯一引擎）。

逐节点按拓扑顺序执行，产出与 StepExecutor 兼容的 ExecutionOutcome。
Phase 9 起：ENABLE_WORKFLOW_GRAPH 分支判断已移除，本类为 PlannedExecutor 的唯一执行引擎。
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

from src.agent.executor import (
    EventCallback,
    ExecutionOutcome,
    ParamBuilder,
    StepExecutionResult,
    _extract_mcp_tools_from_sources,
)
from src.agent.models.execution_plan import ExecutionPlan
from src.agent.models.skill_result import SkillResult
from src.agent.models.workflow_graph import WorkflowGraph, WorkflowNode
from src.agent.policies.budget_policy import RequestBudgetTracker


class WorkflowExecutor:
    """基于 WorkflowGraph 拓扑顺序执行节点，目前仅支持线性执行（无并发）。"""

    def __init__(self, skill_manager: Any) -> None:
        self._skill_manager = skill_manager

    async def execute(
        self,
        graph: WorkflowGraph,
        plan: ExecutionPlan,
        *,
        query: str,
        param_builder: ParamBuilder,
        event_callback: Optional[EventCallback] = None,
        budget_tracker: Optional[RequestBudgetTracker] = None,
    ) -> ExecutionOutcome:
        """按拓扑顺序逐节点执行。

        `plan` 参数仅用于兼容 Outcome/trace/展示层结构；主执行语义来自 `graph`，
        不再依赖 StepExecutor 的线性计划解释。
        """
        outcome = ExecutionOutcome(plan=plan)

        try:
            ordered_nodes = graph.topological_order()
        except ValueError as exc:
            outcome.halted = True
            outcome.halt_reason = str(exc)
            return outcome

        for node in ordered_nodes:
            step_result, skill_result = await self._execute_node(
                node,
                query=query,
                param_builder=param_builder,
                event_callback=event_callback,
                budget_tracker=budget_tracker,
            )
            outcome.step_results.append(step_result)
            if skill_result is not None:
                outcome.skill_results.append(skill_result)

            if step_result.status == "error" and not node.optional:
                outcome.halted = True
                outcome.halt_reason = (
                    step_result.error or f"required node failed: {node.id}"
                )
                return outcome

        return outcome

    async def _execute_node(
        self,
        node: WorkflowNode,
        *,
        query: str,
        param_builder: ParamBuilder,
        event_callback: Optional[EventCallback],
        budget_tracker: Optional[RequestBudgetTracker],
    ) -> tuple[StepExecutionResult, Optional[SkillResult]]:
        params: Dict[str, Any] = {}
        param_builder_source = ""
        if node.skill:
            if node.inputs:
                params = dict(param_builder(node.skill, query))
                param_builder_source = "plan"
            else:
                params = dict(param_builder(node.skill, query))
                param_builder_source = "fallback_builder"
        if node.inputs:
            params.update(node.inputs)

        await self._emit(
            event_callback,
            "step_start",
            {
                "step_id": node.id,
                "title": node.title or node.skill or node.id,
                "description": "",
                "skill": node.skill,
                "kind": node.kind,
            },
        )

        started = time.perf_counter()
        last_error: Optional[str] = None
        skill_result: Optional[SkillResult] = None

        try:
            if node.kind != "tool" or not node.skill:
                raise ValueError(f"unsupported node kind: {node.kind!r}")
            if budget_tracker:
                budget_tracker.register_tool_call()

            coro = asyncio.to_thread(self._skill_manager.call_skill, node.skill, **params)
            if node.timeout_ms:
                skill_result = await asyncio.wait_for(
                    coro, timeout=node.timeout_ms / 1000.0
                )
            else:
                skill_result = await coro
        except asyncio.TimeoutError:
            last_error = f"node timeout after {node.timeout_ms} ms"
            skill_result = SkillResult.from_error(
                skill_name=node.skill or node.id,
                error_code="NODE_TIMEOUT",
                error_message=last_error,
            )
        except Exception as exc:
            last_error = str(exc)
            skill_result = SkillResult.from_error(
                skill_name=node.skill or node.id,
                error_code="NODE_EXECUTION_ERROR",
                error_message=last_error,
            )

        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        if skill_result and skill_result.latency_ms is None:
            skill_result.latency_ms = latency_ms

        status = "success" if skill_result and skill_result.success else "error"
        if status == "error" and not last_error:
            last_error = (
                skill_result.error_message if skill_result else "unknown error"
            )

        step_result = StepExecutionResult(
            step_id=node.id,
            title=node.title or node.skill or node.id,
            kind=node.kind,
            status=status,
            skill=node.skill,
            input_params=params,
            param_builder_source=param_builder_source,
            mcp_tools_used=_extract_mcp_tools_from_sources(
                list(skill_result.sources) if skill_result else []
            ),
            attempts=1,
            required=not node.optional,
            latency_ms=latency_ms,
            error=None if status == "success" else last_error,
            summary=skill_result.summary if skill_result else "",
            sources=list(skill_result.sources) if skill_result else [],
        )

        if skill_result:
            await self._emit(
                event_callback,
                "step_result",
                {
                    "step_id": node.id,
                    "skill": node.skill,
                    "status": status,
                    "summary": skill_result.summary,
                    "latency_ms": latency_ms,
                },
            )
            for source in skill_result.sources:
                await self._emit(event_callback, "evidence_found", source)

        await self._emit(
            event_callback,
            "step_end",
            {
                "step_id": node.id,
                "title": node.title or node.skill or node.id,
                "status": status,
                "skill": node.skill,
                "latency_ms": latency_ms,
                "error": last_error if status == "error" else None,
            },
        )
        return step_result, skill_result

    async def _emit(
        self,
        event_callback: Optional[EventCallback],
        event_type: str,
        payload: Dict[str, Any],
    ) -> None:
        if not event_callback:
            return
        maybe_result = event_callback(event_type, payload)
        if asyncio.iscoroutine(maybe_result):
            await maybe_result
