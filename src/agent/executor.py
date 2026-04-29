from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.agent.models.execution_plan import ExecutionPlan, PlanStep
from src.agent.models.skill_result import SkillResult
from src.agent.policies.budget_policy import RequestBudgetTracker


EventCallback = Callable[[str, Dict[str, Any]], Awaitable[None] | None]
ParamBuilder = Callable[[str, str], Dict[str, Any]]


@dataclass
class StepExecutionResult:
    step_id: str
    title: str
    kind: str
    status: str
    skill: Optional[str] = None
    input_params: Dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    required: bool = True
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    summary: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "kind": self.kind,
            "status": self.status,
            "skill": self.skill,
            "input_params": dict(self.input_params),
            "attempts": self.attempts,
            "required": self.required,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "summary": self.summary,
            "sources": list(self.sources),
        }

    def to_trace_entry(self) -> "ExecutionTraceEntry":  # noqa: F821
        """转换为统一 trace 模型（Phase 7 引入）。"""
        from src.agent.models.execution_trace_entry import ExecutionTraceEntry
        return ExecutionTraceEntry.from_step_result(self)


@dataclass
class ExecutionOutcome:
    plan: ExecutionPlan
    skill_results: List[SkillResult] = field(default_factory=list)
    step_results: List[StepExecutionResult] = field(default_factory=list)
    halted: bool = False
    halt_reason: Optional[str] = None


class StepExecutor:
    """Deprecated linear ExecutionPlan executor.

    主 planned 路径已迁移到 WorkflowExecutor；本类仍保留给 TaskOrchestrator
    与历史测试/回退路径使用。新 planned 代码不得新增对本类的依赖。
    """

    def __init__(self, skill_manager: Any) -> None:
        self._skill_manager = skill_manager

    async def execute(
        self,
        plan: ExecutionPlan,
        *,
        query: str,
        param_builder: ParamBuilder,
        event_callback: Optional[EventCallback] = None,
        budget_tracker: Optional[RequestBudgetTracker] = None,
    ) -> ExecutionOutcome:
        """Execute a legacy ExecutionPlan linearly for compatibility only."""
        outcome = ExecutionOutcome(plan=plan)
        steps = list(plan.steps)
        index = 0

        while index < len(steps):
            step = steps[index]
            if step.parallel_group:
                group = [step]
                index += 1
                while index < len(steps) and steps[index].parallel_group == step.parallel_group:
                    group.append(steps[index])
                    index += 1
                if budget_tracker:
                    budget_tracker.register_parallelism(len(group))
                results = await asyncio.gather(
                    *[
                        self._execute_step(
                            group_step,
                            query=query,
                            param_builder=param_builder,
                            event_callback=event_callback,
                            budget_tracker=budget_tracker,
                        )
                        for group_step in group
                    ]
                )
            else:
                results = [
                    await self._execute_step(
                        step,
                        query=query,
                        param_builder=param_builder,
                        event_callback=event_callback,
                        budget_tracker=budget_tracker,
                    )
                ]
                index += 1

            for step_result, skill_result in results:
                outcome.step_results.append(step_result)
                if skill_result is not None:
                    outcome.skill_results.append(skill_result)
                if step_result.status == "error" and step_result.required:
                    outcome.halted = True
                    outcome.halt_reason = (
                        step_result.error
                        or f"required step failed: {step_result.step_id}"
                    )
                    return outcome

        return outcome

    async def _execute_step(
        self,
        step: PlanStep,
        *,
        query: str,
        param_builder: ParamBuilder,
        event_callback: Optional[EventCallback],
        budget_tracker: Optional[RequestBudgetTracker],
    ) -> tuple[StepExecutionResult, Optional[SkillResult]]:
        params = {}
        if step.skill:
            params = dict(param_builder(step.skill, query))
        if step.params:
            params.update(step.params)

        await self._emit(
            event_callback,
            "step_start",
            {
                "step_id": step.id,
                "title": step.title or step.skill or step.id,
                "description": step.description,
                "skill": step.skill,
                "kind": step.kind,
            },
        )

        started = time.perf_counter()
        attempts = 0
        last_error: Optional[str] = None
        skill_result: Optional[SkillResult] = None
        for attempts in range(1, step.retry_policy + 2):
            try:
                if step.kind != "tool" or not step.skill:
                    raise ValueError(f"unsupported step kind: {step.kind}")
                if budget_tracker:
                    budget_tracker.register_tool_call()
                coro = asyncio.to_thread(self._skill_manager.call_skill, step.skill, **params)
                if step.timeout_ms:
                    skill_result = await asyncio.wait_for(
                        coro,
                        timeout=step.timeout_ms / 1000.0,
                    )
                else:
                    skill_result = await coro
            except asyncio.TimeoutError:
                last_error = f"step timeout after {step.timeout_ms} ms"
                skill_result = SkillResult.from_error(
                    skill_name=step.skill or step.id,
                    error_code="STEP_TIMEOUT",
                    error_message=last_error,
                )
            except Exception as exc:
                last_error = str(exc)
                skill_result = SkillResult.from_error(
                    skill_name=step.skill or step.id,
                    error_code="STEP_EXECUTION_ERROR",
                    error_message=last_error,
                )

            if skill_result.success:
                break
            last_error = skill_result.error_message or last_error or "unknown error"

        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        if skill_result and skill_result.latency_ms is None:
            skill_result.latency_ms = latency_ms

        status = "success" if skill_result and skill_result.success else "error"
        step_result = StepExecutionResult(
            step_id=step.id,
            title=step.title or step.skill or step.id,
            kind=step.kind,
            status=status,
            skill=step.skill,
            input_params=params,
            attempts=attempts,
            required=step.required,
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
                    "step_id": step.id,
                    "skill": step.skill,
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
                "step_id": step.id,
                "title": step.title or step.skill or step.id,
                "status": status,
                "skill": step.skill,
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
