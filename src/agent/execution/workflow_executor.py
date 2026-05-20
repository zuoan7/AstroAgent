"""WorkflowExecutor — native DAG executor for planned tasks.

Executes ready WorkflowGraph nodes concurrently while respecting dependencies,
step retry policy, failure strategy, and evidence aggregation.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

from src.agent.executor import (
    EvidenceAggregator,
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
    """Execute WorkflowGraph nodes as a dependency-aware DAG."""

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
        """Execute a DAG with bounded concurrency.

        `plan` remains a compatibility view and carries step-level policy
        (`retry_policy`, `required`, `fallback_strategy`, `evidence_key`) for
        graph nodes that originated from PlanStep.
        """
        outcome = ExecutionOutcome(plan=plan)
        evidence = EvidenceAggregator()
        step_by_id = {step.id: step for step in plan.steps}
        node_by_id = {node.id: node for node in graph.nodes}
        dependency_map = graph.dependency_map()
        successor_map = graph.successor_map()

        validation_errors = graph.validate()
        if validation_errors:
            outcome.halted = True
            outcome.halt_reason = "; ".join(validation_errors)
            return outcome

        pending = set(node_by_id)
        terminal: Dict[str, StepExecutionResult] = {}

        while pending:
            ready: List[WorkflowNode] = []
            skipped: List[tuple[WorkflowNode, str]] = []

            for node_id in sorted(pending):
                deps = dependency_map.get(node_id, [])
                if not all(dep in terminal for dep in deps):
                    continue
                block_reason = self._dependency_block_reason(
                    deps,
                    terminal,
                    step_by_id,
                )
                if block_reason:
                    skipped.append((node_by_id[node_id], block_reason))
                else:
                    ready.append(node_by_id[node_id])

            for node, reason in skipped:
                step_result = await self._skip_node(
                    node,
                    reason=reason,
                    dependencies=dependency_map.get(node.id, []),
                    step_by_id=step_by_id,
                    event_callback=event_callback,
                )
                pending.remove(node.id)
                terminal[node.id] = step_result
                outcome.step_results.append(step_result)
                outcome.skipped_step_ids.append(node.id)
                evidence.add(
                    key=step_result.evidence_key or node.output_key or node.id,
                    step_result=step_result,
                    skill_result=None,
                )

            if skipped:
                outcome.evidence_by_key, outcome.evidence_items = evidence.snapshot()

            if not ready:
                if skipped:
                    continue
                outcome.halted = True
                outcome.halt_reason = "no executable DAG nodes; dependency resolution stalled"
                outcome.evidence_by_key, outcome.evidence_items = evidence.snapshot()
                return outcome

            max_parallelism = self._max_parallelism(budget_tracker, len(ready))
            for batch in self._chunks(ready, max_parallelism):
                if budget_tracker:
                    budget_tracker.register_parallelism(len(batch))

                results = await asyncio.gather(
                    *[
                        self._execute_node(
                            node,
                            query=query,
                            param_builder=param_builder,
                            event_callback=event_callback,
                            budget_tracker=budget_tracker,
                            dependencies=dependency_map.get(node.id, []),
                            step_by_id=step_by_id,
                        )
                        for node in batch
                    ]
                )

                halting_failure: Optional[str] = None
                skip_descendants_from: List[str] = []
                for node, (step_result, skill_result) in zip(batch, results):
                    pending.remove(node.id)
                    terminal[node.id] = step_result
                    outcome.step_results.append(step_result)
                    if skill_result is not None:
                        outcome.skill_results.append(skill_result)
                    evidence.add(
                        key=step_result.evidence_key or node.output_key or node.id,
                        step_result=step_result,
                        skill_result=skill_result,
                    )

                    if step_result.status == "error":
                        strategy = self._fallback_strategy(node, step_by_id)
                        if strategy == "skip_dependents":
                            skip_descendants_from.append(node.id)
                        if self._should_halt_on_failure(node, step_by_id):
                            halting_failure = (
                                step_result.error
                                or f"required node failed: {step_result.step_id}"
                            )

                for failed_node_id in skip_descendants_from:
                    for descendant_id in self._descendants(
                        failed_node_id,
                        successor_map,
                    ):
                        if descendant_id not in pending:
                            continue
                        node = node_by_id[descendant_id]
                        step_result = await self._skip_node(
                            node,
                            reason=f"dependency requested dependent skip: {failed_node_id}",
                            dependencies=dependency_map.get(node.id, []),
                            step_by_id=step_by_id,
                            event_callback=event_callback,
                        )
                        pending.remove(descendant_id)
                        terminal[descendant_id] = step_result
                        outcome.step_results.append(step_result)
                        outcome.skipped_step_ids.append(descendant_id)
                        evidence.add(
                            key=step_result.evidence_key or node.output_key or node.id,
                            step_result=step_result,
                            skill_result=None,
                        )

                outcome.evidence_by_key, outcome.evidence_items = evidence.snapshot()
                if halting_failure:
                    outcome.halted = True
                    outcome.halt_reason = halting_failure
                    await self._skip_pending_nodes(
                        pending=pending,
                        node_by_id=node_by_id,
                        dependency_map=dependency_map,
                        step_by_id=step_by_id,
                        event_callback=event_callback,
                        outcome=outcome,
                        terminal=terminal,
                        evidence=evidence,
                        reason=halting_failure,
                    )
                    outcome.evidence_by_key, outcome.evidence_items = evidence.snapshot()
                    return outcome

        outcome.evidence_by_key, outcome.evidence_items = evidence.snapshot()
        return outcome

    @staticmethod
    def _chunks(nodes: List[WorkflowNode], size: int) -> List[List[WorkflowNode]]:
        chunk_size = max(size, 1)
        return [nodes[i : i + chunk_size] for i in range(0, len(nodes), chunk_size)]

    @staticmethod
    def _max_parallelism(
        budget_tracker: Optional[RequestBudgetTracker],
        ready_count: int,
    ) -> int:
        if budget_tracker is None:
            return max(ready_count, 1)
        return max(min(ready_count, budget_tracker.budget.max_parallelism), 1)

    @staticmethod
    def _step_for_node(node: WorkflowNode, step_by_id: Dict[str, Any]) -> Any:
        return step_by_id.get(node.id)

    def _retry_policy(self, node: WorkflowNode, step_by_id: Dict[str, Any]) -> int:
        step = self._step_for_node(node, step_by_id)
        return int(getattr(step, "retry_policy", 0) or 0)

    def _is_required(self, node: WorkflowNode, step_by_id: Dict[str, Any]) -> bool:
        step = self._step_for_node(node, step_by_id)
        if step is not None:
            return bool(getattr(step, "required", True))
        return not node.optional

    def _fallback_strategy(self, node: WorkflowNode, step_by_id: Dict[str, Any]) -> str:
        step = self._step_for_node(node, step_by_id)
        raw = str(getattr(step, "fallback_strategy", "") or "").strip().lower()
        if raw in {"continue", "partial", "partial_answer", "ignore"}:
            return "continue"
        if raw in {"skip", "skip_dependents", "skip_dependent"}:
            return "skip_dependents"
        if raw in {"react", "react_fallback", "fallback_react"}:
            return "react_fallback"
        if raw in {"halt", "fail", "fail_fast"}:
            return "halt"
        return "halt" if self._is_required(node, step_by_id) else "continue"

    def _evidence_key(self, node: WorkflowNode, step_by_id: Dict[str, Any]) -> str:
        step = self._step_for_node(node, step_by_id)
        return (
            str(getattr(step, "evidence_key", "") or "")
            or str(node.output_key or "")
            or node.id
        )

    def _dependency_block_reason(
        self,
        dependencies: List[str],
        terminal: Dict[str, StepExecutionResult],
        step_by_id: Dict[str, Any],
    ) -> str:
        for dep_id in dependencies:
            dep_result = terminal.get(dep_id)
            if dep_result is None or dep_result.status == "success":
                continue
            dep_node = WorkflowNode(
                id=dep_id,
                optional=not dep_result.required,
            )
            strategy = dep_result.fallback_strategy or self._fallback_strategy(
                dep_node,
                step_by_id,
            )
            if dep_result.status == "skipped":
                return f"dependency skipped: {dep_id}"
            if dep_result.required or strategy in {"halt", "skip_dependents", "react_fallback"}:
                return f"dependency failed: {dep_id}"
        return ""

    def _should_halt_on_failure(
        self,
        node: WorkflowNode,
        step_by_id: Dict[str, Any],
    ) -> bool:
        strategy = self._fallback_strategy(node, step_by_id)
        return strategy in {"halt", "react_fallback"} and self._is_required(
            node,
            step_by_id,
        )

    @staticmethod
    def _descendants(
        node_id: str,
        successor_map: Dict[str, List[str]],
    ) -> List[str]:
        seen: set[str] = set()
        stack = list(successor_map.get(node_id, []))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(successor_map.get(current, []))
        return sorted(seen)

    async def _skip_pending_nodes(
        self,
        *,
        pending: set[str],
        node_by_id: Dict[str, WorkflowNode],
        dependency_map: Dict[str, List[str]],
        step_by_id: Dict[str, Any],
        event_callback: Optional[EventCallback],
        outcome: ExecutionOutcome,
        terminal: Dict[str, StepExecutionResult],
        evidence: EvidenceAggregator,
        reason: str,
    ) -> None:
        for node_id in sorted(list(pending)):
            node = node_by_id[node_id]
            step_result = await self._skip_node(
                node,
                reason=f"planned execution halted: {reason}",
                dependencies=dependency_map.get(node.id, []),
                step_by_id=step_by_id,
                event_callback=event_callback,
            )
            pending.remove(node_id)
            terminal[node_id] = step_result
            outcome.step_results.append(step_result)
            outcome.skipped_step_ids.append(node_id)
            evidence.add(
                key=step_result.evidence_key or node.output_key or node.id,
                step_result=step_result,
                skill_result=None,
            )

    async def _execute_node(
        self,
        node: WorkflowNode,
        *,
        query: str,
        param_builder: ParamBuilder,
        event_callback: Optional[EventCallback],
        budget_tracker: Optional[RequestBudgetTracker],
        dependencies: List[str],
        step_by_id: Dict[str, Any],
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
                "depends_on": list(dependencies),
                "evidence_key": self._evidence_key(node, step_by_id),
                "fallback_strategy": self._fallback_strategy(node, step_by_id),
            },
        )

        started = time.perf_counter()
        attempts = 0
        last_error: Optional[str] = None
        skill_result: Optional[SkillResult] = None

        for attempts in range(1, self._retry_policy(node, step_by_id) + 2):
            try:
                if node.kind != "tool" or not node.skill:
                    raise ValueError(f"unsupported node kind: {node.kind!r}")
                if budget_tracker:
                    budget_tracker.register_tool_call()

                coro = asyncio.to_thread(
                    self._skill_manager.call_skill,
                    node.skill,
                    **params,
                )
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

            if skill_result.success:
                break
            last_error = skill_result.error_message or last_error or "unknown error"

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
            logical_skill=getattr(skill_result, "logical_skill", None)
            if skill_result
            else None,
            operation=getattr(skill_result, "operation", None) if skill_result else None,
            expected_mcp_tools=list(
                getattr(skill_result, "expected_mcp_tools", []) or []
            )
            if skill_result
            else [],
            attempts=attempts,
            required=self._is_required(node, step_by_id),
            latency_ms=latency_ms,
            error=None if status == "success" else last_error,
            summary=skill_result.summary if skill_result else "",
            sources=list(skill_result.sources) if skill_result else [],
            depends_on=list(dependencies),
            evidence_key=self._evidence_key(node, step_by_id),
            fallback_strategy=self._fallback_strategy(node, step_by_id),
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
                "depends_on": list(dependencies),
                "evidence_key": step_result.evidence_key,
                "fallback_strategy": step_result.fallback_strategy,
            },
        )
        return step_result, skill_result

    async def _skip_node(
        self,
        node: WorkflowNode,
        *,
        reason: str,
        dependencies: List[str],
        step_by_id: Dict[str, Any],
        event_callback: Optional[EventCallback],
    ) -> StepExecutionResult:
        step_result = StepExecutionResult(
            step_id=node.id,
            title=node.title or node.skill or node.id,
            kind=node.kind,
            status="skipped",
            skill=node.skill,
            attempts=0,
            required=self._is_required(node, step_by_id),
            error=reason,
            depends_on=list(dependencies),
            evidence_key=self._evidence_key(node, step_by_id),
            fallback_strategy=self._fallback_strategy(node, step_by_id),
        )
        await self._emit(
            event_callback,
            "step_end",
            {
                "step_id": node.id,
                "title": step_result.title,
                "status": "skipped",
                "skill": node.skill,
                "error": reason,
                "depends_on": list(dependencies),
                "evidence_key": step_result.evidence_key,
                "fallback_strategy": step_result.fallback_strategy,
            },
        )
        return step_result

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
