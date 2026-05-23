"""WorkflowGraph DAG 执行器，按依赖关系并发执行节点，并处理重试、跳过、失败策略和证据聚合。
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional

from src.core.mcp_protocol import is_tool_error, parse_tool_response
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
    """按依赖关系执行 WorkflowGraph 节点的 DAG 执行器。"""

    def __init__(self, skill_manager: Any) -> None:
        """初始化 WorkflowExecutor 的依赖、配置和内部状态。"""
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
        """在受控并发下执行 DAG。

        plan 仍作为兼容视图存在，用来承载来自 PlanStep 的步骤级策略，
        包括 retry_policy、required、fallback_strategy 和 evidence_key。
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
                outcome.halt_reason = (
                    "no executable DAG nodes; dependency resolution stalled"
                )
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
                    outcome.evidence_by_key, outcome.evidence_items = (
                        evidence.snapshot()
                    )
                    return outcome

        outcome.evidence_by_key, outcome.evidence_items = evidence.snapshot()
        return outcome

    @staticmethod
    def _chunks(nodes: List[WorkflowNode], size: int) -> List[List[WorkflowNode]]:
        """按最大并发数切分可执行节点批次。"""
        chunk_size = max(size, 1)
        return [nodes[i : i + chunk_size] for i in range(0, len(nodes), chunk_size)]

    @staticmethod
    def _max_parallelism(
        budget_tracker: Optional[RequestBudgetTracker],
        ready_count: int,
    ) -> int:
        """根据预算和就绪节点数计算本批最大并发数。"""
        if budget_tracker is None:
            return max(ready_count, 1)
        return max(min(ready_count, budget_tracker.budget.max_parallelism), 1)

    @staticmethod
    def _step_for_node(node: WorkflowNode, step_by_id: Dict[str, Any]) -> Any:
        """从兼容计划中查找当前 DAG 节点对应的 PlanStep。"""
        return step_by_id.get(node.id)

    def _retry_policy(self, node: WorkflowNode, step_by_id: Dict[str, Any]) -> int:
        """读取节点对应的重试次数策略。"""
        step = self._step_for_node(node, step_by_id)
        return int(getattr(step, "retry_policy", 0) or 0)

    def _is_required(self, node: WorkflowNode, step_by_id: Dict[str, Any]) -> bool:
        """判断节点是否为必需步骤。"""
        step = self._step_for_node(node, step_by_id)
        if step is not None:
            return bool(getattr(step, "required", True))
        return not node.optional

    def _fallback_strategy(self, node: WorkflowNode, step_by_id: Dict[str, Any]) -> str:
        """解析节点失败后的 fallback 策略。"""
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
        """计算节点结果应写入的证据键。"""
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
        """判断依赖结果是否会阻塞当前节点执行。"""
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
            if dep_result.required or strategy in {
                "halt",
                "skip_dependents",
                "react_fallback",
            }:
                return f"dependency failed: {dep_id}"
        return ""

    def _should_halt_on_failure(
        self,
        node: WorkflowNode,
        step_by_id: Dict[str, Any],
    ) -> bool:
        """判断节点失败后是否应中止整个 DAG。"""
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
        """计算某节点在 DAG 中的所有后代节点。"""
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
        """在 DAG 中止后把尚未执行的节点标记为 skipped。"""
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
        """执行单个 DAG 节点并生成步骤结果和 SkillResult。"""
        executable = self._resolve_executable(node)
        capability_kind = executable["capability_kind"]
        capability_name = executable["capability_name"]
        executable_skill = executable["skill"]
        executable_tool = executable["tool"]
        params, param_builder_source = self._build_node_params(
            node,
            executable=executable,
            query=query,
            param_builder=param_builder,
        )

        await self._emit(
            event_callback,
            "step_start",
            {
                "step_id": node.id,
                "title": node.title or capability_name or node.skill or node.id,
                "description": "",
                "skill": node.skill,
                "capability_kind": capability_kind,
                "capability_name": capability_name,
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
                if node.kind != "tool" or not (executable_skill or executable_tool):
                    raise ValueError(f"unsupported node kind: {node.kind!r}")
                if budget_tracker:
                    budget_tracker.register_tool_call()

                coro = asyncio.to_thread(
                    self._execute_capability,
                    executable=executable,
                    params=params,
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
                    skill_name=executable_skill or executable_tool or node.id,
                    error_code="NODE_TIMEOUT",
                    error_message=last_error,
                )
            except Exception as exc:
                last_error = str(exc)
                skill_result = SkillResult.from_error(
                    skill_name=executable_skill or executable_tool or node.id,
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
            last_error = skill_result.error_message if skill_result else "unknown error"

        step_result = StepExecutionResult(
            step_id=node.id,
            title=node.title or capability_name or node.skill or node.id,
            kind=node.kind,
            status=status,
            skill=node.skill,
            capability_kind=capability_kind,
            capability_name=capability_name,
            capability_reason=(
                "workflow_node_capability" if capability_name else ""
            ),
            input_params=params,
            param_builder_source=param_builder_source,
            mcp_tools_used=_extract_mcp_tools_from_sources(
                list(skill_result.sources) if skill_result else []
            ),
            logical_skill=(
                (
                    getattr(skill_result, "logical_skill", None)
                    or executable_skill
                    or executable_tool
                )
                if skill_result
                else None
            ),
            operation=(
                getattr(skill_result, "operation", None) if skill_result else None
            ),
            expected_mcp_tools=(
                list(getattr(skill_result, "expected_mcp_tools", []) or [])
                if skill_result
                else []
            ),
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
                    "capability_kind": capability_kind,
                    "capability_name": capability_name,
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
                "title": node.title or capability_name or node.skill or node.id,
                "status": status,
                "skill": node.skill,
                "capability_kind": capability_kind,
                "capability_name": capability_name,
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
        """把单个节点标记为 skipped 并发出结束事件。"""
        step_result = StepExecutionResult(
            step_id=node.id,
            title=(
                node.title
                or getattr(node, "capability_name", "")
                or node.skill
                or node.id
            ),
            kind=node.kind,
            status="skipped",
            skill=node.skill,
            capability_kind=getattr(node, "capability_kind", "") or "",
            capability_name=getattr(node, "capability_name", "") or "",
            capability_reason="workflow_node_capability",
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
                "capability_kind": step_result.capability_kind,
                "capability_name": step_result.capability_name,
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
        """调用可选事件回调，兼容同步和异步回调。"""
        if not event_callback:
            return
        maybe_result = event_callback(event_type, payload)
        if asyncio.iscoroutine(maybe_result):
            await maybe_result

    @staticmethod
    def _resolve_executable(node: WorkflowNode) -> Dict[str, Optional[str]]:
        """从 WorkflowNode 的能力字段解析可执行技能或原子工具。"""
        capability_kind = getattr(node, "capability_kind", "") or ""
        capability_name = getattr(node, "capability_name", "") or ""
        return {
            "capability_kind": capability_kind,
            "capability_name": capability_name,
            "skill": capability_name if capability_kind == "skill" else None,
            "tool": capability_name if capability_kind == "tool" else None,
        }

    def _build_node_params(
        self,
        node: WorkflowNode,
        *,
        executable: Dict[str, Optional[str]],
        query: str,
        param_builder: ParamBuilder,
    ) -> tuple[Dict[str, Any], str]:
        """合并计划输入和参数构建器结果，得到节点调用参数。"""
        capability_kind = executable["capability_kind"] or ""
        capability_name = executable["capability_name"] or ""
        executable_skill = executable["skill"]
        params: Dict[str, Any] = {}

        if executable_skill:
            params = self._build_capability_params(
                param_builder,
                "skill",
                executable_skill,
                query,
            )
        elif executable["tool"] and not node.inputs:
            params = self._build_capability_params(
                param_builder,
                capability_kind,
                capability_name,
                query,
            )

        if node.inputs:
            params.update(node.inputs)
            return params, "plan"
        return params, "fallback_builder" if params else ""

    @staticmethod
    def _build_capability_params(
        param_builder: ParamBuilder,
        capability_kind: str,
        capability_name: str,
        query: str,
    ) -> Dict[str, Any]:
        """通过参数构建器为技能或工具能力生成参数。"""
        if hasattr(param_builder, "build_for_capability"):
            return dict(
                param_builder.build_for_capability(  # type: ignore[attr-defined]
                    capability_kind,
                    capability_name,
                    query,
                )
            )
        return dict(param_builder(capability_name, query))

    def _execute_capability(
        self,
        *,
        executable: Dict[str, Optional[str]],
        params: Dict[str, Any],
    ) -> SkillResult:
        """根据可执行能力调用高层技能或原子工具。"""
        if executable.get("tool"):
            return self._call_atomic_tool_as_skill_result(
                str(executable["tool"]),
                params,
            )
        if executable.get("skill"):
            return self._skill_manager.call_skill(
                str(executable["skill"]),
                **params,
            )
        raise ValueError("workflow node has no executable capability")

    def _call_atomic_tool_as_skill_result(
        self,
        tool_name: str,
        params: Dict[str, Any],
    ) -> SkillResult:
        """调用原子 MCP 工具并包装为 SkillResult。"""
        if not hasattr(self._skill_manager, "call_mcp_tool"):
            raise ValueError(
                f"skill manager does not support atomic tool calls: {tool_name}"
            )

        raw = self._skill_manager.call_mcp_tool(tool_name, **params)
        if is_tool_error(raw):
            envelope = parse_tool_response(raw)
            error_msg = ""
            if envelope is not None and hasattr(envelope, "error"):
                error_msg = getattr(envelope.error, "message", "")
            result = SkillResult.from_error(
                skill_name=tool_name,
                error_code="TOOL_CALL_FAILED",
                error_message=error_msg or str(raw)[:500],
            )
            result.logical_skill = tool_name
            result.expected_mcp_tools = [tool_name]
            result.allowed_child_tools = [tool_name]
            result.sources = [
                {"kind": "mcp_tool", "tool": tool_name, "snippet": str(raw)[:240]}
            ]
            return result

        envelope = parse_tool_response(raw)
        if envelope is not None and hasattr(envelope, "data"):
            payload = envelope.data
        else:
            try:
                payload = json.loads(raw)
            except Exception:
                payload = raw

        data = payload if isinstance(payload, dict) else {"raw": payload}
        summary = (
            payload
            if isinstance(payload, str)
            else json.dumps(payload, ensure_ascii=False)
        )
        return SkillResult(
            skill_name=tool_name,
            success=True,
            data=data,
            summary=summary,
            sources=[
                {"kind": "mcp_tool", "tool": tool_name, "snippet": str(raw)[:240]}
            ],
            logical_skill=tool_name,
            expected_mcp_tools=[tool_name],
            allowed_child_tools=[tool_name],
        )
