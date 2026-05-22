"""PlannedExecutor — structured DAG execution path for planned tasks."""
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
from src.capabilities.param_builder import CapabilityParamBuilder
from src.core.config import settings


class PlannedExecutor:
    """Run planned tasks as plan_graph -> DAG execution -> evidence synthesis."""

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
        context: Optional[Any] = None,
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
            param_builder=CapabilityParamBuilder(
                self._param_builder,
                chat_history=chat_history,
                user_profile=user_profile,
            ),
            event_callback=event_callback,
            budget_tracker=budget_tracker,
        )

        fallback_decision = self._fallback_policy.decide_for_execution(
            outcome=outcome,
            plan=plan,
        )

        versions_payload = {
            "router_policy_version": str(
                getattr(settings, "ROUTER_POLICY_VERSION", "router_v1")
            ),
            "planner_version": str(getattr(settings, "PLANNER_VERSION", "planner_v2")),
            "schema_version": str(getattr(settings, "SCHEMA_VERSION", "schema_v2")),
            "synth_prompt_version": str(
                getattr(
                    self._synthesizer,
                    "prompt_version",
                    getattr(settings, "SYNTH_PROMPT_VERSION", "synth_prompt_v3"),
                )
            ),
            "fallback_policy_version": self._fallback_policy.version,
            "budget_policy_version": (
                budget_tracker.budget.policy_version if budget_tracker else "budget_v1"
            ),
        }

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
            versions=versions_payload,
            evidence=outcome.evidence_by_key,
        )
        response.route = decision.route
        response.task_type = decision.task_type
        response.sources = self._merge_sources(response.sources, outcome.evidence_items)
        response.execution_plan = plan.to_dict()
        response.execution_trace = [step.to_dict() for step in outcome.step_results]
        response.route_decision = decision.to_meta()
        response.fallback_path = [fallback_decision.to_dict()] if fallback_decision else []
        response.budget_usage = budget_tracker.snapshot() if budget_tracker else None
        response.versions = response.versions or versions_payload
        response.audit_metadata = self._build_observability_metadata(
            decision=decision,
            plan=plan,
            execution_trace=response.execution_trace,
            evidence_by_key=outcome.evidence_by_key,
            skipped_step_ids=outcome.skipped_step_ids,
            context=context,
        )
        response.execution_events = self._build_execution_events(
            response=response,
            fallback_decision=fallback_decision,
        )
        return response

    @staticmethod
    def _merge_sources(
        existing: list[dict[str, Any]],
        evidence_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for item in list(existing or []) + list(evidence_items or []):
            if not isinstance(item, dict):
                continue
            identity = (
                str(item.get("source_id") or ""),
                str(item.get("title") or ""),
                str(item.get("snippet") or ""),
            )
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(dict(item))
        return merged

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

    def repair_plan(
        self,
        decision: RouteDecision,
        query: str,
        *,
        chat_history: str = "",
        user_profile: str = "",
        failed_response: Optional[FinalResponse] = None,
        error: str = "",
    ) -> Optional[ExecutionPlan]:
        """Build a one-shot repaired plan for structural/parameter failures.

        This is intentionally narrow: it only normalizes an existing planned
        response by fixing bad dependencies, converting repairable tool nodes,
        and rebuilding params for failed tool steps. Tool outages and timeouts
        are handled by fallback instead of broad replanning.
        """
        if failed_response is None or not failed_response.execution_plan:
            return None

        try:
            plan = ExecutionPlan.from_dict(failed_response.execution_plan)
        except Exception:
            return None

        if not plan.steps:
            return None

        failed_step_ids = self._failed_step_ids_from_response(failed_response)
        trace_by_step = {
            str(trace.get("step_id", "")): trace
            for trace in failed_response.execution_trace
            if isinstance(trace, dict)
        }
        valid_ids = {step.id for step in plan.steps}
        changed = False

        for step in plan.steps:
            valid_depends_on = [
                dep for dep in step.depends_on if dep in valid_ids and dep != step.id
            ]
            if valid_depends_on != step.depends_on:
                step.depends_on = valid_depends_on
                changed = True

            if step.kind != "tool" and step.skill:
                step.kind = "tool"
                changed = True

            trace = trace_by_step.get(step.id, {})
            step_error = str(trace.get("error") or error or "")
            if (
                step.id in failed_step_ids
                and step.skill
                and self._is_param_repair_error(step_error)
            ):
                try:
                    step.params = self._param_builder.build(
                        step.skill,
                        query,
                        chat_history=chat_history,
                        user_profile=user_profile,
                    )
                    changed = True
                except Exception:
                    pass

        if not changed:
            return None

        plan.planner_type = f"{plan.planner_type}_repair"
        plan.rationale = (
            f"{plan.rationale}\n"
            "受控计划修复：规范依赖/节点类型，并为失败步骤重建参数。"
        ).strip()
        return plan

    @staticmethod
    def _failed_step_ids_from_response(response: FinalResponse) -> set[str]:
        ids: set[str] = set()
        for fallback in response.fallback_path or []:
            metadata = fallback.get("metadata") or {}
            for key in ("required_failed_steps", "optional_failed_steps"):
                for step_id in metadata.get(key) or []:
                    ids.add(str(step_id))
        for trace in response.execution_trace or []:
            if not isinstance(trace, dict):
                continue
            if trace.get("status") in {"error", "skipped"}:
                ids.add(str(trace.get("step_id", "")))
        return {step_id for step_id in ids if step_id}

    @staticmethod
    def _is_param_repair_error(error: str) -> bool:
        text = str(error or "").lower()
        return any(
            hint in text
            for hint in (
                "missing required",
                "required parameter",
                "missing parameter",
                "invalid param",
                "参数",
                "缺少",
            )
        )

    def _resolve_plan_and_graph(
        self,
        decision: RouteDecision,
        query: str,
        *,
        chat_history: str = "",
        user_profile: str = "",
        execution_plan: Optional[ExecutionPlan] = None,
    ) -> tuple[ExecutionPlan, WorkflowGraph]:
        """Resolve the native WorkflowGraph and compatibility ExecutionPlan view."""
        if execution_plan is not None:
            return execution_plan, WorkflowGraph.from_execution_plan(execution_plan)

        if hasattr(self._planner, "plan_graph"):
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

    def _build_observability_metadata(
        self,
        *,
        decision: RouteDecision,
        plan: ExecutionPlan,
        execution_trace: list[dict[str, Any]],
        evidence_by_key: dict[str, Any],
        skipped_step_ids: list[str],
        context: Optional[Any] = None,
    ) -> dict[str, Any]:
        route_meta = decision.to_meta()
        capability = getattr(context, "capability_decision", None)
        capability_payload = (
            capability.to_dict()
            if capability is not None and hasattr(capability, "to_dict")
            else None
        )
        param_sources: list[str] = []
        handler_mcp_tools: list[str] = []
        operations: list[dict[str, Any]] = []
        expected_mcp_tools: list[str] = []
        for trace in execution_trace:
            source = trace.get("param_builder_source")
            if source and source not in param_sources:
                param_sources.append(str(source))
            for tool_name in trace.get("mcp_tools_used") or []:
                if tool_name not in handler_mcp_tools:
                    handler_mcp_tools.append(tool_name)
            for tool_name in trace.get("expected_mcp_tools") or []:
                if tool_name not in expected_mcp_tools:
                    expected_mcp_tools.append(tool_name)
            if trace.get("operation"):
                operations.append(
                    {
                        "step_id": trace.get("step_id"),
                        "logical_skill": trace.get("logical_skill") or trace.get("skill"),
                        "operation": trace.get("operation"),
                        "expected_mcp_tools": list(
                            trace.get("expected_mcp_tools") or []
                        ),
                        "actual_mcp_tools": list(trace.get("mcp_tools_used") or []),
                    }
                )

        if len(param_sources) == 1:
            param_builder_source = param_sources[0]
        elif len(param_sources) > 1:
            param_builder_source = "mixed"
        else:
            param_builder_source = ""

        return {
            "router_source": route_meta.get("router_source"),
            "rule_confidence": route_meta.get("rule_confidence"),
            "llm_confidence": route_meta.get("llm_confidence"),
            "tool_necessity_action": route_meta.get("tool_necessity_action"),
            "tool_necessity_reason": route_meta.get("tool_necessity_reason"),
            "tool_necessity_confidence": route_meta.get(
                "tool_necessity_confidence"
            ),
            "tool_necessity_missing_params": route_meta.get(
                "tool_necessity_missing_params", []
            ),
            "tool_necessity_allowed_skill_hints": route_meta.get(
                "tool_necessity_allowed_skill_hints", []
            ),
            "tool_necessity_forbidden_skill_hints": route_meta.get(
                "tool_necessity_forbidden_skill_hints", []
            ),
            "planner_source": plan.planner_type,
            "plan_steps_with_params": [
                {
                    "id": step.id,
                    "skill": step.skill,
                    "capability_kind": step.capability_kind,
                    "capability_name": step.capability_name,
                    "params": dict(step.params),
                    "planner_source": step.planner_source or plan.planner_type,
                    "purpose": step.purpose,
                    "success_criteria": step.success_criteria,
                    "evidence_key": step.evidence_key,
                }
                for step in plan.steps
            ],
            "param_builder_source": param_builder_source,
            "param_builder_sources": param_sources,
            "handler_mcp_tools_used": handler_mcp_tools,
            "operations": operations,
            "expected_mcp_tools": expected_mcp_tools,
            "capability_decision": capability_payload,
            "capability_kind": (
                capability_payload.get("kind", "") if capability_payload else ""
            ),
            "capability_name": (
                capability_payload.get("name", "") if capability_payload else ""
            ),
            "capability_reason": (
                capability_payload.get("reason", "") if capability_payload else ""
            ),
            "dag_node_count": len(plan.steps),
            "dag_evidence_keys": sorted(evidence_by_key),
            "dag_evidence": evidence_by_key,
            "skipped_step_ids": list(skipped_step_ids),
        }

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
