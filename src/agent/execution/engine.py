"""ExecutionEngine — 统一执行引擎（Phase 4 引入，Phase 8/9 为主路径）。

根据 ExecutionDecision.mode 分发到 DirectExecutor / PlannedExecutor / ReactExecutor。
Phase 9 起：ENABLE_UNIFIED_EXECUTION_ENGINE=True，本引擎为默认主路径；
            ENABLE_WORKFLOW_GRAPH flag 已移除，PlannedExecutor 直接使用 WorkflowExecutor。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

from src.agent.execution.direct_executor import DirectExecutor
from src.agent.execution.planned_executor import PlannedExecutor
from src.agent.execution.react_executor import ReactExecutor
from src.agent.executor import EventCallback
from src.agent.models.execution_event import ExecutionEvent
from src.agent.models.final_response import FinalResponse
from src.agent.planner import Planner
from src.agent.policies.budget_policy import RequestBudgetTracker
from src.agent.policies.fallback_policy import FallbackPolicy
from src.agent.request_router import RouteDecision
from src.agent.response_synthesizer import ResponseSynthesizer

if TYPE_CHECKING:
    from src.agent.models.execution_context import ExecutionContext
    from src.agent.models.execution_decision import ExecutionDecision
    from src.agent.models.execution_plan import ExecutionPlan


class ExecutionEngine:
    """统一执行入口，根据 ExecutionDecision.mode 分发到对应执行器。"""

    def __init__(
        self,
        skill_manager: Any,
        rag_retriever: Any,
        llm: Any,
        *,
        synthesizer: Optional[ResponseSynthesizer] = None,
        planner: Optional[Planner] = None,
        fallback_policy: Optional[FallbackPolicy] = None,
        agent_executor: Optional[Any] = None,
        agent_executor_factory: Optional[Any] = None,
    ) -> None:
        _synthesizer = synthesizer or ResponseSynthesizer(llm=llm)

        self._direct = DirectExecutor(
            skill_manager=skill_manager,
            rag_retriever=rag_retriever,
            llm=llm,
            synthesizer=_synthesizer,
        )
        self._planned = PlannedExecutor(
            skill_manager=skill_manager,
            llm=llm,
            synthesizer=_synthesizer,
            planner=planner,
            fallback_policy=fallback_policy,
        )
        self._react = ReactExecutor(
            agent_executor=agent_executor,
            agent_executor_factory=agent_executor_factory,
        )

    @property
    def direct(self) -> DirectExecutor:
        return self._direct

    @property
    def planned(self) -> PlannedExecutor:
        return self._planned

    @property
    def react(self) -> ReactExecutor:
        return self._react

    async def run(
        self,
        decision: "ExecutionDecision",
        legacy_decision: RouteDecision,
        query: str,
        *,
        chat_history: str = "",
        user_profile: str = "",
        execution_plan: Optional["ExecutionPlan"] = None,
        event_callback: Optional[EventCallback] = None,
        budget_tracker: Optional[RequestBudgetTracker] = None,
        context: Optional["ExecutionContext"] = None,
    ) -> FinalResponse:
        """根据 ExecutionDecision.mode 分发到对应执行器。

        参数：
            decision: Phase 3 引入的 ExecutionDecision，驱动分发逻辑。
            legacy_decision: 原有 RouteDecision，传递给各 Executor 的内部逻辑。
            query / chat_history / user_profile: 请求参数。
            execution_plan: 可选预构建计划（planned 路径）。
            event_callback: 步骤事件回调（planned 路径）。
            budget_tracker: 预算追踪器（planned 路径）。
            context: ExecutionContext（可选，Phase 2 引入），当前仅用于观测。
        """
        mode = decision.mode

        if mode == "direct":
            response = await self._direct.run(
                legacy_decision,
                query,
                chat_history=chat_history,
                user_profile=user_profile,
            )
            self._attach_engine_events(
                response,
                decision=decision,
                legacy_decision=legacy_decision,
                context=context,
            )
            return response

        if mode == "planned":
            response = await self._run_planned_with_recovery(
                legacy_decision,
                query,
                chat_history=chat_history,
                user_profile=user_profile,
                execution_plan=execution_plan,
                event_callback=event_callback,
                budget_tracker=budget_tracker,
            )
            self._attach_engine_events(
                response,
                decision=decision,
                legacy_decision=legacy_decision,
                context=context,
            )
            return response

        if mode == "react":
            response = await self._react.run(
                legacy_decision,
                query,
                chat_history=chat_history,
                user_profile=user_profile,
            )
            self._attach_engine_events(
                response,
                decision=decision,
                legacy_decision=legacy_decision,
                context=context,
            )
            return response

        raise ValueError(f"unsupported execution mode: {mode!r}")

    async def _run_planned_with_recovery(
        self,
        legacy_decision: RouteDecision,
        query: str,
        *,
        chat_history: str,
        user_profile: str,
        execution_plan: Optional["ExecutionPlan"],
        event_callback: Optional[EventCallback],
        budget_tracker: Optional[RequestBudgetTracker],
    ) -> FinalResponse:
        try:
            response = await self._planned.run(
                legacy_decision,
                query,
                chat_history=chat_history,
                user_profile=user_profile,
                execution_plan=execution_plan,
                event_callback=event_callback,
                budget_tracker=budget_tracker,
            )
        except Exception as exc:
            return await self._recover_from_planned_exception(
                legacy_decision,
                query,
                error=exc,
                chat_history=chat_history,
                user_profile=user_profile,
                event_callback=event_callback,
                budget_tracker=budget_tracker,
            )

        return await self._recover_planned_response(
            response,
            legacy_decision,
            query,
            chat_history=chat_history,
            user_profile=user_profile,
            event_callback=event_callback,
            budget_tracker=budget_tracker,
        )

    async def _recover_from_planned_exception(
        self,
        legacy_decision: RouteDecision,
        query: str,
        *,
        error: Exception,
        chat_history: str,
        user_profile: str,
        event_callback: Optional[EventCallback],
        budget_tracker: Optional[RequestBudgetTracker],
    ) -> FinalResponse:
        fallback = {
            "strategy": "plan_repair",
            "reason": "planned_execution_exception",
            "metadata": {
                "halt_reason": str(error),
                "task_type": legacy_decision.task_type,
                "recovery_mode": "plan_repair",
                "executed": False,
                "source_plan_step_ids": [],
            },
        }
        repaired_plan = self._build_repair_plan(
            legacy_decision,
            query,
            chat_history=chat_history,
            user_profile=user_profile,
            failed_response=None,
            error=str(error),
        )
        if repaired_plan is not None:
            repair_entry = self._with_recovery_metadata(
                fallback,
                mode="plan_repair",
                executed=True,
                source_plan_step_ids=[],
            )
            try:
                repaired_response = await self._planned.run(
                    legacy_decision,
                    query,
                    chat_history=chat_history,
                    user_profile=user_profile,
                    execution_plan=repaired_plan,
                    event_callback=event_callback,
                    budget_tracker=budget_tracker,
                )
                repaired_response.fallback_path = [
                    repair_entry,
                    *list(repaired_response.fallback_path or []),
                ]
                repaired_response.execution_events = [
                    self._plan_repaired_event(repaired_plan, repair_entry),
                    *list(repaired_response.execution_events or []),
                ]
                return await self._finalize_repaired_response(
                    repaired_response,
                    legacy_decision,
                    query,
                    chat_history=chat_history,
                    user_profile=user_profile,
                )
            except Exception as repair_error:
                fallback = {
                    "strategy": "react_fallback",
                    "reason": "plan_repair_failed",
                    "metadata": {
                        "halt_reason": str(repair_error),
                        "task_type": legacy_decision.task_type,
                    },
                }

        return await self._run_react_fallback(
            legacy_decision,
            query,
            chat_history=chat_history,
            user_profile=user_profile,
            planned_response=None,
            fallback_entry=fallback
            if fallback.get("strategy") == "react_fallback"
            else {
                "strategy": "react_fallback",
                "reason": "planned_execution_exception",
                "metadata": {
                    "halt_reason": str(error),
                    "task_type": legacy_decision.task_type,
                },
            },
        )

    async def _recover_planned_response(
        self,
        response: FinalResponse,
        legacy_decision: RouteDecision,
        query: str,
        *,
        chat_history: str,
        user_profile: str,
        event_callback: Optional[EventCallback],
        budget_tracker: Optional[RequestBudgetTracker],
    ) -> FinalResponse:
        fallback = self._primary_fallback(response)
        if fallback is None:
            return response

        strategy = str(fallback.get("strategy") or "")
        if strategy == "partial_answer":
            partial_entry = self._with_recovery_metadata(
                fallback,
                mode="partial_answer",
                executed=True,
                source_plan_step_ids=self._trace_step_ids(response),
            )
            response.fallback_path = [
                partial_entry,
                *list(response.fallback_path[1:]),
            ]
            response.execution_events = self._replace_fallback_event(
                response.execution_events,
                partial_entry,
            )
            return response

        if strategy == "plan_repair":
            repaired = await self._attempt_plan_repair(
                response,
                legacy_decision,
                query,
                fallback_entry=fallback,
                chat_history=chat_history,
                user_profile=user_profile,
                event_callback=event_callback,
                budget_tracker=budget_tracker,
            )
            if repaired is not None:
                return repaired
            fallback = {
                "strategy": "react_fallback",
                "reason": "plan_repair_unavailable",
                "metadata": dict(fallback.get("metadata") or {}),
            }

        if strategy == "react_fallback" or fallback.get("strategy") == "react_fallback":
            return await self._run_react_fallback(
                legacy_decision,
                query,
                chat_history=chat_history,
                user_profile=user_profile,
                planned_response=response,
                fallback_entry=fallback,
            )

        return response

    async def _attempt_plan_repair(
        self,
        response: FinalResponse,
        legacy_decision: RouteDecision,
        query: str,
        *,
        fallback_entry: Dict[str, Any],
        chat_history: str,
        user_profile: str,
        event_callback: Optional[EventCallback],
        budget_tracker: Optional[RequestBudgetTracker],
    ) -> Optional[FinalResponse]:
        repaired_plan = self._build_repair_plan(
            legacy_decision,
            query,
            chat_history=chat_history,
            user_profile=user_profile,
            failed_response=response,
            error=str((fallback_entry.get("metadata") or {}).get("halt_reason") or ""),
        )
        if repaired_plan is None:
            return None

        repair_entry = self._with_recovery_metadata(
            fallback_entry,
            mode="plan_repair",
            executed=True,
            source_plan_step_ids=self._trace_step_ids(response),
        )
        repaired_response = await self._planned.run(
            legacy_decision,
            query,
            chat_history=chat_history,
            user_profile=user_profile,
            execution_plan=repaired_plan,
            event_callback=event_callback,
            budget_tracker=budget_tracker,
        )
        repaired_response.fallback_path = [
            repair_entry,
            *list(repaired_response.fallback_path or []),
        ]
        repaired_response.execution_events = [
            *self._planned_context_events(response, repair_entry),
            self._plan_repaired_event(repaired_plan, repair_entry),
            *list(repaired_response.execution_events or []),
        ]

        return await self._finalize_repaired_response(
            repaired_response,
            legacy_decision,
            query,
            chat_history=chat_history,
            user_profile=user_profile,
        )

    async def _finalize_repaired_response(
        self,
        repaired_response: FinalResponse,
        legacy_decision: RouteDecision,
        query: str,
        *,
        chat_history: str,
        user_profile: str,
    ) -> FinalResponse:
        next_fallback = (
            repaired_response.fallback_path[1]
            if len(repaired_response.fallback_path) > 1
            else None
        )
        if not next_fallback:
            return repaired_response

        next_strategy = next_fallback.get("strategy")
        if next_strategy == "partial_answer":
            partial_entry = self._with_recovery_metadata(
                next_fallback,
                mode="partial_answer",
                executed=True,
                source_plan_step_ids=self._trace_step_ids(repaired_response),
            )
            repaired_response.fallback_path[1] = partial_entry
            repaired_response.execution_events.append(
                ExecutionEvent(
                    type="fallback_triggered",
                    payload=partial_entry,
                    source="planned",
                ).to_dict()
            )
            return repaired_response

        if next_strategy in {"react_fallback", "plan_repair"}:
            return await self._run_react_fallback(
                legacy_decision,
                query,
                chat_history=chat_history,
                user_profile=user_profile,
                planned_response=repaired_response,
                fallback_entry=next_fallback,
            )
        return repaired_response

    def _build_repair_plan(
        self,
        legacy_decision: RouteDecision,
        query: str,
        *,
        chat_history: str,
        user_profile: str,
        failed_response: Optional[FinalResponse],
        error: str,
    ) -> Optional["ExecutionPlan"]:
        repair_plan = getattr(self._planned, "repair_plan", None)
        if not callable(repair_plan):
            return None
        return repair_plan(
            legacy_decision,
            query,
            chat_history=chat_history,
            user_profile=user_profile,
            failed_response=failed_response,
            error=error,
        )

    async def _run_react_fallback(
        self,
        legacy_decision: RouteDecision,
        query: str,
        *,
        chat_history: str,
        user_profile: str,
        planned_response: Optional[FinalResponse],
        fallback_entry: Dict[str, Any],
    ) -> FinalResponse:
        react_response = await self._react.run(
            legacy_decision,
            query,
            chat_history=chat_history,
            user_profile=user_profile,
        )
        source_step_ids = (
            self._trace_step_ids(planned_response) if planned_response else []
        )
        react_entry = self._with_recovery_metadata(
            fallback_entry,
            mode="react",
            executed=True,
            source_plan_step_ids=source_step_ids,
        )

        if planned_response is not None:
            react_response.execution_plan = planned_response.execution_plan
            react_response.execution_trace = list(planned_response.execution_trace or [])
            react_response.route_decision = planned_response.route_decision
            react_response.budget_usage = planned_response.budget_usage
            react_response.versions = planned_response.versions
            react_response.audit_metadata = self._audit_with_recovery(
                planned_response.audit_metadata,
                react_entry,
            )
            react_response.sources = self._merge_unique_dicts(
                list(planned_response.sources or []),
                list(react_response.sources or []),
            )
            react_response.tools_used = [
                *list(planned_response.tools_used or []),
                *list(react_response.tools_used or []),
            ]
            existing_path = [
                item
                for item in list(planned_response.fallback_path or [])
                if item is not fallback_entry and item != fallback_entry
            ]
            react_response.fallback_path = [react_entry, *existing_path]
            react_response.execution_events = [
                *self._planned_context_events(planned_response, react_entry),
                *list(react_response.execution_events or []),
            ]
        else:
            react_response.fallback_path = [react_entry]
            react_response.audit_metadata = self._audit_with_recovery(
                react_response.audit_metadata,
                react_entry,
            )
            react_response.execution_events = [
                ExecutionEvent(
                    type="fallback_triggered",
                    payload=react_entry,
                    source="planned",
                ).to_dict(),
                *list(react_response.execution_events or []),
            ]
        return react_response

    @staticmethod
    def _primary_fallback(response: FinalResponse) -> Optional[Dict[str, Any]]:
        if not response.fallback_path:
            return None
        first = response.fallback_path[0]
        return dict(first) if isinstance(first, dict) else None

    @staticmethod
    def _trace_step_ids(response: Optional[FinalResponse]) -> list[str]:
        if response is None:
            return []
        return [
            str(trace.get("step_id"))
            for trace in list(response.execution_trace or [])
            if isinstance(trace, dict) and trace.get("step_id")
        ]

    @staticmethod
    def _with_recovery_metadata(
        fallback: Dict[str, Any],
        *,
        mode: str,
        executed: bool,
        source_plan_step_ids: list[str],
    ) -> Dict[str, Any]:
        payload = dict(fallback or {})
        metadata = dict(payload.get("metadata") or {})
        metadata["recovery_mode"] = mode
        metadata["executed"] = executed
        metadata["source_plan_step_ids"] = list(source_plan_step_ids)
        payload["metadata"] = metadata
        return payload

    @staticmethod
    def _planned_context_events(
        response: FinalResponse,
        fallback_entry: Dict[str, Any],
    ) -> list[dict[str, Any]]:
        events = [
            event
            for event in list(response.execution_events or [])
            if event.get("type")
            not in {"answer_ready", "final_answer", "fallback_triggered"}
        ]
        events.append(
            ExecutionEvent(
                type="fallback_triggered",
                payload=fallback_entry,
                source="planned",
            ).to_dict()
        )
        return events

    @staticmethod
    def _replace_fallback_event(
        events: list[dict[str, Any]],
        fallback_entry: Dict[str, Any],
    ) -> list[dict[str, Any]]:
        updated: list[dict[str, Any]] = []
        inserted = False
        for event in list(events or []):
            if event.get("type") == "fallback_triggered":
                if not inserted:
                    updated.append(
                        ExecutionEvent(
                            type="fallback_triggered",
                            payload=fallback_entry,
                            source=event.get("source") or "planned",
                        ).to_dict()
                    )
                    inserted = True
                continue
            updated.append(event)
        if not inserted:
            updated.append(
                ExecutionEvent(
                    type="fallback_triggered",
                    payload=fallback_entry,
                    source="planned",
                ).to_dict()
            )
        return updated

    @staticmethod
    def _plan_repaired_event(
        repaired_plan: "ExecutionPlan",
        fallback_entry: Dict[str, Any],
    ) -> dict[str, Any]:
        return ExecutionEvent(
            type="plan_repaired",
            payload={
                "plan": repaired_plan.to_dict()
                if hasattr(repaired_plan, "to_dict")
                else repaired_plan,
                "fallback": fallback_entry,
            },
            source="planned",
        ).to_dict()

    @staticmethod
    def _audit_with_recovery(
        audit_metadata: Optional[Dict[str, Any]],
        fallback_entry: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = dict(audit_metadata or {})
        payload["recovery"] = dict(fallback_entry)
        return payload

    @staticmethod
    def _merge_unique_dicts(
        first: list[dict[str, Any]],
        second: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for item in [*first, *second]:
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

    async def astream_events(
        self,
        decision: "ExecutionDecision",
        legacy_decision: RouteDecision,
        query: str,
        *,
        chat_history: str = "",
        user_profile: str = "",
        version: str = "v1",
        context: Optional["ExecutionContext"] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """统一的流式事件入口。

        当前仅 react 路径需要原始流式事件；direct/planned 继续通过 FinalResponse/trace 适配。
        """
        if decision.mode != "react":
            raise ValueError(
                "ExecutionEngine.astream_events() currently only supports react mode"
            )

        agent_input = self._react.build_agent_input(
            getattr(context, "query", query),
            chat_history=getattr(context, "chat_history", chat_history),
            user_profile=getattr(context, "user_profile", user_profile),
        )
        async for event in self._react.astream_events(agent_input, version=version):
            yield event

    def preview_plan(
        self,
        decision: "ExecutionDecision",
        legacy_decision: RouteDecision,
        query: str,
        *,
        chat_history: str = "",
        user_profile: str = "",
        execution_plan: Optional["ExecutionPlan"] = None,
    ) -> Optional["ExecutionPlan"]:
        """为 StreamingService 提供展示层所需的兼容计划视图。"""
        if decision.mode != "planned":
            return None
        return self._planned.preview_plan(
            legacy_decision,
            query,
            chat_history=chat_history,
            user_profile=user_profile,
            execution_plan=execution_plan,
        )

    def _attach_engine_events(
        self,
        response: FinalResponse,
        *,
        decision: "ExecutionDecision",
        legacy_decision: RouteDecision,
        context: Optional["ExecutionContext"],
    ) -> None:
        events = []
        if context is not None and getattr(context, "profile", None) is not None:
            events.append(
                ExecutionEvent(
                    type="task_profile",
                    payload=context.profile.to_dict(),
                    source="router",
                ).to_dict()
            )
        events.append(
            ExecutionEvent(
                type="route_decided",
                payload=legacy_decision.to_meta(),
                source="router",
            ).to_dict()
        )
        events.append(
            ExecutionEvent(
                type="execution_decision",
                payload=decision.to_dict(),
                source="policy",
            ).to_dict()
        )
        if response.execution_events:
            events.extend(list(response.execution_events))
        else:
            events.append(
                ExecutionEvent(
                    type="answer_ready",
                    payload={
                        "answer": response.answer,
                        "summary": response.summary,
                        "route": response.route,
                        "task_type": response.task_type,
                    },
                    source=decision.mode,
                ).to_dict()
            )
        response.execution_events = events
