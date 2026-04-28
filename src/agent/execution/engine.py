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
            response = await self._planned.run(
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
