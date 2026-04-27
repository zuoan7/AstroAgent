"""ExecutionEngine — 统一执行引擎（Phase 4 引入，Phase 8/9 为主路径）。

根据 ExecutionDecision.mode 分发到 DirectExecutor / PlannedExecutor / ReactExecutor。
Phase 9 起：ENABLE_UNIFIED_EXECUTION_ENGINE=True，本引擎为默认主路径；
            ENABLE_WORKFLOW_GRAPH flag 已移除，PlannedExecutor 直接使用 WorkflowExecutor。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from src.agent.execution.direct_executor import DirectExecutor
from src.agent.execution.planned_executor import PlannedExecutor
from src.agent.execution.react_executor import ReactExecutor
from src.agent.executor import EventCallback
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
            return await self._direct.run(
                legacy_decision,
                query,
                chat_history=chat_history,
                user_profile=user_profile,
            )

        if mode == "planned":
            return await self._planned.run(
                legacy_decision,
                query,
                chat_history=chat_history,
                user_profile=user_profile,
                execution_plan=execution_plan,
                event_callback=event_callback,
                budget_tracker=budget_tracker,
            )

        if mode == "react":
            raise NotImplementedError(
                "ExecutionEngine.run() 不支持 react 流式路径；"
                "请通过 engine.react.astream_events() 进行流式调用。"
            )

        raise ValueError(f"unsupported execution mode: {mode!r}")
