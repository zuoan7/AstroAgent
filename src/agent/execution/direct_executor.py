"""DirectExecutor — 直接任务执行器（Phase 4 引入）。

抽取自 TaskOrchestrator._run_direct_task()，逻辑完全等价。
Phase 9 起作为主路径，循环依赖已消除（改用 SkillParamBuilder）。
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from src.agent.executor import _extract_mcp_tools_from_sources
from src.agent.models.execution_event import ExecutionEvent
from src.agent.models.final_response import FinalResponse
from src.agent.policies.prompt_budget import PromptBudgetManager, PromptSection
from src.agent.request_router import RouteDecision
from src.agent.skill_param_builder import SkillParamBuilder
from src.core.config import settings


class DirectExecutor:
    """封装 direct_task 子路径。"""

    def __init__(
        self,
        skill_manager: Any,
        rag_retriever: Any,
        llm: Any,
        synthesizer: Any,
    ) -> None:
        self._skill_manager = skill_manager
        self._rag = rag_retriever
        self._llm = llm
        self._synthesizer = synthesizer
        self._param_builder = SkillParamBuilder(skill_manager)

    async def run(
        self,
        decision: RouteDecision,
        query: str,
        *,
        chat_history: str = "",
        user_profile: str = "",
    ) -> FinalResponse:
        if decision.task_type == "smalltalk":
            response = self._synthesizer.synthesize_smalltalk(
                self._smalltalk_reply(query)
            )
            self._attach_response_metadata(response, decision=decision)
            self._attach_execution_events(response)
            return response

        if decision.task_type == "clarification":
            response = self._run_clarification(decision, query)
            self._attach_response_metadata(response, decision=decision)
            self._attach_execution_events(response)
            return response

        if decision.task_type == "direct_answer_no_tool":
            response = self._run_no_tool_answer(decision, query)
            self._attach_response_metadata(response, decision=decision)
            self._attach_execution_events(response)
            return response

        if decision.task_type == "single_tool_lookup":
            return await self._run_tool_task(decision, query)

        if decision.task_type == "simple_qa":
            response = await self._run_simple_qa(
                query, chat_history=chat_history, user_profile=user_profile
            )
            self._attach_response_metadata(response, decision=decision)
            self._attach_execution_events(response)
            return response

        raise ValueError(f"unsupported direct task type: {decision.task_type}")

    def _run_clarification(
        self,
        decision: RouteDecision,
        query: str,
    ) -> FinalResponse:
        answer = (
            decision.clarification_prompt
            or "这个请求还缺少关键信息。请补充目标、时间、地点或器材参数后我再继续。"
        )
        return FinalResponse(
            answer=answer,
            summary=answer[:200] if len(answer) > 200 else answer,
            tools_used=[],
            sources=[],
            confidence=decision.confidence,
            route=decision.route,
            task_type=decision.task_type,
            versions=self._synthesizer._default_versions()
            if hasattr(self._synthesizer, "_default_versions")
            else None,
        )

    def _run_no_tool_answer(
        self,
        decision: RouteDecision,
        query: str,
    ) -> FinalResponse:
        answer = decision.answer_hint or self._direct_no_tool_reply(query)
        return FinalResponse(
            answer=answer,
            summary=answer[:200] if len(answer) > 200 else answer,
            tools_used=[],
            sources=[],
            confidence=decision.confidence,
            route=decision.route,
            task_type=decision.task_type,
            versions=self._synthesizer._default_versions()
            if hasattr(self._synthesizer, "_default_versions")
            else None,
        )

    async def _run_tool_task(self, decision: RouteDecision, query: str) -> FinalResponse:
        from src.agent.models.skill_result import SkillResult
        from src.agent.param_parser import ParamParser
        from src.skills import registry

        skill_name = decision.matched_skills[0]
        params = self._build_skill_params(skill_name, query)
        result: SkillResult = await asyncio.to_thread(
            self._skill_manager.call_skill,
            skill_name,
            **params,
        )
        response = self._synthesizer.synthesize_direct(
            query=query,
            task_type=decision.task_type,
            skill_results=[result],
        )
        self._attach_response_metadata(
            response,
            decision=decision,
            param_builder_source="fallback_builder",
            handler_mcp_tools_used=_extract_mcp_tools_from_sources(result.sources),
            logical_skill=getattr(result, "logical_skill", None),
            operation=getattr(result, "operation", None),
            expected_mcp_tools=getattr(result, "expected_mcp_tools", []),
        )
        self._attach_execution_events(
            response,
            tool_name=skill_name,
            tool_input=params,
            tool_summary=result.summary,
            tool_status="success" if result.success else "error",
        )
        return response

    def _attach_response_metadata(
        self,
        response: FinalResponse,
        *,
        decision: RouteDecision,
        param_builder_source: str = "",
        handler_mcp_tools_used: Optional[list[str]] = None,
        logical_skill: Optional[str] = None,
        operation: Optional[str] = None,
        expected_mcp_tools: Optional[list[str]] = None,
    ) -> None:
        route_meta = decision.to_meta()
        response.route = decision.route
        response.task_type = decision.task_type
        response.route_decision = route_meta
        response.audit_metadata = {
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
            "planner_source": "",
            "plan_steps_with_params": [],
            "param_builder_source": param_builder_source,
            "param_builder_sources": [param_builder_source] if param_builder_source else [],
            "handler_mcp_tools_used": list(handler_mcp_tools_used or []),
            "logical_skill": logical_skill,
            "operation": operation,
            "expected_mcp_tools": list(expected_mcp_tools or []),
        }

    async def _run_simple_qa(
        self,
        query: str,
        *,
        chat_history: str,
        user_profile: str,
    ) -> FinalResponse:
        retrieval = self._rag.retrieve(query, fast_mode=True)
        context = retrieval.get("context", "")

        if settings.PROMPT_BUDGET_ENABLED:
            mgr = PromptBudgetManager()
            sections = [
                PromptSection(
                    "instruction",
                    "你是天文助手。请基于给定知识，用简洁直接的中文回答。\n"
                    "如果知识不足，要明确说明不确定，不要编造。",
                    priority=100,
                    required=True,
                ),
                PromptSection(
                    "user_profile",
                    user_profile,
                    priority=70,
                    max_chars=800,
                ),
                PromptSection(
                    "chat_history",
                    chat_history,
                    priority=60,
                    max_chars=1200,
                ),
                PromptSection(
                    "rag_context",
                    context,
                    priority=50,
                    max_chars=3000,
                ),
                PromptSection(
                    "query",
                    f"问题：{query}\n\n回答：",
                    priority=100,
                    required=True,
                ),
            ]
            result = mgr.fit_sections(sections)
            prompt = result.text
        else:
            prompt = (
                "你是天文助手。请基于给定知识，用简洁直接的中文回答。\n"
                "如果知识不足，要明确说明不确定，不要编造。\n\n"
                f"用户画像：\n{user_profile[:400]}\n\n"
                f"最近对话：\n{chat_history[:800]}\n\n"
                f"知识：\n{context[:2400]}\n\n"
                f"问题：{query}\n\n回答："
            )

        answer = await asyncio.to_thread(self._invoke_llm, prompt)
        return self._synthesizer.synthesize_qa(
            query=query,
            answer=answer,
            rag_context=context,
            retrieval=retrieval,
        )

    def _attach_execution_events(
        self,
        response: FinalResponse,
        *,
        tool_name: Optional[str] = None,
        tool_input: Optional[Dict[str, Any]] = None,
        tool_summary: str = "",
        tool_status: str = "success",
    ) -> None:
        events = []
        if tool_name:
            events.append(
                ExecutionEvent(
                    type="tool_called",
                    payload={
                        "tool": tool_name,
                        "input": dict(tool_input or {}),
                        "status": "running",
                    },
                    source="direct",
                ).to_dict()
            )
            events.append(
                ExecutionEvent(
                    type="tool_result",
                    payload={
                        "tool": tool_name,
                        "output_summary": tool_summary,
                        "status": tool_status,
                    },
                    source="direct",
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
                source="direct",
            ).to_dict()
        )
        response.execution_events = events

    def _invoke_llm(self, prompt: str) -> str:
        result = self._llm.invoke(prompt)
        return getattr(result, "content", None) or str(result)

    def _smalltalk_reply(self, query: str) -> str:
        normalized = (query or "").strip().lower()
        if "谢谢" in query or "thanks" in normalized:
            return "不客气。如果你想查天象、天气或观测计划，直接告诉我时间和地点即可。"
        if "在吗" in query:
            return "在。可以直接问我天文知识、今晚观测目标、天气或观测计划。"
        return "你好，我可以帮你查询天象、观测条件、天体位置和天文知识。"

    def _direct_no_tool_reply(self, query: str) -> str:
        prompt = (
            "你是天文助手。请不用任何外部工具，直接回答这个稳定知识或经验判断问题。"
            "如果问题缺少实时数据，要明确说明只能给一般性判断。\n\n"
            f"问题：{query}\n\n回答："
        )
        return self._invoke_llm(prompt)

    def _build_skill_params(self, skill_name: str, query: str) -> Dict[str, Any]:
        return self._param_builder.build(skill_name, query)
