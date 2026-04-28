"""DirectExecutor — 直接任务执行器（Phase 4 引入）。

抽取自 TaskOrchestrator._run_direct_task()，逻辑完全等价。
Phase 9 起作为主路径，循环依赖已消除（改用 SkillParamBuilder）。
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from src.agent.models.execution_event import ExecutionEvent
from src.agent.models.final_response import FinalResponse
from src.agent.request_router import RouteDecision
from src.agent.skill_param_builder import SkillParamBuilder


class DirectExecutor:
    """封装 direct_task 三条子路径：smalltalk / single_tool_lookup / simple_qa。"""

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
            self._attach_execution_events(response)
            return response

        if decision.task_type == "single_tool_lookup":
            return await self._run_tool_task(decision, query)

        if decision.task_type == "simple_qa":
            response = await self._run_simple_qa(
                query, chat_history=chat_history, user_profile=user_profile
            )
            self._attach_execution_events(response)
            return response

        raise ValueError(f"unsupported direct task type: {decision.task_type}")

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
        self._attach_execution_events(
            response,
            tool_name=skill_name,
            tool_input=params,
            tool_summary=result.summary,
            tool_status="success" if result.success else "error",
        )
        return response

    async def _run_simple_qa(
        self,
        query: str,
        *,
        chat_history: str,
        user_profile: str,
    ) -> FinalResponse:
        retrieval = self._rag.retrieve(query, fast_mode=True)
        context = retrieval.get("context", "")
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

    def _build_skill_params(self, skill_name: str, query: str) -> Dict[str, Any]:
        return self._param_builder.build(skill_name, query)
