from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from src.agent.param_parser import ParamParser
from src.agent.request_router import RouteDecision


class TaskOrchestrator:
    """Direct execution path for low-latency online requests."""

    def __init__(self, skill_manager: Any, rag_retriever: Any, llm: Any) -> None:
        self._skill_manager = skill_manager
        self._rag = rag_retriever
        self._llm = llm

    async def run(
        self,
        decision: RouteDecision,
        query: str,
        *,
        chat_history: str,
        user_profile: str,
    ) -> Dict[str, Any]:
        if decision.route == "smalltalk":
            return {
                "answer": self._smalltalk_reply(query),
                "route": decision.route,
                "sources": [],
                "tools_used": [],
                "memory_hits": [],
            }

        if decision.route == "tool_task":
            return await self._run_tool_task(decision, query)

        if decision.route == "simple_qa":
            return await self._run_simple_qa(query, chat_history=chat_history, user_profile=user_profile)

        raise ValueError(f"unsupported direct route: {decision.route}")

    async def _run_simple_qa(
        self,
        query: str,
        *,
        chat_history: str,
        user_profile: str,
    ) -> Dict[str, Any]:
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
        return {
            "answer": answer,
            "route": "simple_qa",
            "sources": [
                {
                    "source_id": "rag_fast_path",
                    "kind": "rag_context",
                    "title": "RAG Fast Path",
                    "snippet": context[:240],
                }
            ] if context else [],
            "tools_used": [
                {
                    "run_id": "rag_fast_path",
                    "tool": "RAGRetrieve",
                    "input": query,
                    "output_summary": context[:240],
                    "status": "success" if context else "empty",
                }
            ],
            "retrieval": retrieval,
        }

    async def _run_tool_task(self, decision: RouteDecision, query: str) -> Dict[str, Any]:
        skill_name = decision.matched_skills[0]
        params = self._build_skill_params(skill_name, query)
        result = await asyncio.to_thread(
            self._skill_manager.call_skill,
            skill_name,
            **params,
        )
        return {
            "answer": result,
            "route": "tool_task",
            "sources": [
                {
                    "source_id": skill_name,
                    "kind": "tool_output",
                    "title": skill_name,
                    "snippet": str(result)[:240],
                    "tool": skill_name,
                }
            ],
            "tools_used": [
                {
                    "run_id": skill_name,
                    "tool": skill_name,
                    "input": params,
                    "output_summary": str(result)[:240],
                    "status": "success",
                }
            ],
            "matched_skill": skill_name,
            "params": params,
        }

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
        parsed = ParamParser.parse(query)
        if isinstance(parsed, dict) and parsed:
            return parsed

        if skill_name == "weather-lookup":
            return {"city": query.strip()}
        if skill_name == "observation-planner":
            return {"location": query.strip()}
        if skill_name == "deep-sky-observing-guide":
            return {"target": query.strip()}
        if skill_name == "celestial-position-calculator":
            return {"target": query.strip()}
        return {"query": query.strip()}
