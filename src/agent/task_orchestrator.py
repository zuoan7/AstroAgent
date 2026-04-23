from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional

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
        if decision.route == "direct_task":
            return await self._run_direct_task(
                decision,
                query,
                chat_history=chat_history,
                user_profile=user_profile,
            )

        if decision.route == "planned_task":
            return await self._run_planned_task(
                decision,
                query,
                chat_history=chat_history,
                user_profile=user_profile,
            )

        raise ValueError(f"unsupported orchestrated route: {decision.route}")

    async def _run_direct_task(
        self,
        decision: RouteDecision,
        query: str,
        *,
        chat_history: str,
        user_profile: str,
    ) -> Dict[str, Any]:
        if decision.task_type == "smalltalk":
            return {
                "answer": self._smalltalk_reply(query),
                "route": decision.route,
                "task_type": decision.task_type,
                "sources": [],
                "tools_used": [],
                "memory_hits": [],
            }

        if decision.task_type == "single_tool_lookup":
            return await self._run_tool_task(decision, query)

        if decision.task_type == "simple_qa":
            return await self._run_simple_qa(
                query, chat_history=chat_history, user_profile=user_profile
            )

        raise ValueError(f"unsupported direct task type: {decision.task_type}")

    async def _run_planned_task(
        self,
        decision: RouteDecision,
        query: str,
        *,
        chat_history: str,
        user_profile: str,
    ) -> Dict[str, Any]:
        skills = decision.matched_skills or self._skills_for_task_type(decision.task_type)
        if not skills:
            raise ValueError(f"no planned-task skills resolved for {decision.task_type}")

        tool_timeline: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = []
        collected_outputs: List[str] = []

        for skill_name in skills:
            params = self._build_skill_params(skill_name, query)
            result = await asyncio.to_thread(
                self._skill_manager.call_skill,
                skill_name,
                **params,
            )
            result_text = str(result)
            collected_outputs.append(f"[{skill_name}]\n{result_text}")
            tool_timeline.append(
                {
                    "run_id": skill_name,
                    "tool": skill_name,
                    "input": params,
                    "output_summary": result_text[:240],
                    "status": "success",
                }
            )
            sources.append(
                {
                    "source_id": skill_name,
                    "kind": "tool_output",
                    "title": skill_name,
                    "snippet": result_text[:240],
                    "tool": skill_name,
                }
            )

        answer = await asyncio.to_thread(
            self._invoke_planned_synthesis,
            decision,
            query,
            collected_outputs,
            chat_history,
            user_profile,
        )
        return {
            "answer": answer,
            "route": decision.route,
            "task_type": decision.task_type,
            "sources": sources,
            "tools_used": tool_timeline,
            "plan": {
                "task_type": decision.task_type,
                "expected_output_schema": decision.expected_output_schema,
                "skills": skills,
            },
        }

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
            "route": "direct_task",
            "task_type": "simple_qa",
            "sources": [
                {
                    "source_id": "rag_fast_path",
                    "kind": "rag_context",
                    "title": "RAG Fast Path",
                    "snippet": context[:240],
                }
            ]
            if context
            else [],
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
            "route": "direct_task",
            "task_type": decision.task_type,
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
            return {
                "location": self._extract_location(query) or query.strip(),
                "date": self._extract_date(query),
            }
        if skill_name == "deep-sky-observing-guide":
            return {
                "target": self._extract_target(query) or query.strip(),
                "observer_location": self._extract_location(query),
                "date": self._extract_date(query),
                "equipment": self._extract_equipment(query),
            }
        if skill_name == "celestial-events-forecast":
            return {
                "event_type": self._extract_event_type(query),
                "start_date": self._extract_date(query),
                "end_date": None,
            }
        if skill_name == "astrophotography-calculator":
            return {
                "target": self._extract_target(query) or query.strip(),
                "camera": self._extract_camera(query) or "未指定相机",
                "location": self._extract_location(query),
                "date": self._extract_date(query),
            }
        if skill_name == "celestial-position-calculator":
            return {
                "target": self._extract_target(query) or query.strip(),
                "location": self._extract_location(query),
                "datetime": self._extract_datetime(query),
            }
        return {"query": query.strip()}

    def _skills_for_task_type(self, task_type: str) -> List[str]:
        mapping = {
            "observation_recommendation": [
                "weather-lookup",
                "observation-planner",
            ],
            "celestial_event_analysis": ["celestial-events-forecast"],
            "deep_sky_guidance": ["deep-sky-observing-guide"],
            "astrophotography_advice": [
                "astrophotography-calculator",
                "weather-lookup",
            ],
        }
        return list(mapping.get(task_type, []))

    def _invoke_planned_synthesis(
        self,
        decision: RouteDecision,
        query: str,
        collected_outputs: List[str],
        chat_history: str,
        user_profile: str,
    ) -> str:
        prompt = (
            "你是天文助手。请基于已经执行完成的计划步骤，为用户输出最终答案。\n"
            "要求：\n"
            "1. 先直接回答，再给出关键建议。\n"
            "2. 明确说明哪些建议来自天气、天象、目标观测或摄影参数。\n"
            "3. 不要虚构未提供的数据。\n\n"
            f"任务类型：{decision.task_type}\n"
            f"输出Schema：{decision.expected_output_schema}\n"
            f"用户画像：\n{user_profile[:400]}\n\n"
            f"最近对话：\n{chat_history[:600]}\n\n"
            f"用户问题：{query}\n\n"
            "已完成步骤结果：\n"
            f"{chr(10).join(collected_outputs)[:5000]}\n\n"
            "请给出整合后的中文回答："
        )
        return self._invoke_llm(prompt)

    def _extract_location(self, query: str) -> Optional[str]:
        for city in ("北京", "上海", "广州", "深圳", "苏州", "杭州", "成都", "南京", "武汉"):
            if city in query:
                return city
        return None

    def _extract_target(self, query: str) -> Optional[str]:
        catalog_match = re.search(r"\b(M\d{1,3}|NGC\s?\d{1,4})\b", query, re.IGNORECASE)
        if catalog_match:
            return catalog_match.group(1).upper().replace(" ", "")
        for target in ("木星", "土星", "火星", "金星", "月球", "太阳", "M31", "M42", "猎户座大星云"):
            if target in query:
                return target
        return None

    def _extract_date(self, query: str) -> Optional[str]:
        for token in ("今天", "明天", "今晚", "明晚", "本周末", "下周一"):
            if token in query:
                return token
        return None

    def _extract_datetime(self, query: str) -> Optional[str]:
        match = re.search(r"\d{4}-\d{2}-\d{2}(?:\s+\d{1,2}:\d{2})?", query)
        if match:
            return match.group(0)
        return self._extract_date(query)

    def _extract_equipment(self, query: str) -> Optional[str]:
        for equipment in ("双筒", "双筒望远镜", "小折射镜", "8寸望远镜", "赤道仪", "三脚架"):
            if equipment in query:
                return equipment
        return None

    def _extract_camera(self, query: str) -> Optional[str]:
        for camera in ("Sony", "Canon", "Nikon", "ZWO", "QHY", "相机"):
            if camera in query:
                return camera
        return None

    def _extract_event_type(self, query: str) -> Optional[str]:
        for event_type in ("流星雨", "月食", "日食", "行星合月", "冲日", "掩星"):
            if event_type in query:
                return event_type
        return None
