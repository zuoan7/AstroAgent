from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agent.models.skill_result import SkillResult
from src.agent.models.final_response import FinalResponse
from src.agent.policies.budget_policy import RequestBudgetTracker
from src.core.logger import logger
from src.core.config import settings


class ResponseSynthesizer:
    """
    统一的答案合成器。

    职责：
    - 多技能结果合成为最终答案（调用 LLM）
    - 单技能/简单场景直接结构化包装（不调 LLM）
    - 统一计算 confidence、收集 sources、构建 tools_used
    """

    def __init__(
        self,
        llm: Any,
        *,
        budget_tracker: Optional[RequestBudgetTracker] = None,
        synth_prompt_version: Optional[str] = None,
    ) -> None:
        self._llm = llm
        self._budget_tracker = budget_tracker
        self._synth_prompt_version = synth_prompt_version or str(
            getattr(settings, "SYNTH_PROMPT_VERSION", "synth_prompt_v2")
        )

    def synthesize(
        self,
        query: str,
        task_type: str,
        output_schema: str,
        skill_results: List[SkillResult],
        *,
        chat_history: str = "",
        user_profile: str = "",
        route: str = "planned_task",
        execution_plan: Optional[Dict[str, Any]] = None,
        execution_trace: Optional[List[Dict[str, Any]]] = None,
        route_decision: Optional[Dict[str, Any]] = None,
        fallback_path: Optional[List[Dict[str, Any]]] = None,
        budget_usage: Optional[Dict[str, Any]] = None,
        versions: Optional[Dict[str, Any]] = None,
    ) -> FinalResponse:
        collected_outputs = []
        sources: List[Dict[str, Any]] = []
        tools_used: List[Dict[str, Any]] = []
        structured_payload: Dict[str, Any] = {}

        for sr in skill_results:
            collected_outputs.append(f"[{sr.skill_name}]\n{sr.summary}")
            tools_used.append(sr.to_tool_timeline_entry())
            sources.extend(sr.sources)
            if sr.success and sr.data:
                structured_payload[sr.skill_name] = sr.data

        prompt = (
            "你是天文助手。请基于已经执行完成的计划步骤，为用户输出最终答案。\n"
            "要求：\n"
            "1. 先直接回答，再给出关键建议。\n"
            "2. 明确说明哪些建议来自天气、天象、目标观测或摄影参数。\n"
            "3. 不要虚构未提供的数据。\n\n"
            f"任务类型：{task_type}\n"
            f"输出Schema：{output_schema}\n"
            f"用户画像：\n{user_profile[:400]}\n\n"
            f"最近对话：\n{chat_history[:600]}\n\n"
            f"用户问题：{query}\n\n"
            "已完成步骤结果：\n"
            f"{chr(10).join(collected_outputs)[:5000]}\n\n"
            "请给出整合后的中文回答："
        )

        answer = self._invoke_llm(prompt)
        confidence = self._compute_confidence(skill_results)

        return FinalResponse(
            answer=answer,
            summary=answer[:200] if len(answer) > 200 else answer,
            sources=sources,
            tools_used=tools_used,
            confidence=confidence,
            structured_payload=structured_payload if structured_payload else None,
            route=route,
            task_type=task_type,
            execution_plan=execution_plan,
            execution_trace=list(execution_trace or []),
            route_decision=route_decision,
            fallback_path=list(fallback_path or []),
            budget_usage=budget_usage,
            versions=versions or self._default_versions(),
        )

    def synthesize_direct(
        self,
        query: str,
        task_type: str,
        skill_results: List[SkillResult],
        *,
        raw_answer: Optional[str] = None,
        route: str = "direct_task",
    ) -> FinalResponse:
        sources: List[Dict[str, Any]] = []
        tools_used: List[Dict[str, Any]] = []
        structured_payload: Dict[str, Any] = {}

        for sr in skill_results:
            tools_used.append(sr.to_tool_timeline_entry())
            sources.extend(sr.sources)
            if sr.success and sr.data:
                structured_payload[sr.skill_name] = sr.data

        if raw_answer is not None:
            answer = raw_answer
        elif skill_results:
            answer = skill_results[0].summary
        else:
            answer = ""

        confidence = self._compute_confidence(skill_results)

        return FinalResponse(
            answer=answer,
            summary=answer[:200] if len(answer) > 200 else answer,
            sources=sources,
            tools_used=tools_used,
            confidence=confidence,
            structured_payload=structured_payload if structured_payload else None,
            route=route,
            task_type=task_type,
            versions=self._default_versions(),
        )

    def synthesize_qa(
        self,
        query: str,
        answer: str,
        *,
        rag_context: str = "",
        retrieval: Optional[Dict[str, Any]] = None,
        route: str = "direct_task",
    ) -> FinalResponse:
        sources = []
        tools_used = []
        if rag_context:
            sources.append({
                "source_id": "rag_fast_path",
                "kind": "rag_context",
                "title": "RAG Fast Path",
                "snippet": rag_context[:240],
            })
            tools_used.append({
                "run_id": "rag_fast_path",
                "tool": "RAGRetrieve",
                "input": query,
                "output_summary": rag_context[:240],
                "status": "success" if rag_context else "empty",
            })

        confidence = 0.6 if rag_context else 0.4

        return FinalResponse(
            answer=answer,
            summary=answer[:200] if len(answer) > 200 else answer,
            sources=sources,
            tools_used=tools_used,
            confidence=confidence,
            route=route,
            task_type="simple_qa",
            versions=self._default_versions(),
        )

    def synthesize_smalltalk(
        self,
        answer: str,
    ) -> FinalResponse:
        return FinalResponse(
            answer=answer,
            summary=answer,
            confidence=0.98,
            route="direct_task",
            task_type="smalltalk",
            versions=self._default_versions(),
        )

    def _invoke_llm(self, prompt: str) -> str:
        if self._budget_tracker:
            self._budget_tracker.register_context_chars(len(prompt))
            self._budget_tracker.register_llm_call()
        result = self._llm.invoke(prompt)
        return getattr(result, "content", None) or str(result)

    def _default_versions(self) -> Dict[str, Any]:
        return {
            "schema_version": str(getattr(settings, "SCHEMA_VERSION", "schema_v2")),
            "synth_prompt_version": self._synth_prompt_version,
        }

    @staticmethod
    def _compute_confidence(skill_results: List[SkillResult]) -> float:
        if not skill_results:
            return 0.35
        confidence = 0.45
        successful = [sr for sr in skill_results if sr.success]
        confidence += min(len(successful) * 0.1, 0.3)
        if successful and all(sr.success for sr in skill_results):
            confidence += 0.1
        total_sources = sum(len(sr.sources) for sr in skill_results)
        if total_sources > 0:
            confidence += 0.1
        return min(round(confidence, 2), 0.95)
