"""Direct 执行器，处理闲聊、简单问答、无需工具回答和单工具查询等低复杂度路径。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

from src.agent.executor import _extract_mcp_tools_from_sources
from src.agent.fast_answers import stable_knowledge_answer
from src.agent.models.execution_event import ExecutionEvent
from src.agent.models.final_response import FinalResponse
from src.agent.prompts import get_prompt_renderer
from src.agent.skill_param_builder import SkillParamBuilder
from src.capabilities.param_builder import CapabilityParamBuilder
from src.core.config import settings
from src.core.mcp_protocol import is_tool_error, parse_tool_response


class DirectExecutor:
    """封装 direct_task 子路径。"""

    def __init__(
        self,
        skill_manager: Any,
        rag_retriever: Any,
        llm: Any,
        synthesizer: Any,
    ) -> None:
        """初始化 DirectExecutor 的依赖、配置和内部状态。"""
        self._skill_manager = skill_manager
        self._rag = rag_retriever
        self._llm = llm
        self._synthesizer = synthesizer
        self._param_builder = SkillParamBuilder(skill_manager)

    async def run_context(
        self,
        context: Any,
    ) -> FinalResponse:
        """使用统一 ExecutionContext 执行当前路径。"""
        profile = context.profile
        query = context.query
        chat_history = context.chat_history
        user_profile = context.user_profile
        task_type = getattr(profile, "task_type", "")

        if task_type == "smalltalk":
            response = self._synthesizer.synthesize_smalltalk(
                self._smalltalk_reply(query)
            )
            self._attach_response_metadata(response, profile=profile)
            self._attach_execution_events(response)
            return response

        if task_type == "clarification":
            response = self._run_clarification(profile)
            self._attach_response_metadata(response, profile=profile)
            self._attach_execution_events(response)
            return response

        if task_type == "direct_answer_no_tool":
            response = self._run_no_tool_answer(profile, query)
            self._attach_response_metadata(response, profile=profile)
            self._attach_execution_events(response)
            return response

        if task_type == "single_tool_lookup":
            return await self._run_tool_task(
                chat_history=chat_history,
                user_profile=user_profile,
                context=context,
            )

        if task_type == "simple_qa":
            response = await self._run_simple_qa(
                query, chat_history=chat_history, user_profile=user_profile
            )
            self._attach_response_metadata(response, profile=profile)
            self._attach_execution_events(response)
            return response

        raise ValueError(f"unsupported direct task type: {task_type}")

    def _run_clarification(
        self,
        profile: Any,
    ) -> FinalResponse:
        """生成缺少关键信息时的澄清回答。"""
        answer = (
            getattr(profile, "clarification_prompt", "")
            or "这个请求还缺少关键信息。请补充目标、时间、地点或器材参数后我再继续。"
        )
        return FinalResponse(
            answer=answer,
            summary=answer[:200] if len(answer) > 200 else answer,
            tools_used=[],
            sources=[],
            confidence=getattr(profile, "confidence", 0.0),
            route=getattr(profile, "legacy_route", ""),
            task_type=getattr(profile, "task_type", ""),
            versions=(
                self._synthesizer._default_versions()
                if hasattr(self._synthesizer, "_default_versions")
                else None
            ),
        )

    def _run_no_tool_answer(
        self,
        profile: Any,
        query: str,
    ) -> FinalResponse:
        """生成不需要工具的直接回答。"""
        answer = (
            getattr(profile, "answer_hint", "")
            or stable_knowledge_answer(query)
            or self._direct_no_tool_reply(query)
        )
        return FinalResponse(
            answer=answer,
            summary=answer[:200] if len(answer) > 200 else answer,
            tools_used=[],
            sources=[],
            confidence=getattr(profile, "confidence", 0.0),
            route=getattr(profile, "legacy_route", ""),
            task_type=getattr(profile, "task_type", ""),
            versions=(
                self._synthesizer._default_versions()
                if hasattr(self._synthesizer, "_default_versions")
                else None
            ),
        )

    async def _run_tool_task(
        self,
        *,
        chat_history: str = "",
        user_profile: str = "",
        context: Any,
    ) -> FinalResponse:
        """执行 direct 单工具或单技能查询。"""
        from src.agent.models.skill_result import SkillResult

        profile = context.profile
        query = context.query
        capability = getattr(context, "capability_decision", None)
        capability_kind = getattr(capability, "kind", "") if capability else ""
        capability_name = getattr(capability, "name", "") if capability else ""
        capability_reason = getattr(capability, "reason", "") if capability else ""

        if capability_kind == "tool" and capability_name:
            params = self._build_atomic_tool_params(capability, query)
            result = await asyncio.to_thread(
                self._call_atomic_tool_as_skill_result,
                capability_name,
                params,
            )
            tool_name = capability_name
        else:
            fallback_hints = list(getattr(profile, "capability_hints", []) or [])
            skill_name = (
                capability_name
                if capability_kind == "skill" and capability_name
                else (fallback_hints[0] if fallback_hints else "")
            )
            if not skill_name:
                raise ValueError("no direct task capability resolved")
            if not capability_kind:
                capability_kind = "skill"
            if not capability_name:
                capability_name = skill_name
            if not capability_reason:
                capability_reason = "route_capability_hint"
            params = self._build_skill_params(
                skill_name,
                query,
                chat_history=chat_history,
                user_profile=user_profile,
            )
            result: SkillResult = await asyncio.to_thread(
                self._skill_manager.call_skill,
                skill_name,
                **params,
            )
            tool_name = skill_name

        mcp_tools_used = _extract_mcp_tools_from_sources(result.sources)
        response = self._synthesizer.synthesize_direct(
            query=query,
            task_type=getattr(profile, "task_type", ""),
            skill_results=[result],
        )
        self._attach_response_metadata(
            response,
            profile=profile,
            param_builder_source="fallback_builder",
            handler_mcp_tools_used=mcp_tools_used,
            logical_skill=getattr(result, "logical_skill", None) or tool_name,
            operation=getattr(result, "operation", None),
            expected_mcp_tools=getattr(result, "expected_mcp_tools", []),
            capability_kind=capability_kind,
            capability_name=capability_name,
            capability_reason=capability_reason,
        )
        self._attach_execution_events(
            response,
            tool_name=tool_name,
            tool_input=params,
            tool_summary=result.summary,
            tool_status="success" if result.success else "error",
            logical_skill=getattr(result, "logical_skill", None) or tool_name,
            operation=getattr(result, "operation", None),
            mcp_tools_used=mcp_tools_used,
            expected_mcp_tools=getattr(result, "expected_mcp_tools", []),
            capability_kind=capability_kind,
            capability_name=capability_name,
            capability_reason=capability_reason,
        )
        return response

    def _attach_response_metadata(
        self,
        response: FinalResponse,
        *,
        profile: Any,
        param_builder_source: str = "",
        handler_mcp_tools_used: Optional[list[str]] = None,
        logical_skill: Optional[str] = None,
        operation: Optional[str] = None,
        expected_mcp_tools: Optional[list[str]] = None,
        capability_kind: str = "",
        capability_name: str = "",
        capability_reason: str = "",
    ) -> None:
        """把路由、工具和能力选择审计元信息挂到响应。"""
        route_meta = profile.to_legacy_route_meta()
        response.route = str(getattr(profile, "legacy_route", ""))
        response.task_type = str(getattr(profile, "task_type", ""))
        response.route_decision = route_meta
        response.audit_metadata = {
            "router_source": route_meta.get("router_source"),
            "rule_confidence": route_meta.get("rule_confidence"),
            "llm_confidence": route_meta.get("llm_confidence"),
            "tool_necessity_action": route_meta.get("tool_necessity_action"),
            "tool_necessity_reason": route_meta.get("tool_necessity_reason"),
            "tool_necessity_confidence": route_meta.get("tool_necessity_confidence"),
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
            "param_builder_sources": (
                [param_builder_source] if param_builder_source else []
            ),
            "handler_mcp_tools_used": list(handler_mcp_tools_used or []),
            "logical_skill": logical_skill,
            "operation": operation,
            "expected_mcp_tools": list(expected_mcp_tools or []),
            "capability_kind": capability_kind,
            "capability_name": capability_name,
            "capability_reason": capability_reason,
        }

    async def _run_simple_qa(
        self,
        query: str,
        *,
        chat_history: str,
        user_profile: str,
    ) -> FinalResponse:
        """执行简单问答快路径，必要时使用 RAG 和 LLM。"""
        fast_answer = stable_knowledge_answer(query)
        if fast_answer:
            return self._synthesizer.synthesize_qa(
                query=query,
                answer=fast_answer,
                rag_context="",
                retrieval={"source": "stable_knowledge_fast_answer"},
            )

        retrieval = self._rag.retrieve(query, fast_mode=True)
        context = retrieval.get("context", "")

        prompt = get_prompt_renderer().render_sections(
            "direct.simple_qa",
            {
                "query": query,
                "user_profile": user_profile,
                "chat_history": chat_history,
                "rag_context": context,
            },
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
        logical_skill: Optional[str] = None,
        operation: Optional[str] = None,
        mcp_tools_used: Optional[list[str]] = None,
        expected_mcp_tools: Optional[list[str]] = None,
        capability_kind: str = "",
        capability_name: str = "",
        capability_reason: str = "",
    ) -> None:
        """为 direct 响应构造工具调用和答案就绪事件。"""
        events = []
        if tool_name:
            events.append(
                ExecutionEvent(
                    type="tool_called",
                    payload={
                        "tool": tool_name,
                        "display_tool": tool_name,
                        "logical_skill": logical_skill or tool_name,
                        "input": dict(tool_input or {}),
                        "status": "running",
                        "operation": operation,
                        "mcp_tools_used": list(mcp_tools_used or []),
                        "expected_mcp_tools": list(expected_mcp_tools or []),
                        "capability_kind": capability_kind,
                        "capability_name": capability_name,
                        "capability_reason": capability_reason,
                    },
                    source="direct",
                ).to_dict()
            )
            events.append(
                ExecutionEvent(
                    type="tool_result",
                    payload={
                        "tool": tool_name,
                        "display_tool": tool_name,
                        "logical_skill": logical_skill or tool_name,
                        "output_summary": tool_summary,
                        "status": tool_status,
                        "operation": operation,
                        "mcp_tools_used": list(mcp_tools_used or []),
                        "expected_mcp_tools": list(expected_mcp_tools or []),
                        "capability_kind": capability_kind,
                        "capability_name": capability_name,
                        "capability_reason": capability_reason,
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
        """同步调用底层 LLM 并抽取文本内容。"""
        result = self._llm.invoke(prompt)
        return getattr(result, "content", None) or str(result)

    def _smalltalk_reply(self, query: str) -> str:
        """生成固定闲聊回复。"""
        normalized = (query or "").strip().lower()
        if "谢谢" in query or "thanks" in normalized:
            return "不客气。如果你想查天象、天气或观测计划，直接告诉我时间和地点即可。"
        if "在吗" in query:
            return "在。可以直接问我天文知识、今晚观测目标、天气或观测计划。"
        return "你好，我可以帮你查询天象、观测条件、天体位置和天文知识。"

    def _direct_no_tool_reply(self, query: str) -> str:
        """调用提示词生成无需工具的直接回答。"""
        prompt = get_prompt_renderer().render(
            "direct.no_tool_answer",
            {"query": query},
        )
        return self._invoke_llm(prompt)

    def _build_skill_params(
        self,
        skill_name: str,
        query: str,
        *,
        chat_history: str = "",
        user_profile: str = "",
    ) -> Dict[str, Any]:
        """根据技能名和上下文构造调用参数。"""
        try:
            return self._param_builder.build(
                skill_name,
                query,
                chat_history=chat_history,
                user_profile=user_profile,
            )
        except KeyError:
            return CapabilityParamBuilder.build_atomic_tool_params(skill_name, query)

    @staticmethod
    def _build_atomic_tool_params(capability: Any, query: str) -> Dict[str, Any]:
        """根据能力选择信息构造原子工具参数。"""
        metadata = getattr(capability, "metadata", {}) or {}
        explicit_params = metadata.get("params")
        tool_name = getattr(capability, "name", "")
        return CapabilityParamBuilder.build_atomic_tool_params(
            tool_name,
            query,
            explicit_params=(
                explicit_params if isinstance(explicit_params, dict) else None
            ),
        )

    def _call_atomic_tool_as_skill_result(
        self,
        tool_name: str,
        params: Dict[str, Any],
    ) -> "SkillResult":
        """调用原子 MCP 工具并包装为 SkillResult。"""
        from src.agent.models.skill_result import SkillResult

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
