"""统一执行上下文模型，聚合 TaskProfile、RequestContext 和能力选择结果。

该模型是 ExecutionEngine 和各执行器的完整输入描述。当前已进入
Router/Policy/ExecutionEngine 主链路，legacy 构造方法仅用于旧接口桥接和测试。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.agent.models.capability_decision import CapabilityDecision
from src.agent.models.task_profile import TaskProfile
from src.agent.models.request_context import RequestContext


@dataclass
class ExecutionContext:
    """统一执行上下文，作为 ExecutionEngine 和各执行器的主输入。"""
    profile: TaskProfile
    request: RequestContext
    capability_decision: Optional[CapabilityDecision] = None

    # 可选扩展字段（Phase 3 前保持 Optional）
    extra_meta: Dict[str, Any] = field(default_factory=dict)

    # ── 便捷属性，减少调用层的字段访问深度 ─────────────────────────────────
    @property
    def query(self) -> str:
        """返回当前请求的用户查询文本。"""
        return self.request.query

    @property
    def chat_history(self) -> str:
        """返回当前请求的短期对话历史。"""
        return self.request.chat_history

    @property
    def user_profile(self) -> str:
        """返回当前请求的长期用户画像上下文。"""
        return self.request.user_profile

    @property
    def request_id(self) -> str:
        """返回当前请求的唯一标识。"""
        return self.request.request_id

    @property
    def task_type(self) -> str:
        """返回任务画像中的任务类型。"""
        return self.profile.task_type

    @property
    def legacy_route(self) -> str:
        """返回兼容旧链路的 route 字符串。"""
        return self.profile.legacy_route

    def to_dict(self) -> Dict[str, Any]:
        """将当前模型转换为可序列化字典。"""
        return {
            "profile": self.profile.to_dict(),
            "request": self.request.to_dict(),
            "capability_decision": (
                self.capability_decision.to_dict()
                if self.capability_decision is not None
                else None
            ),
            "extra_meta": dict(self.extra_meta),
        }

    @classmethod
    def from_legacy_params(
        cls,
        *,
        route: str,
        task_type: str,
        confidence: float,
        query: str,
        chat_history: str = "",
        user_profile: str = "",
        request_id: Optional[str] = None,
        matched_skills: Optional[list] = None,
        capability_hints: Optional[list] = None,
        expected_output_schema: str = "generic_answer_v1",
        use_long_term_memory: bool = True,
    ) -> "ExecutionContext":
        """[Legacy adapter] 从旧式 (RouteDecision + 散落参数) 构造 ExecutionContext。

        主路径已优先直接构造 ExecutionContext；本方法仅保留给兼容桥接和测试。
        """
        profile = TaskProfile.from_legacy_route(
            route=route,
            task_type=task_type,
            confidence=confidence,
            matched_skills=matched_skills,
            capability_hints=capability_hints,
            expected_output_schema=expected_output_schema,
        )
        request = RequestContext.from_legacy_params(
            query=query,
            chat_history=chat_history,
            user_profile=user_profile,
            request_id=request_id,
            use_long_term_memory=use_long_term_memory,
        )
        return cls(profile=profile, request=request)

    @classmethod
    def from_legacy_decision(
        cls,
        decision: Any,
        query: str,
        *,
        chat_history: str = "",
        user_profile: str = "",
        request_id: Optional[str] = None,
        use_long_term_memory: bool = True,
        default_route: str = "fallback_react",
        default_task_type: str = "open_domain_reasoning",
        default_output_schema: str = "generic_answer_v1",
    ) -> "ExecutionContext":
        """[Legacy adapter] Convert RouteDecision-like objects to context.

        This is the only executor-side place that mirrors legacy
        `matched_skills` into the context model. New execution paths should
        construct ExecutionContext directly from TaskProfile and RequestContext.
        """
        profile = TaskProfile.from_legacy_route(
            route=getattr(decision, "route", default_route),
            task_type=getattr(decision, "task_type", default_task_type),
            confidence=getattr(decision, "confidence", 0.0),
            matched_skills=list(getattr(decision, "matched_skills", []) or []),
            capability_hints=list(getattr(decision, "capability_hints", []) or []),
            reason=getattr(decision, "reason", ""),
            expected_output_schema=getattr(
                decision, "expected_output_schema", default_output_schema
            ),
            router_source=getattr(decision, "router_source", "rule"),
            rule_confidence=getattr(decision, "rule_confidence", None),
            llm_confidence=getattr(decision, "llm_confidence", None),
            tool_necessity_action=getattr(decision, "tool_necessity_action", ""),
            tool_necessity_reason=getattr(decision, "tool_necessity_reason", ""),
            tool_necessity_confidence=getattr(
                decision, "tool_necessity_confidence", None
            ),
            answer_hint=getattr(decision, "answer_hint", ""),
            clarification_prompt=getattr(decision, "clarification_prompt", ""),
            tool_necessity_missing_params=list(
                getattr(decision, "tool_necessity_missing_params", []) or []
            ),
            tool_necessity_allowed_skill_hints=list(
                getattr(decision, "tool_necessity_allowed_skill_hints", []) or []
            ),
            tool_necessity_forbidden_skill_hints=list(
                getattr(decision, "tool_necessity_forbidden_skill_hints", []) or []
            ),
        )
        request = RequestContext.from_legacy_params(
            query=query,
            chat_history=chat_history,
            user_profile=user_profile,
            request_id=request_id,
            use_long_term_memory=use_long_term_memory,
        )
        return cls(profile=profile, request=request)
