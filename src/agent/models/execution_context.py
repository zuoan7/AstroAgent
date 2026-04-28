"""
ExecutionContext — 统一执行上下文，Phase 2 引入。

= TaskProfile + RequestContext，作为未来执行引擎的完整输入描述。

当前状态：ExecutionContext 已进入 Router/Policy/ExecutionEngine 主链路；
          from_legacy_params() 仅保留为旧接口桥接与测试辅助。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.agent.models.task_profile import TaskProfile
from src.agent.models.request_context import RequestContext


@dataclass
class ExecutionContext:
    profile: TaskProfile
    request: RequestContext

    # 可选扩展字段（Phase 3 前保持 Optional）
    extra_meta: Dict[str, Any] = field(default_factory=dict)

    # ── 便捷属性，减少调用层的字段访问深度 ─────────────────────────────────
    @property
    def query(self) -> str:
        return self.request.query

    @property
    def chat_history(self) -> str:
        return self.request.chat_history

    @property
    def user_profile(self) -> str:
        return self.request.user_profile

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def task_type(self) -> str:
        return self.profile.task_type

    @property
    def legacy_route(self) -> str:
        return self.profile.legacy_route

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "request": self.request.to_dict(),
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
