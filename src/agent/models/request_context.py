"""
RequestContext — 统一请求上下文，Phase 2 引入。

打包 query / chat_history / user_profile / request_id，
目标替代散落在 TaskOrchestrator.run() 和 StreamingService 中的同名参数。

当前状态：模型已稳定，但主执行路径仍通过散参调用 TaskOrchestrator.run()。
收敛计划：待 UnifiedExecutionEngine 实现后升级为主执行入口的统一输入。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RequestContext:
    query: str
    chat_history: str = ""
    user_profile: str = ""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    use_long_term_memory: bool = True

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "chat_history": self.chat_history,
            "user_profile": self.user_profile,
            "request_id": self.request_id,
            "use_long_term_memory": self.use_long_term_memory,
        }

    @classmethod
    def from_legacy_params(
        cls,
        query: str,
        *,
        chat_history: str = "",
        user_profile: str = "",
        request_id: Optional[str] = None,
        use_long_term_memory: bool = True,
    ) -> "RequestContext":
        """[Legacy adapter] 从旧式散落参数构造 RequestContext。

        当前主路径（StreamingService → TaskOrchestrator）仍通过散参传递，
        本方法仅被 ExecutionContext.from_legacy_params() 内部调用作为兼容桥接。
        待 UnifiedExecutionEngine 实现后，主入口将直接构造 RequestContext，本方法降为可选。
        """
        return cls(
            query=query,
            chat_history=chat_history,
            user_profile=user_profile,
            request_id=request_id or uuid.uuid4().hex[:8],
            use_long_term_memory=use_long_term_memory,
        )
