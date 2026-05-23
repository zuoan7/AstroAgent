"""统一请求上下文模型，保存查询、短期历史、用户画像和请求级元信息。

该模型用于替代 StreamingService 和执行入口中散落传递的同名参数，
legacy 构造方法仅保留给历史兼容入口。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RequestContext:
    """请求上下文，保存用户查询、会话历史、用户画像和请求标识。"""
    query: str
    chat_history: str = ""
    user_profile: str = ""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    use_long_term_memory: bool = True

    def to_dict(self) -> dict:
        """将当前模型转换为可序列化字典。"""
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

        主路径已直接构造 RequestContext；本方法只服务于兼容桥接和测试。
        """
        return cls(
            query=query,
            chat_history=chat_history,
            user_profile=user_profile,
            request_id=request_id or uuid.uuid4().hex[:8],
            use_long_term_memory=use_long_term_memory,
        )
