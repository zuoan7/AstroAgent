from typing import Sequence

from src.core.logger import logger
from src.memory.core.models import Message, SessionMemoryState
from src.memory.short_term_memory.config import ShortTermMemoryConfig, get_memory_settings
from src.memory.short_term_memory.context_builder import ROLE_LABELS


SUMMARY_SYSTEM_PROMPT = """你是一个对话摘要专家。请将以下天文助手与用户的对话历史压缩为简洁的摘要。

要求：
1. 保留关键天文信息（天体名称、观测条件、日期时间、位置等）
2. 保留用户的目标和意图
3. 保留已确认的约束条件
4. 保留中间结论和重要发现
5. 去除重复和冗余信息
6. 摘要应简洁但信息完整

请直接输出摘要内容，不要添加额外说明："""


SUMMARY_USER_TEMPLATE = """请摘要以下对话历史：

{conversation_text}

摘要："""


class ConversationSummarizer:
    def __init__(self, config: ShortTermMemoryConfig, token_counter):
        self.config = config
        self._estimate_tokens = token_counter

    def should_summarize(self, state: SessionMemoryState, messages: Sequence[Message]) -> bool:
        if not self.config.enable_summary:
            return False
        if len(messages) <= self.config.summary_keep_last_n:
            return False
        total_tokens = self._estimate_tokens(state.summary) + sum(self._estimate_tokens(msg.content) for msg in messages)
        return (
            len(messages) >= self.config.summary_trigger_messages
            or total_tokens >= self.config.summary_trigger_tokens
        )

    def summarize(self, messages: Sequence[Message]) -> str:
        text = "\n".join(f"{ROLE_LABELS.get(msg.role, msg.role)}: {msg.content}" for msg in messages)
        if not text.strip():
            return ""
        settings = get_memory_settings()
        if getattr(settings, "DASHSCOPE_API_KEY", None):
            try:
                return self._summarize_with_llm(text)
            except Exception as exc:
                logger.warning(f"LLM摘要生成失败，回退到规则摘要: {exc}")
        return self.fallback_summarize(messages)

    def fallback_summarize(self, messages: Sequence[Message]) -> str:
        lines = []
        for msg in messages:
            prefix = ROLE_LABELS.get(msg.role, msg.role)
            snippet = msg.content.strip()
            if len(snippet) > 120:
                snippet = snippet[:120] + "..."
            lines.append(f"{prefix}: {snippet}")
        summary = "\n".join(lines)
        max_chars = self.config.summary_max_tokens * 2
        return summary[:max_chars]

    def merge_summary(self, old: str, new: str) -> str:
        merged = f"{old}\n\n[new]\n{new}".strip() if old else new.strip()
        max_chars = self.config.summary_max_tokens * 3
        return merged[:max_chars]

    def _summarize_with_llm(self, text: str) -> str:
        from langchain_community.chat_models import ChatTongyi
        from langchain_core.messages import HumanMessage, SystemMessage

        settings = get_memory_settings()
        llm = ChatTongyi(
            model=settings.MODEL_NAME,
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
            temperature=0.0,
            request_timeout=15,
        )
        response = llm.invoke(
            [
                SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=SUMMARY_USER_TEMPLATE.format(conversation_text=text[:3000])),
            ]
        )
        return response.content.strip()
