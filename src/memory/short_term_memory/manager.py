import json
import time
from typing import Any, Dict, List, Optional, Sequence

from src.core.mcp_protocol import parse_tool_response
from src.memory.core.models import Message, SalientFact, SessionMemoryState, ToolCallRecord
from src.memory.short_term_memory.config import ShortTermMemoryConfig
from src.memory.short_term_memory.context_builder import ContextBuilder
from src.memory.short_term_memory.repository import ShortTermMemoryRepository
from src.memory.short_term_memory.summarizer import ConversationSummarizer


class ShortTermMemory:
    def __init__(self, session_id: Optional[str] = None, user_id: Optional[str] = None):
        from src.memory import memory as memory_module

        self._settings = memory_module.settings
        self.config = ShortTermMemoryConfig.from_settings(self._settings)
        self.session_id = session_id or f"session_{int(time.time())}"
        self.user_id = user_id or self._settings.DEFAULT_USER_ID
        self.messages: List[Message] = []
        self.tool_calls: List[ToolCallRecord] = []
        self.salient_facts: List[SalientFact] = []
        self.state = SessionMemoryState(session_id=self.session_id)
        self.last_trimmed_content: List[Dict[str, Any]] = []
        self._restored_from_db = False
        self._repository: Optional[ShortTermMemoryRepository] = None
        self._summarizer = ConversationSummarizer(self.config, self._estimate_tokens)
        self._context_builder = ContextBuilder(self.config, self._estimate_tokens)

        if self.config.enable_persistence:
            self._repository = ShortTermMemoryRepository(self.config.persistence_path)
            self._repository.initialize()
            self.load_session(self.session_id)
            self._ensure_session_state()

    @property
    def summary(self) -> str:
        return self.state.summary

    @summary.setter
    def summary(self, value: str):
        self.state.summary = value or ""

    @property
    def trimmed_count(self) -> int:
        return self.state.trimmed_count

    @trimmed_count.setter
    def trimmed_count(self, value: int):
        self.state.trimmed_count = value

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.25)

    def _ensure_session_state(self):
        if not self._repository:
            return
        self.state.updated_at = time.time()
        self._repository.save_session_state(self.state, self.user_id)

    def load_session(self, session_id: Optional[str] = None) -> bool:
        if session_id:
            self.session_id = session_id
            self.state.session_id = session_id
        if self._restored_from_db or not self._repository:
            return self._restored_from_db
        payload = self._repository.load_session(self.session_id)
        if not payload:
            return False
        self.user_id = payload["user_id"]
        self.state = payload["state"]
        self.messages = payload["messages"]
        self.tool_calls = payload["tool_calls"]
        self.salient_facts = payload["salient_facts"]
        self._restored_from_db = True
        return True

    def _collect_importance_reasons(self, role: str, content: str, message_type: str = "chat") -> List[str]:
        reasons: List[str] = []
        if role in self.config.high_importance_roles:
            reasons.append("high_priority_role")
        if any(keyword in content for keyword in ["目标", "任务", "要求", "约束", "必须", "不要", "计划", "确认"]):
            reasons.append("contains_requirement")
        if role == "user" and ("？" in content or "?" in content):
            reasons.append("user_question")
        if message_type == "tool" and any(token in content.lower() for token in ["error", "错误", "failed"]):
            reasons.append("tool_error")
        return reasons

    def _calculate_importance(self, role: str, content: str, message_type: str = "chat") -> int:
        reasons = self._collect_importance_reasons(role, content, message_type)
        return min(max(len(reasons), 0), 3)

    def add_message(
        self,
        role: str,
        content: str,
        timestamp: Optional[float] = None,
        importance: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        timestamp = timestamp or time.time()
        message_type = (metadata or {}).get("message_type", "chat")
        reasons = self._collect_importance_reasons(role, content, message_type)
        msg = Message(
            session_id=self.session_id,
            role=role,
            content=content,
            timestamp=timestamp,
            importance=importance if importance is not None else min(len(reasons), 3),
            importance_reason=",".join(reasons),
            message_type=message_type,
            metadata=metadata or {},
        )
        self.messages.append(msg)
        self._persist_message(msg)
        self._trim_messages_if_needed()
        self._maybe_summarize()
        self._ensure_session_state()

    def add_tool_call(
        self,
        tool_name: str,
        tool_input: str,
        result: str,
        timestamp: Optional[float] = None,
        success: bool = True,
    ):
        timestamp = timestamp or time.time()
        record = ToolCallRecord(
            tool_name=tool_name,
            timestamp=timestamp,
            input_summary=(tool_input or "")[:200],
            **self._summarize_tool_result(result),
            status="success" if success else "error",
            importance=1 if success else 3,
        )
        self.tool_calls.append(record)
        self.tool_calls = self.tool_calls[-self.config.max_tool_records :]
        if self._repository:
            self._repository.append_tool_call(self.session_id, record)

    def add_salient_fact(
        self,
        fact_type: str,
        content: str,
        source: str = "",
        timestamp: Optional[float] = None,
    ):
        timestamp = timestamp or time.time()
        signature = (fact_type, content)
        if any((item.fact_type, item.content) == signature for item in self.salient_facts):
            return
        fact = SalientFact(
            fact_type=fact_type,
            content=content,
            timestamp=timestamp,
            source=source,
            source_type=source or "short_term_memory",
            source_id=self.session_id,
        )
        self.salient_facts.append(fact)
        self.salient_facts = sorted(
            self.salient_facts,
            key=lambda item: (item.timestamp, item.fact_type, item.content),
        )[-self.config.max_salient_facts :]
        if self._repository:
            self._repository.replace_salient_facts(self.session_id, self.salient_facts)

    def _persist_message(self, message: Message):
        if self._repository:
            self._repository.append_message(self.session_id, message)

    def _select_messages_to_trim(self) -> Sequence[Message]:
        overflow = len(self.messages) - self.config.max_size
        if overflow <= 0:
            return []
        return self.messages[:overflow]

    def _promote_to_salient_fact(self, messages: Sequence[Message]):
        for message in messages:
            if message.importance >= 2:
                self.add_salient_fact(
                    fact_type="important_message",
                    content=f"[{message.role}] {message.content[:200]}",
                    source="context_trimming",
                    timestamp=message.timestamp,
                )

    def _trim_messages_if_needed(self):
        trimmed = list(self._select_messages_to_trim())
        if not trimmed:
            return
        self.last_trimmed_content = [item.to_dict() for item in trimmed]
        self.trimmed_count += len(trimmed)
        self.messages = self.messages[len(trimmed) :]
        self._promote_to_salient_fact(trimmed)
        if self._repository:
            self._repository.replace_messages(self.session_id, self.messages)

    def _should_trigger_summary(self) -> bool:
        return self._summarizer.should_summarize(self.state, self.messages)

    def _maybe_summarize(self):
        if not self._should_trigger_summary():
            return
        keep_n = self.config.summary_keep_last_n
        history = self.messages[:-keep_n]
        if not history:
            return
        summary = self._summarizer.summarize(history)
        if not summary:
            return
        self.summary = self._summarizer.merge_summary(self.summary, summary)
        self._promote_to_salient_fact(history)
        self.messages = self.messages[-keep_n:]
        if self._repository:
            self._repository.replace_messages(self.session_id, self.messages)
            self._repository.update_summary(self.session_id, self.summary, self.trimmed_count)

    def _summarize_tool_result(self, result: str) -> Dict[str, Any]:
        if not result:
            return {
                "output_summary": "",
                "output_is_summary": False,
                "output_is_truncated": False,
            }

        max_len = self.config.tool_result_max_length
        envelope = parse_tool_response(result)
        if envelope is not None:
            summary_text = self._summarize_structured_payload(
                envelope.model_dump(mode="json"),
                max_len=max_len,
            )
            return {
                "output_summary": summary_text,
                "output_is_summary": True,
                "output_is_truncated": "[已截断]" in summary_text,
            }

        try:
            payload = json.loads(result)
            if isinstance(payload, dict):
                summary_text = self._summarize_structured_payload(payload, max_len=max_len)
                return {
                    "output_summary": summary_text,
                    "output_is_summary": True,
                    "output_is_truncated": "[已截断]" in summary_text,
                }
        except (TypeError, json.JSONDecodeError):
            pass

        if len(result) <= max_len:
            return {
                "output_summary": result,
                "output_is_summary": False,
                "output_is_truncated": False,
            }
        return {
            "output_summary": self._truncate_with_marker(result, max_len=max_len, prefix="[摘要][已截断] "),
            "output_is_summary": True,
            "output_is_truncated": True,
        }

    def _summarize_structured_payload(self, payload: Dict[str, Any], max_len: int) -> str:
        summary = self._build_structured_summary(payload)
        if len(summary) <= max_len:
            return f"[摘要] {summary}"
        return self._truncate_with_marker(summary, max_len=max_len, prefix="[摘要][已截断] ")

    def _build_structured_summary(self, payload: Dict[str, Any]) -> str:
        if payload.get("ok") is False:
            error = payload.get("error") or {}
            code = error.get("code", "UNKNOWN_ERROR")
            message = error.get("message", "未知错误")
            details = error.get("details") or {}
            detail_keys = list(details.keys())[:5]
            details_part = f"; details={','.join(detail_keys)}" if detail_keys else ""
            return f"错误结果 code={code}; message={message}{details_part}"

        data = payload.get("data", payload)
        if isinstance(data, dict):
            keys = list(data.keys())
            parts = [f"结构化结果字段={','.join(keys[:8]) or '无'}"]
            for key in keys[:5]:
                parts.append(f"{key}={self._preview_value(data.get(key), 80)}")
            return "; ".join(parts)
        if isinstance(data, list):
            preview_items = ", ".join(self._preview_value(item, 40) for item in data[:3])
            return f"结构化结果列表 count={len(data)}; items={preview_items}"
        return f"结果={self._preview_value(data, 160)}"

    def _preview_value(self, value: Any, max_len: int) -> str:
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            text = str(value)
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _truncate_with_marker(self, text: str, max_len: int, prefix: str) -> str:
        if max_len <= len(prefix):
            return prefix[:max_len]
        available = max_len - len(prefix)
        if len(text) > available:
            text = text[: max(available - 3, 0)] + ("..." if available >= 3 else "")
        return prefix + text

    def get_context(self, max_tokens: Optional[int] = None) -> str:
        return self.build_context(max_tokens=max_tokens)["context_text"]

    def build_context(self, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        if max_tokens and max_tokens != self.config.context_budget:
            original = self.config.context_budget
            self.config.context_budget = max_tokens
            try:
                return self._context_builder.build_context(self.state, self.messages, self.salient_facts, self.tool_calls)
            finally:
                self.config.context_budget = original
        return self._context_builder.build_context(self.state, self.messages, self.salient_facts, self.tool_calls)

    def debug_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        if session_id and session_id != self.session_id and self._repository:
            payload = self._repository.load_session(session_id)
            if payload:
                return {
                    "session_id": session_id,
                    "user_id": payload["user_id"],
                    "message_count": len(payload["messages"]),
                    "tool_call_count": len(payload["tool_calls"]),
                    "salient_fact_count": len(payload["salient_facts"]),
                    "summary_length": len(payload["state"].summary),
                    "trimmed_count": payload["state"].trimmed_count,
                }
        return self.get_debug_info()

    def debug_context(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        if session_id and session_id != self.session_id and self._repository:
            payload = self._repository.load_session(session_id)
            if payload:
                return ContextBuilder(self.config, self._estimate_tokens).build_context(
                    payload["state"], payload["messages"], payload["salient_facts"], payload["tool_calls"]
                )
        return self.get_context_debug_info()

    def get_recent_messages(self, window: Optional[int] = None) -> List[Dict[str, str]]:
        window = window or self.config.memory_window
        return [{"role": msg.role, "content": msg.content} for msg in self.messages[-window:]]

    def clear_session(self):
        self.clear()

    def clear(self):
        self.messages.clear()
        self.tool_calls.clear()
        self.salient_facts.clear()
        self.summary = ""
        self.trimmed_count = 0
        self.last_trimmed_content.clear()
        if self._repository:
            self._repository.clear_session(self.session_id)

    def get_size(self) -> int:
        return len(self.messages)

    def get_debug_info(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "message_count": len(self.messages),
            "tool_call_count": len(self.tool_calls),
            "salient_fact_count": len(self.salient_facts),
            "summary_length": len(self.summary),
            "summary_tokens": self._estimate_tokens(self.summary),
            "trimmed_count": self.trimmed_count,
            "persistence_enabled": self.config.enable_persistence,
            "config": {
                "max_size": self.config.max_size,
                "context_budget": self.config.context_budget,
                "summary_max_tokens": self.config.summary_max_tokens,
                "summary_trigger_messages": self.config.summary_trigger_messages,
                "summary_trigger_tokens": self.config.summary_trigger_tokens,
            },
        }

    def get_context_debug_info(self) -> Dict[str, Any]:
        context = self.build_context()
        return {
            "context_text_preview": context["context_text"][:500],
            "context_total_tokens": context["total_tokens"],
            "key_facts_preview": context.get("key_facts", "")[:300],
            "history_summary_preview": context.get("history_summary", "")[:300],
            "recent_dialog_preview": context.get("recent_dialog", "")[:300],
            "last_trimmed_content": self.last_trimmed_content[:5],
            "trimmed_count": self.trimmed_count,
        }

    def get_all_messages(self) -> List[Dict[str, Any]]:
        return [msg.to_dict() for msg in self.messages]

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.tool_calls]

    def get_salient_facts(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.salient_facts]

    def get_summary(self) -> str:
        return self.summary

    @classmethod
    def restore_session(cls, session_id: str) -> Optional["ShortTermMemory"]:
        config = ShortTermMemoryConfig.from_settings()
        if not config.enable_persistence:
            return None
        repository = ShortTermMemoryRepository(config.persistence_path)
        repository.initialize()
        user_id = repository.load_session_user(session_id)
        if not user_id:
            return None
        return cls(session_id=session_id, user_id=user_id)
