from typing import Dict, List, Sequence, Tuple

from src.memory.core.models import Message, SalientFact, SessionMemoryState, ToolCallRecord
from src.memory.short_term_memory.config import ShortTermMemoryConfig


ROLE_LABELS = {
    "user": "用户",
    "assistant": "助手",
    "tool": "工具",
    "system": "系统",
}


class ContextBuilder:
    def __init__(self, config: ShortTermMemoryConfig, token_counter):
        self.config = config
        self._estimate_tokens = token_counter

    def build_context(
        self,
        session_state: SessionMemoryState,
        messages: Sequence[Message],
        facts: Sequence[SalientFact],
        tool_calls: Sequence[ToolCallRecord],
    ) -> Dict[str, object]:
        budget = self._allocate_budget(self.config.context_budget)
        selected_facts, facts_text = self._select_salient_facts(facts, budget["facts"])
        summary_text = self._truncate_to_budget(session_state.summary, budget["summary"])
        selected_messages, recent_text = self._select_recent_messages(messages, budget["recent"])
        selected_tool_calls, tool_text = self._select_tool_calls(tool_calls, budget["tools"])
        context_text = self._assemble_context_text(facts_text, summary_text, recent_text, tool_text)
        return {
            "context_text": context_text,
            "key_facts": facts_text,
            "history_summary": summary_text,
            "recent_dialog": recent_text,
            "tool_summary": tool_text,
            "total_tokens": self._estimate_tokens(context_text),
            "message_count": len(messages),
            "selected_recent_messages": [msg.to_dict() for msg in selected_messages],
            "selected_salient_facts": [fact.to_dict() for fact in selected_facts],
            "selected_tool_calls": [call.to_dict() for call in selected_tool_calls],
            "summary_tokens": self._estimate_tokens(summary_text),
            "facts_count": len(facts),
            "trimmed_count": session_state.trimmed_count,
        }

    def _allocate_budget(self, total_budget: int) -> Dict[str, int]:
        return {
            "facts": max(256, int(total_budget * 0.2)),
            "summary": max(256, int(total_budget * 0.25)),
            "recent": max(512, int(total_budget * 0.4)),
            "tools": max(128, int(total_budget * 0.15)),
        }

    def _select_recent_messages(self, messages: Sequence[Message], token_budget: int) -> Tuple[List[Message], str]:
        selected: List[Message] = []
        used_tokens = 0
        for message in reversed(messages):
            line = f"{ROLE_LABELS.get(message.role, message.role)}: {message.content}"
            line_tokens = self._estimate_tokens(line)
            if selected and (
                len(selected) >= self.config.max_recent_messages or used_tokens + line_tokens > token_budget
            ):
                continue
            selected.append(message)
            used_tokens += line_tokens
            if len(selected) >= self.config.max_recent_messages and used_tokens >= self.config.max_recent_tokens:
                break
        selected.reverse()
        text = "\n".join(f"{ROLE_LABELS.get(msg.role, msg.role)}: {msg.content}" for msg in selected)
        return selected, text or "无最近对话"

    def _select_salient_facts(self, facts: Sequence[SalientFact], token_budget: int) -> Tuple[List[SalientFact], str]:
        chosen: List[SalientFact] = []
        lines: List[str] = []
        for fact in sorted(facts, key=lambda item: (item.timestamp, item.fact_type))[-self.config.max_salient_facts :]:
            line = f"- [{fact.fact_type}] {fact.content}"
            tentative = "\n".join(lines + [line])
            if self._estimate_tokens(tentative) > token_budget:
                continue
            chosen.append(fact)
            lines.append(line)
        return chosen, "\n".join(lines)

    def _select_tool_calls(
        self,
        tool_calls: Sequence[ToolCallRecord],
        token_budget: int,
    ) -> Tuple[List[ToolCallRecord], str]:
        chosen: List[ToolCallRecord] = []
        lines: List[str] = []
        for call in tool_calls[-self.config.max_tool_records :]:
            status = "✓" if call.status == "success" else "✗"
            tags: List[str] = []
            if call.output_is_summary:
                tags.append("摘要")
            if call.output_is_truncated:
                tags.append("已截断")
            tag_text = f" [{'|'.join(tags)}]" if tags else ""
            line = f"[{status}] {call.tool_name}{tag_text}: {call.output_summary}"
            tentative = "\n".join(lines + [line])
            if self._estimate_tokens(tentative) > token_budget:
                break
            chosen.append(call)
            lines.append(line)
        return chosen, "\n".join(lines)

    def _assemble_context_text(
        self,
        salient_facts: str,
        summary: str,
        recent_messages: str,
        tool_summary: str,
    ) -> str:
        sections: List[str] = []
        if salient_facts:
            sections.append(f"=== salient facts ===\n{salient_facts}")
        if summary:
            sections.append(f"=== summary ===\n{summary}")
        if recent_messages:
            sections.append(f"=== recent messages ===\n{recent_messages}")
        if tool_summary:
            sections.append(f"=== tool summary ===\n{tool_summary}")
        return "\n\n".join(sections) if sections else "无对话上下文"

    def _truncate_to_budget(self, text: str, token_budget: int) -> str:
        if self._estimate_tokens(text) <= token_budget:
            return text
        lines = text.splitlines()
        kept: List[str] = []
        for line in lines:
            tentative = "\n".join(kept + [line])
            if self._estimate_tokens(tentative) > token_budget:
                break
            kept.append(line)
        marker = "[摘要已截断，非完整原文]"
        tentative = "\n".join(kept + [marker]) if kept else marker
        if self._estimate_tokens(tentative) <= token_budget:
            return tentative
        return "\n".join(kept)
