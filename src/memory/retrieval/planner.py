import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

from src.memory.core.models import Message, SalientFact, ToolCallRecord
from src.memory.domain.summary_snapshot import SummarySnapshot
from src.memory.domain.task_state import TaskState

ROLE_LABELS = {"user": "用户", "assistant": "助手", "system": "系统", "tool": "工具"}


@dataclass
class RetrievalPlan:
    """Traceable retrieval decision for a context build."""

    query: str
    query_type: str
    token_budget: int
    selected_message_ids: list[str] = field(default_factory=list)
    selected_fact_ids: list[str] = field(default_factory=list)
    selected_tool_call_ids: list[str] = field(default_factory=list)
    selected_snapshot_id: str = ""
    selected_task_state_version: int = 0


class RetrievalPlanner:
    """P0 query-aware context assembler.

    It keeps hard slots for task state and recent conversation, then ranks
    facts/tool calls/messages by simple query overlap. This deterministic
    behavior makes context selection easy to test and inspect.
    """

    def __init__(self, token_counter):
        self._estimate_tokens = token_counter

    def build_context(
        self,
        query: str,
        token_budget: int,
        task_state: TaskState,
        summary_snapshot: SummarySnapshot | None,
        messages: Sequence[Message],
        facts: Sequence[SalientFact],
        tool_calls: Sequence[ToolCallRecord],
    ) -> Dict[str, Any]:
        query_type = self._classify_query(query)
        plan = RetrievalPlan(
            query=query, query_type=query_type, token_budget=token_budget
        )
        sections: list[tuple[str, str]] = []

        task_text = self._format_task_state(task_state)
        if task_text:
            sections.append(("task state", task_text))
            plan.selected_task_state_version = task_state.version

        if summary_snapshot and summary_snapshot.summary_text:
            summary_text = self._truncate_to_budget(
                summary_snapshot.summary_text, max(256, int(token_budget * 0.2))
            )
            sections.append(("summary snapshot", summary_text))
            plan.selected_snapshot_id = summary_snapshot.snapshot_id

        selected_facts = self._rank_facts(query, facts)[:8]
        facts_text = "\n".join(
            f"- [{fact.fact_type}] {fact.content}" for fact in selected_facts
        )
        if facts_text:
            sections.append(("relevant facts", facts_text))
            plan.selected_fact_ids = [fact.fact_id for fact in selected_facts]

        selected_tools = self._rank_tools(query, tool_calls)[:6]
        tools_text = "\n".join(self._format_tool_call(call) for call in selected_tools)
        if tools_text:
            sections.append(("tool evidence", tools_text))
            plan.selected_tool_call_ids = [call.tool_call_id for call in selected_tools]

        selected_messages = self._rank_messages(query, messages)[:8]
        selected_messages.sort(key=lambda item: item.timestamp)
        message_text = "\n".join(
            f"{ROLE_LABELS.get(msg.role, msg.role)}: {msg.content}"
            for msg in selected_messages
        )
        if message_text:
            sections.append(("recent/relevant messages", message_text))
            plan.selected_message_ids = [msg.message_id for msg in selected_messages]

        context_text = self._fit_sections(sections, token_budget)
        return {
            "context_text": context_text or "无对话上下文",
            "query_type": query_type,
            "retrieval_plan": plan.__dict__,
            "total_tokens": self._estimate_tokens(context_text),
            "selected_recent_messages": [msg.to_dict() for msg in selected_messages],
            "selected_salient_facts": [fact.to_dict() for fact in selected_facts],
            "selected_tool_calls": [call.to_dict() for call in selected_tools],
            "selected_summary_snapshot": (
                summary_snapshot.to_dict() if summary_snapshot else None
            ),
            "selected_task_state": task_state.to_dict(),
            "built_at": time.time(),
        }

    def _classify_query(self, query: str) -> str:
        lower = (query or "").lower()
        if any(token in lower for token in ["tool", "工具", "结果", "证据", "输出"]):
            return "evidence"
        if any(token in lower for token in ["next", "下一步", "计划", "进度", "todo"]):
            return "task_progress"
        if any(token in lower for token in ["why", "为什么", "原因"]):
            return "reasoning"
        return "general"

    def _rank_messages(self, query: str, messages: Sequence[Message]) -> list[Message]:
        return sorted(
            messages,
            key=lambda msg: (self._score(query, msg.content), msg.timestamp),
            reverse=True,
        )

    def _rank_facts(
        self, query: str, facts: Sequence[SalientFact]
    ) -> list[SalientFact]:
        return sorted(
            facts,
            key=lambda fact: (self._score(query, fact.content), fact.timestamp),
            reverse=True,
        )

    def _rank_tools(
        self, query: str, tool_calls: Sequence[ToolCallRecord]
    ) -> list[ToolCallRecord]:
        return sorted(
            tool_calls,
            key=lambda call: (
                self._score(
                    query,
                    " ".join([call.tool_name, call.input_summary, call.output_summary]),
                ),
                call.importance,
                call.timestamp,
            ),
            reverse=True,
        )

    def _score(self, query: str, text: str) -> int:
        query_terms = set(self._terms(query))
        if not query_terms:
            return 0
        text_terms = set(self._terms(text))
        return len(query_terms & text_terms)

    def _terms(self, text: str) -> list[str]:
        return [
            item.lower()
            for item in re.findall(r"[\w\u4e00-\u9fff]+", text or "")
            if len(item) > 1
        ]

    def _format_task_state(self, state: TaskState) -> str:
        parts = []
        if state.current_goal:
            parts.append(f"current_goal: {state.current_goal}")
        if state.active_constraints:
            parts.append(
                "active_constraints: " + "; ".join(state.active_constraints[:8])
            )
        if state.pending_steps:
            parts.append("pending_steps: " + "; ".join(state.pending_steps[:8]))
        if state.blockers:
            parts.append("blockers: " + "; ".join(state.blockers[:5]))
        if state.next_action:
            parts.append(f"next_action: {state.next_action}")
        return "\n".join(parts)

    def _format_tool_call(self, call: ToolCallRecord) -> str:
        tags = []
        if call.raw_artifact_id:
            tags.append(f"artifact={call.raw_artifact_id}")
        if call.output_is_truncated:
            tags.append("truncated")
        tag_text = f" [{'|'.join(tags)}]" if tags else ""
        status = "success" if call.success else "error"
        return f"- {call.tool_name} ({status}){tag_text}: {call.output_summary}"

    def _fit_sections(
        self, sections: Sequence[tuple[str, str]], token_budget: int
    ) -> str:
        rendered: list[str] = []
        for title, body in sections:
            section = f"=== {title} ===\n{body}"
            tentative = "\n\n".join(rendered + [section])
            if self._estimate_tokens(tentative) <= token_budget:
                rendered.append(section)
                continue
            remaining = max(
                token_budget - self._estimate_tokens("\n\n".join(rendered)), 0
            )
            if remaining <= 32:
                break
            truncated = self._truncate_to_budget(body, remaining)
            if truncated:
                rendered.append(f"=== {title} ===\n{truncated}")
            break
        return "\n\n".join(rendered)

    def _truncate_to_budget(self, text: str, token_budget: int) -> str:
        if self._estimate_tokens(text) <= token_budget:
            return text
        kept = []
        for line in text.splitlines():
            tentative = "\n".join(kept + [line])
            if self._estimate_tokens(tentative) > token_budget:
                break
            kept.append(line)
        return "\n".join(kept) if kept else text[: max(token_budget * 2, 32)]
