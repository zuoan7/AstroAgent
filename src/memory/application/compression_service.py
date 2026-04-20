import json
from typing import Any, Iterable, Sequence

from src.memory.application.summary_snapshot_manager import SummarySnapshotManager
from src.memory.domain.events import MemoryEvent
from src.memory.domain.summary_snapshot import SummarySnapshot


class CompressionService:
    """Deterministic P0 compression utilities for tool digests and snapshots."""

    def __init__(self, summary_snapshot_manager: SummarySnapshotManager, max_summary_chars: int = 1800):
        self.summary_snapshot_manager = summary_snapshot_manager
        self.max_summary_chars = max_summary_chars

    def digest_tool_output(self, raw_output: str, max_chars: int = 600) -> str:
        """Create a prompt-friendly digest without replacing the raw artifact."""

        text = raw_output or ""
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return self._truncate(text, max_chars)
        return self._truncate(self._digest_json(payload), max_chars)

    def create_summary_snapshot(
        self,
        tenant_id: str,
        session_id: str,
        events: Sequence[MemoryEvent],
        created_by_model: str = "rule-based",
    ) -> SummarySnapshot:
        """Create a snapshot from raw events using a deterministic fallback summarizer."""

        summary_text = self.summarize_events(events)
        return self.summary_snapshot_manager.create_snapshot(
            tenant_id=tenant_id,
            session_id=session_id,
            summary_text=summary_text,
            covered_events=events,
            quality_score=self._estimate_quality(summary_text, events),
            created_by_model=created_by_model,
        )

    def rebase_summary(
        self,
        tenant_id: str,
        session_id: str,
        base_snapshot: SummarySnapshot | None,
        new_events: Sequence[MemoryEvent],
    ) -> SummarySnapshot:
        """Create a new snapshot from a previous snapshot plus uncovered events."""

        seed = []
        if base_snapshot and base_snapshot.summary_text:
            seed.append(f"已有快照: {base_snapshot.summary_text}")
        event_summary = self.summarize_events(new_events)
        if event_summary:
            seed.append(f"新增事件: {event_summary}")
        summary_text = self._truncate("\n".join(seed), self.max_summary_chars)
        return self.summary_snapshot_manager.create_snapshot(
            tenant_id=tenant_id,
            session_id=session_id,
            summary_text=summary_text,
            covered_events=new_events,
            quality_score=self._estimate_quality(summary_text, new_events),
            created_by_model="rule-based-rebase",
        )

    def summarize_events(self, events: Iterable[MemoryEvent]) -> str:
        lines: list[str] = []
        for event in events:
            payload: dict[str, Any] = event.payload or {}
            if event.event_type == "message_created":
                role = payload.get("role", "message")
                content = payload.get("content", "")
                lines.append(f"{role}: {self._truncate(str(content), 180)}")
            elif event.event_type in {"tool_call_finished", "tool_call_failed"}:
                tool_name = payload.get("tool_name", "tool")
                output = payload.get("output_digest") or payload.get("output_summary", "")
                lines.append(f"tool {tool_name}: {self._truncate(str(output), 180)}")
            elif event.event_type == "task_state_updated":
                state = payload.get("state", {})
                goal = state.get("current_goal") or ""
                next_action = state.get("next_action") or ""
                lines.append(f"task: goal={goal}; next={next_action}")
        return self._truncate("\n".join(line for line in lines if line.strip()), self.max_summary_chars)

    def _digest_json(self, payload: Any) -> str:
        if isinstance(payload, dict):
            keys = list(payload.keys())
            parts = [f"json fields={','.join(keys[:8]) or 'none'}"]
            for key in keys[:5]:
                parts.append(f"{key}={self._truncate(str(payload.get(key)), 80)}")
            return "; ".join(parts)
        if isinstance(payload, list):
            return f"json list count={len(payload)}; sample={self._truncate(str(payload[:3]), 180)}"
        return str(payload)

    def _estimate_quality(self, summary_text: str, events: Sequence[MemoryEvent]) -> float:
        if not events:
            return 0.0
        coverage = min(len(summary_text) / max(len(events) * 80, 1), 1.0)
        return round(max(0.2, coverage), 2)

    def _truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        if max_chars <= 3:
            return text[:max_chars]
        return text[: max_chars - 3] + "..."
