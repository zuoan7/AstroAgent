"""短期记忆压缩工具。

负责把大型工具输出压成 prompt 友好的摘要，并把事件批次压缩成结构化
summary snapshot。结构化摘要默认由确定性规则生成，也允许注入 LLM 摘要器。
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from src.memory.application.summary_snapshot_manager import SummarySnapshotManager
from src.memory.domain.events import MemoryEvent
from src.memory.domain.summary_snapshot import SummarySnapshot

SUMMARY_FIELDS = [
    "topics",
    "decisions",
    "open_questions",
    "established_facts",
    "tool_results_index",
]


@dataclass
class StructuredSummary:
    """短期记忆摘要的稳定 JSON schema。"""

    topics: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    established_facts: list[str] = field(default_factory=list)
    tool_results_index: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """输出固定字段顺序的可序列化字典。"""

        return {
            "topics": self.topics,
            "decisions": self.decisions,
            "open_questions": self.open_questions,
            "established_facts": self.established_facts,
            "tool_results_index": self.tool_results_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuredSummary":
        """从可能不完整的字典恢复并归一化摘要。"""

        return cls(
            topics=_unique_strings(data.get("topics", [])),
            decisions=_unique_strings(data.get("decisions", [])),
            open_questions=_unique_strings(data.get("open_questions", [])),
            established_facts=_unique_strings(data.get("established_facts", [])),
            tool_results_index=_unique_tool_results(data.get("tool_results_index", [])),
        )


class CompressionService:
    """Compression utilities for tool digests and structured summary snapshots."""

    def __init__(
        self,
        summary_snapshot_manager: SummarySnapshotManager,
        max_summary_chars: int = 1800,
        structured_summarizer: Callable[..., Any] | None = None,
    ):
        """初始化摘要快照管理器、摘要长度上限和可选结构化摘要器。"""

        self.summary_snapshot_manager = summary_snapshot_manager
        self.max_summary_chars = max_summary_chars
        self.structured_summarizer = structured_summarizer

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
        """Create L1 segment snapshots plus a global structured working snapshot."""

        event_list = list(events)
        self._create_l1_snapshots(
            tenant_id=tenant_id,
            session_id=session_id,
            events=event_list,
            created_by_model=created_by_model,
        )
        summary_text = self.summarize_events(event_list)
        return self.summary_snapshot_manager.create_snapshot(
            tenant_id=tenant_id,
            session_id=session_id,
            summary_text=summary_text,
            covered_events=event_list,
            snapshot_type="working",
            summary_level="l2",
            quality_score=self._estimate_quality(summary_text, event_list),
            created_by_model=created_by_model,
        )

    def rebase_summary(
        self,
        tenant_id: str,
        session_id: str,
        base_snapshot: SummarySnapshot | None,
        new_events: Sequence[MemoryEvent],
    ) -> SummarySnapshot:
        """Merge an existing structured snapshot with newly uncovered events."""

        event_list = list(new_events)
        self._create_l1_snapshots(
            tenant_id=tenant_id,
            session_id=session_id,
            events=event_list,
            created_by_model="rule-based-l1",
        )
        summary_text = self._summarize_rebase_with_optional_llm(
            base_snapshot,
            event_list,
        )
        return self.summary_snapshot_manager.create_snapshot(
            tenant_id=tenant_id,
            session_id=session_id,
            summary_text=summary_text,
            covered_events=event_list,
            snapshot_type="working",
            summary_level="l2",
            quality_score=self._estimate_quality(summary_text, event_list),
            created_by_model="structured-rebase",
        )

    def summarize_events(self, events: Iterable[MemoryEvent]) -> str:
        """把事件批次摘要为固定 schema JSON 字符串。"""

        event_list = list(events)
        llm_summary = self._summarize_events_with_optional_llm(event_list)
        if llm_summary is not None:
            return self._summary_to_json(llm_summary)
        return self._summary_to_json(self._rule_summary_from_events(event_list))

    def parse_summary(self, summary_text: str) -> StructuredSummary:
        """解析结构化摘要；旧自由文本快照会降级为 established facts。"""

        text = summary_text or ""
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            facts = [
                self._truncate(line.strip(), 220)
                for line in text.splitlines()
                if line.strip()
            ][:8]
            if not facts and text.strip():
                facts = [self._truncate(text.strip(), 220)]
            return StructuredSummary(
                topics=sorted(_extract_summary_entities(text))[:8],
                established_facts=facts,
            )
        if not isinstance(payload, dict):
            return StructuredSummary(established_facts=[self._truncate(str(payload), 220)])
        return StructuredSummary.from_dict(payload)

    def merge_summaries(
        self,
        base: StructuredSummary,
        incoming: StructuredSummary,
    ) -> StructuredSummary:
        """按 topic 去重，并用新 decision 标记旧 decision superseded。"""

        merged = StructuredSummary(
            topics=_merge_strings(base.topics, incoming.topics),
            decisions=list(base.decisions),
            open_questions=_merge_strings(
                base.open_questions,
                incoming.open_questions,
            ),
            established_facts=_merge_strings(
                base.established_facts,
                incoming.established_facts,
            ),
            tool_results_index=_merge_tool_results(
                base.tool_results_index,
                incoming.tool_results_index,
            ),
        )
        for new_decision in incoming.decisions:
            merged.decisions = [
                self._mark_superseded_if_conflict(old_decision, new_decision)
                for old_decision in merged.decisions
            ]
            if _normalize_text(new_decision) not in {
                _normalize_text(decision) for decision in merged.decisions
            }:
                merged.decisions.append(new_decision)
        return merged

    def _summarize_events_with_optional_llm(
        self,
        events: Sequence[MemoryEvent],
    ) -> StructuredSummary | None:
        """调用可选 LLM/外部摘要器，非法输出返回 None 以触发规则兜底。"""

        if self.structured_summarizer is None:
            return None
        try:
            summarizer = self.structured_summarizer
            if hasattr(summarizer, "summarize_events"):
                raw = summarizer.summarize_events(events)
            else:
                raw = summarizer(events)
            return self._coerce_structured_summary(raw)
        except Exception:
            return None

    def _summarize_rebase_with_optional_llm(
        self,
        base_snapshot: SummarySnapshot | None,
        new_events: Sequence[MemoryEvent],
    ) -> str:
        """调用可选 LLM merge，失败时用规则结构化 merge。"""

        if self.structured_summarizer is not None:
            try:
                summarizer = self.structured_summarizer
                if hasattr(summarizer, "rebase_summary"):
                    raw = summarizer.rebase_summary(base_snapshot, new_events)
                    summary = self._coerce_structured_summary(raw)
                    if summary is not None:
                        return self._summary_to_json(summary)
            except Exception:
                pass

        base_summary = (
            self.parse_summary(base_snapshot.summary_text)
            if base_snapshot and base_snapshot.summary_text
            else StructuredSummary()
        )
        incoming_summary = self.parse_summary(self.summarize_events(new_events))
        merged = self.merge_summaries(base_summary, incoming_summary)
        return self._summary_to_json(merged)

    def _coerce_structured_summary(self, raw: Any) -> StructuredSummary | None:
        """把摘要器输出转换为 StructuredSummary，失败返回 None。"""

        if isinstance(raw, StructuredSummary):
            return raw
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return None
        elif isinstance(raw, dict):
            payload = raw
        else:
            return None
        if not isinstance(payload, dict):
            return None
        if not any(field in payload for field in SUMMARY_FIELDS):
            return None
        return StructuredSummary.from_dict(payload)

    def _rule_summary_from_events(
        self,
        events: Sequence[MemoryEvent],
    ) -> StructuredSummary:
        """确定性地从事件中抽取结构化摘要字段。"""

        summary = StructuredSummary()
        for event in events:
            payload: dict[str, Any] = event.payload or {}
            event_text = self._event_text(event)
            summary.topics = _merge_strings(
                summary.topics,
                sorted(_extract_summary_entities(event_text)) or self._topic_fallback(event_text),
            )
            if event.event_type == "message_created":
                self._summarize_message_event(summary, payload)
            elif event.event_type in {"tool_call_finished", "tool_call_failed"}:
                self._summarize_tool_event(summary, payload)
            elif event.event_type == "task_state_updated":
                self._summarize_task_state_event(summary, payload)
            elif event.event_type == "fact_extracted":
                content = str(payload.get("content", "")).strip()
                if content:
                    summary.established_facts = _merge_strings(
                        summary.established_facts,
                        [self._truncate(content, 220)],
                    )
            elif event.event_type == "memory_deleted":
                summary.decisions = _merge_strings(
                    summary.decisions,
                    ["memory deletion applied"],
                )
        return summary

    def _summarize_message_event(
        self,
        summary: StructuredSummary,
        payload: dict[str, Any],
    ) -> None:
        """把 message_created payload 归入 question/decision/fact。"""

        role = str(payload.get("role", "message"))
        content = str(payload.get("content", "")).strip()
        if not content:
            return
        item = f"{role}: {self._truncate(content, 220)}"
        if "?" in content or "？" in content:
            summary.open_questions = _merge_strings(summary.open_questions, [item])
        elif role == "assistant" and self._looks_like_decision(content):
            summary.decisions = _merge_strings(summary.decisions, [item])
        else:
            summary.established_facts = _merge_strings(summary.established_facts, [item])

    def _summarize_tool_event(
        self,
        summary: StructuredSummary,
        payload: dict[str, Any],
    ) -> None:
        """把工具事件写入 facts 与 tool_results_index。"""

        metadata = payload.get("metadata", {}) or {}
        tool_name = str(payload.get("tool_name", "tool"))
        status = str(payload.get("status") or ("success" if payload.get("success", True) else "error"))
        output = str(
            payload.get("output_digest")
            or payload.get("output_summary")
            or payload.get("result_summary")
            or ""
        )
        key_finding = self._truncate(output, 220)
        if key_finding:
            summary.established_facts = _merge_strings(
                summary.established_facts,
                [f"tool {tool_name} ({status}): {key_finding}"],
            )
        summary.tool_results_index = _merge_tool_results(
            summary.tool_results_index,
            [
                {
                    "tool": tool_name,
                    "tool_type": str(metadata.get("tool_type") or ""),
                    "params_hash": str(metadata.get("params_hash") or ""),
                    "status": status,
                    "key_finding": key_finding,
                }
            ],
        )

    def _summarize_task_state_event(
        self,
        summary: StructuredSummary,
        payload: dict[str, Any],
    ) -> None:
        """把 task_state_updated payload 中的目标、问题和步骤并入摘要。"""

        state = payload.get("state", {}) or {}
        goal = str(state.get("current_goal") or "").strip()
        next_action = str(state.get("next_action") or "").strip()
        if goal:
            summary.topics = _merge_strings(summary.topics, [self._truncate(goal, 160)])
        if next_action:
            summary.decisions = _merge_strings(
                summary.decisions,
                [f"next_action: {self._truncate(next_action, 180)}"],
            )
        summary.open_questions = _merge_strings(
            summary.open_questions,
            [self._truncate(str(item), 180) for item in state.get("open_questions", [])],
        )
        summary.decisions = _merge_strings(
            summary.decisions,
            [self._truncate(str(item), 180) for item in state.get("completed_steps", [])],
        )
        summary.established_facts = _merge_strings(
            summary.established_facts,
            [self._truncate(str(item), 180) for item in state.get("active_constraints", [])],
        )

    def _summary_to_json(self, summary: StructuredSummary) -> str:
        """把结构化摘要压缩到长度上限内，并保持 JSON 合法。"""

        compact = self._compact_summary(summary)
        return json.dumps(
            compact.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _compact_summary(self, summary: StructuredSummary) -> StructuredSummary:
        """按字段优先级裁剪结构化摘要，避免字符截断破坏 JSON。"""

        compact = StructuredSummary(
            topics=[self._truncate(item, 120) for item in summary.topics[:12]],
            decisions=[self._truncate(item, 180) for item in summary.decisions[:12]],
            open_questions=[
                self._truncate(item, 180) for item in summary.open_questions[:12]
            ],
            established_facts=[
                self._truncate(item, 180) for item in summary.established_facts[:18]
            ],
            tool_results_index=[
                {
                    **item,
                    "key_finding": self._truncate(item.get("key_finding", ""), 160),
                }
                for item in summary.tool_results_index[:12]
            ],
        )
        removal_order = [
            "topics",
            "tool_results_index",
            "established_facts",
            "decisions",
            "open_questions",
        ]
        while len(json.dumps(compact.to_dict(), ensure_ascii=False)) > self.max_summary_chars:
            removed = False
            for field_name in removal_order:
                values = getattr(compact, field_name)
                if values:
                    values.pop()
                    removed = True
                    break
            if not removed:
                break
        return compact

    def _create_l1_snapshots(
        self,
        tenant_id: str,
        session_id: str,
        events: Sequence[MemoryEvent],
        created_by_model: str,
    ) -> None:
        """为工具链或对话段落创建局部 L1 摘要，不替代 working 快照。"""

        for segment in self._l1_segments(events):
            summary = self._summary_to_json(self._rule_summary_from_events(segment))
            self.summary_snapshot_manager.create_snapshot(
                tenant_id=tenant_id,
                session_id=session_id,
                summary_text=summary,
                covered_events=segment,
                snapshot_type="segment",
                summary_level="l1",
                quality_score=self._estimate_quality(summary, segment),
                created_by_model=f"{created_by_model}-l1",
                supersede_latest=False,
            )

    def _l1_segments(self, events: Sequence[MemoryEvent]) -> list[list[MemoryEvent]]:
        """按工具链和简短对话段落切出 L1 摘要段。"""

        segments: list[list[MemoryEvent]] = []
        index = 0
        while index < len(events):
            event = events[index]
            if event.event_type in {"tool_call_finished", "tool_call_failed"}:
                start = index
                index += 1
                while index < len(events) and events[index].event_type in {
                    "tool_call_finished",
                    "tool_call_failed",
                }:
                    index += 1
                if (
                    index < len(events)
                    and events[index].event_type == "message_created"
                    and (events[index].payload or {}).get("role") == "assistant"
                ):
                    index += 1
                segments.append(list(events[start:index]))
                continue
            index += 1

        paragraph: list[MemoryEvent] = []
        for event in events:
            if event.event_type not in {"message_created", "task_state_updated", "fact_extracted"}:
                continue
            paragraph.append(event)
            is_assistant = (
                event.event_type == "message_created"
                and (event.payload or {}).get("role") == "assistant"
            )
            if is_assistant or len(paragraph) >= 6:
                if len(paragraph) > 1:
                    segments.append(list(paragraph))
                paragraph = []
        if len(paragraph) > 1:
            segments.append(paragraph)
        return segments

    def _mark_superseded_if_conflict(
        self,
        old_decision: str,
        new_decision: str,
    ) -> str:
        """若新旧 decision 明显冲突，则给旧 decision 标 superseded。"""

        if "[superseded]" in old_decision:
            return old_decision
        if not _decision_conflicts(old_decision, new_decision):
            return old_decision
        return f"{old_decision} [superseded]"

    def _looks_like_decision(self, content: str) -> bool:
        """识别助手回复中可进入 decisions 的结论/建议语句。"""

        return any(
            marker in content
            for marker in [
                "结论",
                "决定",
                "建议",
                "可以",
                "适合",
                "不适合",
                "不能",
                "最终",
                "下一步",
            ]
        )

    def _topic_fallback(self, text: str) -> list[str]:
        """实体缺失时用短文本作为 topic fallback。"""

        text = text.strip()
        if not text:
            return []
        return [self._truncate(text, 80)]

    def _event_text(self, event: MemoryEvent) -> str:
        """抽取事件中适合识别 topic/entity 的文本。"""

        payload = event.payload or {}
        if event.event_type == "message_created":
            return str(payload.get("content", ""))
        if event.event_type in {"tool_call_finished", "tool_call_failed"}:
            return " ".join(
                str(payload.get(key, ""))
                for key in [
                    "tool_name",
                    "tool_input",
                    "input_summary",
                    "output_digest",
                    "output_summary",
                ]
            )
        if event.event_type == "task_state_updated":
            return str(payload.get("state", {}))
        if event.event_type == "fact_extracted":
            return str(payload.get("content", ""))
        return str(payload)

    def _digest_json(self, payload: Any) -> str:
        """为 JSON 工具输出生成字段级摘要，避免把完整结构塞进 prompt。"""

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
        """估算结构化摘要质量：字段覆盖、工具索引覆盖与解析成功率。"""

        if not events:
            return 0.0
        summary = self.parse_summary(summary_text)
        non_empty_fields = sum(
            1
            for field_name in SUMMARY_FIELDS
            if getattr(summary, field_name)
        )
        tool_events = [
            event
            for event in events
            if event.event_type in {"tool_call_finished", "tool_call_failed"}
        ]
        tool_coverage = 1.0
        if tool_events:
            tool_coverage = min(
                len(summary.tool_results_index) / max(len(tool_events), 1),
                1.0,
            )
        field_coverage = non_empty_fields / len(SUMMARY_FIELDS)
        event_coverage = min(
            (
                len(summary.decisions)
                + len(summary.open_questions)
                + len(summary.established_facts)
                + len(summary.tool_results_index)
            )
            / max(len(events), 1),
            1.0,
        )
        quality = 0.45 * field_coverage + 0.35 * event_coverage + 0.20 * tool_coverage
        return round(max(0.2, quality), 2)

    def _truncate(self, text: str, max_chars: int) -> str:
        """按字符数截断文本，并在空间足够时追加省略号。"""

        if len(text) <= max_chars:
            return text
        if max_chars <= 3:
            return text[:max_chars]
        return text[: max_chars - 3] + "..."


def _unique_strings(values: Any) -> list[str]:
    """清洗字符串列表，去空并保持首次出现顺序。"""

    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        values = list(values) if isinstance(values, (tuple, set)) else [values]
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = _normalize_text(text)
        if not text or key in seen:
            continue
        result.append(text)
        seen.add(key)
    return result


def _merge_strings(first: Sequence[str], second: Sequence[str]) -> list[str]:
    """合并两个字符串序列并去重。"""

    return _unique_strings([*list(first or []), *list(second or [])])


def _unique_tool_results(values: Any) -> list[dict[str, str]]:
    """清洗 tool_results_index 列表并按工具/参数/status/key 去重。"""

    if not isinstance(values, list):
        return []
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in values:
        if not isinstance(item, dict):
            continue
        normalized = {
            "tool": str(item.get("tool") or item.get("tool_name") or ""),
            "tool_type": str(item.get("tool_type") or ""),
            "params_hash": str(item.get("params_hash") or ""),
            "status": str(item.get("status") or ""),
            "key_finding": str(item.get("key_finding") or ""),
        }
        key = (
            normalized["tool"],
            normalized["params_hash"],
            normalized["status"],
            _normalize_text(normalized["key_finding"]),
        )
        if key in seen:
            continue
        result.append(normalized)
        seen.add(key)
    return result


def _merge_tool_results(
    first: Sequence[dict[str, str]],
    second: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    """合并工具结果索引并去重。"""

    return _unique_tool_results([*list(first or []), *list(second or [])])


def _normalize_text(text: str) -> str:
    """用于摘要项去重的宽松归一化。"""

    return re.sub(r"\s+", "", str(text or "").lower())


def _extract_summary_entities(text: str) -> set[str]:
    """抽取摘要 topic 使用的地点、目标和天象实体。"""

    source = text or ""
    entities = {
        location
        for location in [
            "北京",
            "上海",
            "广州",
            "深圳",
            "杭州",
            "苏州",
            "成都",
            "南京",
            "武汉",
            "西安",
        ]
        if location in source
    }
    entities |= {target.upper() for target in re.findall(r"\b[Mm]\d{1,3}\b", source)}
    aliases = {
        "M42": ["猎户座大星云", "猎户座星云", "Orion Nebula"],
        "M31": ["仙女座星系", "仙女座大星系", "Andromeda"],
        "月球": ["月球", "Moon"],
        "火星": ["火星", "Mars"],
        "木星": ["木星", "Jupiter"],
        "土星": ["土星", "Saturn"],
        "英仙座流星雨": ["英仙座流星雨", "Perseids"],
        "双子座流星雨": ["双子座流星雨", "Geminids"],
    }
    for canonical, names in aliases.items():
        if any(name in source for name in names):
            entities.add(canonical)
    return entities


def _decision_conflicts(old_decision: str, new_decision: str) -> bool:
    """用确定性 cue 判断新旧 decisions 是否冲突。"""

    old = old_decision or ""
    new = new_decision or ""
    old_positive = any(marker in old for marker in ["可以", "适合", "建议"])
    old_negative = any(marker in old for marker in ["不可以", "不适合", "不能", "不要"])
    new_positive = any(marker in new for marker in ["可以", "适合", "建议"])
    new_negative = any(marker in new for marker in ["不可以", "不适合", "不能", "不要"])
    if (old_positive and new_negative) or (old_negative and new_positive):
        old_entities = _extract_summary_entities(old)
        new_entities = _extract_summary_entities(new)
        return not old_entities or not new_entities or bool(old_entities & new_entities)
    if any(marker in new for marker in ["否决", "不是", "改为", "更新为"]):
        return bool(
            _extract_summary_entities(old) & _extract_summary_entities(new)
            or set(_normalize_text(old).split()) & set(_normalize_text(new).split())
        )
    return False
