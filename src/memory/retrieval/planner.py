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


@dataclass
class ToolEvidenceMeta:
    tool_call_id: str
    tool_name: str
    tool_type: str
    locations: set[str]
    targets: set[str]
    is_fresh_marked: bool
    is_stale_marked: bool
    timestamp: float


@dataclass
class RetrievalFocus:
    locations: set[str]
    targets: set[str]
    preferred_tool_types: set[str]
    freshness_intent: str


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

        selected_tools = self._rank_tools_with_focus(query, task_state, tool_calls)
        tools_text = "\n".join(self._format_tool_call(call) for call in selected_tools)
        if tools_text:
            sections.append(("tool evidence", tools_text))
            plan.selected_tool_call_ids = [call.tool_call_id for call in selected_tools]

        selected_messages = self._rank_messages_with_focus(query, task_state, messages)[
            :6
        ]
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

    def _rank_messages_with_focus(
        self,
        query: str,
        task_state: TaskState,
        messages: Sequence[Message],
    ) -> list[Message]:
        focus = self._extract_focus(query, task_state)
        recent_message_ids = {
            message.message_id
            for message in sorted(messages, key=lambda item: item.timestamp)[-4:]
        }
        scored = []
        for message in messages:
            base_score = self._score(query, message.content)
            is_recent = message.message_id in recent_message_ids
            if base_score == 0 and not is_recent:
                continue
            if self._task_state_covers_message_noise(message.content, task_state, focus):
                continue
            if (
                not is_recent
                and self._task_state_covers_message(message.content, task_state, focus)
            ):
                continue

            score = base_score + self._score_message_focus(message.content, focus)
            scored.append((score, message.timestamp, message))
        return [
            item[2]
            for item in sorted(
                scored,
                key=lambda item: (item[0], item[1]),
                reverse=True,
            )
        ]

    def _score_message_focus(self, text: str, focus: RetrievalFocus) -> int:
        score = 0
        locations = self._extract_locations(text)
        targets = self._extract_targets(text)
        if focus.locations:
            if locations & focus.locations:
                score += 2
            elif locations:
                score -= 3
        if focus.targets:
            if targets & focus.targets:
                score += 2
            elif targets:
                score -= 3

        type_keywords = {
            "weather": ["天气", "云量", "湿度", "透明度"],
            "position": ["高度", "升起", "落下", "位置"],
            "photo": ["曝光", "ISO", "拍摄", "摄影", "参数"],
            "event": ["流星雨", "天象", "观测窗口", "峰值"],
            "neo": ["小行星", "NEO", "近地"],
        }
        for tool_type in focus.preferred_tool_types:
            if any(keyword in text for keyword in type_keywords.get(tool_type, [])):
                score += 1
        return score

    def _task_state_covers_message(
        self,
        message_text: str,
        task_state: TaskState,
        focus: RetrievalFocus,
    ) -> bool:
        state_text = self._format_task_state(task_state)
        if not state_text:
            return False

        message_entities = self._extract_locations(message_text) | self._extract_targets(
            message_text
        )
        focus_entities = focus.locations | focus.targets
        if message_entities and message_entities <= focus_entities:
            return all(entity in state_text for entity in message_entities)
        return False

    def _task_state_covers_message_noise(
        self,
        message_text: str,
        task_state: TaskState,
        focus: RetrievalFocus,
    ) -> bool:
        state_text = self._format_task_state(task_state)
        if not state_text:
            return False

        message_entities = self._extract_locations(message_text) | self._extract_targets(
            message_text
        )
        focus_entities = focus.locations | focus.targets
        conflict_entities = message_entities - focus_entities
        if conflict_entities and all(entity in state_text for entity in conflict_entities):
            return True

        stale_markers = ["旧参数", "旧地点", "旧结果"]
        if any(marker in message_text and marker in state_text for marker in stale_markers):
            return True

        if self._is_negative_constraint_text(message_text) and message_entities:
            return all(entity in state_text for entity in message_entities)
        return False

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

    def _rank_tools_with_focus(
        self,
        query: str,
        task_state: TaskState,
        tool_calls: Sequence[ToolCallRecord],
    ) -> list[ToolCallRecord]:
        focus = self._extract_focus(query, task_state)
        scored = []
        for call in tool_calls:
            meta = self._extract_tool_meta(call)
            score = self._score_tool_with_focus(query, focus, call, meta)
            if score > 0:
                scored.append((score, call.importance, meta.timestamp, call))

        if not scored:
            return self._rank_tools(query, tool_calls)[:1]

        ranked = [
            item[3]
            for item in sorted(
                scored,
                key=lambda item: (item[0], item[1], item[2]),
                reverse=True,
            )
        ]
        deduped = self._dedupe_superseded_tools(ranked, focus)
        if not deduped:
            fallback = [
                call
                for call in self._rank_tools(query, tool_calls)
                if not self._extract_tool_meta(call).is_stale_marked
            ]
            return fallback[:1]
        return deduped[:3]

    def _dedupe_superseded_tools(
        self,
        ranked_tools: Sequence[ToolCallRecord],
        focus: RetrievalFocus,
    ) -> list[ToolCallRecord]:
        if focus.freshness_intent in {"historical", "compare"}:
            return list(ranked_tools)

        best_by_group: dict[
            tuple[str, tuple[str, ...], tuple[str, ...]],
            tuple[tuple[int, float], int, ToolCallRecord],
        ] = {}
        passthrough: list[tuple[int, ToolCallRecord]] = []

        for index, call in enumerate(ranked_tools):
            meta = self._extract_tool_meta(call)
            if meta.is_stale_marked:
                continue

            if not meta.locations and not meta.targets:
                passthrough.append((index, call))
                continue

            key = (
                meta.tool_type,
                tuple(sorted(meta.locations)),
                tuple(sorted(meta.targets)),
            )
            preference = (1 if meta.is_fresh_marked else 0, meta.timestamp)
            current = best_by_group.get(key)
            if current is None or preference > current[0]:
                best_by_group[key] = (preference, index, call)

        kept = [(index, call) for _, index, call in best_by_group.values()]
        kept.extend(passthrough)
        kept.sort(key=lambda item: item[0])
        return [call for _, call in kept]

    def _score_tool_with_focus(
        self,
        query: str,
        focus: RetrievalFocus,
        call: ToolCallRecord,
        meta: ToolEvidenceMeta,
    ) -> int:
        text = " ".join([call.tool_name, call.input_summary, call.output_summary])
        score = self._score(query, text)

        if focus.locations:
            if meta.locations & focus.locations:
                score += 5
            elif meta.locations:
                score -= 8

        if focus.targets:
            if meta.targets & focus.targets:
                score += 5
            elif meta.targets:
                score -= 8

        if focus.preferred_tool_types:
            if meta.tool_type in focus.preferred_tool_types:
                score += 3
            elif meta.tool_type != "generic":
                score -= 3

        if focus.freshness_intent == "latest":
            if meta.is_stale_marked:
                score -= 8
            if meta.is_fresh_marked:
                score += 4

        return score

    def _extract_tool_meta(self, call: ToolCallRecord) -> ToolEvidenceMeta:
        text = " ".join([call.tool_name, call.input_summary, call.output_summary])
        return ToolEvidenceMeta(
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            tool_type=self._infer_tool_type(call.tool_name),
            locations=self._extract_locations(text),
            targets=self._extract_targets(text),
            is_fresh_marked=any(
                marker in text
                for marker in [
                    "fresh_marker",
                    "新参数",
                    "新地点",
                    "新结果",
                    "更新",
                    "latest",
                    "new_",
                ]
            ),
            is_stale_marked=any(
                marker in text
                for marker in [
                    "stale_marker",
                    "旧参数",
                    "旧地点",
                    "旧结果",
                    "旧 ",
                    "old_",
                ]
            ),
            timestamp=float(call.timestamp),
        )

    def _extract_focus(self, query: str, task_state: TaskState) -> RetrievalFocus:
        fields = [
            query or "",
            task_state.current_goal or "",
            task_state.next_action or "",
            *list(task_state.active_constraints or []),
        ]
        positive_fields = [
            field for field in fields if not self._is_negative_constraint_text(field)
        ]
        focus_text = "\n".join(positive_fields)
        all_text = "\n".join(fields)
        lower = all_text.lower()

        preferred_tool_types = set()
        type_keywords = {
            "weather": ["天气", "云量", "湿度", "透明度", "气象"],
            "position": ["高度", "升起", "落下", "位置"],
            "photo": ["曝光", "ISO", "拍摄", "摄影", "参数"],
            "event": ["流星雨", "天象", "什么时候看", "观测窗口", "峰值"],
            "neo": ["小行星", "NEO", "neo", "近地"],
        }
        for tool_type, keywords in type_keywords.items():
            if any(keyword in focus_text for keyword in keywords):
                preferred_tool_types.add(tool_type)

        if any(token in all_text for token in ["对比", "前后", "为什么不一样"]):
            freshness_intent = "compare"
        elif any(
            token in all_text
            for token in [
                "现在",
                "最新",
                "更新",
                "最终",
                "后面那次",
                "新结果优先",
                "新参数",
                "新地点",
                "最近一次",
            ]
        ):
            freshness_intent = "latest"
        elif any(token in all_text for token in ["之前", "旧", "原来"]):
            freshness_intent = "historical"
        else:
            freshness_intent = "neutral"

        if "neo" in lower:
            preferred_tool_types.add("neo")

        return RetrievalFocus(
            locations=self._extract_focus_entities(
                [query or "", task_state.next_action or "", task_state.current_goal or ""]
                + list(task_state.active_constraints or []),
                self._extract_locations,
            ),
            targets=self._extract_focus_entities(
                [query or "", task_state.next_action or "", task_state.current_goal or ""]
                + list(task_state.active_constraints or []),
                self._extract_targets,
            ),
            preferred_tool_types=preferred_tool_types,
            freshness_intent=freshness_intent,
        )

    def _extract_focus_entities(self, fields, extractor) -> set[str]:
        for field in fields:
            if self._is_negative_constraint_text(field):
                continue
            entities = extractor(field)
            if entities:
                return entities
        return set()

    def _is_negative_constraint_text(self, text: str) -> bool:
        return any(
            cue in (text or "")
            for cue in [
                "不要",
                "排除",
                "不应",
                "不能",
                "别说",
                "别管",
                "先别",
                "不是",
                "冲突",
            ]
        )

    def _infer_tool_type(self, tool_name: str) -> str:
        name = (tool_name or "").lower()
        if any(token in name for token in ["weather", "天气"]):
            return "weather"
        if any(token in name for token in ["celestial-position", "position", "位置"]):
            return "position"
        if any(
            token in name
            for token in [
                "astrophotography",
                "astrophoto",
                "photo",
                "exposure",
                "calculator",
            ]
        ):
            return "photo"
        if any(token in name for token in ["neo", "asteroid", "小行星", "近地"]):
            return "neo"
        if any(
            token in name
            for token in ["event", "forecast", "calendar", "meteor", "天象", "流星雨"]
        ):
            return "event"
        return "generic"

    def _extract_locations(self, text: str) -> set[str]:
        known_locations = [
            "北京",
            "上海",
            "广州",
            "深圳",
            "杭州",
            "苏州",
            "成都",
            "南京",
            "武汉",
        ]
        return {location for location in known_locations if location in (text or "")}

    def _extract_targets(self, text: str) -> set[str]:
        source = text or ""
        targets = set(re.findall(r"\bM\d+\b", source, flags=re.IGNORECASE))
        targets = {target.upper() for target in targets}

        aliases = {
            "M42": ["猎户座大星云", "猎户座星云", "Orion Nebula"],
            "M31": ["仙女座星系", "仙女座大星系", "Andromeda"],
            "木星": ["木星", "Jupiter"],
            "土星": ["土星", "Saturn"],
            "月球": ["月球", "Moon"],
            "英仙座流星雨": ["英仙座流星雨", "Perseids"],
            "双子座流星雨": ["双子座流星雨", "Geminids"],
        }
        for canonical, names in aliases.items():
            if any(name in source for name in names):
                targets.add(canonical)

        for match in re.findall(r"\b20\d{2}\s*[A-Z]{1,3}\d+\b", source):
            targets.add(re.sub(r"\s+", " ", match).strip())
        return targets

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
