"""短期记忆上下文检索规划器。

该模块实现 select_strategy 中短期上下文和工具证据选择策略：根据 query、
任务状态、摘要快照、最近消息、事实和工具证据做多信号召回、场景自适应
打分、多样性去冗和预算装配，最终生成可追踪的 prompt 上下文与
retrieval_plan。
"""

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Sequence

from src.memory.core.models import Message, SalientFact, ToolCallRecord
from src.memory.domain.summary_snapshot import SummarySnapshot
from src.memory.domain.task_state import TaskState

ROLE_LABELS = {"user": "用户", "assistant": "助手", "system": "系统", "tool": "工具"}

TOOL_TTL_SECONDS = {
    "visibility": 1 * 60 * 60,
    "weather": 6 * 60 * 60,
    "position": 24 * 60 * 60,
    "ephemeris": 24 * 60 * 60,
    "neo": 12 * 60 * 60,
    "event": 24 * 60 * 60,
    "catalog": 0,
    "generic": 24 * 60 * 60,
}

TOOL_SCENE_WEIGHTS = {
    "observation": {
        "loc": 0.18,
        "tgt": 0.10,
        "tool": 0.10,
        "fresh": 0.32,
        "query": 0.12,
        "success": 0.08,
        "error": 0.10,
        "superseded": 0.50,
    },
    "computation": {
        "loc": 0.08,
        "tgt": 0.36,
        "tool": 0.14,
        "fresh": 0.14,
        "query": 0.10,
        "success": 0.08,
        "error": 0.10,
        "superseded": 0.45,
    },
    "learning_qa": {
        "loc": 0.08,
        "tgt": 0.12,
        "tool": 0.08,
        "fresh": 0.08,
        "query": 0.46,
        "success": 0.08,
        "error": 0.10,
        "superseded": 0.35,
    },
    "debugging": {
        "loc": 0.12,
        "tgt": 0.12,
        "tool": 0.14,
        "fresh": 0.14,
        "query": 0.12,
        "success": 0.08,
        "error": 0.28,
        "superseded": 0.30,
    },
    "general": {
        "loc": 0.15,
        "tgt": 0.15,
        "tool": 0.12,
        "fresh": 0.15,
        "query": 0.25,
        "success": 0.08,
        "error": 0.10,
        "superseded": 0.40,
    },
}


class ContextScene(str, Enum):
    """上下文装配场景，用于选择 section 配额和候选上限。"""

    OBSERVATION = "observation"
    COMPUTATION = "computation"
    LEARNING_QA = "learning_qa"
    DEBUGGING = "debugging"
    GENERAL = "general"


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
    context_scene: str = ContextScene.GENERAL.value
    section_budgets: Dict[str, int] = field(default_factory=dict)
    omitted_counts: Dict[str, int] = field(default_factory=dict)
    downgrade_steps: list[str] = field(default_factory=list)
    focus_stack: list[Dict[str, Any]] = field(default_factory=list)
    context_pressure: float = 0.0
    summary_needed: bool = False


@dataclass
class ToolEvidenceMeta:
    """工具证据的轻量元信息，用于 freshness、事件链和焦点匹配。"""

    tool_call_id: str
    tool_name: str
    tool_type: str
    locations: set[str]
    targets: set[str]
    is_fresh_marked: bool
    is_stale_marked: bool
    timestamp: float
    params_hash: str = ""
    produced_at: float = 0.0
    effective_until: float = 0.0
    superseded_by: str = ""
    supersedes_tool_call_ids: list[str] = field(default_factory=list)
    fresh_score: float = 0.0
    expired: bool = False
    query_relevance: float = 0.0
    error_signal: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalFocus:
    """从 query、任务状态和最近用户消息中抽取出的检索焦点。"""

    locations: set[str]
    targets: set[str]
    preferred_tool_types: set[str]
    freshness_intent: str
    boosted_locations: set[str] = field(default_factory=set)
    boosted_targets: set[str] = field(default_factory=set)
    drifted: bool = False


@dataclass
class ContextCandidate:
    """统一表示可进入 prompt 的 message/fact/tool/summary 候选。"""

    candidate_id: str
    source_type: str
    section: str
    text: str
    tokens: int
    score: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    payload: Any = None


@dataclass
class ContextPolicy:
    """场景化上下文装配策略。"""

    scene: str
    section_ratios: Dict[str, float]
    top_k: Dict[str, int]
    mmr_lambda: float = 0.7
    similarity_threshold: float = 0.82
    max_tools: int = 5
    max_per_tool_type: int = 2
    max_per_target: int = 2
    min_section_tokens: int = 80
    downgrade_order: list[str] = field(
        default_factory=lambda: [
            "tool_detail",
            "old_messages",
            "low_score_facts",
            "compact_summary",
        ]
    )


class RetrievalPlanner:
    """Deterministic query-aware context assembler."""

    def __init__(self, token_counter, semantic_ranker=None):
        """注入 token 估算器和可选语义排序器，默认保持确定性排序。"""

        self._estimate_tokens = token_counter
        self._semantic_ranker = semantic_ranker

    def build_context(
        self,
        query: str,
        token_budget: int,
        task_state: TaskState,
        summary_snapshot: SummarySnapshot | None,
        messages: Sequence[Message],
        facts: Sequence[SalientFact],
        tool_calls: Sequence[ToolCallRecord],
        task_type: str | None = None,
        capability_hints: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """按候选召回、场景配额和 token 预算组装短期记忆上下文。"""

        query_type = self._classify_query(query)
        context_scene = self._classify_scene(
            query=query,
            task_state=task_state,
            task_type=task_type,
            capability_hints=capability_hints or {},
        )
        policy = self._policy_for_scene(context_scene)
        focus, focus_stack = self._derive_focus_stack(query, task_state, messages)
        plan = RetrievalPlan(
            query=query,
            query_type=query_type,
            token_budget=token_budget,
            context_scene=context_scene,
            focus_stack=focus_stack,
        )

        rendered_sections: list[str] = []
        selected_messages: list[Message] = []
        selected_facts: list[SalientFact] = []
        selected_tools: list[ToolCallRecord] = []
        selected_snapshot: SummarySnapshot | None = None

        task_text = self._format_task_state(task_state)
        task_tokens = 0
        if task_text:
            rendered_task = self._render_pinned_task_state(
                task_state=task_state,
                token_budget=token_budget,
                plan=plan,
            )
            if rendered_task:
                rendered_sections.append(rendered_task)
                task_tokens = self._estimate_tokens(rendered_task)
                plan.selected_task_state_version = task_state.version

        remaining_for_context = max(token_budget - task_tokens, 0)
        section_budgets = self._section_budgets(
            remaining_for_context, policy.section_ratios
        )
        plan.section_budgets = {"task_state": task_tokens, **section_budgets}

        candidates_by_section = {
            "summary": self._summary_candidates(
                summary_snapshot,
                section_budgets,
                query,
                focus,
            ),
            "facts": self._fact_candidates(query, facts, focus),
            "tools": self._tool_candidates(query, task_state, tool_calls, focus, policy),
            "messages": self._message_candidates(query, task_state, messages, focus),
        }
        plan.context_pressure = self._context_pressure(
            token_budget=token_budget,
            task_tokens=task_tokens,
            candidates_by_section=candidates_by_section,
        )

        carry = 0
        for section in ["summary", "facts", "tools", "messages"]:
            candidates = candidates_by_section[section]
            if not candidates:
                plan.omitted_counts[section] = 0
                carry += section_budgets.get(section, 0)
                continue

            available = self._remaining_budget(rendered_sections, token_budget)
            if available <= 0:
                plan.omitted_counts[section] = len(candidates)
                self._record_downgrade(plan, section, len(candidates))
                continue

            requested = section_budgets.get(section, 0) + carry
            section_budget = min(
                available,
                max(requested, min(policy.min_section_tokens, available)),
            )
            section_text, selected, omitted, used_tokens = self._assemble_section(
                section=section,
                candidates=candidates,
                token_budget=section_budget,
                policy=policy,
                plan=plan,
            )
            if section_text:
                rendered_sections.append(section_text)
                for candidate in selected:
                    if candidate.source_type == "message":
                        selected_messages.append(candidate.payload)
                    elif candidate.source_type == "fact":
                        selected_facts.append(candidate.payload)
                    elif candidate.source_type == "tool":
                        selected_tools.append(candidate.payload)
                    elif candidate.source_type == "summary":
                        selected_snapshot = candidate.payload

            plan.omitted_counts[section] = omitted
            if omitted:
                self._record_downgrade(plan, section, omitted)
            carry = max(requested - used_tokens, 0)

        selected_messages.sort(key=lambda item: item.timestamp)
        context_text = "\n\n".join(rendered_sections)
        if not context_text:
            context_text = "无对话上下文"

        plan.selected_message_ids = [msg.message_id for msg in selected_messages]
        plan.selected_fact_ids = [fact.fact_id for fact in selected_facts]
        plan.selected_tool_call_ids = [call.tool_call_id for call in selected_tools]
        plan.selected_snapshot_id = (
            selected_snapshot.snapshot_id if selected_snapshot else ""
        )
        plan.summary_needed = self._summary_needed(plan)

        return {
            "context_text": context_text,
            "query_type": query_type,
            "retrieval_plan": plan.__dict__,
            "total_tokens": self._estimate_tokens(context_text),
            "selected_recent_messages": [msg.to_dict() for msg in selected_messages],
            "selected_salient_facts": [fact.to_dict() for fact in selected_facts],
            "selected_tool_calls": [call.to_dict() for call in selected_tools],
            "selected_summary_snapshot": (
                selected_snapshot.to_dict() if selected_snapshot else None
            ),
            "selected_task_state": task_state.to_dict(),
            "context_scene": context_scene,
            "section_budgets": dict(plan.section_budgets),
            "omitted_counts": dict(plan.omitted_counts),
            "context_pressure": plan.context_pressure,
            "summary_needed": plan.summary_needed,
            "built_at": time.time(),
        }

    def _classify_query(self, query: str) -> str:
        """粗分 query 类型，用于调试 retrieval_plan 和后续策略扩展。"""

        lower = (query or "").lower()
        if any(token in lower for token in ["tool", "工具", "结果", "证据", "输出"]):
            return "evidence"
        if any(token in lower for token in ["next", "下一步", "计划", "进度", "todo"]):
            return "task_progress"
        if any(token in lower for token in ["why", "为什么", "原因"]):
            return "reasoning"
        return "general"

    def _classify_scene(
        self,
        query: str,
        task_state: TaskState,
        task_type: str | None,
        capability_hints: Dict[str, Any],
    ) -> str:
        """根据 query、任务状态和能力提示推断上下文装配场景。"""

        if task_type in {scene.value for scene in ContextScene}:
            return str(task_type)
        hint_scene = capability_hints.get("context_scene") or capability_hints.get(
            "task_type"
        )
        if hint_scene in {scene.value for scene in ContextScene}:
            return str(hint_scene)

        text = "\n".join(
            [
                query or "",
                task_state.current_goal or "",
                task_state.next_action or "",
                " ".join(task_state.active_constraints or []),
            ]
        )
        lower = text.lower()
        if any(
            token in lower
            for token in ["bug", "debug", "traceback", "exception", "报错", "失败", "修复"]
        ):
            return ContextScene.DEBUGGING.value
        if any(
            token in text
            for token in ["计算", "参数", "曝光", "角度", "公式", "推导", "估算"]
        ) or any(token in lower for token in ["calculate", "compute", "formula"]):
            return ContextScene.COMPUTATION.value
        if any(
            token in text
            for token in ["什么是", "解释", "学习", "为什么", "原理", "教程"]
        ) or any(token in lower for token in ["explain", "learn", "why"]):
            return ContextScene.LEARNING_QA.value
        if any(
            token in text
            for token in [
                "观测",
                "天气",
                "云量",
                "湿度",
                "透明度",
                "高度",
                "升起",
                "落下",
                "流星雨",
                "天象",
            ]
        ):
            return ContextScene.OBSERVATION.value
        return ContextScene.GENERAL.value

    def _policy_for_scene(self, scene: str) -> ContextPolicy:
        """返回指定场景的 section token 配额和候选选择策略。"""

        ratios = {
            ContextScene.OBSERVATION.value: {
                "summary": 0.10,
                "facts": 0.15,
                "tools": 0.50,
                "messages": 0.25,
            },
            ContextScene.COMPUTATION.value: {
                "summary": 0.15,
                "facts": 0.35,
                "tools": 0.35,
                "messages": 0.15,
            },
            ContextScene.LEARNING_QA.value: {
                "summary": 0.30,
                "facts": 0.30,
                "tools": 0.10,
                "messages": 0.30,
            },
            ContextScene.DEBUGGING.value: {
                "summary": 0.25,
                "facts": 0.10,
                "tools": 0.40,
                "messages": 0.25,
            },
            ContextScene.GENERAL.value: {
                "summary": 0.20,
                "facts": 0.25,
                "tools": 0.30,
                "messages": 0.25,
            },
        }
        return ContextPolicy(
            scene=scene,
            section_ratios=ratios.get(scene, ratios[ContextScene.GENERAL.value]),
            top_k={"summary": 1, "facts": 8, "tools": 5, "messages": 6},
        )

    def _section_budgets(
        self, token_budget: int, ratios: Dict[str, float]
    ) -> Dict[str, int]:
        """按比例把剩余 token budget 拆分给各上下文 section。"""

        return {
            section: max(0, int(token_budget * ratio))
            for section, ratio in ratios.items()
        }

    def _summary_candidates(
        self,
        summary_snapshot: SummarySnapshot | None,
        section_budgets: Dict[str, int],
        query: str = "",
        focus: RetrievalFocus | None = None,
    ) -> list[ContextCandidate]:
        """把 summary snapshot 转换为可参与预算装配的候选。"""

        if not summary_snapshot or not summary_snapshot.summary_text:
            return []
        budget = max(64, section_budgets.get("summary", 0))
        text, metadata = self._render_summary_snapshot(
            summary_snapshot.summary_text,
            budget=budget,
            query=query,
            focus=focus,
        )
        return [
            ContextCandidate(
                candidate_id=summary_snapshot.snapshot_id,
                source_type="summary",
                section="summary",
                text=text,
                tokens=self._estimate_tokens(text),
                score=1.0,
                timestamp=summary_snapshot.created_at,
                metadata={
                    "summary_level": summary_snapshot.summary_level,
                    **metadata,
                },
                payload=summary_snapshot,
            )
        ]

    def _render_summary_snapshot(
        self,
        summary_text: str,
        budget: int,
        query: str,
        focus: RetrievalFocus | None,
    ) -> tuple[str, Dict[str, Any]]:
        """把结构化摘要按 query/focus 渲染为可读字段文本；旧文本走截断。"""

        try:
            payload = json.loads(summary_text or "")
        except (TypeError, json.JSONDecodeError):
            return self._truncate_to_budget(summary_text, budget), {
                "structured": False,
                "selected_fields": [],
            }
        if not isinstance(payload, dict) or not any(
            field in payload
            for field in [
                "topics",
                "decisions",
                "open_questions",
                "established_facts",
                "tool_results_index",
            ]
        ):
            return self._truncate_to_budget(summary_text, budget), {
                "structured": False,
                "selected_fields": [],
            }

        selected_fields: list[str] = []
        lines: list[str] = []
        focus_terms = self._summary_focus_terms(query, focus)

        topics = self._summary_string_items(payload.get("topics", []))
        chosen_topics = self._select_summary_items(topics, focus_terms, limit=4)
        if chosen_topics:
            selected_fields.append("topics")
            lines.append("topics: " + "; ".join(chosen_topics))

        open_questions = self._summary_string_items(payload.get("open_questions", []))
        chosen_questions = self._select_summary_items(open_questions, focus_terms, limit=5)
        if chosen_questions:
            selected_fields.append("open_questions")
            lines.append("open_questions:")
            lines.extend(f"- {item}" for item in chosen_questions)

        decisions = self._summary_string_items(payload.get("decisions", []))
        chosen_decisions = self._select_summary_items(decisions, focus_terms, limit=6)
        if chosen_decisions:
            selected_fields.append("decisions")
            lines.append("decisions:")
            lines.extend(f"- {item}" for item in chosen_decisions)

        facts = self._summary_string_items(payload.get("established_facts", []))
        chosen_facts = self._select_summary_items(facts, focus_terms, limit=6)
        if chosen_facts:
            selected_fields.append("established_facts")
            lines.append("established_facts:")
            lines.extend(f"- {item}" for item in chosen_facts)

        tools = self._summary_tool_items(payload.get("tool_results_index", []))
        chosen_tools = self._select_summary_items(tools, focus_terms, limit=5)
        if chosen_tools:
            selected_fields.append("tool_results_index")
            lines.append("tool_results_index:")
            lines.extend(f"- {item}" for item in chosen_tools)

        rendered = "\n".join(lines).strip()
        if not rendered:
            rendered = self._truncate_to_budget(summary_text, budget)
        else:
            rendered = self._truncate_to_budget(rendered, budget)
        return rendered, {
            "structured": True,
            "selected_fields": selected_fields,
        }

    def _summary_focus_terms(
        self,
        query: str,
        focus: RetrievalFocus | None,
    ) -> set[str]:
        """生成结构化摘要字段选择用的 query/focus 词项。"""

        terms = set(self._terms(query))
        if focus is not None:
            terms |= {item.lower() for item in focus.locations | focus.targets}
            terms |= {item.lower() for item in focus.preferred_tool_types}
        return {term for term in terms if term}

    def _summary_string_items(self, value: Any) -> list[str]:
        """把结构化摘要字段归一为字符串列表。"""

        if isinstance(value, str):
            return [value] if value else []
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _summary_tool_items(self, value: Any) -> list[str]:
        """把 tool_results_index 渲染为紧凑可读文本。"""

        if not isinstance(value, list):
            return []
        items: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool") or "tool")
            status = str(item.get("status") or "")
            params_hash = str(item.get("params_hash") or "")
            key_finding = str(item.get("key_finding") or "")
            tag_parts = [
                part
                for part in [
                    status,
                    f"params={params_hash}" if params_hash else "",
                ]
                if part
            ]
            tag = f" [{'|'.join(tag_parts)}]" if tag_parts else ""
            items.append(f"{tool}{tag}: {key_finding}".strip())
        return items

    def _select_summary_items(
        self,
        items: Sequence[str],
        focus_terms: set[str],
        limit: int,
    ) -> list[str]:
        """优先保留与 query/focus 匹配的摘要项，不足时按原顺序补齐。"""

        if limit <= 0:
            return []
        if not items:
            return []
        scored: list[tuple[int, int, str]] = []
        for index, item in enumerate(items):
            item_terms = set(self._terms(item)) | {
                term.lower() for term in focus_terms if term in item.lower()
            }
            overlap = len(item_terms & focus_terms)
            scored.append((overlap, -index, item))
        matched = [
            item
            for overlap, _, item in sorted(scored, reverse=True)
            if overlap > 0
        ]
        if len(matched) >= limit:
            return matched[:limit]
        seen = set(matched)
        for item in items:
            if item in seen:
                continue
            matched.append(item)
            seen.add(item)
            if len(matched) >= limit:
                break
        return matched[:limit]

    def _fact_candidates(
        self,
        query: str,
        facts: Sequence[SalientFact],
        focus: RetrievalFocus,
    ) -> list[ContextCandidate]:
        """通过 recent、lexical、focus 三路召回生成事实候选。"""

        texts_by_id = {
            fact.fact_id: f"{fact.fact_type} {fact.content}" for fact in facts
        }
        recent_ids = self._recent_recall_ids(
            facts,
            id_attr="fact_id",
            limit=8,
        )
        lexical_scores = self._lexical_recall_scores(query, texts_by_id)
        candidates = []
        for fact in facts:
            text = f"- [{fact.fact_type}] {fact.content}"
            focus_score = self._focus_recall_score(fact.content, focus)
            recall_sources = self._recall_sources(
                candidate_id=fact.fact_id,
                recent_ids=recent_ids,
                lexical_scores=lexical_scores,
                focus_score=focus_score,
            )
            if not recall_sources:
                continue
            score = (
                lexical_scores.get(fact.fact_id, 0.0)
                + focus_score * 0.8
                + (0.25 if "recent" in recall_sources else 0.0)
            )
            candidates.append(
                ContextCandidate(
                    candidate_id=fact.fact_id,
                    source_type="fact",
                    section="facts",
                    text=text,
                    tokens=self._estimate_tokens(text),
                    score=max(float(score), 0.05),
                    timestamp=float(fact.timestamp),
                    metadata={
                        "recall_sources": recall_sources,
                        "lexical_score": round(lexical_scores.get(fact.fact_id, 0.0), 4),
                        "focus_score": round(focus_score, 4),
                    },
                    payload=fact,
                )
            )
        return self._rank_candidates(candidates)

    def _tool_candidates(
        self,
        query: str,
        task_state: TaskState,
        tool_calls: Sequence[ToolCallRecord],
        focus: RetrievalFocus,
        policy: ContextPolicy,
    ) -> list[ContextCandidate]:
        """通过多路召回和 freshness 排序生成工具证据候选。"""

        texts_by_id = {
            call.tool_call_id: " ".join(
                [call.tool_name, call.input_summary, call.output_summary]
            )
            for call in tool_calls
        }
        recent_ids = self._recent_recall_ids(
            tool_calls,
            id_attr="tool_call_id",
            limit=max(policy.max_tools * 2, 8),
        )
        lexical_scores = self._lexical_recall_scores(query, texts_by_id)
        recall_sources_by_id: dict[str, list[str]] = {}
        recalled_calls: list[ToolCallRecord] = []
        for call in tool_calls:
            meta = self._extract_tool_meta(call)
            focus_score = self._tool_focus_recall_score(meta, focus)
            recall_sources = self._recall_sources(
                candidate_id=call.tool_call_id,
                recent_ids=recent_ids,
                lexical_scores=lexical_scores,
                focus_score=focus_score,
            )
            if not recall_sources:
                continue
            recall_sources_by_id[call.tool_call_id] = recall_sources
            recalled_calls.append(call)

        ranked_tools = self._rank_tools_with_focus(
            query, task_state, recalled_calls, focus=focus, policy=policy
        )
        candidates = []
        for call in ranked_tools:
            meta = self._extract_tool_meta(call)
            text = self._format_tool_call(call)
            focus_score = self._tool_focus_recall_score(meta, focus)
            recall_sources = recall_sources_by_id.get(call.tool_call_id, [])
            score = self._float_metadata(
                call.metadata.get("tool_score"),
                self._score_tool_with_focus(query, focus, call, meta, policy.scene),
            )
            candidates.append(
                ContextCandidate(
                    candidate_id=call.tool_call_id,
                    source_type="tool",
                    section="tools",
                    text=text,
                    tokens=self._estimate_tokens(text),
                    score=float(score),
                    timestamp=float(call.timestamp),
                    metadata={
                        "tool_type": meta.tool_type,
                        "targets": sorted(meta.targets),
                        "locations": sorted(meta.locations),
                        "params_hash": meta.params_hash,
                        "produced_at": meta.produced_at,
                        "effective_until": meta.effective_until,
                        "fresh_score": round(meta.fresh_score, 6),
                        "expired": bool(meta.expired or self._tool_is_expired(meta)),
                        "superseded_by": meta.superseded_by,
                        "query_relevance": round(meta.query_relevance, 6),
                        "error_signal": round(meta.error_signal, 6),
                        "tool_score": round(score, 6),
                        "selection_reason": call.metadata.get(
                            "selection_reason", "ranked"
                        ),
                        "recall_sources": recall_sources,
                        "lexical_score": round(
                            lexical_scores.get(call.tool_call_id, 0.0), 4
                        ),
                        "focus_score": round(focus_score, 4),
                    },
                    payload=call,
                )
            )
        return self._rank_candidates(candidates)

    def _message_candidates(
        self,
        query: str,
        task_state: TaskState,
        messages: Sequence[Message],
        focus: RetrievalFocus,
    ) -> list[ContextCandidate]:
        """通过 recent、lexical、focus 三路召回生成消息候选。"""

        texts_by_id = {message.message_id: message.content for message in messages}
        recent_ids = self._recent_recall_ids(
            messages,
            id_attr="message_id",
            limit=4,
        )
        lexical_scores = self._lexical_recall_scores(query, texts_by_id)
        candidates = []
        for msg in messages:
            if self._task_state_covers_message_noise(msg.content, task_state, focus):
                continue
            focus_score = self._focus_recall_score(msg.content, focus)
            recall_sources = self._recall_sources(
                candidate_id=msg.message_id,
                recent_ids=recent_ids,
                lexical_scores=lexical_scores,
                focus_score=focus_score,
            )
            if not recall_sources:
                continue
            text = f"{ROLE_LABELS.get(msg.role, msg.role)}: {msg.content}"
            score = (
                lexical_scores.get(msg.message_id, 0.0)
                + focus_score * 0.8
                + (0.25 if "recent" in recall_sources else 0.0)
            )
            candidates.append(
                ContextCandidate(
                    candidate_id=msg.message_id,
                    source_type="message",
                    section="messages",
                    text=text,
                    tokens=self._estimate_tokens(text),
                    score=max(float(score), 0.05),
                    timestamp=float(msg.timestamp),
                    metadata={
                        "recall_sources": recall_sources,
                        "lexical_score": round(
                            lexical_scores.get(msg.message_id, 0.0), 4
                        ),
                        "focus_score": round(focus_score, 4),
                    },
                    payload=msg,
                )
            )
        return self._rank_candidates(candidates)

    def _recent_recall_ids(
        self,
        items: Sequence[Any],
        id_attr: str,
        limit: int,
    ) -> set[str]:
        """返回按时间排序的最近候选 id 集合。"""

        if limit <= 0:
            return set()
        recent = sorted(
            items,
            key=lambda item: float(getattr(item, "timestamp", 0.0) or 0.0),
        )[-limit:]
        return {str(getattr(item, id_attr)) for item in recent if getattr(item, id_attr, "")}

    def _lexical_recall_scores(
        self,
        query: str,
        texts_by_id: Dict[str, str],
    ) -> dict[str, float]:
        """计算轻量 BM25-like 词袋召回分数，不依赖持久索引。"""

        query_terms = set(self._terms(query))
        if not query_terms or not texts_by_id:
            return {}

        doc_terms: dict[str, list[str]] = {
            item_id: self._terms(text) for item_id, text in texts_by_id.items()
        }
        doc_count = max(len(doc_terms), 1)
        document_frequency: dict[str, int] = {}
        for terms in doc_terms.values():
            for term in set(terms):
                if term in query_terms:
                    document_frequency[term] = document_frequency.get(term, 0) + 1

        scores: dict[str, float] = {}
        for item_id, terms in doc_terms.items():
            if not terms:
                continue
            term_counts: dict[str, int] = {}
            for term in terms:
                term_counts[term] = term_counts.get(term, 0) + 1
            score = 0.0
            for term in query_terms:
                tf = term_counts.get(term, 0)
                if not tf:
                    continue
                df = document_frequency.get(term, 0)
                idf = math.log((doc_count + 1) / (df + 1)) + 1.0
                score += (tf / (tf + 1.2)) * idf
            if score > 0:
                scores[item_id] = score
        return scores

    def _recall_sources(
        self,
        candidate_id: str,
        recent_ids: set[str],
        lexical_scores: dict[str, float],
        focus_score: float,
    ) -> list[str]:
        """记录候选来自 recent、lexical、focus 中的哪些召回路径。"""

        sources = []
        if candidate_id in recent_ids:
            sources.append("recent")
        if lexical_scores.get(candidate_id, 0.0) > 0:
            sources.append("lexical")
        if focus_score > 0:
            sources.append("focus")
        return sources

    def _focus_recall_score(self, text: str, focus: RetrievalFocus) -> float:
        """计算文本与当前地点、目标和工具类型焦点的匹配强度。"""

        score = 0.0
        locations = self._extract_locations(text)
        targets = self._extract_targets(text)
        if focus.locations and locations & focus.locations:
            score += 1.0
        if focus.targets and targets & focus.targets:
            score += 1.0
        if locations & focus.boosted_locations:
            score += 0.5
        if targets & focus.boosted_targets:
            score += 0.5
        for tool_type in focus.preferred_tool_types:
            if any(keyword in text for keyword in self._tool_type_keywords(tool_type)):
                score += 0.5
        return score

    def _tool_focus_recall_score(
        self,
        meta: ToolEvidenceMeta,
        focus: RetrievalFocus,
    ) -> float:
        """计算工具证据元信息与当前检索焦点的匹配强度。"""

        score = 0.0
        if focus.locations and meta.locations & focus.locations:
            score += 1.0
        if focus.targets and meta.targets & focus.targets:
            score += 1.0
        if focus.boosted_locations and meta.locations & focus.boosted_locations:
            score += 0.5
        if focus.boosted_targets and meta.targets & focus.boosted_targets:
            score += 0.5
        if focus.preferred_tool_types and meta.tool_type in focus.preferred_tool_types:
            score += 1.0
        return score

    def _tool_type_keywords(self, tool_type: str) -> list[str]:
        """返回工具类型对应的中文/英文关键词，用于 focus 召回。"""

        return {
            "visibility": ["透明度", "能见度", "视宁度", "seeing"],
            "weather": ["天气", "云量", "湿度", "透明度"],
            "position": ["高度", "升起", "落下", "位置"],
            "ephemeris": ["星历", "高度", "赤经", "赤纬"],
            "photo": ["曝光", "ISO", "拍摄", "摄影", "参数"],
            "event": ["流星雨", "天象", "观测窗口", "峰值"],
            "neo": ["小行星", "NEO", "近地"],
            "catalog": ["星表", "目录", "catalog", "simbad"],
        }.get(tool_type, [])

    def _normalize_candidate_scores(
        self, candidates: Sequence[ContextCandidate]
    ) -> list[ContextCandidate]:
        """把候选原始分数归一到可跨来源比较的范围。"""

        normalized = list(candidates)
        if not normalized:
            return normalized
        raw_scores = [candidate.score for candidate in normalized]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        timestamps = [candidate.timestamp for candidate in normalized]
        min_timestamp = min(timestamps)
        max_timestamp = max(timestamps)

        for candidate in normalized:
            if max_score == min_score:
                relevance = 1.0
            else:
                relevance = (candidate.score - min_score) / (max_score - min_score)
            if max_timestamp == min_timestamp:
                recency = 0.0
            else:
                recency = (candidate.timestamp - min_timestamp) / (
                    max_timestamp - min_timestamp
                )
            source_bonus = 0.04 * len(candidate.metadata.get("recall_sources", []))
            candidate.score = round(0.82 * relevance + 0.14 * recency + source_bonus, 6)
        return normalized

    def _rank_candidates(
        self, candidates: Sequence[ContextCandidate]
    ) -> list[ContextCandidate]:
        """按归一化分数和时间排序候选，并保留语义 ranker 扩展点。"""

        candidates = self._normalize_candidate_scores(candidates)
        ranked = sorted(
            candidates,
            key=lambda item: (item.score, item.timestamp),
            reverse=True,
        )
        if self._semantic_ranker is None:
            return ranked
        return self._semantic_ranker.rank(ranked)

    def _assemble_section(
        self,
        section: str,
        candidates: Sequence[ContextCandidate],
        token_budget: int,
        policy: ContextPolicy,
        plan: RetrievalPlan,
    ) -> tuple[str, list[ContextCandidate], int, int]:
        """在 section budget 内用 MMR 选择候选并渲染该 section。"""

        mmr_candidates = self._mmr_select(
            list(candidates),
            limit=policy.top_k.get(section, len(candidates)),
            lambda_value=policy.mmr_lambda,
            similarity_threshold=policy.similarity_threshold,
        )
        title = self._section_title(section)
        selected: list[ContextCandidate] = []
        omitted = len(candidates) - len(mmr_candidates)

        for candidate in mmr_candidates:
            trial = selected + [candidate]
            body = self._section_body(section, trial)
            section_text = f"=== {title} ===\n{body}"
            if self._estimate_tokens(section_text) <= token_budget:
                selected.append(candidate)
                continue
            omitted += 1

        if not selected and section == "summary" and mmr_candidates:
            compact = self._fit_body_to_section_budget(
                title=title,
                body=mmr_candidates[0].text,
                token_budget=token_budget,
            )
            if compact:
                selected = [
                    ContextCandidate(
                        **{
                            **mmr_candidates[0].__dict__,
                            "text": compact,
                            "tokens": self._estimate_tokens(compact),
                        }
                    )
                ]
                omitted = max(0, len(candidates) - 1)
                plan.downgrade_steps.append("compact_summary")

        if not selected:
            placeholder = self._omitted_placeholder(section, omitted)
            section_text = f"=== {title} ===\n{placeholder}"
            if omitted and self._estimate_tokens(section_text) <= token_budget:
                return section_text, [], omitted, self._estimate_tokens(section_text)
            return "", [], omitted, 0

        body = self._section_body(section, selected)
        if omitted:
            placeholder = self._omitted_placeholder(section, omitted)
            trial_body = f"{body}\n{placeholder}"
            trial_section = f"=== {title} ===\n{trial_body}"
            if self._estimate_tokens(trial_section) <= token_budget:
                body = trial_body
        section_text = f"=== {title} ===\n{body}"
        return section_text, selected, omitted, self._estimate_tokens(section_text)

    def _section_body(
        self, section: str, candidates: Sequence[ContextCandidate]
    ) -> str:
        """按 section 类型把候选正文合并为可读文本。"""

        ordered = list(candidates)
        if section == "messages":
            ordered.sort(key=lambda item: item.timestamp)
        return "\n".join(candidate.text for candidate in ordered if candidate.text)

    def _section_title(self, section: str) -> str:
        """把内部 section key 转换为 prompt 中的标题。"""

        return {
            "summary": "summary snapshot",
            "facts": "relevant facts",
            "tools": "tool evidence",
            "messages": "recent/relevant messages",
        }.get(section, section)

    def _fit_body_to_section_budget(
        self, title: str, body: str, token_budget: int
    ) -> str:
        """把单个 section body 截断到可容纳的 token 预算内。"""

        if token_budget <= 0:
            return ""
        prefix = f"=== {title} ===\n"
        available = max(token_budget - self._estimate_tokens(prefix), 0)
        if available <= 0:
            return ""
        return self._truncate_to_budget(body, available)

    def _render_pinned_task_state(
        self,
        task_state: TaskState,
        token_budget: int,
        plan: RetrievalPlan,
    ) -> str:
        """优先渲染任务状态，并在预算不足时降级为紧凑形式。"""

        title = "task state"
        body = self._format_task_state(task_state)
        section = f"=== {title} ===\n{body}"
        if self._estimate_tokens(section) <= token_budget:
            return section

        compact = self._format_task_state(task_state, compact=True)
        compact_section = f"=== {title} ===\n{compact}"
        if self._estimate_tokens(compact_section) <= token_budget:
            plan.downgrade_steps.append("compact_task_state")
            return compact_section

        compact_body = self._fit_body_to_section_budget(
            title=title,
            body=compact,
            token_budget=token_budget,
        )
        if compact_body:
            plan.downgrade_steps.append("compact_task_state")
            return f"=== {title} ===\n{compact_body}"
        return ""

    def _remaining_budget(self, rendered_sections: Sequence[str], token_budget: int) -> int:
        """计算已渲染 section 之后仍可使用的 token budget。"""

        return max(token_budget - self._estimate_tokens("\n\n".join(rendered_sections)), 0)

    def _context_pressure(
        self,
        token_budget: int,
        task_tokens: int,
        candidates_by_section: Dict[str, Sequence[ContextCandidate]],
    ) -> float:
        """估算候选总量相对 token budget 的上下文压力。"""

        if token_budget <= 0:
            return 0.0
        candidate_tokens = task_tokens + sum(
            candidate.tokens
            for candidates in candidates_by_section.values()
            for candidate in candidates
        )
        return round(candidate_tokens / max(token_budget, 1), 3)

    def _summary_needed(self, plan: RetrievalPlan) -> bool:
        """根据上下文压力和省略数量判断是否建议创建 summary。"""

        omitted_total = sum(plan.omitted_counts.values())
        return plan.context_pressure >= 1.2 or omitted_total >= 8

    def _record_downgrade(
        self, plan: RetrievalPlan, section: str, omitted_count: int
    ) -> None:
        """在 retrieval plan 中记录因预算不足发生的降级步骤。"""

        if omitted_count <= 0:
            return
        step = {
            "tools": "tool_detail",
            "messages": "old_messages",
            "facts": "low_score_facts",
            "summary": "compact_summary",
        }.get(section)
        if step and step not in plan.downgrade_steps:
            plan.downgrade_steps.append(step)

    def _omitted_placeholder(self, section: str, omitted_count: int) -> str:
        """生成 section 内省略候选数量的可读占位提示。"""

        labels = {
            "summary": "摘要内容",
            "facts": "低分事实",
            "tools": "工具证据",
            "messages": "旧消息",
        }
        return f"- 已省略 {max(omitted_count, 0)} 条{labels.get(section, '候选')}"

    def _mmr_select(
        self,
        candidates: list[ContextCandidate],
        limit: int,
        lambda_value: float,
        similarity_threshold: float,
    ) -> list[ContextCandidate]:
        """用 MMR 在相关性和多样性之间折中选择候选。"""

        if limit <= 0:
            return []
        pending = list(candidates)
        selected: list[ContextCandidate] = []
        max_score = max((candidate.score for candidate in pending), default=1.0) or 1.0

        while pending and len(selected) < limit:
            best_index = 0
            best_score = float("-inf")
            for index, candidate in enumerate(pending):
                similarity = max(
                    (
                        self._candidate_similarity(candidate, other)
                        for other in selected
                    ),
                    default=0.0,
                )
                if similarity >= similarity_threshold:
                    mmr_score = -1.0
                else:
                    mmr_score = lambda_value * (
                        candidate.score / max_score
                    ) - (1 - lambda_value) * similarity
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = index
            if best_score < 0 and selected:
                break
            selected.append(pending.pop(best_index))
        return selected

    def _candidate_similarity(
        self, first: ContextCandidate, second: ContextCandidate
    ) -> float:
        """用词项 Jaccard 相似度估算两个候选的重复程度。"""

        first_terms = set(self._terms(first.text))
        second_terms = set(self._terms(second.text))
        if not first_terms or not second_terms:
            return 0.0
        return len(first_terms & second_terms) / len(first_terms | second_terms)

    def _rank_messages(self, query: str, messages: Sequence[Message]) -> list[Message]:
        """按 query 词项重叠和时间对消息做旧版排序。"""

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
        focus: RetrievalFocus | None = None,
    ) -> list[Message]:
        """结合最近性、query 重叠和任务焦点选择消息。"""

        focus = focus or self._extract_focus(query, task_state)
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
        """按地点、目标和工具类型焦点评估消息相关性。"""

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
        if locations & focus.boosted_locations:
            score += 1
        if targets & focus.boosted_targets:
            score += 1

        type_keywords = {
            "visibility": ["透明度", "能见度", "视宁度", "seeing"],
            "weather": ["天气", "云量", "湿度", "透明度"],
            "position": ["高度", "升起", "落下", "位置"],
            "ephemeris": ["星历", "赤经", "赤纬"],
            "photo": ["曝光", "ISO", "拍摄", "摄影", "参数"],
            "event": ["流星雨", "天象", "观测窗口", "峰值"],
            "neo": ["小行星", "NEO", "近地"],
            "catalog": ["星表", "目录", "catalog", "simbad"],
        }
        for tool_type in focus.preferred_tool_types:
            if any(keyword in text for keyword in type_keywords.get(tool_type, [])):
                score += 1
        return score

    def _score_entity_focus(self, text: str, focus: RetrievalFocus) -> float:
        """按实体焦点为事实或消息提供加权/惩罚分。"""

        score = 0.0
        locations = self._extract_locations(text)
        targets = self._extract_targets(text)
        if focus.locations:
            score += 2.0 if locations & focus.locations else 0.0
            if locations and not locations & focus.locations:
                score -= 2.0
        if focus.targets:
            score += 2.0 if targets & focus.targets else 0.0
            if targets and not targets & focus.targets:
                score -= 2.0
        if locations & focus.boosted_locations:
            score *= 1.3
        if targets & focus.boosted_targets:
            score *= 1.3
        return score

    def _task_state_covers_message(
        self,
        message_text: str,
        task_state: TaskState,
        focus: RetrievalFocus,
    ) -> bool:
        """判断消息中的焦点实体是否已被 task state 覆盖。"""

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
        """判断消息是否是已由 task state 表达的冲突或旧上下文噪声。"""

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
        """按 query 词项重叠和时间对事实做旧版排序。"""

        return sorted(
            facts,
            key=lambda fact: (self._score(query, fact.content), fact.timestamp),
            reverse=True,
        )

    def _rank_tools(
        self, query: str, tool_calls: Sequence[ToolCallRecord]
    ) -> list[ToolCallRecord]:
        """按 query 重叠、重要性和时间对工具调用做旧版排序。"""

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
        focus: RetrievalFocus | None = None,
        policy: ContextPolicy | None = None,
    ) -> list[ToolCallRecord]:
        """按结构化信号、事件链和多样性约束选择工具证据。"""

        focus = focus or self._extract_focus(query, task_state)
        policy = policy or self._policy_for_scene(ContextScene.GENERAL.value)
        metas_by_id = self._derive_tool_evidence_metas(
            tool_calls,
            query=query,
            focus=focus,
        )
        scored = []
        for call in tool_calls:
            meta = metas_by_id.get(call.tool_call_id) or self._extract_tool_meta(call)
            if not self._tool_is_focus_eligible(meta, focus):
                continue
            details = self._tool_score_details(
                query=query,
                focus=focus,
                call=call,
                meta=meta,
                scene=policy.scene,
            )
            score = details["score"]
            if score > 0 and self._tool_has_selection_signal(details, focus, meta):
                scored.append(
                    (
                        score,
                        call.importance,
                        meta.produced_at or meta.timestamp,
                        self._with_tool_selection_metadata(call, meta, details),
                    )
                )

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
        return self._select_diverse_tool_evidence(
            deduped,
            max_tools=policy.max_tools,
            max_per_tool_type=policy.max_per_tool_type,
            max_per_target=policy.max_per_target,
            focus=focus,
        )

    def _derive_tool_evidence_metas(
        self,
        tool_calls: Sequence[ToolCallRecord],
        query: str = "",
        focus: RetrievalFocus | None = None,
    ) -> dict[str, ToolEvidenceMeta]:
        """派生 retrieval 内部 metadata，包括 freshness、显式/隐式 supersession。"""

        metas_by_id = {
            call.tool_call_id: self._extract_tool_meta(call) for call in tool_calls
        }
        calls_by_id = {call.tool_call_id: call for call in tool_calls}
        explicit_superseded: set[str] = set()
        for call in tool_calls:
            meta = metas_by_id[call.tool_call_id]
            text = " ".join([call.tool_name, call.input_summary, call.output_summary])
            meta.query_relevance = self._query_relevance(query, text)
            meta.expired = self._tool_is_expired(meta)
            meta.fresh_score = self._tool_fresh_score(meta)
            for old_id in meta.supersedes_tool_call_ids:
                old_meta = metas_by_id.get(old_id)
                if old_meta is None:
                    continue
                old_meta.superseded_by = call.tool_call_id
                explicit_superseded.add(old_id)

        groups: dict[str, list[ToolEvidenceMeta]] = {}
        for meta in metas_by_id.values():
            group_key = self._tool_evidence_group_key(meta)
            if group_key:
                groups.setdefault(group_key, []).append(meta)

        for group in groups.values():
            ordered = sorted(
                group,
                key=lambda meta: (meta.produced_at or meta.timestamp, meta.timestamp),
            )
            previous_successes: list[ToolEvidenceMeta] = []
            latest_error: ToolEvidenceMeta | None = None
            for meta in ordered:
                call = calls_by_id.get(meta.tool_call_id)
                if call is None:
                    continue
                if call.success:
                    for old_meta in previous_successes:
                        if old_meta.tool_call_id not in explicit_superseded:
                            old_meta.superseded_by = meta.tool_call_id
                    previous_successes.append(meta)
                elif latest_error is None or (
                    meta.produced_at or meta.timestamp,
                    meta.timestamp,
                ) > (
                    latest_error.produced_at or latest_error.timestamp,
                    latest_error.timestamp,
                ):
                    latest_error = meta
            if latest_error and self._error_evidence_is_relevant(
                latest_error,
                focus,
                chain_size=len(group),
            ):
                latest_error.error_signal = 1.0

        return metas_by_id

    def _dedupe_superseded_tools(
        self,
        ranked_tools: Sequence[ToolCallRecord],
        focus: RetrievalFocus,
    ) -> list[ToolCallRecord]:
        """按工具类型和参数链保留最新成功证据，并保留代表性错误。"""

        if focus.freshness_intent in {"historical", "compare"}:
            return list(ranked_tools)

        best_success: dict[str, tuple[tuple[float, ...], int, ToolCallRecord]] = {}
        best_error: dict[str, tuple[tuple[float, ...], int, ToolCallRecord]] = {}
        passthrough: list[tuple[int, ToolCallRecord]] = []

        for index, call in enumerate(ranked_tools):
            meta = self._extract_tool_meta(call)
            if call.success and meta.is_stale_marked:
                continue

            group_key = self._tool_evidence_group_key(meta)
            if not group_key:
                if not call.success and meta.error_signal <= 0:
                    continue
                passthrough.append((index, call))
                continue

            bucket = best_success if call.success else best_error
            if call.success:
                preference = (
                    0 if self._tool_is_expired(meta) else 1,
                    1 if meta.is_fresh_marked else 0,
                    meta.produced_at or meta.timestamp,
                    meta.timestamp,
                )
            else:
                preference = (
                    meta.produced_at or meta.timestamp,
                    meta.timestamp,
                )
            current = bucket.get(group_key)
            if current is None or preference > current[0]:
                bucket[group_key] = (preference, index, call)

        kept = [(index, call) for _, index, call in best_success.values()]
        kept.extend((index, call) for _, index, call in best_error.values())
        kept.extend(passthrough)
        kept.sort(key=lambda item: item[0])
        return [call for _, call in kept]

    def _tool_evidence_group_key(self, meta: ToolEvidenceMeta) -> str:
        """按工具类型和参数链生成证据去重分组键。"""

        if meta.params_hash:
            return f"{meta.tool_type}:{meta.params_hash}"
        if meta.locations or meta.targets:
            locations = ",".join(sorted(meta.locations))
            targets = ",".join(sorted(meta.targets))
            return f"{meta.tool_type}:loc={locations}:target={targets}"
        return ""

    def _select_diverse_tool_evidence(
        self,
        ranked_tools: Sequence[ToolCallRecord],
        max_tools: int,
        max_per_tool_type: int,
        max_per_target: int,
        focus: RetrievalFocus | None = None,
    ) -> list[ToolCallRecord]:
        """限制总量、同类工具和同目标数量，并为非 latest 意图保留对照证据。"""

        selected: list[ToolCallRecord] = []
        per_type: dict[str, int] = {}
        per_target: dict[str, int] = {}
        focus = focus or RetrievalFocus(set(), set(), set(), "neutral")
        contrast = self._find_contrast_tool_evidence(ranked_tools, focus)
        for call in ranked_tools:
            if not self._tool_diversity_allows(
                call,
                per_type=per_type,
                per_target=per_target,
                max_per_tool_type=max_per_tool_type,
                max_per_target=max_per_target,
            ):
                continue
            selected.append(call)
            self._record_tool_diversity(call, per_type, per_target)
            if len(selected) >= max_tools:
                break

        if (
            contrast is not None
            and all(call.tool_call_id != contrast.tool_call_id for call in selected)
        ):
            self._inject_contrast_tool_evidence(
                selected=selected,
                contrast=contrast,
                max_tools=max_tools,
                max_per_tool_type=max_per_tool_type,
                max_per_target=max_per_target,
            )
        return selected

    def _limit_tool_evidence(
        self,
        ranked_tools: Sequence[ToolCallRecord],
        max_tools: int,
        max_per_tool_type: int,
        max_per_target: int,
    ) -> list[ToolCallRecord]:
        """兼容旧调用方的工具证据数量限制入口。"""

        return self._select_diverse_tool_evidence(
            ranked_tools,
            max_tools=max_tools,
            max_per_tool_type=max_per_tool_type,
            max_per_target=max_per_target,
            focus=RetrievalFocus(set(), set(), set(), "neutral"),
        )

    def _tool_diversity_allows(
        self,
        call: ToolCallRecord,
        per_type: dict[str, int],
        per_target: dict[str, int],
        max_per_tool_type: int,
        max_per_target: int,
    ) -> bool:
        """判断候选是否满足工具类型和目标实体多样性约束。"""

        meta = self._extract_tool_meta(call)
        if per_type.get(meta.tool_type, 0) >= max_per_tool_type:
            return False
        if meta.targets and any(
            per_target.get(target, 0) >= max_per_target for target in meta.targets
        ):
            return False
        return True

    def _record_tool_diversity(
        self,
        call: ToolCallRecord,
        per_type: dict[str, int],
        per_target: dict[str, int],
    ) -> None:
        """记录已选工具证据占用的类型和目标配额。"""

        meta = self._extract_tool_meta(call)
        per_type[meta.tool_type] = per_type.get(meta.tool_type, 0) + 1
        for target in meta.targets:
            per_target[target] = per_target.get(target, 0) + 1

    def _find_contrast_tool_evidence(
        self,
        ranked_tools: Sequence[ToolCallRecord],
        focus: RetrievalFocus,
    ) -> ToolCallRecord | None:
        """为 compare/historical/neutral 意图寻找一条合格旧证据作对照。"""

        if focus.freshness_intent == "latest":
            return None
        for call in ranked_tools:
            meta = self._extract_tool_meta(call)
            tool_score = self._float_metadata(call.metadata.get("tool_score"), 0.0)
            if tool_score < 0.25:
                continue
            if meta.expired or meta.superseded_by or meta.is_stale_marked:
                return call
        return None

    def _inject_contrast_tool_evidence(
        self,
        selected: list[ToolCallRecord],
        contrast: ToolCallRecord,
        max_tools: int,
        max_per_tool_type: int,
        max_per_target: int,
    ) -> None:
        """在不破坏硬上限的前提下，把合格对照证据放入选择结果。"""

        per_type: dict[str, int] = {}
        per_target: dict[str, int] = {}
        for call in selected:
            self._record_tool_diversity(call, per_type, per_target)
        if self._tool_diversity_allows(
            contrast,
            per_type=per_type,
            per_target=per_target,
            max_per_tool_type=max_per_tool_type,
            max_per_target=max_per_target,
        ):
            if len(selected) < max_tools:
                selected.append(contrast)
                return
            replacement_index = self._contrast_replacement_index(
                selected,
                contrast,
                max_per_tool_type=max_per_tool_type,
                max_per_target=max_per_target,
            )
            if replacement_index is not None:
                selected[replacement_index] = contrast
            return

        replacement_index = self._contrast_replacement_index(
            selected,
            contrast,
            max_per_tool_type=max_per_tool_type,
            max_per_target=max_per_target,
        )
        if replacement_index is not None:
            selected[replacement_index] = contrast

    def _contrast_replacement_index(
        self,
        selected: Sequence[ToolCallRecord],
        contrast: ToolCallRecord,
        max_per_tool_type: int,
        max_per_target: int,
    ) -> int | None:
        """寻找替换一个已选证据后仍满足多样性约束的位置。"""

        for index in range(len(selected) - 1, -1, -1):
            per_type: dict[str, int] = {}
            per_target: dict[str, int] = {}
            for selected_index, call in enumerate(selected):
                if selected_index == index:
                    continue
                self._record_tool_diversity(call, per_type, per_target)
            if self._tool_diversity_allows(
                contrast,
                per_type=per_type,
                per_target=per_target,
                max_per_tool_type=max_per_tool_type,
                max_per_target=max_per_target,
            ):
                return index
        return None

    def _score_tool_with_focus(
        self,
        query: str,
        focus: RetrievalFocus,
        call: ToolCallRecord,
        meta: ToolEvidenceMeta,
        scene: str = ContextScene.GENERAL.value,
    ) -> float:
        """按场景化权重对已归一化信号求和，返回工具证据分。"""

        return self._tool_score_details(query, focus, call, meta, scene)["score"]

    def _tool_score_details(
        self,
        query: str,
        focus: RetrievalFocus,
        call: ToolCallRecord,
        meta: ToolEvidenceMeta,
        scene: str,
    ) -> dict[str, float]:
        """返回工具证据打分细节，供排序和调试 metadata 复用。"""

        signals = self._tool_score_signals(query, focus, call, meta)
        weights = self._tool_scene_weights(scene)
        score = (
            weights["loc"] * signals["match_loc"]
            + weights["tgt"] * signals["match_tgt"]
            + weights["tool"] * signals["match_tool"]
            + weights["fresh"] * signals["fresh_score"]
            + weights["query"] * signals["query_relevance"]
            + weights["success"] * signals["success_signal"]
            + weights["error"] * signals["error_signal"]
            - weights["superseded"] * signals["superseded_penalty"]
        )
        return {**signals, "score": round(max(score, 0.0), 6)}

    def _tool_score_signals(
        self,
        query: str,
        focus: RetrievalFocus,
        call: ToolCallRecord,
        meta: ToolEvidenceMeta,
    ) -> dict[str, float]:
        """把地点、目标、工具类型、freshness、query 和状态信号归一到 [0, 1]。"""

        text = " ".join([call.tool_name, call.input_summary, call.output_summary])
        query_relevance = meta.query_relevance
        if query_relevance <= 0:
            query_relevance = self._query_relevance(query, text)
        fresh_score = meta.fresh_score or self._tool_fresh_score(meta)
        expired = meta.expired or self._tool_is_expired(meta)
        if expired:
            cap = 0.4 if focus.freshness_intent in {"compare", "historical"} else 0.15
            fresh_score = min(fresh_score, cap)

        superseded_penalty = 1.0 if meta.superseded_by else 0.0
        if focus.freshness_intent in {"compare", "historical"}:
            superseded_penalty = 0.0

        return {
            "match_loc": self._entity_match_signal(
                focus.locations | focus.boosted_locations,
                meta.locations,
            ),
            "match_tgt": self._entity_match_signal(
                focus.targets | focus.boosted_targets,
                meta.targets,
            ),
            "match_tool": 1.0
            if focus.preferred_tool_types
            and meta.tool_type in focus.preferred_tool_types
            else 0.0,
            "fresh_score": self._clamp01(fresh_score),
            "query_relevance": self._clamp01(query_relevance),
            "success_signal": 1.0 if call.success else 0.0,
            "error_signal": self._clamp01(meta.error_signal if not call.success else 0.0),
            "superseded_penalty": superseded_penalty,
        }

    def _tool_scene_weights(self, scene: str) -> dict[str, float]:
        """返回工具证据场景化权重表，未知场景回退 general。"""

        return TOOL_SCENE_WEIGHTS.get(scene, TOOL_SCENE_WEIGHTS[ContextScene.GENERAL.value])

    def _entity_match_signal(self, focus_entities: set[str], meta_entities: set[str]) -> float:
        """计算焦点实体与证据实体的二值匹配信号。"""

        if not focus_entities:
            return 0.0
        return 1.0 if meta_entities & focus_entities else 0.0

    def _query_relevance(self, query: str, text: str) -> float:
        """用确定性词项重叠近似 semantic_sim，结果归一到 [0, 1]。"""

        query_terms = set(self._terms(query))
        if not query_terms:
            return 0.0
        text_terms = set(self._terms(text))
        if not text_terms:
            return 0.0
        return min(1.0, len(query_terms & text_terms) / len(query_terms))

    def _clamp01(self, value: float) -> float:
        """把浮点信号限制到 [0, 1]。"""

        return max(0.0, min(1.0, float(value or 0.0)))

    def _tool_has_selection_signal(
        self,
        details: dict[str, float],
        focus: RetrievalFocus,
        meta: ToolEvidenceMeta,
    ) -> bool:
        """避免仅因 success 常量分选入完全无关的工具证据。"""

        if any(
            details.get(key, 0.0) > 0
            for key in [
                "match_loc",
                "match_tgt",
                "match_tool",
                "query_relevance",
                "error_signal",
            ]
        ):
            return True
        return focus.freshness_intent in {"latest", "compare", "historical"} and (
            meta.is_fresh_marked
            or meta.is_stale_marked
            or bool(meta.superseded_by)
            or meta.expired
        )

    def _tool_is_focus_eligible(
        self,
        meta: ToolEvidenceMeta,
        focus: RetrievalFocus,
    ) -> bool:
        """用硬约束排除明确属于其他地点、目标或工具类型的证据。"""

        if focus.locations and meta.locations and not (meta.locations & focus.locations):
            return False
        if focus.targets and meta.targets and not (meta.targets & focus.targets):
            return False
        if (
            focus.preferred_tool_types
            and meta.tool_type not in focus.preferred_tool_types
            and meta.tool_type != "generic"
        ):
            return False
        return True

    def _with_tool_selection_metadata(
        self,
        call: ToolCallRecord,
        meta: ToolEvidenceMeta,
        details: dict[str, float],
    ) -> ToolCallRecord:
        """返回带 retrieval 调试 metadata 的工具调用副本，不回写历史事件。"""

        metadata = dict(call.metadata or {})
        metadata.update(
            {
                "tool_type": meta.tool_type,
                "params_hash": meta.params_hash,
                "produced_at": meta.produced_at,
                "effective_until": meta.effective_until,
                "fresh_score": round(details["fresh_score"], 6),
                "expired": bool(meta.expired or self._tool_is_expired(meta)),
                "superseded_by": meta.superseded_by,
                "query_relevance": round(details["query_relevance"], 6),
                "error_signal": round(details["error_signal"], 6),
                "tool_score": round(details["score"], 6),
                "selection_reason": self._tool_selection_reason(meta, details),
            }
        )
        if meta.supersedes_tool_call_ids:
            metadata["supersedes_tool_call_ids"] = list(meta.supersedes_tool_call_ids)
        return replace(call, metadata=metadata)

    def _tool_selection_reason(
        self,
        meta: ToolEvidenceMeta,
        details: dict[str, float],
    ) -> str:
        """生成短调试说明，解释工具证据为何被选入候选。"""

        reasons = []
        if details.get("match_loc", 0.0) > 0:
            reasons.append("location_match")
        if details.get("match_tgt", 0.0) > 0:
            reasons.append("target_match")
        if details.get("match_tool", 0.0) > 0:
            reasons.append("tool_type_match")
        if details.get("query_relevance", 0.0) > 0:
            reasons.append("query_relevance")
        if details.get("error_signal", 0.0) > 0:
            reasons.append("representative_error")
        if meta.expired:
            reasons.append("expired")
        if meta.superseded_by:
            reasons.append("superseded")
        if details.get("fresh_score", 0.0) >= 0.75:
            reasons.append("fresh")
        return ",".join(reasons) or "ranked"

    def _error_evidence_is_relevant(
        self,
        meta: ToolEvidenceMeta,
        focus: RetrievalFocus | None,
        chain_size: int,
    ) -> bool:
        """只让同焦点或同参数链的最新失败证据携带 error signal。"""

        if chain_size > 1:
            return True
        if meta.query_relevance > 0:
            return True
        if focus is None:
            return False
        return (
            bool(focus.locations and meta.locations & focus.locations)
            or bool(focus.targets and meta.targets & focus.targets)
            or bool(
                focus.preferred_tool_types
                and meta.tool_type in focus.preferred_tool_types
            )
        )

    def _extract_tool_meta(self, call: ToolCallRecord) -> ToolEvidenceMeta:
        """从工具名称、输入摘要、输出摘要和 metadata 中抽取证据元信息。"""

        metadata = dict(call.metadata or {})
        text = " ".join([call.tool_name, call.input_summary, call.output_summary])
        tool_type = str(metadata.get("tool_type") or self._infer_tool_type(call.tool_name))
        produced_at = self._float_metadata(metadata.get("produced_at"), call.timestamp)
        effective_raw = metadata.get("effective_until")
        effective_until = self._float_metadata(effective_raw, 0.0)
        if effective_raw is None and produced_at > 0:
            ttl_seconds = self._tool_ttl_seconds(tool_type)
            effective_until = 0.0 if ttl_seconds <= 0 else produced_at + ttl_seconds
        params_hash = str(metadata.get("params_hash") or self._params_hash(call.input_summary))
        supersedes_tool_call_ids = self._string_list_metadata(
            metadata.get("supersedes_tool_call_ids")
        )
        superseded_by = str(metadata.get("superseded_by") or "")
        return ToolEvidenceMeta(
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            tool_type=tool_type,
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
                ]
            ),
            is_stale_marked=bool(metadata.get("superseded"))
            or any(
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
            params_hash=params_hash,
            produced_at=produced_at,
            effective_until=effective_until,
            superseded_by=superseded_by,
            supersedes_tool_call_ids=supersedes_tool_call_ids,
            fresh_score=self._float_metadata(metadata.get("fresh_score"), 0.0),
            expired=bool(metadata.get("expired", False)),
            query_relevance=self._float_metadata(metadata.get("query_relevance"), 0.0),
            error_signal=self._float_metadata(metadata.get("error_signal"), 0.0),
            metadata=metadata,
        )

    def _float_metadata(self, value: Any, fallback: float) -> float:
        """把 metadata 值安全转换为 float，失败时使用 fallback。"""

        try:
            return float(value)
        except (TypeError, ValueError):
            return float(fallback)

    def _string_list_metadata(self, value: Any) -> list[str]:
        """把 metadata 中的 id 列表兼容解析成字符串列表。"""

        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if item]
        return [str(value)]

    def _tool_ttl_seconds(self, tool_type: str) -> float:
        """返回工具类型对应的默认证据有效期秒数。"""

        return TOOL_TTL_SECONDS.get(tool_type, TOOL_TTL_SECONDS["generic"])

    def _tool_tau_seconds(self, tool_type: str) -> float:
        """返回 freshness 指数衰减的 tau；默认与 TTL 策略一致。"""

        return self._tool_ttl_seconds(tool_type)

    def _tool_fresh_score(
        self,
        meta: ToolEvidenceMeta,
        reference_time: float | None = None,
    ) -> float:
        """用 exp(-age/tau_tool_type) 计算结构化 freshness 分数。"""

        tau = self._tool_tau_seconds(meta.tool_type)
        if tau <= 0:
            return 1.0
        now = time.time() if reference_time is None else reference_time
        produced_at = meta.produced_at or meta.timestamp
        age = max(0.0, now - produced_at)
        return self._clamp01(math.exp(-age / tau))

    def _tool_is_expired(
        self,
        meta: ToolEvidenceMeta,
        reference_time: float | None = None,
    ) -> bool:
        """判断工具证据在给定参考时间下是否已过期。"""

        if meta.effective_until <= 0:
            return False
        now = time.time() if reference_time is None else reference_time
        return meta.effective_until < now

    def _params_hash(self, tool_input: str) -> str:
        """为工具输入生成稳定短哈希，用于识别参数链。"""

        normalized = self._normalize_tool_input(tool_input)
        if not normalized:
            return ""
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def _normalize_tool_input(self, tool_input: str) -> str:
        """规范化工具输入，JSON 使用排序键以保证哈希稳定。"""

        raw = (tool_input or "").strip()
        if not raw:
            return ""
        try:
            parsed = json.loads(raw)
        except Exception:
            return raw
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _derive_focus_stack(
        self,
        query: str,
        task_state: TaskState,
        messages: Sequence[Message],
    ) -> tuple[RetrievalFocus, list[Dict[str, Any]]]:
        """合并 query、task state 和近期用户消息形成检索焦点栈。"""

        query_focus = self._extract_focus(query, task_state)
        recent_user_messages = [
            message
            for message in sorted(messages, key=lambda item: item.timestamp)
            if message.role == "user"
        ][-5:]

        recent_locations: set[str] = set()
        recent_targets: set[str] = set()
        entity_counts: dict[str, int] = {}
        stack = [
            {
                "source": "query_task",
                "locations": sorted(query_focus.locations),
                "targets": sorted(query_focus.targets),
                "intent": query_focus.freshness_intent,
            }
        ]

        for message in recent_user_messages:
            if self._is_negative_constraint_text(message.content):
                continue
            locations = self._extract_locations(message.content)
            targets = self._extract_targets(message.content)
            recent_locations |= locations
            recent_targets |= targets
            for entity in locations | targets:
                entity_counts[entity] = entity_counts.get(entity, 0) + 1
            stack.append(
                {
                    "source": "recent_user",
                    "message_id": message.message_id,
                    "locations": sorted(locations),
                    "targets": sorted(targets),
                }
            )

        drifted = self._focus_drifted(
            query_focus.locations | query_focus.targets,
            recent_locations | recent_targets,
        )
        locations = set(query_focus.locations)
        targets = set(query_focus.targets)
        if not drifted:
            if not locations:
                locations |= recent_locations
            if not targets:
                targets |= recent_targets

        task_entities = self._extract_locations(task_state.current_goal) | self._extract_targets(
            task_state.current_goal
        )
        for entity in (query_focus.locations | query_focus.targets | task_entities):
            entity_counts[entity] = entity_counts.get(entity, 0) + 1

        boosted_locations = {
            entity
            for entity, count in entity_counts.items()
            if count >= 3 and entity in (locations | recent_locations)
        } & (locations | recent_locations)
        boosted_targets = {
            entity
            for entity, count in entity_counts.items()
            if count >= 3 and entity in (targets | recent_targets)
        } & (targets | recent_targets)
        boosted_locations &= self._extract_locations(" ".join(boosted_locations))
        boosted_targets &= self._extract_targets(" ".join(boosted_targets))

        focus = RetrievalFocus(
            locations=locations,
            targets=targets,
            preferred_tool_types=set(query_focus.preferred_tool_types),
            freshness_intent=query_focus.freshness_intent,
            boosted_locations=boosted_locations if not drifted else set(),
            boosted_targets=boosted_targets if not drifted else set(),
            drifted=drifted,
        )
        stack[0]["drifted"] = drifted
        stack[0]["boosted_locations"] = sorted(focus.boosted_locations)
        stack[0]["boosted_targets"] = sorted(focus.boosted_targets)
        return focus, stack

    def _focus_drifted(self, current_entities: set[str], previous_entities: set[str]) -> bool:
        """用实体 Jaccard distance 判断当前追问是否发生话题漂移。"""

        if not current_entities or not previous_entities:
            return False
        union = current_entities | previous_entities
        if not union:
            return False
        distance = 1 - (len(current_entities & previous_entities) / len(union))
        return distance > 0.6

    def _extract_focus(self, query: str, task_state: TaskState) -> RetrievalFocus:
        """从 query 与任务状态中抽取地点、目标、工具类型和新旧意图。"""

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
            "visibility": ["透明度", "能见度", "视宁度", "seeing"],
            "weather": ["天气", "云量", "湿度", "透明度", "气象"],
            "position": ["高度", "升起", "落下", "位置"],
            "ephemeris": ["星历", "赤经", "赤纬"],
            "photo": ["曝光", "ISO", "拍摄", "摄影", "参数"],
            "event": ["流星雨", "天象", "什么时候看", "观测窗口", "峰值"],
            "neo": ["小行星", "NEO", "neo", "近地"],
            "catalog": ["星表", "目录", "catalog", "simbad"],
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

        focus_fields = [
            query or "",
            task_state.next_action or "",
            task_state.current_goal or "",
        ] + list(task_state.active_constraints or [])
        return RetrievalFocus(
            locations=self._extract_focus_entities(
                focus_fields,
                self._extract_locations,
            ),
            targets=self._extract_focus_entities(
                focus_fields,
                self._extract_targets,
            ),
            preferred_tool_types=preferred_tool_types,
            freshness_intent=freshness_intent,
        )

    def _extract_focus_entities(self, fields, extractor) -> set[str]:
        """从正向字段中按优先级抽取第一组焦点实体。"""

        for field in fields:
            if self._is_negative_constraint_text(field):
                continue
            entities = extractor(field)
            if entities:
                return entities
        return set()

    def _is_negative_constraint_text(self, text: str) -> bool:
        """识别表达排除、冲突或否定约束的文本。"""

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
        """根据工具名称推断工具证据类型。"""

        name = (tool_name or "").lower()
        if any(token in name for token in ["visibility", "seeing", "透明度", "能见度"]):
            return "visibility"
        if any(token in name for token in ["weather", "天气"]):
            return "weather"
        if any(token in name for token in ["celestial-position", "position", "位置"]):
            return "position"
        if any(token in name for token in ["ephemeris", "星历"]):
            return "ephemeris"
        if any(token in name for token in ["catalog", "simbad", "messier", "ngc", "gaia"]):
            return "catalog"
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
        """从文本中抽取当前支持的城市地点实体。"""

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
            "西安",
        ]
        return {location for location in known_locations if location in (text or "")}

    def _extract_targets(self, text: str) -> set[str]:
        """从文本中抽取天体、梅西耶编号和天象目标实体。"""

        source = text or ""
        targets = set(re.findall(r"\bM\d+\b", source, flags=re.IGNORECASE))
        targets = {target.upper() for target in targets}

        aliases = {
            "M42": ["猎户座大星云", "猎户座星云", "Orion Nebula"],
            "M31": ["仙女座星系", "仙女座大星系", "Andromeda"],
            "木星": ["木星", "Jupiter"],
            "土星": ["土星", "Saturn"],
            "月球": ["月球", "Moon"],
            "火星": ["火星", "Mars"],
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
        """计算 query 与文本之间的去重词项重叠数量。"""

        query_terms = set(self._terms(query))
        if not query_terms:
            return 0
        text_terms = set(self._terms(text))
        return len(query_terms & text_terms)

    def _terms(self, text: str) -> list[str]:
        """把中英文混合文本切为用于召回和相似度的轻量词项。"""

        terms: list[str] = []
        for item in re.findall(r"[\w\u4e00-\u9fff]+", text or ""):
            lower = item.lower()
            if len(lower) <= 1:
                continue
            terms.append(lower)
            if re.fullmatch(r"[\u4e00-\u9fff]+", item) and len(item) > 2:
                terms.extend(item[index : index + 2] for index in range(len(item) - 1))
        return terms

    def _format_task_state(self, state: TaskState, compact: bool = False) -> str:
        """把结构化任务状态渲染为 prompt 中的短文本区块。"""

        if compact:
            parts = []
            if state.current_goal:
                parts.append(f"current_goal: {state.current_goal}")
            if state.next_action:
                parts.append(f"next_action: {state.next_action}")
            if state.active_constraints:
                parts.append("active_constraints: " + "; ".join(state.active_constraints[:3]))
            if not parts and state.status and state.status != "active":
                parts.append(f"status: {state.status}")
            return " | ".join(parts)

        parts = []
        has_task_content = bool(
            state.current_goal
            or state.active_constraints
            or state.pending_steps
            or state.completed_steps
            or state.open_questions
            or state.assumptions
            or state.blockers
            or state.next_action
        )
        if state.status and (state.status != "active" or has_task_content):
            parts.append(f"status: {state.status}")
        if state.current_goal:
            parts.append(f"current_goal: {state.current_goal}")
        if state.active_constraints:
            parts.append(
                "active_constraints: " + "; ".join(state.active_constraints[:8])
            )
        if state.pending_steps:
            parts.append("pending_steps: " + "; ".join(state.pending_steps[:8]))
        if state.completed_steps:
            parts.append("completed_steps: " + "; ".join(state.completed_steps[:8]))
        if state.open_questions:
            parts.append("open_questions: " + "; ".join(state.open_questions[:6]))
        if state.assumptions:
            parts.append("assumptions: " + "; ".join(state.assumptions[:6]))
        if state.blockers:
            parts.append("blockers: " + "; ".join(state.blockers[:5]))
        if state.next_action:
            parts.append(f"next_action: {state.next_action}")
        return "\n".join(parts)

    def _format_tool_call(self, call: ToolCallRecord) -> str:
        """把工具调用证据渲染为 prompt 中的一行文本。"""

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
        """兼容旧测试的区块装配助手。"""

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
        """优先按行截断文本，避免把长区块硬切到不可读。"""

        if token_budget <= 0:
            return ""
        if self._estimate_tokens(text) <= token_budget:
            return text
        kept = []
        for line in text.splitlines():
            tentative = "\n".join(kept + [line])
            if self._estimate_tokens(tentative) > token_budget:
                break
            kept.append(line)
        if kept:
            return "\n".join(kept)
        low, high = 0, len(text)
        best = ""
        while low <= high:
            middle = (low + high) // 2
            candidate = text[:middle]
            if self._estimate_tokens(candidate) <= token_budget:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        return best
