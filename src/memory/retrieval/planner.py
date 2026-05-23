"""短期记忆上下文检索规划器。

该模块根据 query、任务状态、摘要快照、最近消息、事实和工具证据，
在 token 预算内生成可追踪的 prompt 上下文和 retrieval_plan。
"""

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Sequence

from src.memory.core.models import Message, SalientFact, ToolCallRecord
from src.memory.domain.summary_snapshot import SummarySnapshot
from src.memory.domain.task_state import TaskState

ROLE_LABELS = {"user": "用户", "assistant": "助手", "system": "系统", "tool": "工具"}


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
    """工具证据的轻量元信息，用于新旧证据去重和焦点匹配。"""

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
            "summary": self._summary_candidates(summary_snapshot, section_budgets),
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
    ) -> list[ContextCandidate]:
        """把 summary snapshot 转换为可参与预算装配的候选。"""

        if not summary_snapshot or not summary_snapshot.summary_text:
            return []
        budget = max(64, section_budgets.get("summary", 0))
        text = self._truncate_to_budget(summary_snapshot.summary_text, budget)
        return [
            ContextCandidate(
                candidate_id=summary_snapshot.snapshot_id,
                source_type="summary",
                section="summary",
                text=text,
                tokens=self._estimate_tokens(text),
                score=1.0,
                timestamp=summary_snapshot.created_at,
                payload=summary_snapshot,
            )
        ]

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
            score = (
                self._score_tool_with_focus(query, focus, call, meta)
                + lexical_scores.get(call.tool_call_id, 0.0) * 0.25
                + focus_score * 0.2
                + (0.15 if "recent" in recall_sources else 0.0)
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
                        "expired": self._tool_is_expired(meta),
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
            "weather": ["天气", "云量", "湿度", "透明度"],
            "position": ["高度", "升起", "落下", "位置"],
            "photo": ["曝光", "ISO", "拍摄", "摄影", "参数"],
            "event": ["流星雨", "天象", "观测窗口", "峰值"],
            "neo": ["小行星", "NEO", "近地"],
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
        """优先选择与地点/目标/工具类型匹配且未被标记为旧结果的工具证据。"""

        focus = focus or self._extract_focus(query, task_state)
        policy = policy or self._policy_for_scene(ContextScene.GENERAL.value)
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
        return self._limit_tool_evidence(
            deduped,
            max_tools=policy.max_tools,
            max_per_tool_type=policy.max_per_tool_type,
            max_per_target=policy.max_per_target,
        )

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

    def _limit_tool_evidence(
        self,
        ranked_tools: Sequence[ToolCallRecord],
        max_tools: int,
        max_per_tool_type: int,
        max_per_target: int,
    ) -> list[ToolCallRecord]:
        """限制工具证据总量、每类工具数量和每个目标数量。"""

        selected: list[ToolCallRecord] = []
        per_type: dict[str, int] = {}
        per_target: dict[str, int] = {}
        for call in ranked_tools:
            meta = self._extract_tool_meta(call)
            if per_type.get(meta.tool_type, 0) >= max_per_tool_type:
                continue
            if meta.targets and any(
                per_target.get(target, 0) >= max_per_target
                for target in meta.targets
            ):
                continue
            selected.append(call)
            per_type[meta.tool_type] = per_type.get(meta.tool_type, 0) + 1
            for target in meta.targets:
                per_target[target] = per_target.get(target, 0) + 1
            if len(selected) >= max_tools:
                break
        return selected

    def _score_tool_with_focus(
        self,
        query: str,
        focus: RetrievalFocus,
        call: ToolCallRecord,
        meta: ToolEvidenceMeta,
    ) -> float:
        """给工具证据打分，地点、目标、工具类型、freshness 和状态都会影响排序。"""

        text = " ".join([call.tool_name, call.input_summary, call.output_summary])
        query_terms = set(self._terms(query))
        relevance = (
            len(query_terms & set(self._terms(text))) / len(query_terms)
            if query_terms
            else 0.0
        )
        score = relevance

        if focus.locations:
            if meta.locations & focus.locations:
                score += 0.35
            elif meta.locations:
                score -= 0.45

        if focus.targets:
            if meta.targets & focus.targets:
                score += 0.35
            elif meta.targets:
                score -= 0.45

        if focus.preferred_tool_types:
            if meta.tool_type in focus.preferred_tool_types:
                score += 0.25
            elif meta.tool_type != "generic":
                score -= 0.30

        if focus.boosted_locations and meta.locations & focus.boosted_locations:
            score *= 1.3
        if focus.boosted_targets and meta.targets & focus.boosted_targets:
            score *= 1.3

        is_expired = self._tool_is_expired(meta)
        if focus.freshness_intent == "latest":
            if meta.is_stale_marked:
                score -= 0.50
            if meta.is_fresh_marked:
                score += 0.25
            if is_expired:
                score -= 1.00
            elif meta.effective_until:
                score += 0.20
        elif focus.freshness_intent == "compare":
            if meta.is_stale_marked or meta.is_fresh_marked:
                score += 0.20
        elif focus.freshness_intent == "historical":
            if is_expired or meta.is_stale_marked:
                score += 0.10

        if score != 0:
            score += 0.05 if call.success else -0.05
        return score

    def _extract_tool_meta(self, call: ToolCallRecord) -> ToolEvidenceMeta:
        """从工具名称、输入摘要、输出摘要和 metadata 中抽取证据元信息。"""

        metadata = dict(call.metadata or {})
        text = " ".join([call.tool_name, call.input_summary, call.output_summary])
        tool_type = str(metadata.get("tool_type") or self._infer_tool_type(call.tool_name))
        produced_at = self._float_metadata(metadata.get("produced_at"), call.timestamp)
        effective_until = self._float_metadata(metadata.get("effective_until"), 0.0)
        if effective_until <= 0 and produced_at > 0:
            effective_until = produced_at + self._tool_ttl_seconds(tool_type)
        params_hash = str(metadata.get("params_hash") or self._params_hash(call.input_summary))
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
                    "latest",
                    "new_",
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
            metadata=metadata,
        )

    def _float_metadata(self, value: Any, fallback: float) -> float:
        """把 metadata 值安全转换为 float，失败时使用 fallback。"""

        try:
            return float(value)
        except (TypeError, ValueError):
            return float(fallback)

    def _tool_ttl_seconds(self, tool_type: str) -> float:
        """返回工具类型对应的默认证据有效期秒数。"""

        return {
            "weather": 6 * 60 * 60,
            "neo": 12 * 60 * 60,
            "event": 24 * 60 * 60,
            "position": 2 * 60 * 60,
            "ephemeris": 2 * 60 * 60,
        }.get(tool_type, 24 * 60 * 60)

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
