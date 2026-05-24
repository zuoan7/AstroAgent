"""长期记忆 prompt 注入器。

本文件负责把长期记忆检索结果变成可注入 prompt 的上下文：先调用规则+
语义检索，再对策略 Top30 做可降级 rerank，最后在 token、类型配额和去重
约束下选择记忆，并产出 explain trace。
"""

from typing import Any, Dict, List, Optional

from src.memory.long_term_memory.models import (
    MemoryEvent,
    MemoryItem,
    MemoryType,
)
from src.memory.long_term_memory.embedding import MemoryEmbeddingService
from src.memory.long_term_memory.profile_projection import ProfileProjection
from src.memory.long_term_memory.repository import LongTermMemoryRepository
from src.memory.long_term_memory.retrieval import (
    LongTermMemoryRetriever,
    RetrievalHit,
)
from src.memory.selection_strategy_config import (
    MemorySelectionStrategyConfig,
    get_memory_selection_strategy_config,
)
from src.memory.task_context import TaskContextProfile, coerce_task_context_profile
from src.rag.reranker import DashScopeReranker

_UNSET = object()
_OBSERVATION_CRITICAL_MEMORY_KEYS = {
    "location",
    "location_info",
    "device_info",
    "equipment",
    "skill_level",
    "timezone",
    "unit_preference",
}


class PromptInjector:
    """选择并格式化可注入 prompt 的长期记忆。"""

    def __init__(
        self,
        repository: LongTermMemoryRepository,
        max_prompt_tokens: Any = _UNSET,
        max_memories: Any = _UNSET,
        relevance_threshold: Any = _UNSET,
        preference_weight: Any = 1.0,
        habit_weight: Any = 0.7,
        constraint_weight: Any = 1.2,
        background_weight: Any = 0.8,
        fact_weight: Any = 0.9,
        embedding_service: Optional[MemoryEmbeddingService] = None,
        rerank_enabled: bool = True,
        rerank_timeout_seconds: float = 1.2,
        strategy_config: MemorySelectionStrategyConfig | None = None,
    ):
        """初始化检索器、reranker、注入预算和类型权重配置。"""

        self._repo = repository
        self._strategy_config = (
            strategy_config or get_memory_selection_strategy_config()
        )
        injection_config = self._strategy_config.long_term.injection
        self._projection = ProfileProjection(repository)
        self.max_prompt_tokens = (
            injection_config.max_prompt_tokens
            if max_prompt_tokens is _UNSET
            else int(max_prompt_tokens)
        )
        self.max_memories = (
            injection_config.max_memories
            if max_memories is _UNSET
            else int(max_memories)
        )
        self.relevance_threshold = (
            injection_config.relevance_threshold
            if relevance_threshold is _UNSET
            else float(relevance_threshold)
        )
        self.rerank_enabled = bool(rerank_enabled)
        self.rerank_timeout_seconds = max(float(rerank_timeout_seconds or 1.2), 0.1)
        self.type_weights = {
            MemoryType.PREFERENCE: preference_weight,
            MemoryType.HABIT: habit_weight,
            MemoryType.CONSTRAINT: constraint_weight,
            MemoryType.BACKGROUND: background_weight,
            MemoryType.FACT: fact_weight,
        }
        self._retriever = LongTermMemoryRetriever(
            repository,
            relevance_threshold=self.relevance_threshold,
            max_memories=self.max_memories,
            embedding_service=embedding_service,
            strategy_config=self._strategy_config,
        )
        self._reranker = self._build_reranker()
        self._last_selection_trace: Dict[str, Any] = {}

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """用轻量字符启发式估算长期记忆行的 token 占用。"""

        if not text:
            return 0
        chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.25)

    def classify_task_type(self, query: str) -> str:
        """复用长期记忆检索器的任务类型分类。"""

        return self._retriever.classify_task_type(query)

    def compute_relevance(self, item: MemoryItem, query: str, task_type: str) -> float:
        """计算单条记忆对 query 的相关性分数。"""

        score, _ = self._retriever.score(item, query, task_type)
        return score

    def get_last_selection_trace(self) -> Dict[str, Any]:
        """返回最近一次 query-aware 注入选择 trace。"""

        return dict(self._last_selection_trace or {})

    def _resolve_task_context_profile(
        self,
        query: str,
        task_type: Optional[str] = None,
        task_context_profile: Optional[Any] = None,
    ) -> TaskContextProfile:
        """把可选统一画像和旧 task_type 入参合并为长期记忆可用画像。"""

        profile = coerce_task_context_profile(
            task_context_profile,
            query=query,
            task_type=task_type,
        )
        if profile is not None:
            return profile
        return TaskContextProfile.from_memory_inputs(
            query=query,
            task_type=task_type or self.classify_task_type(query),
            context_scene=task_type,
            intent="long_term_injection",
            source="long_term_injection_rule",
        )

    def select_memories(
        self,
        user_id: str,
        query: str,
        task_type: Optional[str] = None,
        total_context_budget: Optional[int] = None,
        task_context_profile: Optional[Any] = None,
    ) -> List[MemoryItem]:
        """在数量和 token 预算内选择最相关的长期记忆。"""

        hits = self.select_memory_hits(
            user_id,
            query,
            task_type=task_type,
            total_context_budget=total_context_budget,
            task_context_profile=task_context_profile,
            include_omitted=False,
            record_access=True,
        )
        return [hit.item for hit in hits]

    def select_memory_hits(
        self,
        user_id: str,
        query: str,
        task_type: Optional[str] = None,
        total_context_budget: Optional[int] = None,
        include_omitted: bool = False,
        record_access: bool = False,
        task_context_profile: Optional[Any] = None,
    ) -> List[RetrievalHit]:
        """返回带 trace 信息的长期记忆选择结果。"""

        profile = self._resolve_task_context_profile(
            query=query,
            task_type=task_type,
            task_context_profile=task_context_profile,
        )
        resolved_task_type = profile.task_type
        memory_budget = self._effective_memory_budget(total_context_budget)
        selected, omitted = self._build_selection(
            user_id=user_id,
            query=query,
            task_type=resolved_task_type,
            memory_budget=memory_budget,
            task_context_profile=profile,
        )

        if record_access:
            for hit in selected:
                self._repo.increment_access_count(hit.item.id)

        all_hits = selected + omitted
        self._last_selection_trace = {
            "trace_version": "memory_selection_v1",
            "strategy_config_version": self._strategy_config.version,
            "task_type": resolved_task_type,
            "task_context_profile": profile.to_dict(),
            "profile": profile.to_dict(),
            "memory_budget": memory_budget,
            "selected_memory_ids": [hit.item.id for hit in selected],
            "selected_count": len(selected),
            "selected_tokens": sum(hit.token_estimate for hit in selected),
            "selected": {
                "memories": [hit.item.id for hit in selected],
            },
            "omitted": self._omitted_reason_counts(omitted),
            "fallbacks": self._trace_fallbacks(all_hits),
            "scores": {
                "selected_tokens": sum(hit.token_estimate for hit in selected),
                "relevance_threshold": self.relevance_threshold,
                "max_memories": self.max_memories,
            },
            "hits": [hit.to_trace_dict() for hit in all_hits],
        }
        return all_hits if include_omitted else selected

    def _build_selection(
        self,
        user_id: str,
        query: str,
        task_type: str,
        memory_budget: int,
        task_context_profile: Optional[TaskContextProfile] = None,
    ) -> tuple[List[RetrievalHit], List[RetrievalHit]]:
        """完成召回、rerank、阈值过滤、去重、配额和预算裁剪。"""

        hits = self._retriever.retrieve(
            user_id,
            query,
            task_type,
            limit=100,
            include_below_threshold=True,
            task_context_profile=task_context_profile,
        )
        if not hits or memory_budget <= 0:
            for hit in hits:
                self._omit(hit, "token_budget")
            return [], hits

        injection_config = self._strategy_config.long_term.injection
        strategy_hits = hits[: injection_config.rerank_top_k]
        omitted = []
        for hit in hits[injection_config.rerank_top_k :]:
            self._omit(hit, hit.omitted_reason or "rerank_limit")
            omitted.append(hit)

        strategy_hits = self._rerank_strategy_hits(query, strategy_hits)

        deduped: List[RetrievalHit] = []
        seen_keys = set()
        for hit in strategy_hits:
            hit.token_estimate = self._estimate_tokens(
                self._format_single_memory(hit.item)
            )
            threshold_score = max(
                hit.score,
                float(hit.components.get("policy_score", hit.score) or 0.0),
            )
            if threshold_score < self.relevance_threshold:
                self._omit(hit, "below_threshold")
                omitted.append(hit)
                continue
            if hit.omitted_reason == "below_threshold":
                hit.omitted_reason = None
            if self._is_unmatched_background_noise(hit, task_type):
                self._omit(hit, "background_without_query_match")
                omitted.append(hit)
                continue
            normalized_key = self._normalized_key(hit.item)
            if normalized_key in seen_keys:
                self._omit(hit, "duplicate_key")
                omitted.append(hit)
                continue
            seen_keys.add(normalized_key)
            deduped.append(hit)

        if injection_config.constraint_priority:
            top_constraint = next(
                (
                    hit
                    for hit in deduped
                    if hit.item.memory_type == MemoryType.CONSTRAINT
                ),
                None,
            )
            ordered = (
                [top_constraint] + [hit for hit in deduped if hit is not top_constraint]
                if top_constraint
                else deduped
            )
        else:
            ordered = deduped

        selected: List[RetrievalHit] = []
        type_counts: Dict[str, int] = {}
        selected_tokens = 0
        for hit in ordered:
            if len(selected) >= self.max_memories:
                self._omit(hit, "max_memories")
                omitted.append(hit)
                continue

            memory_type = hit.item.memory_type
            if type_counts.get(memory_type, 0) >= injection_config.per_type_quota:
                self._omit(hit, "type_quota")
                omitted.append(hit)
                continue

            if selected_tokens + hit.token_estimate > memory_budget:
                self._omit(hit, "token_budget")
                omitted.append(hit)
                continue

            hit.selected = True
            hit.omitted_reason = None
            selected.append(hit)
            type_counts[memory_type] = type_counts.get(memory_type, 0) + 1
            selected_tokens += hit.token_estimate

        return selected, omitted

    def _build_reranker(self) -> Optional[DashScopeReranker]:
        """创建长期记忆注入专用 reranker，失败时返回 None 降级。"""

        if not self.rerank_enabled:
            return None
        try:
            reranker = DashScopeReranker(
                top_n=self._strategy_config.long_term.injection.rerank_top_k,
                request_timeout=self.rerank_timeout_seconds,
                enabled=True,
            )
            # The compatible HTTP endpoint accepts a short requests timeout; SDK
            # calls do not expose a stable timeout contract across versions.
            if reranker.enabled:
                reranker._use_sdk = False
            return reranker
        except Exception:
            return None

    def _rerank_strategy_hits(
        self, query: str, hits: List[RetrievalHit]
    ) -> List[RetrievalHit]:
        """对策略 Top30 做可降级 rerank，失败时保持 policy score。"""

        if not hits:
            return hits
        for hit in hits:
            hit.components["policy_score"] = round(hit.score, 3)
            hit.components.setdefault("rerank_score", 0.0)
        if len(hits) == 1:
            return hits
        if not self._reranker or not self._reranker.enabled:
            for hit in hits:
                hit.reasons.append("rerank降级: disabled_or_missing_api_key")
            return hits

        documents = [self._format_single_memory(hit.item) for hit in hits]
        try:
            results = self._reranker.rerank(
                query,
                documents,
                top_n=min(
                    self._strategy_config.long_term.injection.rerank_top_k,
                    len(documents),
                ),
            )
        except Exception as exc:
            for hit in hits:
                hit.reasons.append(f"rerank降级: {type(exc).__name__}")
            return hits

        scored: List[RetrievalHit] = []
        seen_indices = set()
        for result in results:
            if result.index < 0 or result.index >= len(hits):
                continue
            rerank_score = self._clamp(float(result.relevance_score or 0.0))
            if rerank_score <= 0:
                continue
            hit = hits[result.index]
            policy_score = float(hit.components.get("policy_score", hit.score) or 0.0)
            hit.components["rerank_score"] = round(rerank_score, 3)
            injection_config = self._strategy_config.long_term.injection
            hit.score = round(
                injection_config.rerank_weight * rerank_score
                + injection_config.policy_weight * policy_score,
                3,
            )
            hit.reasons.append(f"rerank_score={round(rerank_score, 3)}")
            scored.append(hit)
            seen_indices.add(result.index)

        if not scored:
            for hit in hits:
                hit.components["rerank_score"] = 0.0
                hit.reasons.append("rerank降级: model_unavailable_or_timeout")
            return hits

        for index, hit in enumerate(hits):
            if index not in seen_indices:
                hit.reasons.append("rerank未返回，保持策略分")
                scored.append(hit)
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored

    def _format_single_memory(self, item: MemoryItem) -> str:
        """把单条长期记忆格式化为一行 prompt 文本。"""

        type_labels = {
            MemoryType.PREFERENCE: "偏好",
            MemoryType.HABIT: "习惯",
            MemoryType.CONSTRAINT: "约束",
            MemoryType.BACKGROUND: "背景",
            MemoryType.FACT: "事实",
        }
        label = type_labels.get(item.memory_type, item.memory_type)
        value_str = self._value_to_prompt_text(item.value)
        return f"- [{label}] {item.key}: {value_str}"

    def format_for_prompt(
        self,
        user_id: str,
        query: str,
        task_type: Optional[str] = None,
        total_context_budget: Optional[int] = None,
        task_context_profile: Optional[Any] = None,
    ) -> str:
        """按类型分组渲染 query-aware 长期记忆上下文。"""

        memories = self.select_memories(
            user_id,
            query,
            task_type,
            total_context_budget=total_context_budget,
            task_context_profile=task_context_profile,
        )
        if not memories:
            return "暂无用户偏好信息"

        grouped: Dict[str, List[MemoryItem]] = {}
        for item in memories:
            grouped.setdefault(item.memory_type, []).append(item)

        type_labels = {
            MemoryType.PREFERENCE: "用户偏好",
            MemoryType.HABIT: "用户习惯",
            MemoryType.CONSTRAINT: "约束条件",
            MemoryType.BACKGROUND: "用户背景",
            MemoryType.FACT: "稳定事实",
        }
        type_order = [
            MemoryType.CONSTRAINT,
            MemoryType.PREFERENCE,
            MemoryType.BACKGROUND,
            MemoryType.FACT,
            MemoryType.HABIT,
        ]

        parts = []
        for mem_type in type_order:
            items = grouped.get(mem_type)
            if not items:
                continue
            label = type_labels.get(mem_type, mem_type)
            lines = [self._format_single_memory(item) for item in items]
            parts.append(f"【{label}】\n" + "\n".join(lines))

        return "\n\n".join(parts) if parts else "暂无用户偏好信息"

    def _effective_memory_budget(
        self, total_context_budget: Optional[int] = None
    ) -> int:
        """按总上下文预算折算长期记忆可用预算，并套用上下限。"""

        if total_context_budget is None:
            return self.max_prompt_tokens
        try:
            coupled = int(
                float(total_context_budget)
                * self._strategy_config.long_term.injection.memory_budget_ratio
            )
        except (TypeError, ValueError):
            return self.max_prompt_tokens
        injection_config = self._strategy_config.long_term.injection
        coupled = max(
            injection_config.memory_budget_min,
            min(injection_config.memory_budget_max, coupled),
        )
        return min(self.max_prompt_tokens, coupled)

    def _value_to_prompt_text(self, value: Any, max_chars: int = 180) -> str:
        """把 memory value 压缩成单行 prompt 文本，避免长值撑爆预算。"""

        if isinstance(value, list):
            value_str = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            value_str = ", ".join(f"{k}: {v}" for k, v in value.items())
        else:
            value_str = "" if value is None else str(value)
        if len(value_str) > max_chars:
            return value_str[: max_chars - 3].rstrip() + "..."
        return value_str

    def _normalized_key(self, item: MemoryItem) -> str:
        """生成去重 key，优先使用业务 key，缺失时回退到类型/分类/值。"""

        key = str(item.key or "").strip().lower()
        if key:
            return key
        return f"{item.memory_type}:{item.category}:{self._value_to_prompt_text(item.value, 40)}"

    def _is_unmatched_background_noise(self, hit: RetrievalHit, task_type: str) -> bool:
        """Filter generic background memories that matched only by type prior."""

        memory_type = getattr(hit.item.memory_type, "value", hit.item.memory_type)
        if memory_type != MemoryType.BACKGROUND.value:
            return False
        if self._has_query_or_semantic_match(hit):
            return False
        if task_type == "observation" and self._is_observation_critical_key(hit.item):
            return False
        return True

    def _has_query_or_semantic_match(self, hit: RetrievalHit) -> bool:
        components = hit.components or {}
        return (
            float(components.get("query_relevance", 0.0) or 0.0) > 0.0
            or float(components.get("semantic_similarity", 0.0) or 0.0) > 0.0
        )

    def _is_observation_critical_key(self, item: MemoryItem) -> bool:
        return bool(
            {str(item.key or ""), str(item.category or "")}
            & _OBSERVATION_CRITICAL_MEMORY_KEYS
        )

    def _omit(self, hit: RetrievalHit, reason: str) -> None:
        """标记命中未入选，并记录可解释的省略原因。"""

        hit.selected = False
        hit.omitted_reason = reason

    def _omitted_reason_counts(self, hits: List[RetrievalHit]) -> Dict[str, int]:
        """统计长期记忆候选未入选原因，供统一 trace 展示。"""

        counts: Dict[str, int] = {}
        for hit in hits:
            reason = hit.omitted_reason or "unknown"
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def _trace_fallbacks(self, hits: List[RetrievalHit]) -> List[str]:
        """从命中原因中抽取语义召回和 rerank 降级说明。"""

        fallbacks: List[str] = []
        for hit in hits:
            for reason in hit.reasons:
                if "降级" not in reason and "fallback" not in reason:
                    continue
                if reason not in fallbacks:
                    fallbacks.append(reason)
        return fallbacks

    def _clamp(self, value: float) -> float:
        """把 rerank 分数限制在 0 到 1 区间。"""

        return max(0.0, min(float(value), 1.0))

    def _format_profile(self, profile: Dict[str, Any]) -> str:
        """把完整用户画像投影渲染为 prompt 文本。"""

        parts = []
        if profile.get("preferences"):
            lines = [f"- {k}: {v}" for k, v in profile["preferences"].items()]
            parts.append("【用户偏好】\n" + "\n".join(lines))
        if profile.get("habits"):
            lines = []
            for k, v in profile["habits"].items():
                if isinstance(v, list):
                    lines.append(f"- {k}: {', '.join(str(i) for i in v[:12])}")
                else:
                    lines.append(f"- {k}: {v}")
            parts.append("【用户习惯】\n" + "\n".join(lines))
        if profile.get("constraints"):
            lines = [f"- {c}" for c in profile["constraints"]]
            parts.append("【约束条件】\n" + "\n".join(lines))
        if profile.get("background"):
            lines = [f"- {k}: {v}" for k, v in profile["background"].items()]
            parts.append("【用户背景】\n" + "\n".join(lines))
        if profile.get("facts"):
            lines = [
                f"- {f.get('key', '')}: {f.get('value', '')}" for f in profile["facts"]
            ]
            parts.append("【稳定事实】\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def _select_events_for_prompt(
        self, user_id: str, task_type: Optional[str] = None
    ) -> List[MemoryEvent]:
        """选择兼容旧事件模型的高置信记忆事件。"""

        events = self._repo.get_active_events(user_id, limit=self.max_memories)
        return sorted(
            events,
            key=lambda event: (
                event.confidence,
                event.last_confirmed_at or event.created_at,
                event.created_at,
            ),
            reverse=True,
        )[: self.max_memories]

    def _format_events(self, events: List[MemoryEvent]) -> str:
        """把旧版 memory_events 渲染为 prompt 附加区块。"""

        if not events:
            return ""
        lines = [f"- {event.event_type}.{event.key}: {event.value}" for event in events]
        return "【近期记忆事件】\n" + "\n".join(lines)

    def format_profile_for_prompt(
        self, user_id: str, task_type: Optional[str] = None
    ) -> str:
        """渲染完整画像和旧版 active events，主要用于兼容路径。"""

        profile = self._projection.build(user_id)
        if not any(
            profile.get(key)
            for key in ["preferences", "habits", "constraints", "background", "facts"]
        ):
            profile = self._repo.load_profile(user_id)
            if not profile:
                return "暂无用户偏好信息"
        parts = []
        formatted_profile = self._format_profile(profile)
        if formatted_profile:
            parts.append(formatted_profile)
        formatted_events = self._format_events(
            self._select_events_for_prompt(user_id, task_type=task_type)
        )
        if formatted_events:
            parts.append(formatted_events)
        return "\n\n".join(parts) if parts else "暂无用户偏好信息"
