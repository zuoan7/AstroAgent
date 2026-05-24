"""Memory selection strategy configuration.

This module externalizes strategy parameters and vocabularies into an optional
YAML file while preserving code defaults as the stable fallback. Runtime rules
stay in the strategy modules; only weights, thresholds, quotas, TTLs and terms
are loaded here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from src.core.config import resolve_path, settings
from src.core.logger import logger

DEFAULT_STRATEGY_CONFIG_VERSION = "memory_selection_strategy_v1"

_SECTIONS = ["summary", "facts", "tools", "messages"]
_TOOL_WEIGHT_KEYS = [
    "loc",
    "tgt",
    "tool",
    "fresh",
    "query",
    "success",
    "error",
    "superseded",
]
_LTM_SCORE_KEYS = [
    "confidence",
    "type_weight",
    "source_bonus",
    "query_relevance",
    "semantic_similarity",
    "recency",
    "constraint_bonus",
    "stale_penalty",
]
_MEMORY_TYPES = ["preference", "habit", "constraint", "background", "fact"]
_OPEN_MAP_PATH_SUFFIXES = (
    ".short_term.tool_evidence.ttl_seconds",
)


@dataclass(frozen=True)
class ShortTermContextPolicyConfig:
    scene_section_ratios: Dict[str, Dict[str, float]]
    top_k: Dict[str, int]
    mmr_lambda: float
    similarity_threshold: float
    max_tools: int
    max_per_tool_type: int
    max_per_target: int
    min_section_tokens: int
    downgrade_order: list[str]
    summary_needed_context_pressure: float
    summary_needed_omitted_total: int


@dataclass(frozen=True)
class ToolEvidenceStrategyConfig:
    ttl_seconds: Dict[str, float]
    scene_weights: Dict[str, Dict[str, float]]
    expired_freshness_cap: float
    compare_historical_expired_freshness_cap: float
    contrast_min_score: float


@dataclass(frozen=True)
class SummaryTriggerStrategyConfig:
    snapshot_batch_size: int
    initial_uncovered_event_threshold: int
    rebase_uncovered_event_threshold: int
    uncovered_token_threshold: int
    fixed_uncovered_event_threshold: int
    fixed_uncovered_token_threshold: int
    context_pressure_threshold: float
    omitted_total_threshold: int
    topic_drift_distance_threshold: float
    snapshottable_event_types: list[str]


@dataclass(frozen=True)
class ShortTermStrategyConfig:
    context_policy: ShortTermContextPolicyConfig
    tool_evidence: ToolEvidenceStrategyConfig
    summary_trigger: SummaryTriggerStrategyConfig


@dataclass(frozen=True)
class LongTermRetrievalStrategyConfig:
    task_scoring_weights: Dict[str, Dict[str, float]]
    task_type_priors: Dict[str, Dict[str, float]]
    astronomy_keywords: list[str]
    observation_keywords: list[str]
    learning_keywords: list[str]
    semantic_weight_cap: float
    stale_miss_threshold: int
    source_bonus: Dict[str, float]


@dataclass(frozen=True)
class LongTermInjectionStrategyConfig:
    relevance_threshold: float
    max_memories: int
    max_prompt_tokens: int
    memory_budget_ratio: float
    memory_budget_min: int
    memory_budget_max: int
    rerank_top_k: int
    rerank_weight: float
    policy_weight: float
    per_type_quota: int
    constraint_priority: bool


@dataclass(frozen=True)
class LongTermPromotionStrategyConfig:
    type_thresholds: Dict[str, float]
    source_weights: Dict[str, float]
    decay_days: float
    auto_promotable_background_categories: list[str]
    expandable_keys: list[str]
    mutually_exclusive_keys: list[str]
    background_consistency_threshold: float
    single_auto_effective_count_threshold: float


@dataclass(frozen=True)
class ExtractionGatingStrategyConfig:
    explicit_signal_patterns: list[str]
    stable_indicators: list[str]
    temporary_indicators: list[str]
    revocation_patterns: list[str]
    equipment_terms: list[str]
    location_terms: list[str]
    topic_keywords: list[str]
    extraction_keywords: list[str]
    window_size: int
    signal_weights: Dict[str, float]
    gating_thresholds: Dict[str, float]
    implicit_extraction_confidence: Dict[str, float]


@dataclass(frozen=True)
class LongTermStrategyConfig:
    retrieval: LongTermRetrievalStrategyConfig
    injection: LongTermInjectionStrategyConfig
    promotion: LongTermPromotionStrategyConfig
    extraction_gating: ExtractionGatingStrategyConfig


@dataclass(frozen=True)
class MemorySelectionStrategyConfig:
    version: str
    short_term: ShortTermStrategyConfig
    long_term: LongTermStrategyConfig


def get_memory_selection_strategy_config(
    overrides: Optional[Dict[str, Any]] = None,
) -> MemorySelectionStrategyConfig:
    """Load memory selection strategy config with defaults -> YAML -> overrides."""

    defaults = _default_config_data()
    data = deepcopy(defaults)
    yaml_data = _load_yaml_config()
    if yaml_data:
        data = _merge_known(data, yaml_data, path="selection_strategy.yaml")
    if overrides:
        data = _merge_known(data, overrides, path="selection_strategy.overrides")
    sanitized = _sanitize_config_data(data, defaults)
    return _build_config(sanitized)


def _load_yaml_config() -> Dict[str, Any]:
    """Read the optional strategy YAML. Missing or invalid files are non-fatal."""

    configured_path = getattr(
        settings,
        "MEMORY_SELECTION_STRATEGY_CONFIG_PATH",
        "config/memory/selection_strategy.yaml",
    )
    try:
        path = Path(resolve_path(str(configured_path)))
    except Exception:
        path = Path(str(configured_path))
    if not path.exists():
        logger.warning("memory selection strategy config not found: %s", path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    except Exception as exc:
        logger.warning(
            "memory selection strategy config load failed: %s (%s)",
            path,
            exc,
        )
        return {}
    if not isinstance(loaded, dict):
        logger.warning("memory selection strategy config must be a mapping: %s", path)
        return {}
    return loaded


def _merge_known(
    base: Dict[str, Any], overlay: Mapping[str, Any], path: str
) -> Dict[str, Any]:
    """Deep-merge known config fields and ignore unknown top-level fields."""

    if not isinstance(overlay, Mapping):
        logger.warning("invalid strategy config section at %s: expected mapping", path)
        return base

    result = deepcopy(base)
    for key, value in overlay.items():
        if key not in result:
            if _is_open_map_path(path):
                result[str(key)] = value
                continue
            logger.warning(
                "unknown memory strategy config field ignored: %s.%s", path, key
            )
            continue
        current = result[key]
        if isinstance(current, dict):
            if not isinstance(value, Mapping):
                logger.warning(
                    "invalid strategy config field ignored: %s.%s", path, key
                )
                continue
            result[key] = _merge_known(current, value, f"{path}.{key}")
        else:
            result[key] = value
    return result


def _is_open_map_path(path: str) -> bool:
    return any(path.endswith(suffix) for suffix in _OPEN_MAP_PATH_SUFFIXES)


def _sanitize_config_data(
    data: Dict[str, Any], defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate field-level values so a bad field falls back without failing boot."""

    sanitized = deepcopy(defaults)
    sanitized["version"] = _string_value(
        data.get("version"), defaults["version"], "version"
    )

    st = data.get("short_term") if isinstance(data.get("short_term"), dict) else {}
    st_default = defaults["short_term"]
    cp = st.get("context_policy") if isinstance(st.get("context_policy"), dict) else {}
    cp_default = st_default["context_policy"]
    sanitized["short_term"]["context_policy"] = {
        "scene_section_ratios": _nested_float_map(
            cp.get("scene_section_ratios"),
            cp_default["scene_section_ratios"],
            "short_term.context_policy.scene_section_ratios",
            allowed_inner_keys=set(_SECTIONS),
            min_value=0.0,
            max_value=1.0,
            require_positive_sum=True,
        ),
        "top_k": _int_map(
            cp.get("top_k"),
            cp_default["top_k"],
            "short_term.context_policy.top_k",
            allowed_keys=set(_SECTIONS),
            min_value=0,
        ),
        "mmr_lambda": _float_value(
            cp.get("mmr_lambda"),
            cp_default["mmr_lambda"],
            "short_term.context_policy.mmr_lambda",
            min_value=0.0,
            max_value=1.0,
        ),
        "similarity_threshold": _float_value(
            cp.get("similarity_threshold"),
            cp_default["similarity_threshold"],
            "short_term.context_policy.similarity_threshold",
            min_value=0.0,
            max_value=1.0,
        ),
        "max_tools": _int_value(
            cp.get("max_tools"),
            cp_default["max_tools"],
            "short_term.context_policy.max_tools",
            min_value=0,
        ),
        "max_per_tool_type": _int_value(
            cp.get("max_per_tool_type"),
            cp_default["max_per_tool_type"],
            "short_term.context_policy.max_per_tool_type",
            min_value=1,
        ),
        "max_per_target": _int_value(
            cp.get("max_per_target"),
            cp_default["max_per_target"],
            "short_term.context_policy.max_per_target",
            min_value=1,
        ),
        "min_section_tokens": _int_value(
            cp.get("min_section_tokens"),
            cp_default["min_section_tokens"],
            "short_term.context_policy.min_section_tokens",
            min_value=0,
        ),
        "downgrade_order": _string_list(
            cp.get("downgrade_order"),
            cp_default["downgrade_order"],
            "short_term.context_policy.downgrade_order",
        ),
        "summary_needed_context_pressure": _float_value(
            cp.get("summary_needed_context_pressure"),
            cp_default["summary_needed_context_pressure"],
            "short_term.context_policy.summary_needed_context_pressure",
            min_value=0.0,
        ),
        "summary_needed_omitted_total": _int_value(
            cp.get("summary_needed_omitted_total"),
            cp_default["summary_needed_omitted_total"],
            "short_term.context_policy.summary_needed_omitted_total",
            min_value=0,
        ),
    }

    te = st.get("tool_evidence") if isinstance(st.get("tool_evidence"), dict) else {}
    te_default = st_default["tool_evidence"]
    sanitized["short_term"]["tool_evidence"] = {
        "ttl_seconds": _float_map(
            te.get("ttl_seconds"),
            te_default["ttl_seconds"],
            "short_term.tool_evidence.ttl_seconds",
            min_value=0.0,
        ),
        "scene_weights": _nested_float_map(
            te.get("scene_weights"),
            te_default["scene_weights"],
            "short_term.tool_evidence.scene_weights",
            allowed_inner_keys=set(_TOOL_WEIGHT_KEYS),
            min_value=0.0,
        ),
        "expired_freshness_cap": _float_value(
            te.get("expired_freshness_cap"),
            te_default["expired_freshness_cap"],
            "short_term.tool_evidence.expired_freshness_cap",
            min_value=0.0,
            max_value=1.0,
        ),
        "compare_historical_expired_freshness_cap": _float_value(
            te.get("compare_historical_expired_freshness_cap"),
            te_default["compare_historical_expired_freshness_cap"],
            "short_term.tool_evidence.compare_historical_expired_freshness_cap",
            min_value=0.0,
            max_value=1.0,
        ),
        "contrast_min_score": _float_value(
            te.get("contrast_min_score"),
            te_default["contrast_min_score"],
            "short_term.tool_evidence.contrast_min_score",
            min_value=0.0,
            max_value=1.0,
        ),
    }

    trig = (
        st.get("summary_trigger") if isinstance(st.get("summary_trigger"), dict) else {}
    )
    trig_default = st_default["summary_trigger"]
    sanitized["short_term"]["summary_trigger"] = {
        "snapshot_batch_size": _int_value(
            trig.get("snapshot_batch_size"),
            trig_default["snapshot_batch_size"],
            "short_term.summary_trigger.snapshot_batch_size",
            min_value=1,
        ),
        "initial_uncovered_event_threshold": _int_value(
            trig.get("initial_uncovered_event_threshold"),
            trig_default["initial_uncovered_event_threshold"],
            "short_term.summary_trigger.initial_uncovered_event_threshold",
            min_value=0,
        ),
        "rebase_uncovered_event_threshold": _int_value(
            trig.get("rebase_uncovered_event_threshold"),
            trig_default["rebase_uncovered_event_threshold"],
            "short_term.summary_trigger.rebase_uncovered_event_threshold",
            min_value=0,
        ),
        "uncovered_token_threshold": _int_value(
            trig.get("uncovered_token_threshold"),
            trig_default["uncovered_token_threshold"],
            "short_term.summary_trigger.uncovered_token_threshold",
            min_value=0,
        ),
        "fixed_uncovered_event_threshold": _int_value(
            trig.get("fixed_uncovered_event_threshold"),
            trig_default["fixed_uncovered_event_threshold"],
            "short_term.summary_trigger.fixed_uncovered_event_threshold",
            min_value=1,
        ),
        "fixed_uncovered_token_threshold": _int_value(
            trig.get("fixed_uncovered_token_threshold"),
            trig_default["fixed_uncovered_token_threshold"],
            "short_term.summary_trigger.fixed_uncovered_token_threshold",
            min_value=1,
        ),
        "context_pressure_threshold": _float_value(
            trig.get("context_pressure_threshold"),
            trig_default["context_pressure_threshold"],
            "short_term.summary_trigger.context_pressure_threshold",
            min_value=0.0,
        ),
        "omitted_total_threshold": _int_value(
            trig.get("omitted_total_threshold"),
            trig_default["omitted_total_threshold"],
            "short_term.summary_trigger.omitted_total_threshold",
            min_value=0,
        ),
        "topic_drift_distance_threshold": _float_value(
            trig.get("topic_drift_distance_threshold"),
            trig_default["topic_drift_distance_threshold"],
            "short_term.summary_trigger.topic_drift_distance_threshold",
            min_value=0.0,
            max_value=1.0,
        ),
        "snapshottable_event_types": _string_list(
            trig.get("snapshottable_event_types"),
            trig_default["snapshottable_event_types"],
            "short_term.summary_trigger.snapshottable_event_types",
        ),
    }

    lt = data.get("long_term") if isinstance(data.get("long_term"), dict) else {}
    lt_default = defaults["long_term"]
    ret = lt.get("retrieval") if isinstance(lt.get("retrieval"), dict) else {}
    ret_default = lt_default["retrieval"]
    sanitized["long_term"]["retrieval"] = {
        "task_scoring_weights": _nested_float_map(
            ret.get("task_scoring_weights"),
            ret_default["task_scoring_weights"],
            "long_term.retrieval.task_scoring_weights",
            allowed_inner_keys=set(_LTM_SCORE_KEYS),
            min_value=0.0,
        ),
        "task_type_priors": _nested_float_map(
            ret.get("task_type_priors"),
            ret_default["task_type_priors"],
            "long_term.retrieval.task_type_priors",
            allowed_inner_keys=set(_MEMORY_TYPES),
            min_value=0.0,
            max_value=1.0,
        ),
        "astronomy_keywords": _string_list(
            ret.get("astronomy_keywords"),
            ret_default["astronomy_keywords"],
            "long_term.retrieval.astronomy_keywords",
        ),
        "observation_keywords": _string_list(
            ret.get("observation_keywords"),
            ret_default["observation_keywords"],
            "long_term.retrieval.observation_keywords",
        ),
        "learning_keywords": _string_list(
            ret.get("learning_keywords"),
            ret_default["learning_keywords"],
            "long_term.retrieval.learning_keywords",
        ),
        "semantic_weight_cap": _float_value(
            ret.get("semantic_weight_cap"),
            ret_default["semantic_weight_cap"],
            "long_term.retrieval.semantic_weight_cap",
            min_value=0.0,
            max_value=1.0,
        ),
        "stale_miss_threshold": _int_value(
            ret.get("stale_miss_threshold"),
            ret_default["stale_miss_threshold"],
            "long_term.retrieval.stale_miss_threshold",
            min_value=1,
        ),
        "source_bonus": _float_map(
            ret.get("source_bonus"),
            ret_default["source_bonus"],
            "long_term.retrieval.source_bonus",
            min_value=0.0,
            max_value=1.0,
        ),
    }

    inj = lt.get("injection") if isinstance(lt.get("injection"), dict) else {}
    inj_default = lt_default["injection"]
    sanitized["long_term"]["injection"] = {
        "relevance_threshold": _float_value(
            inj.get("relevance_threshold"),
            inj_default["relevance_threshold"],
            "long_term.injection.relevance_threshold",
            min_value=0.0,
            max_value=1.0,
        ),
        "max_memories": _int_value(
            inj.get("max_memories"),
            inj_default["max_memories"],
            "long_term.injection.max_memories",
            min_value=0,
        ),
        "max_prompt_tokens": _int_value(
            inj.get("max_prompt_tokens"),
            inj_default["max_prompt_tokens"],
            "long_term.injection.max_prompt_tokens",
            min_value=0,
        ),
        "memory_budget_ratio": _float_value(
            inj.get("memory_budget_ratio"),
            inj_default["memory_budget_ratio"],
            "long_term.injection.memory_budget_ratio",
            min_value=0.0,
            max_value=1.0,
        ),
        "memory_budget_min": _int_value(
            inj.get("memory_budget_min"),
            inj_default["memory_budget_min"],
            "long_term.injection.memory_budget_min",
            min_value=0,
        ),
        "memory_budget_max": _int_value(
            inj.get("memory_budget_max"),
            inj_default["memory_budget_max"],
            "long_term.injection.memory_budget_max",
            min_value=0,
        ),
        "rerank_top_k": _int_value(
            inj.get("rerank_top_k"),
            inj_default["rerank_top_k"],
            "long_term.injection.rerank_top_k",
            min_value=1,
        ),
        "rerank_weight": _float_value(
            inj.get("rerank_weight"),
            inj_default["rerank_weight"],
            "long_term.injection.rerank_weight",
            min_value=0.0,
            max_value=1.0,
        ),
        "policy_weight": _float_value(
            inj.get("policy_weight"),
            inj_default["policy_weight"],
            "long_term.injection.policy_weight",
            min_value=0.0,
            max_value=1.0,
        ),
        "per_type_quota": _int_value(
            inj.get("per_type_quota"),
            inj_default["per_type_quota"],
            "long_term.injection.per_type_quota",
            min_value=1,
        ),
        "constraint_priority": _bool_value(
            inj.get("constraint_priority"),
            inj_default["constraint_priority"],
            "long_term.injection.constraint_priority",
        ),
    }
    if (
        sanitized["long_term"]["injection"]["memory_budget_min"]
        > sanitized["long_term"]["injection"]["memory_budget_max"]
    ):
        logger.warning(
            "invalid memory budget bounds; fallback to defaults for long_term.injection"
        )
        sanitized["long_term"]["injection"]["memory_budget_min"] = inj_default[
            "memory_budget_min"
        ]
        sanitized["long_term"]["injection"]["memory_budget_max"] = inj_default[
            "memory_budget_max"
        ]

    prom = lt.get("promotion") if isinstance(lt.get("promotion"), dict) else {}
    prom_default = lt_default["promotion"]
    sanitized["long_term"]["promotion"] = {
        "type_thresholds": _float_map(
            prom.get("type_thresholds"),
            prom_default["type_thresholds"],
            "long_term.promotion.type_thresholds",
            allowed_keys=set(_MEMORY_TYPES) - {"fact"},
            min_value=0.0,
            max_value=1.0,
        ),
        "source_weights": _float_map(
            prom.get("source_weights"),
            prom_default["source_weights"],
            "long_term.promotion.source_weights",
            min_value=0.0,
            max_value=1.0,
        ),
        "decay_days": _float_value(
            prom.get("decay_days"),
            prom_default["decay_days"],
            "long_term.promotion.decay_days",
            min_value=0.001,
        ),
        "auto_promotable_background_categories": _string_list(
            prom.get("auto_promotable_background_categories"),
            prom_default["auto_promotable_background_categories"],
            "long_term.promotion.auto_promotable_background_categories",
        ),
        "expandable_keys": _string_list(
            prom.get("expandable_keys"),
            prom_default["expandable_keys"],
            "long_term.promotion.expandable_keys",
        ),
        "mutually_exclusive_keys": _string_list(
            prom.get("mutually_exclusive_keys"),
            prom_default["mutually_exclusive_keys"],
            "long_term.promotion.mutually_exclusive_keys",
        ),
        "background_consistency_threshold": _float_value(
            prom.get("background_consistency_threshold"),
            prom_default["background_consistency_threshold"],
            "long_term.promotion.background_consistency_threshold",
            min_value=0.0,
            max_value=1.0,
        ),
        "single_auto_effective_count_threshold": _float_value(
            prom.get("single_auto_effective_count_threshold"),
            prom_default["single_auto_effective_count_threshold"],
            "long_term.promotion.single_auto_effective_count_threshold",
            min_value=0.0,
        ),
    }

    gate = (
        lt.get("extraction_gating")
        if isinstance(lt.get("extraction_gating"), dict)
        else {}
    )
    gate_default = lt_default["extraction_gating"]
    sanitized["long_term"]["extraction_gating"] = {
        "explicit_signal_patterns": _string_list(
            gate.get("explicit_signal_patterns"),
            gate_default["explicit_signal_patterns"],
            "long_term.extraction_gating.explicit_signal_patterns",
        ),
        "stable_indicators": _string_list(
            gate.get("stable_indicators"),
            gate_default["stable_indicators"],
            "long_term.extraction_gating.stable_indicators",
        ),
        "temporary_indicators": _string_list(
            gate.get("temporary_indicators"),
            gate_default["temporary_indicators"],
            "long_term.extraction_gating.temporary_indicators",
        ),
        "revocation_patterns": _string_list(
            gate.get("revocation_patterns"),
            gate_default["revocation_patterns"],
            "long_term.extraction_gating.revocation_patterns",
        ),
        "equipment_terms": _string_list(
            gate.get("equipment_terms"),
            gate_default["equipment_terms"],
            "long_term.extraction_gating.equipment_terms",
        ),
        "location_terms": _string_list(
            gate.get("location_terms"),
            gate_default["location_terms"],
            "long_term.extraction_gating.location_terms",
        ),
        "topic_keywords": _string_list(
            gate.get("topic_keywords"),
            gate_default["topic_keywords"],
            "long_term.extraction_gating.topic_keywords",
        ),
        "extraction_keywords": _string_list(
            gate.get("extraction_keywords"),
            gate_default["extraction_keywords"],
            "long_term.extraction_gating.extraction_keywords",
        ),
        "window_size": _int_value(
            gate.get("window_size"),
            gate_default["window_size"],
            "long_term.extraction_gating.window_size",
            min_value=1,
        ),
        "signal_weights": _float_map(
            gate.get("signal_weights"),
            gate_default["signal_weights"],
            "long_term.extraction_gating.signal_weights",
        ),
        "gating_thresholds": _float_map(
            gate.get("gating_thresholds"),
            gate_default["gating_thresholds"],
            "long_term.extraction_gating.gating_thresholds",
        ),
        "implicit_extraction_confidence": _float_map(
            gate.get("implicit_extraction_confidence"),
            gate_default["implicit_extraction_confidence"],
            "long_term.extraction_gating.implicit_extraction_confidence",
            min_value=0.0,
            max_value=1.0,
        ),
    }
    return sanitized


def _build_config(data: Dict[str, Any]) -> MemorySelectionStrategyConfig:
    """Convert sanitized dictionaries into frozen dataclasses."""

    st = data["short_term"]
    lt = data["long_term"]
    return MemorySelectionStrategyConfig(
        version=data["version"],
        short_term=ShortTermStrategyConfig(
            context_policy=ShortTermContextPolicyConfig(**st["context_policy"]),
            tool_evidence=ToolEvidenceStrategyConfig(**st["tool_evidence"]),
            summary_trigger=SummaryTriggerStrategyConfig(**st["summary_trigger"]),
        ),
        long_term=LongTermStrategyConfig(
            retrieval=LongTermRetrievalStrategyConfig(**lt["retrieval"]),
            injection=LongTermInjectionStrategyConfig(**lt["injection"]),
            promotion=LongTermPromotionStrategyConfig(**lt["promotion"]),
            extraction_gating=ExtractionGatingStrategyConfig(**lt["extraction_gating"]),
        ),
    )


def _default_config_data() -> Dict[str, Any]:
    """Return code defaults seeded from existing env-backed settings."""

    return {
        "version": DEFAULT_STRATEGY_CONFIG_VERSION,
        "short_term": {
            "context_policy": {
                "scene_section_ratios": {
                    "observation": {
                        "summary": 0.10,
                        "facts": 0.15,
                        "tools": 0.50,
                        "messages": 0.25,
                    },
                    "computation": {
                        "summary": 0.15,
                        "facts": 0.35,
                        "tools": 0.35,
                        "messages": 0.15,
                    },
                    "learning_qa": {
                        "summary": 0.30,
                        "facts": 0.30,
                        "tools": 0.10,
                        "messages": 0.30,
                    },
                    "debugging": {
                        "summary": 0.25,
                        "facts": 0.10,
                        "tools": 0.40,
                        "messages": 0.25,
                    },
                    "general": {
                        "summary": 0.20,
                        "facts": 0.25,
                        "tools": 0.30,
                        "messages": 0.25,
                    },
                },
                "top_k": {"summary": 1, "facts": 8, "tools": 5, "messages": 6},
                "mmr_lambda": 0.7,
                "similarity_threshold": 0.82,
                "max_tools": 5,
                "max_per_tool_type": 2,
                "max_per_target": 2,
                "min_section_tokens": 80,
                "downgrade_order": [
                    "tool_detail",
                    "old_messages",
                    "low_score_facts",
                    "compact_summary",
                ],
                "summary_needed_context_pressure": 1.2,
                "summary_needed_omitted_total": 8,
            },
            "tool_evidence": {
                "ttl_seconds": {
                    "visibility": 3600,
                    "weather": 10800,
                    "position": 7200,
                    "ephemeris": 43200,
                    "neo": 21600,
                    "event": 604800,
                    "catalog": 0,
                    "photo": 86400,
                    "generic": 86400,
                },
                "scene_weights": {
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
                },
                "expired_freshness_cap": 0.15,
                "compare_historical_expired_freshness_cap": 0.40,
                "contrast_min_score": 0.25,
            },
            "summary_trigger": {
                "snapshot_batch_size": 200,
                "initial_uncovered_event_threshold": int(
                    getattr(settings, "MEMORY_SUMMARY_TRIGGER_MESSAGES", 10)
                ),
                "rebase_uncovered_event_threshold": int(
                    getattr(settings, "MEMORY_SUMMARY_MIN_NEW_EVENTS", 6)
                ),
                "uncovered_token_threshold": int(
                    getattr(settings, "MEMORY_SUMMARY_TRIGGER_TOKENS", 3000)
                ),
                "fixed_uncovered_event_threshold": 30,
                "fixed_uncovered_token_threshold": 6000,
                "context_pressure_threshold": 1.2,
                "omitted_total_threshold": 8,
                "topic_drift_distance_threshold": 0.6,
                "snapshottable_event_types": [
                    "message_created",
                    "tool_call_finished",
                    "tool_call_failed",
                    "fact_extracted",
                    "task_state_updated",
                    "memory_deleted",
                ],
            },
        },
        "long_term": {
            "retrieval": {
                "task_scoring_weights": {
                    "observation": {
                        "confidence": 0.20,
                        "type_weight": 0.18,
                        "source_bonus": 0.10,
                        "query_relevance": 0.27,
                        "recency": 0.08,
                        "constraint_bonus": 0.14,
                        "stale_penalty": 0.03,
                    },
                    "learning": {
                        "confidence": 0.24,
                        "type_weight": 0.20,
                        "source_bonus": 0.12,
                        "query_relevance": 0.25,
                        "recency": 0.08,
                        "constraint_bonus": 0.08,
                        "stale_penalty": 0.03,
                    },
                    "qa": {
                        "confidence": 0.24,
                        "type_weight": 0.20,
                        "source_bonus": 0.10,
                        "query_relevance": 0.27,
                        "recency": 0.08,
                        "constraint_bonus": 0.08,
                        "stale_penalty": 0.03,
                    },
                    "general": {
                        "confidence": 0.25,
                        "type_weight": 0.18,
                        "source_bonus": 0.12,
                        "query_relevance": 0.25,
                        "recency": 0.10,
                        "constraint_bonus": 0.07,
                        "stale_penalty": 0.03,
                    },
                },
                "task_type_priors": {
                    "observation": {
                        "constraint": 1.0,
                        "background": 0.90,
                        "habit": 0.85,
                        "preference": 0.75,
                        "fact": 0.75,
                    },
                    "learning": {
                        "preference": 0.90,
                        "background": 0.85,
                        "fact": 0.70,
                        "constraint": 0.65,
                        "habit": 0.45,
                    },
                    "qa": {
                        "fact": 0.90,
                        "background": 0.85,
                        "preference": 0.70,
                        "constraint": 0.65,
                        "habit": 0.45,
                    },
                    "general": {
                        "constraint": 0.90,
                        "preference": 0.85,
                        "background": 0.75,
                        "fact": 0.70,
                        "habit": 0.65,
                    },
                },
                "astronomy_keywords": [
                    "观测",
                    "望远镜",
                    "行星",
                    "恒星",
                    "星系",
                    "星云",
                    "星团",
                    "流星",
                    "彗星",
                    "月相",
                    "日食",
                    "月食",
                    "冲日",
                    "合日",
                    "拍摄",
                    "摄影",
                    "深空",
                    "赤道仪",
                    "导星",
                    "曝光",
                    "木星",
                    "土星",
                    "火星",
                    "金星",
                    "月球",
                    "太阳",
                    "多大",
                    "距离",
                    "质量",
                    "半径",
                    "亮度",
                ],
                "observation_keywords": [
                    "今晚",
                    "观测",
                    "天气",
                    "天气怎么样",
                    "适合看",
                    "看什么",
                    "云量",
                    "透明度",
                    "视宁度",
                    "光害",
                    "月相",
                    "可见",
                    "升起",
                    "落下",
                    "最佳",
                    "推荐",
                    "目标",
                    "望远镜",
                    "拍摄",
                ],
                "learning_keywords": [
                    "什么是",
                    "为什么",
                    "如何",
                    "原理",
                    "解释",
                    "科普",
                    "入门",
                    "学习",
                    "了解",
                ],
                "semantic_weight_cap": 0.08,
                "stale_miss_threshold": 5,
                "source_bonus": {
                    "confirmed": 1.0,
                    "manual": 0.95,
                    "explicit": 0.90,
                    "auto": 0.55,
                    "default": 0.55,
                },
            },
            "injection": {
                "relevance_threshold": 0.45,
                "max_memories": int(
                    getattr(settings, "LTM_MAX_MEMORIES_IN_PROMPT", 15)
                ),
                "max_prompt_tokens": int(
                    getattr(settings, "LTM_MAX_PROMPT_TOKENS", 800)
                ),
                "memory_budget_ratio": 0.1,
                "memory_budget_min": 200,
                "memory_budget_max": 1200,
                "rerank_top_k": 30,
                "rerank_weight": 0.65,
                "policy_weight": 0.35,
                "per_type_quota": 3,
                "constraint_priority": True,
            },
            "promotion": {
                "type_thresholds": {
                    "preference": 0.5,
                    "habit": 0.5,
                    "constraint": 0.65,
                    "background": 0.6,
                },
                "source_weights": {
                    "solid": 1.0,
                    "tentative": 0.85,
                    "inferred": 0.6,
                },
                "decay_days": 30.0,
                "auto_promotable_background_categories": [
                    "device_info",
                    "location",
                    "skill_level",
                    "domain_experience",
                ],
                "expandable_keys": [
                    "location",
                    "location_info",
                    "frequent_topics",
                    "domain_experience",
                    "observation_type",
                ],
                "mutually_exclusive_keys": [
                    "response_style",
                    "knowledge_level",
                    "skill_level",
                    "device_info",
                    "equipment",
                    "timezone",
                    "unit_preference",
                ],
                "background_consistency_threshold": 0.9,
                "single_auto_effective_count_threshold": 1.5,
            },
            "extraction_gating": {
                "explicit_signal_patterns": [
                    r"我(喜欢|偏好|希望|要求|习惯)(.{1,30})",
                    r"(不要|别|避免)(.{1,30})",
                    r"给我(.{1,30})",
                    r"我(是|有)(.{1,30})(经验|基础|背景)",
                    r"请(用|以|按)(.{1,30})",
                    r"我(的)(.{1,30})(是|叫|在)",
                    r"记住(.{1,50})",
                    r"以后(都|一直|总是)(.{1,30})",
                    r"永远(不要|别)(.{1,30})",
                ],
                "temporary_indicators": [
                    "这次",
                    "本次",
                    "这回",
                    "暂时",
                    "仅此一次",
                    "今天",
                    "今晚",
                    "这一次",
                    "临时",
                    "just this time",
                    "for now",
                    "temporarily",
                    "today",
                    "tonight",
                ],
                "stable_indicators": [
                    "记住",
                    "以后",
                    "以后默认",
                    "下次",
                    "下回",
                    "每次",
                    "每次都",
                    "总是",
                    "一直",
                    "通常",
                    "默认",
                    "长期",
                    "永远",
                    "remember",
                    "next time",
                    "always",
                    "usually",
                    "default",
                ],
                "revocation_patterns": [
                    r"忘掉",
                    r"别记",
                    r"不要记",
                    r"不用记",
                    r"不再",
                    r"现在不再",
                    r"不是了",
                    r"作废",
                    r"撤回",
                    r"取消之前",
                    r"之前.*(不算|作废)",
                    r"那次只是(开玩笑|临时)",
                    r"只是开玩笑",
                    r"以后.*改成",
                    r"默认.*改成",
                    r"记住.*改成",
                    r"下次.*改成",
                    r"长期.*改成",
                    r"forget",
                    r"do not remember",
                    r"don't remember",
                    r"no longer",
                    r"revoke",
                ],
                "equipment_terms": [
                    "望远镜",
                    "镜",
                    "相机",
                    "赤道仪",
                    "镜头",
                    "目镜",
                    "CCD",
                    "CMOS",
                    "道布森",
                    "小黑",
                    "星特朗",
                    "信达",
                    "佳能",
                    "尼康",
                    "索尼",
                    "双筒",
                    "三脚架",
                    "滤镜",
                    "口径",
                    "焦距",
                    "导星镜",
                    "ZWO",
                    "ASI",
                    "QHY",
                    "Seestar",
                    "telescope",
                    "camera",
                    "mount",
                    "eyepiece",
                ],
                "location_terms": [
                    "观测地",
                    "观测地点",
                    "地点",
                    "位置",
                    "经纬度",
                    "纬度",
                    "经度",
                    "北京",
                    "上海",
                    "广州",
                    "深圳",
                    "杭州",
                    "苏州",
                    "成都",
                    "南京",
                    "武汉",
                    "location",
                    "site",
                    "latitude",
                    "longitude",
                    "timezone",
                    "time zone",
                ],
                "topic_keywords": [
                    "火星",
                    "木星",
                    "土星",
                    "金星",
                    "月球",
                    "太阳",
                    "黑洞",
                    "星系",
                    "星云",
                    "星团",
                    "流星雨",
                    "彗星",
                    "银河",
                    "深空",
                    "望远镜",
                    "赤道仪",
                    "拍摄",
                    "摄影",
                    "观测",
                ],
                "extraction_keywords": [
                    "简短",
                    "详细",
                    "专业",
                    "通俗",
                    "易懂",
                    "不要",
                    "喜欢",
                    "偏好",
                    "习惯",
                    "经常",
                    "总是",
                    "希望",
                    "要求",
                    "建议",
                    "初学者",
                    "入门",
                    "高级",
                    "进阶",
                    "望远镜",
                    "相机",
                    "拍摄",
                    "观测",
                    "深空",
                    "行星",
                    "月相",
                    "流星雨",
                    "日食",
                    "月食",
                    "星系",
                    "星云",
                    "星团",
                ],
                "window_size": 4,
                "signal_weights": {
                    "self_reference": 1.0,
                    "action_modal": 1.0,
                    "stable_marker": 1.5,
                    "equipment": 0.8,
                    "location": 0.8,
                    "correction": 1.0,
                    "repeated_signal": 1.5,
                    "temporary_penalty": -1.0,
                },
                "gating_thresholds": {
                    "stable_profile_signal": 2.5,
                    "window_repeated_signal": 3.0,
                },
                "implicit_extraction_confidence": {
                    "repeated_equipment": 0.45,
                    "repeated_location": 0.45,
                    "repeated_correction": 0.45,
                    "language_detected": 0.35,
                    "unit_detected": 0.35,
                    "timezone_detected": 0.35,
                },
            },
        },
    }


def _string_value(value: Any, fallback: str, path: str) -> str:
    if isinstance(value, str) and value:
        return value
    logger.warning("invalid memory strategy config field %s; using default", path)
    return fallback


def _bool_value(value: Any, fallback: bool, path: str) -> bool:
    if isinstance(value, bool):
        return value
    logger.warning("invalid memory strategy config field %s; using default", path)
    return fallback


def _int_value(
    value: Any,
    fallback: int,
    path: str,
    *,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    try:
        if isinstance(value, bool):
            raise ValueError
        converted = int(value)
    except (TypeError, ValueError):
        logger.warning("invalid memory strategy config field %s; using default", path)
        return fallback
    if min_value is not None and converted < min_value:
        logger.warning("invalid memory strategy config field %s; using default", path)
        return fallback
    if max_value is not None and converted > max_value:
        logger.warning("invalid memory strategy config field %s; using default", path)
        return fallback
    return converted


def _float_value(
    value: Any,
    fallback: float,
    path: str,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError
        converted = float(value)
    except (TypeError, ValueError):
        logger.warning("invalid memory strategy config field %s; using default", path)
        return fallback
    if min_value is not None and converted < min_value:
        logger.warning("invalid memory strategy config field %s; using default", path)
        return fallback
    if max_value is not None and converted > max_value:
        logger.warning("invalid memory strategy config field %s; using default", path)
        return fallback
    return converted


def _string_list(value: Any, fallback: list[str], path: str) -> list[str]:
    if value is None:
        return list(fallback)
    if not isinstance(value, list):
        logger.warning("invalid memory strategy config field %s; using default", path)
        return list(fallback)
    normalized = [str(item) for item in value if str(item)]
    if len(normalized) != len(value):
        logger.warning("invalid memory strategy config item in %s ignored", path)
    return normalized


def _float_map(
    value: Any,
    fallback: Dict[str, float],
    path: str,
    *,
    allowed_keys: Optional[set[str]] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Dict[str, float]:
    if value is None:
        return dict(fallback)
    if not isinstance(value, Mapping):
        logger.warning("invalid memory strategy config field %s; using default", path)
        return dict(fallback)
    result = dict(fallback)
    for key, raw in value.items():
        normalized_key = str(key)
        if allowed_keys is not None and normalized_key not in allowed_keys:
            logger.warning(
                "unknown memory strategy config key ignored: %s.%s", path, key
            )
            continue
        result[normalized_key] = _float_value(
            raw,
            fallback.get(normalized_key, 0.0),
            f"{path}.{normalized_key}",
            min_value=min_value,
            max_value=max_value,
        )
    return result


def _int_map(
    value: Any,
    fallback: Dict[str, int],
    path: str,
    *,
    allowed_keys: Optional[set[str]] = None,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> Dict[str, int]:
    if value is None:
        return dict(fallback)
    if not isinstance(value, Mapping):
        logger.warning("invalid memory strategy config field %s; using default", path)
        return dict(fallback)
    result = dict(fallback)
    for key, raw in value.items():
        normalized_key = str(key)
        if allowed_keys is not None and normalized_key not in allowed_keys:
            logger.warning(
                "unknown memory strategy config key ignored: %s.%s", path, key
            )
            continue
        result[normalized_key] = _int_value(
            raw,
            fallback.get(normalized_key, 0),
            f"{path}.{normalized_key}",
            min_value=min_value,
            max_value=max_value,
        )
    return result


def _nested_float_map(
    value: Any,
    fallback: Dict[str, Dict[str, float]],
    path: str,
    *,
    allowed_inner_keys: Optional[set[str]] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    require_positive_sum: bool = False,
) -> Dict[str, Dict[str, float]]:
    if value is None:
        return deepcopy(fallback)
    if not isinstance(value, Mapping):
        logger.warning("invalid memory strategy config field %s; using default", path)
        return deepcopy(fallback)
    result = deepcopy(fallback)
    for outer_key, raw_map in value.items():
        scene = str(outer_key)
        scene_fallback = fallback.get(scene) or fallback.get("general") or {}
        if not isinstance(raw_map, Mapping):
            logger.warning(
                "invalid memory strategy config field %s.%s; using default", path, scene
            )
            result[scene] = dict(scene_fallback)
            continue
        merged = dict(scene_fallback)
        for inner_key, raw_value in raw_map.items():
            normalized_inner = str(inner_key)
            if (
                allowed_inner_keys is not None
                and normalized_inner not in allowed_inner_keys
            ):
                logger.warning(
                    "unknown memory strategy config key ignored: %s.%s.%s",
                    path,
                    scene,
                    inner_key,
                )
                continue
            merged[normalized_inner] = _float_value(
                raw_value,
                scene_fallback.get(normalized_inner, 0.0),
                f"{path}.{scene}.{normalized_inner}",
                min_value=min_value,
                max_value=max_value,
            )
        if require_positive_sum and sum(merged.values()) <= 0:
            logger.warning(
                "invalid memory strategy config field %s.%s; using default", path, scene
            )
            merged = dict(scene_fallback)
        result[scene] = merged
    return result


__all__ = [
    "DEFAULT_STRATEGY_CONFIG_VERSION",
    "ExtractionGatingStrategyConfig",
    "LongTermInjectionStrategyConfig",
    "LongTermPromotionStrategyConfig",
    "LongTermRetrievalStrategyConfig",
    "LongTermStrategyConfig",
    "MemorySelectionStrategyConfig",
    "ShortTermContextPolicyConfig",
    "ShortTermStrategyConfig",
    "SummaryTriggerStrategyConfig",
    "ToolEvidenceStrategyConfig",
    "get_memory_selection_strategy_config",
]
