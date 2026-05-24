from types import SimpleNamespace

from src.core.config import settings
from src.memory.core.models import ToolCallRecord
from src.memory.domain.task_state import TaskState
from src.memory.long_term_memory.extractor import MemoryExtractor
from src.memory.long_term_memory.models import MemoryItem, MemoryType
from src.memory.long_term_memory.retrieval import LongTermMemoryRetriever
from src.memory.retrieval.planner import RetrievalPlan, RetrievalPlanner
from src.memory.selection_strategy_config import (
    get_memory_selection_strategy_config,
)


def test_missing_yaml_returns_env_backed_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(
        settings,
        "MEMORY_SELECTION_STRATEGY_CONFIG_PATH",
        str(tmp_path / "missing.yaml"),
    )
    monkeypatch.setattr(settings, "LTM_MAX_PROMPT_TOKENS", 321)

    config = get_memory_selection_strategy_config()

    assert config.long_term.injection.max_prompt_tokens == 321
    assert config.short_term.context_policy.top_k["tools"] == 5


def test_partial_yaml_deep_merges_without_losing_defaults(tmp_path, monkeypatch):
    path = tmp_path / "strategy.yaml"
    path.write_text(
        """
short_term:
  context_policy:
    scene_section_ratios:
      observation:
        tools: 0.80
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "MEMORY_SELECTION_STRATEGY_CONFIG_PATH", str(path))

    config = get_memory_selection_strategy_config()

    ratios = config.short_term.context_policy.scene_section_ratios["observation"]
    assert ratios["tools"] == 0.80
    assert ratios["facts"] == 0.15
    assert config.short_term.context_policy.top_k["messages"] == 6


def test_invalid_ratio_field_falls_back_only_for_bad_value(tmp_path, monkeypatch):
    path = tmp_path / "strategy.yaml"
    path.write_text(
        """
short_term:
  context_policy:
    scene_section_ratios:
      observation:
        summary: 0.30
        tools: -0.1
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "MEMORY_SELECTION_STRATEGY_CONFIG_PATH", str(path))

    config = get_memory_selection_strategy_config()

    ratios = config.short_term.context_policy.scene_section_ratios["observation"]
    assert ratios["summary"] == 0.30
    assert ratios["tools"] == 0.50


def test_constructor_overrides_win_over_yaml(tmp_path, monkeypatch):
    path = tmp_path / "strategy.yaml"
    path.write_text(
        """
long_term:
  injection:
    max_memories: 2
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "MEMORY_SELECTION_STRATEGY_CONFIG_PATH", str(path))

    config = get_memory_selection_strategy_config(
        overrides={"long_term": {"injection": {"max_memories": 7}}}
    )

    assert config.long_term.injection.max_memories == 7


def test_override_section_ratios_changes_retrieval_budgets():
    planner = RetrievalPlanner(
        lambda text: len(text or ""),
        strategy_overrides={
            "short_term": {
                "context_policy": {
                    "scene_section_ratios": {
                        "observation": {
                            "summary": 0.05,
                            "facts": 0.05,
                            "tools": 0.80,
                            "messages": 0.10,
                        }
                    }
                }
            }
        },
    )

    context = planner.build_context(
        query="北京今晚天气适合观测吗",
        token_budget=100,
        task_state=TaskState(tenant_id="t", session_id="s"),
        summary_snapshot=None,
        messages=[],
        facts=[],
        tool_calls=[],
    )

    assert context["section_budgets"]["tools"] == 80
    assert context["decision_trace"]["strategy_config_version"] == (
        "memory_selection_strategy_v1"
    )


def test_override_tool_ttl_changes_effective_until():
    planner = RetrievalPlanner(
        lambda text: len(text or ""),
        strategy_overrides={
            "short_term": {
                "tool_evidence": {"ttl_seconds": {"weather": 10, "generic": 86400}}
            }
        },
    )
    call = ToolCallRecord(
        tool_call_id="weather",
        tool_name="weather-lookup",
        timestamp=100.0,
        input_summary='{"city":"北京"}',
        output_summary="北京晴",
        metadata={"produced_at": 100.0},
    )

    meta = planner._extract_tool_meta(call)

    assert meta.effective_until == 110.0


def test_open_ttl_map_accepts_new_tool_type_and_invalid_known_falls_back(
    tmp_path, monkeypatch
):
    path = tmp_path / "strategy.yaml"
    path.write_text(
        """
short_term:
  tool_evidence:
    ttl_seconds:
      photo: bad
      custom_sensor: 42
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "MEMORY_SELECTION_STRATEGY_CONFIG_PATH", str(path))

    config = get_memory_selection_strategy_config()

    assert config.short_term.tool_evidence.ttl_seconds["photo"] == 86400
    assert config.short_term.tool_evidence.ttl_seconds["custom_sensor"] == 42


def test_default_long_term_task_classification_uses_revised_keywords():
    retriever = LongTermMemoryRetriever(
        repository=SimpleNamespace(),
        strategy_config=get_memory_selection_strategy_config(),
    )

    assert retriever.classify_task_type("北京天气怎么样") == "observation"
    assert retriever.classify_task_type("今晚适合看什么") == "observation"
    assert retriever.classify_task_type("木星有多大") == "qa"


def test_planner_uses_revised_tool_type_ttls_for_photo_and_position():
    planner = RetrievalPlanner(lambda text: len(text or ""))
    photo = planner._extract_tool_meta(
        ToolCallRecord(
            tool_call_id="photo",
            tool_name="astrophotography-calculator",
            timestamp=100.0,
            input_summary='{"target":"M42"}',
            output_summary="曝光参数",
            metadata={"produced_at": 100.0},
        )
    )
    position = planner._extract_tool_meta(
        ToolCallRecord(
            tool_call_id="altaz",
            tool_name="get_altaz",
            timestamp=100.0,
            input_summary='{"target":"木星"}',
            output_summary="高度 35 度",
            metadata={"produced_at": 100.0},
        )
    )

    assert photo.tool_type == "photo"
    assert photo.effective_until == 86500.0
    assert position.tool_type == "position"
    assert position.effective_until == 7300.0


def test_downgrade_order_controls_recorded_priority():
    planner = RetrievalPlanner(
        lambda text: len(text or ""),
        strategy_overrides={
            "short_term": {
                "context_policy": {
                    "downgrade_order": ["old_messages", "tool_detail"]
                }
            }
        },
    )
    policy = planner._policy_for_scene("general")
    plan = RetrievalPlan(query="", query_type="", token_budget=0)

    planner._record_downgrade(plan, "tools", 1, policy)
    planner._record_downgrade(plan, "messages", 1, policy)

    assert plan.downgrade_steps == ["old_messages", "tool_detail"]


def test_override_type_priors_changes_retrieval_score_order():
    strategy_config = get_memory_selection_strategy_config(
        overrides={
            "long_term": {
                "retrieval": {
                    "task_scoring_weights": {
                        "observation": {
                            "confidence": 0.0,
                            "type_weight": 1.0,
                            "source_bonus": 0.0,
                            "query_relevance": 0.0,
                            "recency": 0.0,
                            "constraint_bonus": 0.0,
                            "stale_penalty": 0.0,
                        }
                    },
                    "task_type_priors": {
                        "observation": {
                            "preference": 0.1,
                            "habit": 0.1,
                            "constraint": 0.1,
                            "background": 0.1,
                            "fact": 1.0,
                        }
                    },
                }
            }
        }
    )
    retriever = LongTermMemoryRetriever(
        repository=SimpleNamespace(),
        strategy_config=strategy_config,
    )
    preference = MemoryItem.create(
        user_id="u1",
        memory_type=MemoryType.PREFERENCE,
        category="style",
        key="style",
        value="简短",
        confidence=0.9,
    )
    fact = MemoryItem.create(
        user_id="u1",
        memory_type=MemoryType.FACT,
        category="location",
        key="city",
        value="北京",
        confidence=0.9,
    )

    assert (
        retriever.score_hit(fact, "任意问题", "observation").score
        > retriever.score_hit(preference, "任意问题", "observation").score
    )


def test_override_extraction_indicators_changes_gating():
    extractor = MemoryExtractor(
        strategy_config=get_memory_selection_strategy_config(
            overrides={
                "long_term": {"extraction_gating": {"stable_indicators": ["偏振模式"]}}
            }
        )
    )

    assert extractor.should_attempt_extraction("我希望偏振模式用短回答")
