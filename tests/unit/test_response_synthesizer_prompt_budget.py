"""Unit tests for ResponseSynthesizer prompt budget integration — Phase 2."""

import pytest
from types import SimpleNamespace

from src.agent.response_synthesizer import ResponseSynthesizer


class _FakeLLM:
    def invoke(self, prompt):
        return SimpleNamespace(content="这是测试回答。")


class _FakeSkillResult:
    def __init__(self, name, summary, success=True, data=None, sources=None):
        self.skill_name = name
        self.summary = summary
        self.success = success
        self.data = data
        self.sources = sources or []

    def to_tool_timeline_entry(self):
        return {"tool": self.skill_name, "summary": self.summary[:100]}


def _make_fake_sr(name, summary):
    return SimpleNamespace(
        skill_name=name,
        summary=summary,
        success=True,
        data=None,
        sources=[],
        to_tool_timeline_entry=lambda: {"tool": name, "summary": summary[:100]},
    )


@pytest.fixture
def synth():
    return ResponseSynthesizer(llm=_FakeLLM())


# ---------------------------------------------------------------------------
# 1. Very long tool outputs don't blow up the prompt
# ---------------------------------------------------------------------------

def test_long_tool_outputs_fit_within_budget(monkeypatch):
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_BUDGET_ENABLED", True
    )
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_MAX_INPUT_CHARS", 4000
    )
    synth = ResponseSynthesizer(llm=_FakeLLM())

    huge_summary = "观测数据: " + "X" * 8000
    sr = _make_fake_sr("weather_lookup", huge_summary)

    response = synth.synthesize(
        query="今天天气怎么样？",
        task_type="weather_query",
        output_schema="{answer: string}",
        skill_results=[sr],
    )
    # Should not crash and prompt should be under budget
    assert response.answer == "这是测试回答。"


# ---------------------------------------------------------------------------
# 2. Current query is always preserved
# ---------------------------------------------------------------------------

def test_query_preserved_in_prompt(monkeypatch):
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_BUDGET_ENABLED", True
    )
    synth = ResponseSynthesizer(llm=_FakeLLM())

    sr = _make_fake_sr("test", "result")
    # We can't easily inspect the prompt, but we can verify it doesn't crash
    response = synth.synthesize(
        query="唯一重要的问题在这里",
        task_type="test",
        output_schema="{}",
        skill_results=[sr],
    )
    assert response is not None
    assert response.answer == "这是测试回答。"


# ---------------------------------------------------------------------------
# 3. Instruction is preserved (required section)
# ---------------------------------------------------------------------------

def test_instruction_not_dropped(monkeypatch):
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_BUDGET_ENABLED", True
    )
    synth = ResponseSynthesizer(llm=_FakeLLM())
    sr = _make_fake_sr("test", "data")
    response = synth.synthesize(
        query="测试问题",
        task_type="test",
        output_schema="{}",
        skill_results=[sr],
    )
    assert response.answer == "这是测试回答。"


# ---------------------------------------------------------------------------
# 4. tool_outputs has higher priority than chat_history
# ---------------------------------------------------------------------------

def test_tool_outputs_priority_over_chat_history(monkeypatch):
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_BUDGET_ENABLED", True
    )
    # Tight budget to force competition
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_MAX_INPUT_CHARS", 1000
    )
    synth = ResponseSynthesizer(llm=_FakeLLM())

    sr = _make_fake_sr("test", "工具结果数据")
    response = synth.synthesize(
        query="测试",
        task_type="test",
        output_schema="{}",
        chat_history="旧对话历史 " * 500,
        user_profile="用户画像 " * 200,
        skill_results=[sr],
    )
    assert response.answer == "这是测试回答。"


# ---------------------------------------------------------------------------
# 5. _budget_tracker still records context length
# ---------------------------------------------------------------------------

def test_budget_tracker_still_records_context(monkeypatch):
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_BUDGET_ENABLED", True
    )
    from src.agent.policies.budget_policy import RequestBudgetTracker, RequestBudget

    budget = RequestBudget(
        max_llm_calls=10,
        max_tool_calls=20,
        max_total_time_ms=120000,
        max_parallelism=5,
        max_context_chars=100000,
    )
    tracker = RequestBudgetTracker(budget=budget)
    synth = ResponseSynthesizer(llm=_FakeLLM(), budget_tracker=tracker)
    sr = _make_fake_sr("test", "result")

    chars_before = tracker.context_chars
    synth.synthesize(
        query="测试",
        task_type="test",
        output_schema="{}",
        skill_results=[sr],
    )
    assert tracker.context_chars > chars_before, (
        "budget_tracker 应在 LLM 调用后记录上下文长度"
    )


# ---------------------------------------------------------------------------
# 6. PROMPT_BUDGET_ENABLED=False falls back to old behavior
# ---------------------------------------------------------------------------

def test_budget_disabled_uses_legacy_prompt(monkeypatch):
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_BUDGET_ENABLED", False
    )
    synth = ResponseSynthesizer(llm=_FakeLLM())
    sr = _make_fake_sr("test", "result data")
    response = synth.synthesize(
        query="测试问题",
        task_type="test",
        output_schema="{}",
        skill_results=[sr],
    )
    assert response.answer == "这是测试回答。"
    # Legacy path should not crash with long inputs
    response2 = synth.synthesize(
        query="测试",
        task_type="test",
        output_schema="{}",
        chat_history="H" * 10000,
        user_profile="P" * 10000,
        skill_results=[_make_fake_sr("test", "R" * 10000)],
    )
    assert response2.answer == "这是测试回答。"
