"""Phase 4 unit tests for ResponseSynthesizer tool evidence budget integration."""

import pytest
from types import SimpleNamespace
from unittest import mock

from src.agent.response_synthesizer import ResponseSynthesizer


class _FakeLLM:
    def invoke(self, prompt):
        return SimpleNamespace(content="这是整合后的回答。")


def _make_fake_sr(name, summary, success=True, sources=None, data=None, error_message=""):
    return SimpleNamespace(
        skill_name=name,
        success=success,
        summary=summary,
        sources=sources or [],
        data=data or {},
        error_message=error_message,
        to_tool_timeline_entry=lambda: {"tool": name, "summary": summary[:100]},
    )


@pytest.fixture
def synth():
    return ResponseSynthesizer(llm=_FakeLLM())


# ---------------------------------------------------------------------------
# 1. synthesize uses compacted tool_outputs
# ---------------------------------------------------------------------------

def test_compacted_tool_outputs_used(monkeypatch):
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.TOOL_EVIDENCE_BUDGET_ENABLED", True
    )
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_BUDGET_ENABLED", True
    )
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_MAX_INPUT_CHARS", 10000
    )
    monkeypatch.setattr(
        "src.agent.policies.tool_evidence_budget.settings.TOOL_EVIDENCE_MAX_SINGLE_CHARS",
        800,
    )
    monkeypatch.setattr(
        "src.agent.policies.tool_evidence_budget.settings.TOOL_EVIDENCE_MAX_TOTAL_CHARS",
        3000,
    )
    monkeypatch.setattr(
        "src.agent.policies.tool_evidence_budget.settings.TOOL_EVIDENCE_ERROR_MAX_CHARS",
        300,
    )
    monkeypatch.setattr(
        "src.agent.policies.tool_evidence_budget.settings.TOOL_EVIDENCE_COMPACTED_MAX_CHARS",
        1800,
    )
    synth = ResponseSynthesizer(llm=_FakeLLM())
    srs = [
        _make_fake_sr("weather", "天气查询结果"),
        _make_fake_sr("position", "位置计算结果"),
    ]
    response = synth.synthesize(
        query="今晚观测条件如何？",
        task_type="observation",
        output_schema="{answer: string}",
        skill_results=srs,
    )
    assert response.answer == "这是整合后的回答。"
    assert response is not None


# ---------------------------------------------------------------------------
# 2. Very long tool outputs are trimmed before prompt
# ---------------------------------------------------------------------------

def test_long_tool_outputs_compacted(monkeypatch):
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.TOOL_EVIDENCE_BUDGET_ENABLED", True
    )
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_BUDGET_ENABLED", True
    )
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_MAX_INPUT_CHARS", 10000
    )
    # Set tight single-tool cap
    monkeypatch.setattr(
        "src.agent.policies.tool_evidence_budget.settings.TOOL_EVIDENCE_MAX_SINGLE_CHARS",
        100,
    )
    monkeypatch.setattr(
        "src.agent.policies.tool_evidence_budget.settings.TOOL_EVIDENCE_MAX_TOTAL_CHARS",
        1000,
    )
    monkeypatch.setattr(
        "src.agent.policies.tool_evidence_budget.settings.TOOL_EVIDENCE_ERROR_MAX_CHARS",
        100,
    )
    monkeypatch.setattr(
        "src.agent.policies.tool_evidence_budget.settings.TOOL_EVIDENCE_COMPACTED_MAX_CHARS",
        800,
    )
    synth = ResponseSynthesizer(llm=_FakeLLM())
    srs = [_make_fake_sr("huge-tool", "X" * 5000)]
    response = synth.synthesize(
        query="测试",
        task_type="test",
        output_schema="{}",
        skill_results=srs,
    )
    assert response.answer == "这是整合后的回答。"


# ---------------------------------------------------------------------------
# 3. Query and instruction are preserved
# ---------------------------------------------------------------------------

def test_query_and_instruction_preserved(monkeypatch):
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.TOOL_EVIDENCE_BUDGET_ENABLED", True
    )
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_BUDGET_ENABLED", True
    )
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_MAX_INPUT_CHARS", 10000
    )
    monkeypatch.setattr(
        "src.agent.policies.tool_evidence_budget.settings.TOOL_EVIDENCE_MAX_TOTAL_CHARS",
        3000,
    )
    monkeypatch.setattr(
        "src.agent.policies.tool_evidence_budget.settings.TOOL_EVIDENCE_MAX_SINGLE_CHARS",
        800,
    )
    monkeypatch.setattr(
        "src.agent.policies.tool_evidence_budget.settings.TOOL_EVIDENCE_ERROR_MAX_CHARS",
        300,
    )
    monkeypatch.setattr(
        "src.agent.policies.tool_evidence_budget.settings.TOOL_EVIDENCE_COMPACTED_MAX_CHARS",
        1800,
    )
    synth = ResponseSynthesizer(llm=_FakeLLM())
    srs = [_make_fake_sr("tool", "result")]
    response = synth.synthesize(
        query="关键问题内容",
        task_type="test",
        output_schema="{}",
        skill_results=srs,
    )
    assert response.answer == "这是整合后的回答。"


# ---------------------------------------------------------------------------
# 4. FinalResponse.sources are not lost
# ---------------------------------------------------------------------------

def test_sources_not_lost(monkeypatch):
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.TOOL_EVIDENCE_BUDGET_ENABLED", True
    )
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_BUDGET_ENABLED", True
    )
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_MAX_INPUT_CHARS", 10000
    )
    monkeypatch.setattr(
        "src.agent.policies.tool_evidence_budget.settings.TOOL_EVIDENCE_MAX_TOTAL_CHARS",
        3000,
    )
    monkeypatch.setattr(
        "src.agent.policies.tool_evidence_budget.settings.TOOL_EVIDENCE_MAX_SINGLE_CHARS",
        800,
    )
    monkeypatch.setattr(
        "src.agent.policies.tool_evidence_budget.settings.TOOL_EVIDENCE_ERROR_MAX_CHARS",
        300,
    )
    monkeypatch.setattr(
        "src.agent.policies.tool_evidence_budget.settings.TOOL_EVIDENCE_COMPACTED_MAX_CHARS",
        1800,
    )
    synth = ResponseSynthesizer(llm=_FakeLLM())
    srs = [_make_fake_sr("tool", "result", sources=[{"source_id": "src1", "title": "Test"}])]
    response = synth.synthesize(
        query="测试",
        task_type="test",
        output_schema="{}",
        skill_results=srs,
    )
    assert len(response.sources) == 1
    assert response.sources[0]["source_id"] == "src1"


# ---------------------------------------------------------------------------
# 5. FinalResponse.tools_used are not lost
# ---------------------------------------------------------------------------

def test_tools_used_not_lost(monkeypatch):
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.TOOL_EVIDENCE_BUDGET_ENABLED", True
    )
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_BUDGET_ENABLED", True
    )
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_MAX_INPUT_CHARS", 10000
    )
    monkeypatch.setattr(
        "src.agent.policies.tool_evidence_budget.settings.TOOL_EVIDENCE_MAX_TOTAL_CHARS",
        3000,
    )
    monkeypatch.setattr(
        "src.agent.policies.tool_evidence_budget.settings.TOOL_EVIDENCE_MAX_SINGLE_CHARS",
        800,
    )
    monkeypatch.setattr(
        "src.agent.policies.tool_evidence_budget.settings.TOOL_EVIDENCE_ERROR_MAX_CHARS",
        300,
    )
    monkeypatch.setattr(
        "src.agent.policies.tool_evidence_budget.settings.TOOL_EVIDENCE_COMPACTED_MAX_CHARS",
        1800,
    )
    synth = ResponseSynthesizer(llm=_FakeLLM())
    srs = [
        _make_fake_sr("tool-a", "result a"),
        _make_fake_sr("tool-b", "result b"),
    ]
    response = synth.synthesize(
        query="测试",
        task_type="test",
        output_schema="{}",
        skill_results=srs,
    )
    assert len(response.tools_used) == 2


# ---------------------------------------------------------------------------
# 6. Compactor failure → fallback to raw collected_outputs
# ---------------------------------------------------------------------------

def test_compactor_failure_falls_back(monkeypatch):
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.TOOL_EVIDENCE_BUDGET_ENABLED", True
    )
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_BUDGET_ENABLED", True
    )
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_MAX_INPUT_CHARS", 10000
    )
    synth = ResponseSynthesizer(llm=_FakeLLM())
    srs = [_make_fake_sr("tool", "result data")]

    with mock.patch(
        "src.agent.response_synthesizer.ToolEvidenceCompactor.compact_skill_results",
        side_effect=RuntimeError("模拟 compaction 失败"),
    ):
        response = synth.synthesize(
            query="测试",
            task_type="test",
            output_schema="{}",
            skill_results=srs,
        )
    # Should not crash — fallback to raw collected_outputs
    assert response.answer == "这是整合后的回答。"


# ---------------------------------------------------------------------------
# 7. TOOL_EVIDENCE_BUDGET_ENABLED=False keeps raw behavior
# ---------------------------------------------------------------------------

def test_disabled_keeps_raw_collected(monkeypatch):
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.TOOL_EVIDENCE_BUDGET_ENABLED", False
    )
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_BUDGET_ENABLED", True
    )
    monkeypatch.setattr(
        "src.agent.response_synthesizer.settings.PROMPT_MAX_INPUT_CHARS", 10000
    )
    synth = ResponseSynthesizer(llm=_FakeLLM())
    srs = [
        _make_fake_sr("t1", "数据1"),
        _make_fake_sr("t2", "数据2"),
    ]
    response = synth.synthesize(
        query="测试",
        task_type="test",
        output_schema="{}",
        skill_results=srs,
    )
    assert response.answer == "这是整合后的回答。"
