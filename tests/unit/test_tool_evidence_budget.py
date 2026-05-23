"""Phase 4 unit tests for tool evidence budget governance."""

import pytest
from types import SimpleNamespace

from src.agent.policies.tool_evidence_budget import (
    ToolEvidenceCompactor,
    ToolEvidenceItem,
    ToolEvidenceBudgetResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill_result(name, summary, success=True, sources=None, data=None, error_message=""):
    sources = sources or []
    data = data or {}
    sr = SimpleNamespace(
        skill_name=name,
        success=success,
        summary=summary,
        sources=sources,
        data=data,
        error_message=error_message,
        to_tool_timeline_entry=lambda: {"tool": name, "summary": summary[:100]},
    )
    return sr


def _compactor(**overrides):
    return ToolEvidenceCompactor(
        max_single_chars=overrides.get("max_single_chars", 800),
        max_total_chars=overrides.get("max_total_chars", 3000),
        error_max_chars=overrides.get("error_max_chars", 300),
        compacted_max_chars=overrides.get("compacted_max_chars", 1800),
    )


# ---------------------------------------------------------------------------
# 1. Empty skill_results → empty evidence
# ---------------------------------------------------------------------------

def test_empty_skill_results():
    compactor = _compactor()
    result = compactor.compact_skill_results([])
    assert result.text == ""
    assert result.total_chars_before == 0
    assert result.total_chars_after == 0


# ---------------------------------------------------------------------------
# 2. Single overlong tool summary → trimmed
# ---------------------------------------------------------------------------

def test_single_overlong_summary_trimmed():
    compactor = _compactor(max_single_chars=200)
    sr = _make_skill_result("weather-lookup", "W" * 500)
    result = compactor.compact_skill_results([sr])
    assert result.total_chars_after <= 250  # header + trimmed content
    assert "weather-lookup" in result.trimmed_tools


# ---------------------------------------------------------------------------
# 3. Multiple tool summaries within total budget
# ---------------------------------------------------------------------------

def test_multiple_tools_within_total_budget():
    compactor = _compactor(max_total_chars=2000, max_single_chars=300)
    srs = [
        _make_skill_result("tool-a", "A" * 200),
        _make_skill_result("tool-b", "B" * 250),
        _make_skill_result("tool-c", "C" * 150),
    ]
    result = compactor.compact_skill_results(srs, max_total_chars=2000)
    assert result.total_chars_after <= 2000
    assert result.text != ""


def test_multiple_tools_exceed_total_budget():
    compactor = _compactor(max_total_chars=500, max_single_chars=300)
    srs = [
        _make_skill_result("tool-a", "A" * 400),
        _make_skill_result("tool-b", "B" * 400),
        _make_skill_result("tool-c", "C" * 400),
    ]
    result = compactor.compact_skill_results(srs, max_total_chars=500)
    assert result.total_chars_after <= 500
    # At least one tool trimmed or dropped
    assert len(result.trimmed_tools) + len(result.dropped_tools) >= 1


# ---------------------------------------------------------------------------
# 4. Successful tools preferred over failed tools
# ---------------------------------------------------------------------------

def test_success_preferred_over_error():
    compactor = _compactor(max_total_chars=800, max_single_chars=400)
    srs = [
        _make_skill_result("good-tool", "G" * 300, success=True),
        _make_skill_result("bad-tool", "B" * 300, success=False, error_message="failed"),
    ]
    result = compactor.compact_skill_results(srs, max_total_chars=800)
    # Success tool should be present, error tool may be trimmed
    assert "good-tool" in result.text
    assert result.text != ""


def test_error_tool_dropped_when_budget_tight():
    compactor = _compactor(max_total_chars=300, max_single_chars=300)
    srs = [
        _make_skill_result("good-tool", "G" * 250, success=True),
        _make_skill_result("bad-tool", "B" * 250, success=False, error_message="failed"),
    ]
    result = compactor.compact_skill_results(srs, max_total_chars=300)
    # Success tool should be present, error may be dropped
    assert result.text != ""
    assert "good-tool" in result.text


# ---------------------------------------------------------------------------
# 5. Tools with sources get higher priority
# ---------------------------------------------------------------------------

def test_sources_boost_priority():
    compactor = _compactor(max_total_chars=500, max_single_chars=400)
    srs = [
        _make_skill_result("no-source", "N" * 300, sources=[]),
        _make_skill_result("has-source", "H" * 300, sources=[{"id": "src1"}]),
    ]
    result = compactor.compact_skill_results(srs, max_total_chars=500)
    # has-source should be present, no-source may be trimmed
    assert result.text != ""


# ---------------------------------------------------------------------------
# 6. Error tool truncated to error max chars
# ---------------------------------------------------------------------------

def test_error_tool_truncated_to_error_max():
    compactor = _compactor(error_max_chars=100)
    sr = _make_skill_result("bad-tool", "", success=False, error_message="E" * 500)
    result = compactor.compact_skill_results([sr])
    # The error chunk includes "[bad-tool] error\nerror: " prefix + error message
    # Total should be <= prefix + error_max_chars (~100 + ~20)
    assert result.total_chars_after <= 200
    # Should be flagged as trimmed
    assert "bad-tool" in result.trimmed_tools or len(result.text) < 500


# ---------------------------------------------------------------------------
# 7. dropped_tools correctly recorded
# ---------------------------------------------------------------------------

def test_dropped_tools_recorded():
    compactor = _compactor(max_total_chars=200, max_single_chars=100)
    srs = [_make_skill_result(f"tool-{i}", "X" * 200) for i in range(10)]
    result = compactor.compact_skill_results(srs, max_total_chars=200)
    assert len(result.dropped_tools) >= 1
    # Trimmed or dropped should be non-empty
    assert len(result.trimmed_tools) + len(result.dropped_tools) >= 1


# ---------------------------------------------------------------------------
# 8. TOOL_EVIDENCE_BUDGET_ENABLED=False — behavior compatible
# ---------------------------------------------------------------------------

def test_disabled_still_produces_output():
    """When compactor is not used (disabled), the raw collected_outputs path works."""
    # This test verifies that the compactor itself works when called directly
    # The ResponseSynthesizer disable path is tested in the integration test
    compactor = _compactor(max_total_chars=99999)
    srs = [
        _make_skill_result("a", "summary a"),
        _make_skill_result("b", "summary b"),
    ]
    result = compactor.compact_skill_results(srs, max_total_chars=99999)
    assert "a" in result.text
    assert "b" in result.text
    assert not result.dropped_tools


# ---------------------------------------------------------------------------
# 9. compact_items output format is stable
# ---------------------------------------------------------------------------

def test_output_format_stable():
    compactor = _compactor(max_total_chars=99999)
    items = [
        ToolEvidenceItem(tool_name="t1", status="success", summary="result 1", priority=60),
        ToolEvidenceItem(tool_name="t2", status="error", summary="", error_message="failed",
                        priority=30),
    ]
    result = compactor.compact_items(items, max_total_chars=99999)
    assert "[t1] success" in result.text
    assert "result 1" in result.text
    assert "[t2] error" in result.text
    assert "error: failed" in result.text


def test_output_format_multiple_success():
    compactor = _compactor(max_total_chars=99999)
    items = [
        ToolEvidenceItem(tool_name="w", status="success", summary="weather ok", priority=60),
        ToolEvidenceItem(tool_name="p", status="success", summary="position ok", priority=60),
    ]
    result = compactor.compact_items(items, max_total_chars=99999)
    assert "[w] success" in result.text
    assert "[p] success" in result.text
    assert "weather ok" in result.text
    assert "position ok" in result.text


# ---------------------------------------------------------------------------
# 10. No external LLM dependency
# ---------------------------------------------------------------------------

def test_no_llm_dependency():
    """Verify that the compactor has no LLM imports or dependencies."""
    import inspect
    source = inspect.getsource(ToolEvidenceCompactor.compact_items)
    assert "llm" not in source.lower()
    assert "invoke" not in source.lower()
    assert "chat" not in source.lower()


# ---------------------------------------------------------------------------
# 11. Edge cases
# ---------------------------------------------------------------------------

def test_zero_budget():
    compactor = _compactor(max_total_chars=0)
    srs = [_make_skill_result("tool", "data")]
    result = compactor.compact_skill_results(srs, max_total_chars=1)
    # Should not crash; result may be empty or minimal
    assert isinstance(result.text, str)


def test_all_error_tools():
    compactor = _compactor(max_total_chars=3000, error_max_chars=200)
    srs = [
        _make_skill_result("e1", "", success=False, error_message="err1"),
        _make_skill_result("e2", "", success=False, error_message="err2"),
    ]
    result = compactor.compact_skill_results(srs)
    assert "error" in result.text
    assert result.total_chars_after > 0


def test_tool_char_counts_populated():
    compactor = _compactor(max_total_chars=99999)
    srs = [
        _make_skill_result("t1", "hello"),
        _make_skill_result("t2", "world"),
    ]
    result = compactor.compact_skill_results(srs)
    assert "t1" in result.tool_char_counts
    assert "t2" in result.tool_char_counts
    assert result.tool_char_counts["t1"] > 0
    assert result.tool_char_counts["t2"] > 0


def test_single_tool_within_bounds_not_trimmed():
    compactor = _compactor(max_single_chars=800)
    sr = _make_skill_result("tool", "short summary")
    result = compactor.compact_skill_results([sr])
    assert "tool" not in result.trimmed_tools
    assert result.text != ""
