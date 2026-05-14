"""Unit tests for PromptBudgetManager — Phase 2 prompt budget governance."""

import pytest

from src.agent.policies.prompt_budget import (
    PromptBudgetManager,
    PromptBudgetResult,
    PromptSection,
)


@pytest.fixture
def mgr():
    return PromptBudgetManager(max_chars=2000, section_min_chars=40, log_trimmed=False)


# ---------------------------------------------------------------------------
# 1. Required sections are always preserved
# ---------------------------------------------------------------------------

def test_required_sections_preserved(mgr):
    sections = [
        PromptSection("instruction", "You are a helpful assistant.", priority=100, required=True),
        PromptSection("query", "What is the sky?", priority=100, required=True),
    ]
    result = mgr.fit_sections(sections, max_chars=2000)
    assert "instruction" in result.text
    assert "query" in result.text
    assert "instruction" not in result.trimmed_sections
    assert "query" not in result.dropped_sections


def test_required_section_trimmed_when_insufficient_budget(mgr):
    """When budget is extremely tight, even required sections are trimmed — but NOT dropped."""
    long_inst = "X" * 500
    sections = [
        PromptSection("instruction", long_inst, priority=100, required=True),
    ]
    result = mgr.fit_sections(sections, max_chars=150)
    assert "instruction" in result.text
    assert "instruction" not in result.dropped_sections
    # Required section may be trimmed but never dropped
    assert result.total_chars_after <= 150


# ---------------------------------------------------------------------------
# 2. Higher-priority sections are preferred
# ---------------------------------------------------------------------------

def test_high_priority_preferred(mgr):
    sections = [
        PromptSection("low", "A" * 1000, priority=10),
        PromptSection("high", "B" * 1000, priority=90),
    ]
    result = mgr.fit_sections(sections, max_chars=800)
    # high-priority section should survive better
    assert result.section_char_counts.get("high", 0) > result.section_char_counts.get("low", 0)


# ---------------------------------------------------------------------------
# 3. Low-priority sections trimmed or dropped on budget overflow
# ---------------------------------------------------------------------------

def test_low_priority_dropped(mgr):
    sections = [
        PromptSection("required_inst", "Keep this.", priority=100, required=True),
        PromptSection("required_query", "Query here.", priority=100, required=True),
        PromptSection("low_priority", "X" * 3000, priority=10),
    ]
    result = mgr.fit_sections(sections, max_chars=500)
    # low priority should be trimmed or dropped
    assert "low_priority" in result.trimmed_sections or "low_priority" in result.dropped_sections
    # required sections intact
    assert "required_inst" not in result.dropped_sections


def test_low_priority_trimmed_not_dropped_when_room(mgr):
    sections = [
        PromptSection("high", "A" * 200, priority=80),
        PromptSection("low", "B" * 2000, priority=20),
    ]
    result = mgr.fit_sections(sections, max_chars=800)
    # low should be trimmed (still room), not fully dropped
    assert "low" not in result.dropped_sections
    assert "low" in result.trimmed_sections


# ---------------------------------------------------------------------------
# 4. Section max_chars is enforced before global fitting
# ---------------------------------------------------------------------------

def test_section_max_chars_cap(mgr):
    sections = [
        PromptSection("capped", "C" * 5000, priority=50, max_chars=300),
    ]
    result = mgr.fit_sections(sections, max_chars=2000)
    assert result.section_char_counts.get("capped", 0) <= 300


def test_section_max_chars_noop_when_content_fits(mgr):
    sections = [
        PromptSection("small", "Hi", priority=50, max_chars=300),
    ]
    result = mgr.fit_sections(sections, max_chars=2000)
    assert result.section_char_counts.get("small", 0) == 2


# ---------------------------------------------------------------------------
# 5. trimmed_sections / dropped_sections are populated
# ---------------------------------------------------------------------------

def test_trimmed_and_dropped_reported(mgr):
    sections = [
        PromptSection("required", "R" * 100, priority=100, required=True),
        PromptSection("trimmed", "T" * 3000, priority=30),
        PromptSection("dropped", "D" * 2000, priority=5),
    ]
    result = mgr.fit_sections(sections, max_chars=400)
    assert isinstance(result.trimmed_sections, list)
    assert isinstance(result.dropped_sections, list)
    # At least one section was affected
    assert len(result.trimmed_sections) + len(result.dropped_sections) >= 1


# ---------------------------------------------------------------------------
# 6. Output text does not exceed max_chars
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("budget", [100, 500, 1000, 2000, 5000])
def test_output_within_budget(budget):
    mgr = PromptBudgetManager(max_chars=budget, section_min_chars=20, log_trimmed=False)
    sections = [
        PromptSection("inst", "I" * 300, priority=100, required=True),
        PromptSection("query", "Q" * 100, priority=100, required=True),
        PromptSection("a", "A" * 800, priority=70),
        PromptSection("b", "B" * 2000, priority=50),
        PromptSection("c", "C" * 4000, priority=10),
    ]
    result = mgr.fit_sections(sections)
    assert result.total_chars_after <= budget, (
        f"output {result.total_chars_after} > budget {budget}"
    )


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

def test_empty_sections(mgr):
    result = mgr.fit_sections([], max_chars=1000)
    assert result.text == ""
    assert result.total_chars_before == 0
    assert result.total_chars_after == 0


def test_all_empty_content(mgr):
    sections = [
        PromptSection("a", "", priority=100, required=True),
        PromptSection("b", "", priority=50),
    ]
    result = mgr.fit_sections(sections, max_chars=1000)
    # Sections with empty content should still render headers
    assert "=== a ===" in result.text


def test_tiny_budget_still_produces_something(mgr):
    sections = [
        PromptSection("must", "hello world", priority=100, required=True),
    ]
    # With a very small budget, the header itself is 13 chars, so output can exceed.
    # The key property: we always produce *something*.
    result = mgr.fit_sections(sections, max_chars=30)
    assert len(result.text) > 0
    # With budget=30, header(13) + "hello world"(11) = 24 fits OK
    assert len(result.text) <= 30


def test_huge_budget_fits_everything(mgr):
    sections = [
        PromptSection("a", "A" * 500, priority=50),
        PromptSection("b", "B" * 300, priority=30),
    ]
    result = mgr.fit_sections(sections, max_chars=100000)
    assert result.total_chars_after >= result.total_chars_before
    assert not result.trimmed_sections
    assert not result.dropped_sections


def test_multiple_required_sections_tight_budget(mgr):
    sections = [
        PromptSection("r1", "R1" * 300, priority=100, required=True),
        PromptSection("r2", "R2" * 300, priority=100, required=True),
        PromptSection("opt", "O" * 500, priority=30),
    ]
    result = mgr.fit_sections(sections, max_chars=800)
    # Both required sections are present (possibly trimmed)
    assert "r1" not in result.dropped_sections
    assert "r2" not in result.dropped_sections


def test_preserve_head_behavior(mgr):
    """preserve_head=True trims from the end."""
    text = "abcdefghijklmnopqrstuvwxyz"
    section = PromptSection("test", text, priority=50, preserve_head=True)
    result = mgr.fit_sections([section], max_chars=200)
    trimmed = result.text
    assert trimmed.startswith("=== test ===")
    # Content should start with "abc..." not "...xyz"
    content_part = trimmed.split("\n", 1)[1] if "\n" in trimmed else ""
    if len(text) > 10 and content_part:
        assert "abc" in content_part[:20] or "..." in content_part


def test_fit_text_tool(mgr):
    text = "hello world this is a test"
    result = mgr.fit_text("test", text, 10, preserve_head=True)
    assert len(result) <= 10
    assert result.startswith("hello")


def test_fit_text_preserve_tail(mgr):
    text = "hello world this is a test"
    result = mgr.fit_text("test", text, 10, preserve_head=False)
    assert len(result) <= 10
    assert result.endswith("test")


def test_fit_text_noop_when_within_limit(mgr):
    text = "short"
    result = mgr.fit_text("test", text, 100, preserve_head=True)
    assert result == text


def test_fit_text_zero_max(mgr):
    result = mgr.fit_text("test", "content", 0, preserve_head=True)
    assert result == ""


def test_section_char_counts_accurate(mgr):
    sections = [
        PromptSection("inst", "Hello", priority=100, required=True),
        PromptSection("query", "What?", priority=100, required=True),
    ]
    result = mgr.fit_sections(sections, max_chars=2000)
    assert result.section_char_counts.get("inst") == 5
    assert result.section_char_counts.get("query") == 5


def test_total_chars_before_tracks_original(mgr):
    sections = [
        PromptSection("a", "A" * 100, priority=50),
        PromptSection("b", "B" * 200, priority=50),
    ]
    result = mgr.fit_sections(sections, max_chars=2000)
    assert result.total_chars_before == 300


# ---------------------------------------------------------------------------
# 8. Config-based defaults
# ---------------------------------------------------------------------------

def test_manager_uses_settings_defaults(monkeypatch):
    monkeypatch.setattr(
        "src.agent.policies.prompt_budget.settings.PROMPT_MAX_INPUT_CHARS", 4000
    )
    monkeypatch.setattr(
        "src.agent.policies.prompt_budget.settings.PROMPT_SECTION_MIN_CHARS", 80
    )
    monkeypatch.setattr(
        "src.agent.policies.prompt_budget.settings.PROMPT_BUDGET_LOG_TRIMMED", False
    )
    mgr = PromptBudgetManager()
    assert mgr._max_chars == 4000
    assert mgr._section_min_chars == 80
    assert mgr._log_trimmed is False
