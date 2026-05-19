from __future__ import annotations

import pytest

from src.agent.tool_necessity_gate import ToolNecessityGate


@pytest.mark.parametrize(
    ("query", "expected_action", "expected_reason"),
    [
        (
            "帮我算一下拍银河曝光多久合适。",
            "clarify",
            "astrophotography_exposure_missing_required_context",
        ),
        (
            "帮我查一下三体星今晚在哪个方向。",
            "clarify",
            "unrecognized_or_fictional_celestial_target",
        ),
        (
            "我想今晚在上海市中心用肉眼看 M101，能给我安排一下吗？",
            "answer_without_tool",
            "unrealistic_observing_request_can_be_answered_without_tools",
        ),
        (
            "湿度很高会不会镜头起雾？",
            "answer_without_tool",
            "observing_experience_dew_without_live_weather",
        ),
        (
            "天象预报说可见，我在城市里也能看到吗？",
            "answer_without_tool",
            "observing_experience_city_visibility_without_tools",
        ),
        (
            "为什么会有流星雨？",
            "answer_without_tool",
            "stable_knowledge_meteor_shower_mechanism",
        ),
        (
            "星云和星系有什么区别？",
            "answer_without_tool",
            "stable_knowledge_nebula_galaxy_difference",
        ),
    ],
)
def test_gate_blocks_no_tool_and_clarification_golden_cases(
    query: str,
    expected_action: str,
    expected_reason: str,
):
    decision = ToolNecessityGate().evaluate(query)

    assert decision.action == expected_action
    assert decision.reason == expected_reason
    assert decision.confidence >= 0.85


@pytest.mark.parametrize(
    ("query", "expected_skill"),
    [
        ("广州今晚适合出门观星吗？", "weather-lookup"),
        ("我这边今晚月亮高度怎么样？", "celestial-position-calculator"),
        ("这个月有什么比较重要的天象？", "celestial-events-forecast"),
        ("我想带朋友看天象，有没有适合普通人看的？", "celestial-events-forecast"),
        ("今晚月亮比较亮，观测计划要怎么调整？", "observation-planner"),
        ("如果今晚云突然变多，观测计划怎么备选？", "observation-planner"),
    ],
)
def test_gate_allows_tool_golden_cases_with_skill_hints(query: str, expected_skill: str):
    decision = ToolNecessityGate().evaluate(query)

    assert decision.action == "use_tool"
    assert expected_skill in decision.allowed_skill_hints


def test_gate_vetoes_weather_as_separate_observation_plan_step():
    decision = ToolNecessityGate().validate_tool_route(
        "如果今晚云突然变多，观测计划怎么备选？",
        ["weather-lookup", "observation-planner"],
    )

    assert decision.action == "use_tool"
    assert decision.allowed_skill_hints == ["observation-planner"]
    assert "weather-lookup" in decision.forbidden_skill_hints
