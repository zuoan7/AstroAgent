from __future__ import annotations

import pytest

from src.agent.request_router import RequestRouter


@pytest.mark.parametrize(
    ("query", "expected_task_type"),
    [
        ("帮我算一下拍银河曝光多久合适。", "clarification"),
        ("帮我查一下三体星今晚在哪个方向。", "clarification"),
        ("湿度很高会不会镜头起雾？", "direct_answer_no_tool"),
        ("天象预报说可见，我在城市里也能看到吗？", "direct_answer_no_tool"),
        ("为什么会有流星雨？", "direct_answer_no_tool"),
        ("APOD 是什么意思？", "direct_answer_no_tool"),
        ("什么是天球？", "direct_answer_no_tool"),
        ("我在北京。", "direct_answer_no_tool"),
        ("不对，我临时改到杭州了。", "direct_answer_no_tool"),
    ],
)
def test_router_gate_blocks_no_tool_cases_before_skill_matching(
    query: str,
    expected_task_type: str,
):
    profile = RequestRouter().profile(query)

    assert profile.legacy_route == "direct_task"
    assert profile.task_type == expected_task_type
    assert profile.tool_need == "none"
    assert profile.matched_skills == []
    assert profile.tool_necessity_action in {"clarify", "answer_without_tool"}


@pytest.mark.parametrize(
    ("query", "expected_skill", "expected_route"),
    [
        ("广州今晚适合出门观星吗？", "weather-lookup", "direct_task"),
        ("我这边今晚月亮高度怎么样？", "celestial-position-calculator", "direct_task"),
        ("这个月有什么比较重要的天象？", "celestial-events-forecast", "direct_task"),
        ("我想带朋友看天象，有没有适合普通人看的？", "celestial-events-forecast", "direct_task"),
        ("今晚月亮比较亮，观测计划要怎么调整？", "observation-planner", "planned_task"),
        ("如果今晚云突然变多，观测计划怎么备选？", "observation-planner", "planned_task"),
        ("今天的 NASA 每日天文图是什么？", "get_nasa_apod", "direct_task"),
        ("帮我查一下最近韦布望远镜有什么新结果", "web_search", "direct_task"),
    ],
)
def test_router_gate_promotes_tool_cases_with_constrained_skill_hints(
    query: str,
    expected_skill: str,
    expected_route: str,
):
    profile = RequestRouter().profile(query)

    assert profile.legacy_route == expected_route
    assert profile.matched_skills == [expected_skill]
    assert profile.tool_necessity_action == "use_tool"


def test_valid_messier_boundary_is_not_rejected_as_invalid_catalog_id():
    profile = RequestRouter().profile("M110 适合怎么观测？")

    assert profile.legacy_route == "direct_task"
    assert profile.matched_skills == ["deep-sky-observing-guide"]
    assert profile.task_type != "clarification"


def test_router_gate_removes_weather_as_separate_observation_plan_skill():
    profile = RequestRouter().profile("如果今晚云突然变多，观测计划怎么备选？")

    assert profile.legacy_route == "planned_task"
    assert profile.matched_skills == ["observation-planner"]
    assert "weather-lookup" in profile.tool_necessity_forbidden_skill_hints
