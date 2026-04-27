"""
Phase 1 TaskProfile 测试
目标：验证 TaskProfile 构造、RequestRouter.profile() 行为及 legacy_route 映射。

当前状态：TaskProfile 模型已稳定，但主执行路径仍使用 RouteDecision。
          RequestRouter.profile() 仅供旁路观测，主路径调用 route()。
收敛计划：待 UnifiedExecutionEngine 实现后，profile() 成为主路由入口，
          TaskProfile 替代 RouteDecision 驱动执行决策。
"""
from __future__ import annotations

import sys

import pytest

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()

from src.agent.models.task_profile import TaskProfile, LEGACY_ROUTE_MAP
from src.agent.request_router import RequestRouter


class TestTaskProfileConstruction:
    """验证 TaskProfile 直接构造与 from_legacy_route 工厂方法。"""

    def test_from_legacy_smalltalk(self):
        p = TaskProfile.from_legacy_route(
            route="direct_task",
            task_type="smalltalk",
            confidence=0.98,
        )
        assert p.complexity == "low"
        assert p.openness == "low"
        assert p.tool_need == "none"
        assert p.legacy_route == "direct_task"
        assert p.task_type == "smalltalk"

    def test_from_legacy_simple_qa(self):
        p = TaskProfile.from_legacy_route(
            route="direct_task",
            task_type="simple_qa",
            confidence=0.8,
        )
        assert p.complexity == "low"
        assert p.tool_need == "none"
        assert p.legacy_route == "direct_task"

    def test_from_legacy_single_tool(self):
        p = TaskProfile.from_legacy_route(
            route="direct_task",
            task_type="single_tool_lookup",
            confidence=0.9,
            matched_skills=["weather-lookup"],
        )
        assert p.tool_need == "single"
        assert p.complexity == "low"
        assert p.matched_skills == ["weather-lookup"]

    def test_from_legacy_planned_multi_skill(self):
        p = TaskProfile.from_legacy_route(
            route="planned_task",
            task_type="observation_recommendation",
            confidence=0.82,
            matched_skills=["weather-lookup", "observation-planner"],
        )
        assert p.tool_need == "multi"
        assert p.complexity == "high"
        assert p.openness == "low"
        assert len(p.matched_skills) == 2

    def test_from_legacy_planned_single_skill(self):
        p = TaskProfile.from_legacy_route(
            route="planned_task",
            task_type="celestial_event_analysis",
            confidence=0.74,
            matched_skills=["celestial-events-forecast"],
        )
        assert p.tool_need == "single"
        assert p.complexity == "medium"

    def test_from_legacy_fallback_react(self):
        p = TaskProfile.from_legacy_route(
            route="fallback_react",
            task_type="open_domain_reasoning",
            confidence=0.45,
        )
        assert p.complexity == "high"
        assert p.openness == "high"
        assert p.tool_need == "none"
        assert p.legacy_route == "fallback_react"

    def test_to_dict_has_all_mvp_fields(self):
        p = TaskProfile.from_legacy_route(
            route="direct_task",
            task_type="smalltalk",
            confidence=0.98,
            expected_output_schema="chat_answer_v1",
        )
        d = p.to_dict()
        required = {
            "task_type", "complexity", "openness", "tool_need",
            "matched_skills", "confidence", "expected_output_schema", "legacy_route",
        }
        assert required.issubset(d.keys())
        assert d["expected_output_schema"] == "chat_answer_v1"
        assert d["confidence"] == 0.98

    def test_legacy_route_map_covers_all_routes(self):
        for route in ("direct_task", "planned_task", "fallback_react"):
            assert route in LEGACY_ROUTE_MAP
            assert LEGACY_ROUTE_MAP[route] == route


class TestRequestRouterProfile:
    """验证 RequestRouter.profile() 输出符合 TaskProfile 规范。"""

    def setup_method(self):
        self.router = RequestRouter()

    def test_profile_returns_task_profile_instance(self):
        p = self.router.profile("你好")
        assert isinstance(p, TaskProfile)

    def test_smalltalk_profile_low_complexity_no_tools(self):
        p = self.router.profile("你好")
        assert p.task_type == "smalltalk"
        assert p.complexity == "low"
        assert p.openness == "low"
        assert p.tool_need == "none"
        assert p.legacy_route == "direct_task"

    def test_single_skill_profile_tool_need_single(self):
        p = self.router.profile("帮我查一下北京今天天气怎么样")
        assert p.tool_need == "single"
        assert "weather-lookup" in p.matched_skills
        assert p.legacy_route == "direct_task"

    def test_multi_skill_profile_planned_route(self):
        # 该查询被路由为 planned_task（含复杂词"同时"）
        p = self.router.profile("帮我看下北京今晚观测条件，同时查查有没有天象活动")
        assert p.legacy_route == "planned_task"
        assert p.task_type != "smalltalk"

    def test_multi_skill_profile_tool_need_multi_explicit(self):
        # 明确触发多 skill 的查询
        p = self.router.profile("帮我查北京的天气，并做摄影计划")
        assert p.legacy_route == "planned_task"
        if len(p.matched_skills) >= 2:
            assert p.tool_need == "multi"
        else:
            assert p.tool_need in ("single", "multi")

    def test_open_ended_profile_openness_high(self):
        p = self.router.profile("帮我写一篇关于宇宙的科幻小说")
        assert p.openness == "high"
        assert p.complexity == "high"
        assert p.tool_need == "none"
        assert p.legacy_route == "fallback_react"

    def test_profile_confidence_matches_route(self):
        p = self.router.profile("你好")
        decision = self.router.route("你好")
        assert p.confidence == decision.confidence

    def test_profile_expected_output_schema_matches_route(self):
        p = self.router.profile("你好")
        decision = self.router.route("你好")
        assert p.expected_output_schema == decision.expected_output_schema

    def test_route_still_returns_route_decision(self):
        from src.agent.request_router import RouteDecision
        decision = self.router.route("你好")
        assert isinstance(decision, RouteDecision)
        assert decision.route == "direct_task"

    def test_profile_and_route_consistent_for_same_query(self):
        query = "帮我查一下北京今天天气怎么样"
        decision = self.router.route(query)
        p = self.router.profile(query)
        assert p.legacy_route == decision.route
        assert p.task_type == decision.task_type
        assert p.matched_skills == decision.matched_skills
