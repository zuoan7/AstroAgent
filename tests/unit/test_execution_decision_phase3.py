"""
Phase 3 ExecutionDecision 测试
目标：验证 ExecutionDecision 构造、AgentExecutionPolicy.decide() 行为，
      以及在关键样本下与 choose_path() 结果基本一致。

当前状态：ExecutionDecision 模型已稳定，decide() 受 ENABLE_EXECUTION_DECISION 控制。
          主路径仍调用 choose_path(route)；flag 默认 False，decide() 仅供观测/测试。
收敛计划：待 UnifiedExecutionEngine 实现后，decide() 接入主路径，
          choose_path() 降为兼容别名。
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()

from src.agent.models.execution_decision import ExecutionDecision, VALID_EXECUTION_MODES
from src.agent.models.task_profile import TaskProfile
from src.agent.governance import AgentExecutionPolicy


# ─────────────────────────────────────────────────────────────────
# 辅助工厂
# ─────────────────────────────────────────────────────────────────

def _profile(route: str, task_type: str, confidence: float = 0.9, matched_skills=None) -> TaskProfile:
    return TaskProfile.from_legacy_route(
        route=route,
        task_type=task_type,
        confidence=confidence,
        matched_skills=matched_skills or [],
    )


def _policy(**kwargs) -> AgentExecutionPolicy:
    defaults = dict(mode="hybrid", enable_planner=False, enable_react_fallback=True)
    defaults.update(kwargs)
    return AgentExecutionPolicy(**defaults)


# ─────────────────────────────────────────────────────────────────


class TestExecutionDecisionConstruction:
    """验证 ExecutionDecision 数据类构造与约束。"""

    def test_valid_modes(self):
        for mode in ("direct", "planned", "react"):
            d = ExecutionDecision(mode=mode, reason="test")
            assert d.mode == mode

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            ExecutionDecision(mode="unknown_mode", reason="x")

    def test_default_fallback_modes_empty(self):
        d = ExecutionDecision(mode="direct", reason="r")
        assert d.fallback_modes == []

    def test_default_legacy_execution_path(self):
        d = ExecutionDecision(mode="direct", reason="r")
        assert d.legacy_execution_path == "unknown"

    def test_to_dict_structure(self):
        d = ExecutionDecision(
            mode="planned",
            reason="multi_tool",
            fallback_modes=["react"],
            legacy_execution_path="planned",
        )
        result = d.to_dict()
        assert result["mode"] == "planned"
        assert result["reason"] == "multi_tool"
        assert result["fallback_modes"] == ["react"]
        assert result["legacy_execution_path"] == "planned"

    def test_frozen_immutable(self):
        d = ExecutionDecision(mode="direct", reason="r")
        with pytest.raises((AttributeError, TypeError)):
            d.mode = "react"  # type: ignore[misc]


class TestDecideFeatureFlagDisabled:
    """ENABLE_EXECUTION_DECISION=False 时委托 choose_path，结果可观测但不接入主路径。"""

    def _decide_with_flag_off(self, route, task_type, policy_kwargs=None, matched_skills=None):
        policy = _policy(**(policy_kwargs or {}))
        profile = _profile(route, task_type, matched_skills=matched_skills)
        with patch("src.agent.governance.settings") as mock_settings:
            mock_settings.ENABLE_EXECUTION_DECISION = False
            mock_settings.AGENT_MODE = policy.mode
            return policy.decide(profile)

    def test_direct_task_delegates_to_choose_path(self):
        d = self._decide_with_flag_off("direct_task", "smalltalk")
        assert d.mode == "direct"
        assert "delegate_to_choose_path" in d.reason

    def test_planned_task_delegates_to_choose_path(self):
        d = self._decide_with_flag_off("planned_task", "observation_recommendation", matched_skills=["weather-lookup", "observation-planner"])
        assert d.mode == "planned"

    def test_fallback_react_delegates_to_choose_path(self):
        d = self._decide_with_flag_off("fallback_react", "open_domain_reasoning")
        assert d.mode == "react"


class TestDecideFeatureFlagEnabled:
    """ENABLE_EXECUTION_DECISION=True 时按 TaskProfile 三维规则决策。"""

    def _decide(self, route, task_type, policy_kwargs=None, matched_skills=None):
        policy = _policy(**(policy_kwargs or {}))
        profile = _profile(route, task_type, matched_skills=matched_skills)
        with patch("src.agent.governance.settings") as mock_settings:
            mock_settings.ENABLE_EXECUTION_DECISION = True
            return policy.decide(profile)

    # 规则 1：complexity low + tool_need none -> direct
    def test_smalltalk_direct(self):
        d = self._decide("direct_task", "smalltalk")
        assert d.mode == "direct"
        assert d.reason == "low_complexity_no_tools"

    def test_simple_qa_direct(self):
        d = self._decide("direct_task", "simple_qa")
        assert d.mode == "direct"
        assert d.reason == "low_complexity_no_tools"

    # 规则 2：tool_need single + openness!=high -> direct
    def test_single_tool_direct(self):
        d = self._decide("direct_task", "single_tool_lookup", matched_skills=["weather-lookup"])
        assert d.mode == "direct"
        assert d.reason == "single_tool_low_openness"

    # 规则 3：tool_need multi 或 high complexity -> planned
    def test_multi_tool_planned(self):
        d = self._decide("planned_task", "observation_recommendation", matched_skills=["weather-lookup", "observation-planner"])
        assert d.mode == "planned"
        assert "multi_tool" in d.reason or "high_complexity" in d.reason

    def test_high_complexity_single_skill_planned(self):
        # planned_task with single skill => complexity=medium, tool_need=single
        # 按规则 2 应为 direct（openness=low）
        d = self._decide("planned_task", "observation_recommendation", matched_skills=["weather-lookup"])
        assert d.mode in ("direct", "planned")  # medium complexity, single tool -> direct

    # 规则 4：openness high -> react
    def test_open_domain_react(self):
        d = self._decide("fallback_react", "open_domain_reasoning")
        assert d.mode == "react"
        assert d.reason == "high_openness_react"

    # 全局 mode=react 覆盖
    def test_global_react_mode_override(self):
        d = self._decide("direct_task", "smalltalk", policy_kwargs={"mode": "react"})
        assert d.mode == "react"
        assert d.reason == "global_mode_override_react"


class TestDecideConsistencyWithChoosePath:
    """验证 decide() 在关键样本上与 choose_path() 基本一致（核心兼容性保证）。"""

    SAMPLES = [
        # (route, task_type, matched_skills, expected_choose_path)
        ("direct_task", "smalltalk", [], "direct"),
        ("direct_task", "simple_qa", [], "direct"),
        ("direct_task", "single_tool_lookup", ["weather-lookup"], "direct"),
        ("planned_task", "observation_recommendation", ["weather-lookup", "observation-planner"], "planned"),
        ("fallback_react", "open_domain_reasoning", [], "react"),
    ]

    def test_consistency_flag_enabled(self):
        """flag=True 时，decide().legacy_execution_path 与 choose_path() 结果相同。"""
        policy = _policy()
        mismatches = []
        for route, task_type, skills, expected in self.SAMPLES:
            profile = _profile(route, task_type, matched_skills=skills)
            legacy_path = policy.choose_path(route)
            with patch("src.agent.governance.settings") as mock_settings:
                mock_settings.ENABLE_EXECUTION_DECISION = True
                d = policy.decide(profile)
            if d.legacy_execution_path != legacy_path:
                mismatches.append(
                    f"{route}/{task_type}: decide={d.legacy_execution_path}, choose_path={legacy_path}"
                )
        assert not mismatches, f"Mismatches found:\n" + "\n".join(mismatches)

    def test_consistency_flag_disabled(self):
        """flag=False 时，decide() 完全委托 choose_path()，结果必须 100% 一致。"""
        policy = _policy()
        for route, task_type, skills, expected in self.SAMPLES:
            profile = _profile(route, task_type, matched_skills=skills)
            legacy_path = policy.choose_path(route)
            with patch("src.agent.governance.settings") as mock_settings:
                mock_settings.ENABLE_EXECUTION_DECISION = False
                d = policy.decide(profile)
            assert d.mode == legacy_path, (
                f"{route}/{task_type}: decide={d.mode}, choose_path={legacy_path}"
            )

    def test_fallback_react_not_broken(self):
        """fallback_react 场景不论 flag 状态均应产出 react。"""
        policy = _policy()
        profile = _profile("fallback_react", "open_domain_reasoning")
        for flag_val in (True, False):
            with patch("src.agent.governance.settings") as mock_settings:
                mock_settings.ENABLE_EXECUTION_DECISION = flag_val
                d = policy.decide(profile)
            assert d.mode == "react", f"flag={flag_val}: expected react, got {d.mode}"


class TestDecideEdgeCases:
    """边界情况测试。"""

    def test_decide_returns_execution_decision_type(self):
        policy = _policy()
        profile = _profile("direct_task", "smalltalk")
        with patch("src.agent.governance.settings") as mock_settings:
            mock_settings.ENABLE_EXECUTION_DECISION = True
            result = policy.decide(profile)
        assert isinstance(result, ExecutionDecision)

    def test_decide_with_none_context_ok(self):
        """context 参数为 None 时不应抛出。"""
        policy = _policy()
        profile = _profile("direct_task", "smalltalk")
        with patch("src.agent.governance.settings") as mock_settings:
            mock_settings.ENABLE_EXECUTION_DECISION = True
            result = policy.decide(profile, context=None)
        assert result.mode == "direct"

    def test_decide_multi_tool_has_fallback_react(self):
        """多工具 -> planned，且有 react 兜底（enable_react_fallback=True）。"""
        policy = _policy(enable_react_fallback=True)
        profile = _profile("planned_task", "observation_recommendation", matched_skills=["weather-lookup", "observation-planner"])
        with patch("src.agent.governance.settings") as mock_settings:
            mock_settings.ENABLE_EXECUTION_DECISION = True
            d = policy.decide(profile)
        assert d.mode == "planned"
        assert "react" in d.fallback_modes

    def test_decide_multi_tool_no_fallback_react(self):
        """enable_react_fallback=False 时，fallback_modes 不含 react。"""
        policy = _policy(enable_react_fallback=False)
        profile = _profile("planned_task", "observation_recommendation", matched_skills=["weather-lookup", "observation-planner"])
        with patch("src.agent.governance.settings") as mock_settings:
            mock_settings.ENABLE_EXECUTION_DECISION = True
            d = policy.decide(profile)
        assert d.mode == "planned"
        assert "react" not in d.fallback_modes

    def test_choose_path_still_works_after_decide_added(self):
        """choose_path() 在新增 decide() 后仍完整可用。"""
        policy = _policy()
        assert policy.choose_path("direct_task") == "direct"
        assert policy.choose_path("planned_task") == "planned"
        assert policy.choose_path("fallback_react") == "react"
        assert policy.choose_path(None) == "react"  # enable_react_fallback=True
