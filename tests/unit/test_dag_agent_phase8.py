"""Phase 8 DAG Agent 收口测试

目标：
1. config flags（ENABLE_UNIFIED_EXECUTION_ENGINE / ENABLE_WORKFLOW_GRAPH /
   ENABLE_UNIFIED_EXECUTION_TRACE / ENABLE_UNIFIED_EXECUTION_EVENTS）默认均为 True
2. ExecutionEngine 新主路径可被正确调用（direct / planned / react 三种模式）
3. ENABLE_UNIFIED_EXECUTION_ENGINE=False 时回退到旧 TaskOrchestrator 路径
4. planned 主路径通过 WorkflowExecutor 稳定执行
5. 旧兼容接口仍可用，并带有 deprecated/legacy 标记
6. StreamingService._run_orchestrated_path 在新主路径下能正常返回 FinalResponse
7. TaskOrchestrator 保留可用（兼容层不被误删）
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()

from src.agent.execution.engine import ExecutionEngine
from src.agent.execution.direct_executor import DirectExecutor
from src.agent.execution.planned_executor import PlannedExecutor
from src.agent.models.execution_decision import ExecutionDecision
from src.agent.models.final_response import FinalResponse
from src.agent.models.execution_plan import ExecutionPlan, PlanStep
from src.agent.models.skill_result import SkillResult
from src.agent.request_router import RouteDecision
from src.agent.task_orchestrator import TaskOrchestrator


# ─────────────────────────────────────────────────────────────────
# 辅助工厂
# ─────────────────────────────────────────────────────────────────

def _make_final_response(answer: str = "测试答案") -> FinalResponse:
    return FinalResponse(
        answer=answer,
        summary=answer,
        route="direct_task",
        task_type="smalltalk",
    )


def _route_decision(route: str = "direct_task", task_type: str = "smalltalk") -> RouteDecision:
    return RouteDecision(
        route=route,
        task_type=task_type,
        confidence=0.9,
        reason="test",
        matched_skills=[],
        expected_output_schema="generic_answer_v1",
    )


def _mock_synthesizer() -> MagicMock:
    synth = MagicMock()
    fr = _make_final_response()
    synth.synthesize_smalltalk.return_value = fr
    synth.synthesize_direct.return_value = fr
    synth.synthesize_qa.return_value = fr
    synth.synthesize.return_value = fr
    return synth


def _ok_skill_result(name: str = "weather-lookup") -> SkillResult:
    return SkillResult(skill_name=name, success=True, summary=f"{name} ok", data={})


def _skill_manager_mock() -> MagicMock:
    mgr = MagicMock()
    mgr.call_skill.return_value = _ok_skill_result()
    return mgr


# ─────────────────────────────────────────────────────────────────
# 1. config flags 默认值
# ─────────────────────────────────────────────────────────────────

class TestPhase8FlagsDefault:

    def test_enable_unified_execution_engine_default_true(self):
        from src.core.config import settings
        assert getattr(settings, "ENABLE_UNIFIED_EXECUTION_ENGINE", False) is True

    def test_enable_workflow_graph_default_true(self):
        from src.core.config import settings
        assert getattr(settings, "ENABLE_WORKFLOW_GRAPH", False) is True

    def test_enable_unified_execution_trace_default_true(self):
        from src.core.config import settings
        assert getattr(settings, "ENABLE_UNIFIED_EXECUTION_TRACE", False) is True

    def test_enable_unified_execution_events_default_true(self):
        from src.core.config import settings
        assert getattr(settings, "ENABLE_UNIFIED_EXECUTION_EVENTS", False) is True


# ─────────────────────────────────────────────────────────────────
# 2. ExecutionEngine 直接调用测试（direct / planned）
# ─────────────────────────────────────────────────────────────────

class TestExecutionEngineNewPaths:

    def _make_engine(self) -> ExecutionEngine:
        skill_mgr = _skill_manager_mock()
        synth = _mock_synthesizer()
        engine = ExecutionEngine.__new__(ExecutionEngine)
        engine._direct = DirectExecutor(
            skill_manager=skill_mgr,
            rag_retriever=MagicMock(),
            llm=MagicMock(),
            synthesizer=synth,
        )
        engine._planned = PlannedExecutor(
            skill_manager=skill_mgr,
            llm=MagicMock(),
            synthesizer=synth,
        )
        engine._react = MagicMock()
        return engine

    def test_engine_direct_mode(self):
        engine = self._make_engine()
        decision = ExecutionDecision(mode="direct", reason="test")
        route_decision = _route_decision("direct_task", "smalltalk")

        result = asyncio.get_event_loop().run_until_complete(
            engine.run(decision, route_decision, "你好")
        )
        assert isinstance(result, FinalResponse)

    def test_engine_planned_mode(self):
        engine = self._make_engine()
        decision = ExecutionDecision(mode="planned", reason="test")
        route_decision = _route_decision("planned_task", "observation_recommendation")

        plan = ExecutionPlan(
            task_type="observation_recommendation",
            output_schema="observation_answer_v1",
            steps=[
                PlanStep(id="s1", kind="tool", title="天气", skill="weather-lookup"),
            ],
        )

        with patch.object(engine._planned, "_planner") as mock_planner:
            mock_planner.plan.return_value = plan
            result = asyncio.get_event_loop().run_until_complete(
                engine.run(decision, route_decision, "今晚北京天气", execution_plan=plan)
            )
        assert isinstance(result, FinalResponse)

    def test_engine_react_mode(self):
        engine = self._make_engine()
        engine._react.run = AsyncMock(return_value=_make_final_response("react 答案"))
        decision = ExecutionDecision(mode="react", reason="test")
        route_decision = _route_decision()

        result = asyncio.get_event_loop().run_until_complete(
            engine.run(decision, route_decision, "测试")
        )
        assert isinstance(result, FinalResponse)
        assert result.answer == "react 答案"


# ─────────────────────────────────────────────────────────────────
# 3. StreamingService._run_orchestrated_path 分支切换
# ─────────────────────────────────────────────────────────────────

class TestOrchestratedPathBranching:
    """验证 flag 控制下 _run_orchestrated_path 走正确分支。"""

    def _make_service(self, with_engine: bool):
        from src.agent.streaming_service import BaseStreamingGenerator
        svc = BaseStreamingGenerator.__new__(BaseStreamingGenerator)
        svc._event_processors = []
        svc._long_term_memory = None
        svc._current_query = ""
        svc._memory = MagicMock()
        svc._memory.build_context.return_value = {"context_text": ""}
        svc._user_id = "test_user"

        if with_engine:
            skill_mgr = _skill_manager_mock()
            synth = _mock_synthesizer()
            engine = ExecutionEngine.__new__(ExecutionEngine)
            engine._direct = DirectExecutor(
                skill_manager=skill_mgr,
                rag_retriever=MagicMock(),
                llm=MagicMock(),
                synthesizer=synth,
            )
            engine._planned = PlannedExecutor(
                skill_manager=skill_mgr,
                llm=MagicMock(),
                synthesizer=synth,
            )
            engine._react = MagicMock()
            svc._execution_engine = engine
            svc._task_orchestrator = None
        else:
            svc._execution_engine = None
            orch = MagicMock()
            orch.run = AsyncMock(return_value=_make_final_response("旧路径答案"))
            svc._task_orchestrator = orch

        return svc

    def test_new_engine_path_used_when_flag_true(self):
        svc = self._make_service(with_engine=True)
        decision = _route_decision("direct_task", "smalltalk")
        from src.agent.latency import LatencyTracker
        from src.core import config as config_module

        orig = config_module.settings.ENABLE_UNIFIED_EXECUTION_ENGINE
        try:
            config_module.settings.ENABLE_UNIFIED_EXECUTION_ENGINE = True
            result = asyncio.get_event_loop().run_until_complete(
                svc._run_orchestrated_path(
                    "你好",
                    decision,
                    use_long_term_memory=False,
                    latency=LatencyTracker(),
                )
            )
        finally:
            config_module.settings.ENABLE_UNIFIED_EXECUTION_ENGINE = orig
        assert isinstance(result, FinalResponse)

    def test_legacy_path_used_when_flag_false(self):
        svc = self._make_service(with_engine=False)
        decision = _route_decision("direct_task", "smalltalk")
        from src.agent.latency import LatencyTracker
        from src.core import config as config_module

        orig = config_module.settings.ENABLE_UNIFIED_EXECUTION_ENGINE
        try:
            config_module.settings.ENABLE_UNIFIED_EXECUTION_ENGINE = False
            result = asyncio.get_event_loop().run_until_complete(
                svc._run_orchestrated_path(
                    "你好",
                    decision,
                    use_long_term_memory=False,
                    latency=LatencyTracker(),
                )
            )
        finally:
            config_module.settings.ENABLE_UNIFIED_EXECUTION_ENGINE = orig
        assert isinstance(result, FinalResponse)
        assert result.answer == "旧路径答案"

    def test_legacy_path_used_when_engine_not_injected(self):
        """engine 为 None 时即使 flag=True 也回退到旧路径。"""
        svc = self._make_service(with_engine=False)
        decision = _route_decision("direct_task", "smalltalk")
        from src.agent.latency import LatencyTracker
        from src.core import config as config_module

        orig = config_module.settings.ENABLE_UNIFIED_EXECUTION_ENGINE
        try:
            config_module.settings.ENABLE_UNIFIED_EXECUTION_ENGINE = True
            result = asyncio.get_event_loop().run_until_complete(
                svc._run_orchestrated_path(
                    "你好",
                    decision,
                    use_long_term_memory=False,
                    latency=LatencyTracker(),
                )
            )
        finally:
            config_module.settings.ENABLE_UNIFIED_EXECUTION_ENGINE = orig
        assert result.answer == "旧路径答案"


# ─────────────────────────────────────────────────────────────────
# 4. PlannedExecutor ENABLE_WORKFLOW_GRAPH 分支
# ─────────────────────────────────────────────────────────────────

class TestPlannedExecutorWorkflowBranch:

    def _make_executor(self) -> PlannedExecutor:
        skill_mgr = _skill_manager_mock()
        synth = _mock_synthesizer()
        return PlannedExecutor(
            skill_manager=skill_mgr,
            llm=MagicMock(),
            synthesizer=synth,
        )

    def _plan(self) -> ExecutionPlan:
        return ExecutionPlan(
            task_type="observation_recommendation",
            output_schema="observation_answer_v1",
            steps=[
                PlanStep(id="s1", kind="tool", title="天气", skill="weather-lookup"),
            ],
        )

    def test_workflow_executor_is_called(self):
        """Phase 9: ENABLE_WORKFLOW_GRAPH flag 已移除，WorkflowExecutor 为唯一执行引擎。"""
        executor = self._make_executor()
        plan = self._plan()
        decision = _route_decision("planned_task", "observation_recommendation")

        wf_executor = MagicMock()
        wf_executor.execute = AsyncMock(
            return_value=MagicMock(
                step_results=[],
                skill_results=[],
                halted=False,
            )
        )
        executor._workflow_executor = wf_executor
        result = asyncio.get_event_loop().run_until_complete(
            executor.run(decision, "今晚天气", execution_plan=plan)
        )
        wf_executor.execute.assert_called_once()
        assert isinstance(result, FinalResponse)


# ─────────────────────────────────────────────────────────────────
# 5. TaskOrchestrator 兼容层完好
# ─────────────────────────────────────────────────────────────────

class TestTaskOrchestratorCompatibility:
    """TaskOrchestrator 保留可用，不被误删。"""

    def test_task_orchestrator_importable(self):
        from src.agent.task_orchestrator import TaskOrchestrator
        assert TaskOrchestrator is not None

    def test_task_orchestrator_has_deprecated_notice(self):
        """docstring 中含有 deprecated 标记。"""
        assert "deprecated" in TaskOrchestrator.__doc__.lower()

    def test_request_router_route_has_compatibility_notice(self):
        from src.agent.request_router import RequestRouter

        assert "compatibility" in (RequestRouter.route.__doc__ or "").lower()

    def test_choose_path_has_deprecated_notice(self):
        from src.agent.governance import AgentExecutionPolicy

        assert "deprecated" in (AgentExecutionPolicy.choose_path.__doc__ or "").lower()

    def test_task_orchestrator_run_direct(self):
        skill_mgr = _skill_manager_mock()
        synth = _mock_synthesizer()
        orch = TaskOrchestrator.__new__(TaskOrchestrator)
        orch._skill_manager = skill_mgr
        orch._synthesizer = synth
        orch._rag = MagicMock()
        orch._llm = MagicMock()
        from src.agent.planner import Planner
        from src.agent.executor import StepExecutor
        from src.agent.policies.fallback_policy import FallbackPolicy
        orch._planner = MagicMock()
        orch._executor = StepExecutor(skill_manager=skill_mgr)
        orch._fallback_policy = FallbackPolicy()

        decision = _route_decision("direct_task", "smalltalk")
        result = asyncio.get_event_loop().run_until_complete(
            orch.run(decision, "你好", chat_history="", user_profile="")
        )
        assert isinstance(result, FinalResponse)
