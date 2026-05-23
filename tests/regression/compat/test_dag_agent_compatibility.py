"""Phase 8 DAG Agent 收口测试

目标：
1. ENABLE_UNIFIED_EXECUTION_ENGINE 已删除，旧 env var 由 settings extra=ignore 忽略
2. ExecutionEngine 新主路径可被正确调用（direct / planned / react 三种模式）
3. ExecutionEngine 未注入时明确失败，不再回退旧 orchestrator
4. planned 主路径通过 WorkflowExecutor 稳定执行
5. 旧兼容接口仍可用，并带有 deprecated/legacy 标记
6. StreamingService._run_orchestrated_path 在新主路径下能正常返回 FinalResponse
7. 旧 orchestrator / StepExecutor 已删除
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()

from src.agent.execution.engine import ExecutionEngine
from src.agent.execution.direct_executor import DirectExecutor
from src.agent.execution.planned_executor import PlannedExecutor
from src.agent.governance import AgentExecutionPolicy
from src.agent.models.execution_context import ExecutionContext
from src.agent.models.execution_decision import ExecutionDecision
from src.agent.models.final_response import FinalResponse
from src.agent.models.execution_plan import ExecutionPlan, PlanStep
from src.agent.models.request_context import RequestContext
from src.agent.models.skill_result import SkillResult
from src.agent.models.task_profile import TaskProfile
from src.agent.request_router import RouteDecision
from src.agent.request_router import RequestRouter


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

    def test_enable_unified_execution_engine_flag_removed(self):
        from src.core.config import settings
        assert not hasattr(settings, "ENABLE_UNIFIED_EXECUTION_ENGINE")

    def test_deprecated_non_branching_flags_are_removed(self):
        from src.core.config import settings
        removed = {
            "ENABLE_TASK_PROFILE",
            "ENABLE_EXECUTION_CONTEXT",
            "ENABLE_EXECUTION_DECISION",
            "ENABLE_WORKFLOW_GRAPH",
            "ENABLE_UNIFIED_EXECUTION_TRACE",
            "ENABLE_UNIFIED_EXECUTION_EVENTS",
            "ENABLE_UNIFIED_EXECUTION_ENGINE",
        }
        assert all(not hasattr(settings, name) for name in removed)


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

        result = asyncio.run(engine.run(decision, route_decision, "你好"))
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
            result = asyncio.run(
                engine.run(decision, route_decision, "今晚北京天气", execution_plan=plan)
            )
        assert isinstance(result, FinalResponse)

    def test_engine_react_mode(self):
        engine = self._make_engine()
        engine._react.run_context = AsyncMock(return_value=_make_final_response("react 答案"))
        decision = ExecutionDecision(mode="react", reason="test")
        route_decision = _route_decision()

        result = asyncio.run(engine.run(decision, route_decision, "测试"))
        assert isinstance(result, FinalResponse)
        assert result.answer == "react 答案"

    def test_engine_planned_required_failure_runs_react_fallback(self):
        engine = ExecutionEngine.__new__(ExecutionEngine)
        planned_response = FinalResponse(
            answer="planned failed answer",
            summary="planned failed answer",
            route="planned_task",
            task_type="observation_recommendation",
            execution_plan={"task_type": "observation_recommendation", "steps": []},
            execution_trace=[
                {
                    "step_id": "weather",
                    "title": "天气",
                    "kind": "tool",
                    "status": "error",
                    "required": True,
                    "error": "tool failed",
                }
            ],
            fallback_path=[
                {
                    "strategy": "react_fallback",
                    "reason": "required_step_failed",
                    "metadata": {"required_failed_steps": ["weather"]},
                }
            ],
            execution_events=[
                {"type": "plan_created", "payload": {"plan": {}}, "source": "planned"},
                {
                    "type": "fallback_triggered",
                    "payload": {
                        "strategy": "react_fallback",
                        "reason": "required_step_failed",
                        "metadata": {"required_failed_steps": ["weather"]},
                    },
                    "source": "planned",
                },
                {
                    "type": "answer_ready",
                    "payload": {"answer": "planned failed answer"},
                    "source": "planned",
                },
            ],
        )
        react_response = FinalResponse(
            answer="react recovered answer",
            summary="react recovered answer",
            route="planned_task",
            task_type="observation_recommendation",
            execution_events=[
                {
                    "type": "answer_ready",
                    "payload": {"answer": "react recovered answer"},
                    "source": "react",
                }
            ],
        )
        engine._planned = MagicMock()
        engine._planned.run_context = AsyncMock(return_value=planned_response)
        engine._react = MagicMock()
        engine._react.run_context = AsyncMock(return_value=react_response)

        result = asyncio.run(
            engine.run(
                ExecutionDecision(mode="planned", reason="test"),
                _route_decision("planned_task", "observation_recommendation"),
                "今晚北京观测条件",
            )
        )

        engine._react.run_context.assert_awaited_once()
        assert result.answer == "react recovered answer"
        assert result.execution_plan == planned_response.execution_plan
        assert result.execution_trace == planned_response.execution_trace
        assert result.fallback_path[0]["strategy"] == "react_fallback"
        assert result.fallback_path[0]["metadata"]["executed"] is True
        assert result.fallback_path[0]["metadata"]["recovery_mode"] == "react"
        answer_events = [
            event
            for event in result.execution_events
            if event["type"] == "answer_ready"
        ]
        assert answer_events[-1]["payload"]["answer"] == "react recovered answer"

    def test_engine_planned_optional_failure_stays_partial_answer(self):
        engine = ExecutionEngine.__new__(ExecutionEngine)
        planned_response = FinalResponse(
            answer="partial planned answer",
            summary="partial planned answer",
            route="planned_task",
            task_type="observation_recommendation",
            fallback_path=[
                {
                    "strategy": "partial_answer",
                    "reason": "optional_step_failed",
                    "metadata": {"optional_failed_steps": ["weather"]},
                }
            ],
            execution_events=[
                {
                    "type": "fallback_triggered",
                    "payload": {
                        "strategy": "partial_answer",
                        "reason": "optional_step_failed",
                        "metadata": {"optional_failed_steps": ["weather"]},
                    },
                    "source": "planned",
                }
            ],
        )
        engine._planned = MagicMock()
        engine._planned.run_context = AsyncMock(return_value=planned_response)
        engine._react = MagicMock()
        engine._react.run_context = AsyncMock()

        result = asyncio.run(
            engine.run(
                ExecutionDecision(mode="planned", reason="test"),
                _route_decision("planned_task", "observation_recommendation"),
                "今晚北京观测条件",
            )
        )

        engine._react.run_context.assert_not_awaited()
        assert result.answer == "partial planned answer"
        assert result.fallback_path[0]["metadata"]["executed"] is True
        assert result.fallback_path[0]["metadata"]["recovery_mode"] == "partial_answer"

    def test_engine_planned_plan_repair_runs_once(self):
        engine = ExecutionEngine.__new__(ExecutionEngine)
        failed_response = FinalResponse(
            answer="failed planned answer",
            summary="failed planned answer",
            route="planned_task",
            task_type="observation_recommendation",
            execution_plan={"task_type": "observation_recommendation", "steps": []},
            execution_trace=[
                {
                    "step_id": "bad_step",
                    "title": "坏步骤",
                    "kind": "bad",
                    "status": "error",
                    "required": True,
                    "error": "unsupported node kind: 'bad'",
                }
            ],
            fallback_path=[
                {
                    "strategy": "plan_repair",
                    "reason": "repairable_plan_failure",
                    "metadata": {"required_failed_steps": ["bad_step"]},
                }
            ],
            execution_events=[
                {"type": "plan_created", "payload": {"plan": {}}, "source": "planned"},
                {
                    "type": "fallback_triggered",
                    "payload": {
                        "strategy": "plan_repair",
                        "reason": "repairable_plan_failure",
                        "metadata": {"required_failed_steps": ["bad_step"]},
                    },
                    "source": "planned",
                },
                {
                    "type": "answer_ready",
                    "payload": {"answer": "failed planned answer"},
                    "source": "planned",
                },
            ],
        )
        repaired_response = FinalResponse(
            answer="repaired planned answer",
            summary="repaired planned answer",
            route="planned_task",
            task_type="observation_recommendation",
            execution_plan={"task_type": "observation_recommendation", "steps": []},
            execution_events=[
                {
                    "type": "answer_ready",
                    "payload": {"answer": "repaired planned answer"},
                    "source": "planned",
                }
            ],
        )
        repaired_plan = ExecutionPlan(
            task_type="observation_recommendation",
            output_schema="observation_answer_v1",
            steps=[PlanStep(id="fixed", kind="tool", skill="weather-lookup")],
        )
        engine._planned = MagicMock()
        engine._planned.run_context = AsyncMock(side_effect=[failed_response, repaired_response])
        engine._planned.repair_plan_context.return_value = repaired_plan
        engine._react = MagicMock()
        engine._react.run_context = AsyncMock()

        result = asyncio.run(
            engine.run(
                ExecutionDecision(mode="planned", reason="test"),
                _route_decision("planned_task", "observation_recommendation"),
                "今晚北京观测条件",
            )
        )

        assert engine._planned.run_context.await_count == 2
        engine._react.run_context.assert_not_awaited()
        assert result.answer == "repaired planned answer"
        assert result.fallback_path[0]["strategy"] == "plan_repair"
        assert result.fallback_path[0]["metadata"]["executed"] is True
        assert result.fallback_path[0]["metadata"]["recovery_mode"] == "plan_repair"
        assert any(event["type"] == "plan_repaired" for event in result.execution_events)


# ─────────────────────────────────────────────────────────────────
# 3. StreamingService._run_orchestrated_path engine requirement
# ─────────────────────────────────────────────────────────────────

class TestOrchestratedPathBranching:
    """验证 _run_orchestrated_path 只接受 ExecutionEngine 主路径。"""

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
            svc._task_orchestrator = MagicMock()

        return svc

    def test_engine_path_used_when_engine_present(self):
        svc = self._make_service(with_engine=True)
        decision = _route_decision("direct_task", "smalltalk")
        from src.agent.latency import LatencyTracker

        result = asyncio.run(
            svc._run_orchestrated_path(
                "你好",
                decision,
                use_long_term_memory=False,
                latency=LatencyTracker(),
            )
        )
        assert isinstance(result, FinalResponse)

    def test_no_legacy_path_when_engine_missing(self):
        svc = self._make_service(with_engine=False)
        decision = _route_decision("direct_task", "smalltalk")
        from src.agent.latency import LatencyTracker

        with pytest.raises(ValueError, match="execution engine is not configured"):
            asyncio.run(
                svc._run_orchestrated_path(
                    "你好",
                    decision,
                    use_long_term_memory=False,
                    latency=LatencyTracker(),
                )
            )

    def test_engine_not_injected_fails_instead_of_legacy_orchestrator(self):
        """engine 为 None 时不再回退旧路径。"""
        svc = self._make_service(with_engine=False)
        decision = _route_decision("direct_task", "smalltalk")
        from src.agent.latency import LatencyTracker

        with pytest.raises(ValueError, match="execution engine is not configured"):
            asyncio.run(
                svc._run_orchestrated_path(
                    "你好",
                    decision,
                    use_long_term_memory=False,
                    latency=LatencyTracker(),
                )
            )


class TestDeprecatedRefactorFlagsCompatibility:
    """已退场的重构配置位不再参与 profile/context/decision 主路径。"""

    @pytest.mark.parametrize(
        ("query", "expected_mode", "expected_route"),
        [
            ("你好", "direct", "direct_task"),
            ("帮我看下北京今晚观测条件，同时查查有没有天象活动", "planned", "planned_task"),
            ("帮我写一篇关于宇宙的科幻小说", "react", "fallback_react"),
        ],
    )
    def test_profile_context_decision_main_path_is_fixed(
        self,
        query: str,
        expected_mode: str,
        expected_route: str,
    ):
        from src.agent.streaming_service import BaseStreamingGenerator

        svc = BaseStreamingGenerator.__new__(BaseStreamingGenerator)
        svc._execution_policy = AgentExecutionPolicy(
            mode="hybrid",
            enable_react_fallback=True,
        )
        svc._request_router = RequestRouter()

        decision, profile, context = svc._resolve_execution_decision(
            query,
            legacy_decision=None,
            use_long_term_memory=False,
        )

        assert decision.mode == expected_mode
        assert profile is not None and profile.legacy_route == expected_route
        assert context is not None and isinstance(context, ExecutionContext)

    def test_trace_and_event_artifacts_are_always_attached(self):
        engine = ExecutionEngine.__new__(ExecutionEngine)
        response = FinalResponse(
            answer="测试答案",
            summary="测试答案",
            route="direct_task",
            task_type="smalltalk",
            execution_trace=[{"step_id": "s1", "status": "success"}],
        )
        decision = ExecutionDecision(mode="direct", reason="test")
        legacy_decision = _route_decision("direct_task", "smalltalk")
        context = ExecutionContext(
            profile=TaskProfile.from_legacy_route(
                route="direct_task",
                task_type="smalltalk",
                confidence=0.9,
            ),
            request=RequestContext(query="你好"),
        )

        engine._attach_engine_events(
            response,
            decision=decision,
            legacy_decision=legacy_decision,
            context=context,
        )

        assert response.execution_trace == [{"step_id": "s1", "status": "success"}]
        event_types = [event["type"] for event in response.execution_events]
        assert "task_profile" in event_types
        assert "execution_decision" in event_types

    def test_streaming_service_resolve_execution_decision_does_not_call_choose_path(self):
        from src.agent.streaming_service import BaseStreamingGenerator

        svc = BaseStreamingGenerator.__new__(BaseStreamingGenerator)
        svc._request_router = None
        svc._execution_policy = SimpleNamespace(
            mode="hybrid",
            enable_planner=False,
            enable_react_fallback=True,
            choose_path=lambda route: (_ for _ in ()).throw(
                AssertionError("choose_path should not be used by internal main path")
            ),
            decide=lambda profile, context=None: ExecutionDecision(
                mode="react",
                reason="compat_profile_without_router_or_legacy_decision",
                fallback_modes=[],
                legacy_execution_path="react",
            ),
        )

        decision, profile, context = svc._resolve_execution_decision(
            "写一篇宇宙随笔",
            legacy_decision=None,
            use_long_term_memory=False,
        )

        assert decision.mode == "react"
        assert profile is not None and profile.legacy_route == "fallback_react"
        assert context is not None and context.query == "写一篇宇宙随笔"


# ─────────────────────────────────────────────────────────────────
# 4. PlannedExecutor WorkflowExecutor branch
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
            return_value=SimpleNamespace(
                step_results=[],
                skill_results=[],
                evidence_by_key={},
                evidence_items=[],
                skipped_step_ids=[],
                halted=False,
            )
        )
        executor._workflow_executor = wf_executor
        context = ExecutionContext.from_legacy_decision(decision, "今晚天气")
        result = asyncio.run(
            executor.run_context(context, execution_plan=plan)
        )
        wf_executor.execute.assert_called_once()
        assert isinstance(result, FinalResponse)


class TestPlannedRecoveryPolicy:
    def test_required_tool_failure_stays_react_fallback(self):
        from src.agent.policies.fallback_policy import FallbackPolicy

        outcome = SimpleNamespace(
            halted=False,
            step_results=[
                SimpleNamespace(
                    step_id="weather",
                    required=True,
                    status="error",
                    error="weather service unavailable",
                )
            ],
        )
        plan = SimpleNamespace(task_type="observation_recommendation")

        decision = FallbackPolicy().decide_for_execution(
            outcome=outcome,
            plan=plan,
        )

        assert decision.strategy == "react_fallback"
        assert decision.metadata["recovery_mode"] == "react"
        assert decision.metadata["executed"] is False

    def test_repairable_required_failure_returns_plan_repair(self):
        from src.agent.policies.fallback_policy import FallbackPolicy

        outcome = SimpleNamespace(
            halted=False,
            step_results=[
                SimpleNamespace(
                    step_id="bad_step",
                    required=True,
                    status="error",
                    error="unsupported node kind: 'bad'",
                )
            ],
        )
        plan = SimpleNamespace(task_type="observation_recommendation")

        decision = FallbackPolicy().decide_for_execution(
            outcome=outcome,
            plan=plan,
        )

        assert decision.strategy == "plan_repair"
        assert decision.metadata["recovery_mode"] == "plan_repair"
        assert decision.metadata["executed"] is False


# ─────────────────────────────────────────────────────────────────
# 5. Legacy boundary compatibility
# ─────────────────────────────────────────────────────────────────

class TestLegacyBoundaryCompatibility:
    """仅保留明确的外部兼容壳。"""

    def test_task_orchestrator_file_removed(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        assert not (root / "src" / "agent" / "task_orchestrator.py").exists()

    def test_step_executor_removed(self):
        import src.agent.executor as executor_module

        assert not hasattr(executor_module, "StepExecutor")

    def test_request_router_route_has_compatibility_notice(self):
        from src.agent.request_router import RequestRouter

        assert "compatibility" in (RequestRouter.route.__doc__ or "").lower()

    def test_choose_path_has_deprecated_notice(self):
        from src.agent.governance import AgentExecutionPolicy

        assert "deprecated" in (AgentExecutionPolicy.choose_path.__doc__ or "").lower()
