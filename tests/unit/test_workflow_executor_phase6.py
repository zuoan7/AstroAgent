"""Phase 6 WorkflowExecutor 测试

目标：
1. WorkflowExecutor 能执行线性 WorkflowGraph，产出正确 ExecutionOutcome
2. PlannedExecutor 优先走原生 graph 路径
3. ENABLE_WORKFLOW_GRAPH=False 时回退到 legacy plan->graph 转换，但执行器仍是 WorkflowExecutor
4. 可选节点失败不中断执行；必选节点失败立即 halt
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()

from src.agent.execution.workflow_executor import WorkflowExecutor
from src.agent.execution.planned_executor import PlannedExecutor
from src.agent.executor import ExecutionOutcome
from src.agent.models.execution_plan import ExecutionPlan, PlanStep
from src.agent.models.skill_result import SkillResult
from src.agent.models.workflow_graph import WorkflowGraph, WorkflowNode, WorkflowEdge
from src.agent.request_router import RouteDecision


# ─────────────────────────────────────────────────────────────────
# 辅助工厂
# ─────────────────────────────────────────────────────────────────

def _ok_skill_result(name: str = "weather-lookup") -> SkillResult:
    return SkillResult(skill_name=name, success=True, summary=f"{name} ok", data={})


def _err_skill_result(name: str = "weather-lookup") -> SkillResult:
    return SkillResult.from_error(
        skill_name=name, error_code="ERR", error_message="mock error"
    )


def _skill_manager(results: dict) -> MagicMock:
    """results: {skill_name: SkillResult}"""
    mgr = MagicMock()
    def call_skill(skill_name, **kwargs):
        return results.get(skill_name, _err_skill_result(skill_name))
    mgr.call_skill.side_effect = call_skill
    return mgr


def _param_builder(skill: str, query: str) -> dict:
    return {"query": query}


def _linear_graph() -> tuple[WorkflowGraph, ExecutionPlan]:
    plan = ExecutionPlan(
        task_type="observation_recommendation",
        output_schema="observation_answer_v1",
        steps=[
            PlanStep(id="s1", kind="tool", title="天气", skill="weather-lookup", timeout_ms=8000),
            PlanStep(id="s2", kind="tool", title="观测", skill="observation-planner", timeout_ms=12000),
        ],
    )
    graph = WorkflowGraph.from_execution_plan(plan)
    return graph, plan


def _route_decision() -> RouteDecision:
    return RouteDecision(
        route="planned_task",
        task_type="observation_recommendation",
        confidence=0.9,
        reason="test",
        matched_skills=["weather-lookup"],
        expected_output_schema="observation_answer_v1",
    )


def _mock_synthesizer(answer: str = "合成答案") -> MagicMock:
    synth = MagicMock()
    from src.agent.models.final_response import FinalResponse
    fr = FinalResponse(answer=answer, summary=answer, route="planned_task",
                       task_type="observation_recommendation")
    synth.synthesize.return_value = fr
    return synth


# ─────────────────────────────────────────────────────────────────


class TestWorkflowExecutorLinear:
    """WorkflowExecutor 线性执行基本功能。"""

    def test_linear_graph_all_success(self):
        graph, plan = _linear_graph()
        mgr = _skill_manager({
            "weather-lookup": _ok_skill_result("weather-lookup"),
            "observation-planner": _ok_skill_result("observation-planner"),
        })
        executor = WorkflowExecutor(skill_manager=mgr)
        outcome = asyncio.get_event_loop().run_until_complete(
            executor.execute(graph, plan, query="北京今晚", param_builder=_param_builder)
        )
        assert not outcome.halted
        assert len(outcome.step_results) == 2
        assert len(outcome.skill_results) == 2
        assert outcome.step_results[0].step_id == "s1"
        assert outcome.step_results[1].step_id == "s2"
        assert outcome.step_results[0].status == "success"
        assert outcome.step_results[1].status == "success"

    def test_required_node_failure_halts(self):
        graph, plan = _linear_graph()
        mgr = _skill_manager({
            "weather-lookup": _err_skill_result("weather-lookup"),
        })
        executor = WorkflowExecutor(skill_manager=mgr)
        outcome = asyncio.get_event_loop().run_until_complete(
            executor.execute(graph, plan, query="北京今晚", param_builder=_param_builder)
        )
        assert outcome.halted
        assert outcome.halt_reason is not None
        assert len(outcome.step_results) == 1  # 第二步未执行

    def test_optional_node_failure_continues(self):
        plan = ExecutionPlan(
            task_type="t",
            output_schema="s",
            steps=[
                PlanStep(id="p1", kind="tool", skill="weather-lookup", required=False),
                PlanStep(id="p2", kind="tool", skill="observation-planner"),
            ],
        )
        graph = WorkflowGraph.from_execution_plan(plan)
        mgr = _skill_manager({
            "weather-lookup": _err_skill_result("weather-lookup"),
            "observation-planner": _ok_skill_result("observation-planner"),
        })
        executor = WorkflowExecutor(skill_manager=mgr)
        outcome = asyncio.get_event_loop().run_until_complete(
            executor.execute(graph, plan, query="test", param_builder=_param_builder)
        )
        assert not outcome.halted
        assert len(outcome.step_results) == 2
        assert outcome.step_results[0].status == "error"
        assert outcome.step_results[1].status == "success"

    def test_single_node_graph(self):
        plan = ExecutionPlan(
            task_type="t", output_schema="s",
            steps=[PlanStep(id="only", kind="tool", skill="weather-lookup")],
        )
        graph = WorkflowGraph.from_execution_plan(plan)
        mgr = _skill_manager({"weather-lookup": _ok_skill_result()})
        executor = WorkflowExecutor(skill_manager=mgr)
        outcome = asyncio.get_event_loop().run_until_complete(
            executor.execute(graph, plan, query="test", param_builder=_param_builder)
        )
        assert not outcome.halted
        assert len(outcome.step_results) == 1
        assert outcome.step_results[0].step_id == "only"

    def test_empty_graph_returns_empty_outcome(self):
        plan = ExecutionPlan(task_type="t", output_schema="s", steps=[])
        graph = WorkflowGraph.from_execution_plan(plan)
        mgr = _skill_manager({})
        executor = WorkflowExecutor(skill_manager=mgr)
        outcome = asyncio.get_event_loop().run_until_complete(
            executor.execute(graph, plan, query="test", param_builder=_param_builder)
        )
        assert not outcome.halted
        assert outcome.step_results == []
        assert outcome.skill_results == []

    def test_event_callback_called(self):
        graph, plan = _linear_graph()
        mgr = _skill_manager({
            "weather-lookup": _ok_skill_result("weather-lookup"),
            "observation-planner": _ok_skill_result("observation-planner"),
        })
        executor = WorkflowExecutor(skill_manager=mgr)
        events = []

        def cb(event_type, payload):
            events.append(event_type)

        asyncio.get_event_loop().run_until_complete(
            executor.execute(graph, plan, query="test",
                             param_builder=_param_builder, event_callback=cb)
        )
        assert "step_start" in events
        assert "step_end" in events
        assert "step_result" in events

    def test_timeout_ms_propagated(self):
        """节点设置了 timeout_ms，超时时 skill_result 应为 error。"""
        plan = ExecutionPlan(
            task_type="t", output_schema="s",
            steps=[PlanStep(id="slow", kind="tool", skill="slow-skill", timeout_ms=1)],
        )
        graph = WorkflowGraph.from_execution_plan(plan)

        mgr = MagicMock()
        async def slow_call(*a, **kw):
            await asyncio.sleep(10)
            return _ok_skill_result("slow-skill")
        # asyncio.to_thread 将被真实调用，这里通过 patch 替代
        executor = WorkflowExecutor(skill_manager=mgr)

        import asyncio as _asyncio
        original_to_thread = _asyncio.to_thread

        async def fake_to_thread(func, *args, **kwargs):
            await _asyncio.sleep(10)

        with patch("asyncio.to_thread", side_effect=fake_to_thread):
            outcome = _asyncio.get_event_loop().run_until_complete(
                executor.execute(graph, plan, query="test", param_builder=_param_builder)
            )
        assert outcome.halted
        assert outcome.step_results[0].status == "error"

    def test_cycle_graph_halts_gracefully(self):
        n1 = WorkflowNode(id="a", depends_on=["b"], kind="tool", skill="s")
        n2 = WorkflowNode(id="b", depends_on=["a"], kind="tool", skill="s")
        graph = WorkflowGraph(
            nodes=[n1, n2],
            edges=[WorkflowEdge("a", "b"), WorkflowEdge("b", "a")],
            output_schema="s",
        )
        plan = ExecutionPlan(task_type="t", output_schema="s", steps=[])
        executor = WorkflowExecutor(skill_manager=MagicMock())
        outcome = asyncio.get_event_loop().run_until_complete(
            executor.execute(graph, plan, query="test", param_builder=_param_builder)
        )
        assert outcome.halted
        assert "循环" in (outcome.halt_reason or "")


class TestPlannedExecutorGraphFlag:
    """PlannedExecutor 使用 WorkflowExecutor（Phase 9: ENABLE_WORKFLOW_GRAPH flag 已移除）。"""

    def _make_planned_executor(
        self,
        skill_results: dict,
        *,
        workflow_executor: WorkflowExecutor = None,
    ) -> tuple[PlannedExecutor, MagicMock]:
        plan = ExecutionPlan(
            task_type="observation_recommendation",
            output_schema="observation_answer_v1",
            steps=[
                PlanStep(id="s1", kind="tool", title="天气", skill="weather-lookup"),
                PlanStep(id="s2", kind="tool", title="观测", skill="observation-planner"),
            ],
        )
        mock_planner = MagicMock()
        mock_planner.plan.return_value = plan
        mock_planner.plan_graph.return_value = WorkflowGraph.from_execution_plan(plan)

        synth = _mock_synthesizer()
        mgr = _skill_manager(skill_results)
        llm = MagicMock()

        pe = PlannedExecutor(
            skill_manager=mgr,
            llm=llm,
            synthesizer=synth,
            planner=mock_planner,
            workflow_executor=workflow_executor or WorkflowExecutor(skill_manager=mgr),
        )
        return pe, synth

    def test_workflow_executor_is_always_called(self):
        """WorkflowExecutor 为唯一执行引擎，始终被调用。"""
        skill_results = {
            "weather-lookup": _ok_skill_result("weather-lookup"),
            "observation-planner": _ok_skill_result("observation-planner"),
        }
        mock_wf_exec = MagicMock()
        from src.agent.models.execution_plan import ExecutionPlan as EP
        dummy_plan = EP(task_type="t", output_schema="s", steps=[])
        mock_outcome = ExecutionOutcome(plan=dummy_plan, skill_results=[], step_results=[])

        async def fake_wf_execute(*a, **kw):
            return mock_outcome

        mock_wf_exec.execute = fake_wf_execute
        pe, synth = self._make_planned_executor(skill_results, workflow_executor=mock_wf_exec)

        rd = _route_decision()
        result = asyncio.get_event_loop().run_until_complete(
            pe.run(rd, "北京今晚")
        )

        synth.synthesize.assert_called_once()
        from src.agent.models.final_response import FinalResponse
        assert isinstance(result, FinalResponse)

    def test_plan_graph_is_preferred_when_available(self):
        skill_results = {
            "weather-lookup": _ok_skill_result("weather-lookup"),
            "observation-planner": _ok_skill_result("observation-planner"),
        }
        pe, synth = self._make_planned_executor(skill_results)
        rd = _route_decision()

        result = asyncio.get_event_loop().run_until_complete(
            pe.run(rd, "北京今晚")
        )

        pe._planner.plan_graph.assert_called_once()
        pe._planner.plan.assert_not_called()
        from src.agent.models.final_response import FinalResponse
        assert isinstance(result, FinalResponse)

    def test_planned_main_path_does_not_depend_on_step_executor(self):
        skill_results = {
            "weather-lookup": _ok_skill_result("weather-lookup"),
            "observation-planner": _ok_skill_result("observation-planner"),
        }
        pe, synth = self._make_planned_executor(skill_results)
        rd = _route_decision()

        with patch("src.agent.executor.StepExecutor.execute", side_effect=AssertionError("legacy step executor should not be called")):
            result = asyncio.get_event_loop().run_until_complete(
                pe.run(rd, "北京今晚")
            )

        pe._planner.plan_graph.assert_called_once()
        synth.synthesize.assert_called_once()
        from src.agent.models.final_response import FinalResponse
        assert isinstance(result, FinalResponse)

    def test_flag_true_full_execution(self):
        """flag=True 时端到端执行，skill 被实际调用。"""
        skill_results = {
            "weather-lookup": _ok_skill_result("weather-lookup"),
            "observation-planner": _ok_skill_result("observation-planner"),
        }
        mgr = _skill_manager(skill_results)
        wf_exec = WorkflowExecutor(skill_manager=mgr)
        pe, synth = self._make_planned_executor(skill_results, workflow_executor=wf_exec)

        with patch("src.agent.execution.planned_executor.settings") as mock_settings:
            mock_settings.ENABLE_WORKFLOW_GRAPH = True
            rd = _route_decision()
            result = asyncio.get_event_loop().run_until_complete(
                pe.run(rd, "北京今晚")
            )

        synth.synthesize.assert_called_once()
        call_kwargs = synth.synthesize.call_args
        skill_results_arg = call_kwargs.kwargs.get("skill_results", [])
        assert len(skill_results_arg) == 2

    def test_flag_false_legacy_path_stable(self):
        """ENABLE_WORKFLOW_GRAPH=False 时回退到 legacy plan->graph 转换。"""
        skill_results = {
            "weather-lookup": _ok_skill_result("weather-lookup"),
            "observation-planner": _ok_skill_result("observation-planner"),
        }
        mgr = _skill_manager(skill_results)
        pe, synth = self._make_planned_executor(skill_results)

        with patch("src.agent.execution.planned_executor.settings") as mock_settings:
            mock_settings.ENABLE_WORKFLOW_GRAPH = False
            rd = _route_decision()
            result = asyncio.get_event_loop().run_until_complete(
                pe.run(rd, "北京今晚")
            )

        synth.synthesize.assert_called_once()
        pe._planner.plan.assert_called_once()
        pe._planner.plan_graph.assert_not_called()
        from src.agent.models.final_response import FinalResponse
        assert isinstance(result, FinalResponse)

    def test_plan_graph_failure_falls_back_to_plan(self):
        skill_results = {
            "weather-lookup": _ok_skill_result("weather-lookup"),
            "observation-planner": _ok_skill_result("observation-planner"),
        }
        pe, synth = self._make_planned_executor(skill_results)
        pe._planner.plan_graph.side_effect = RuntimeError("graph planning failed")
        rd = _route_decision()

        result = asyncio.get_event_loop().run_until_complete(
            pe.run(rd, "北京今晚")
        )

        pe._planner.plan_graph.assert_called_once()
        pe._planner.plan.assert_called_once()
        from src.agent.models.final_response import FinalResponse
        assert isinstance(result, FinalResponse)
