from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agent.execution.direct_executor import DirectExecutor
from src.agent.execution.engine import ExecutionEngine
from src.agent.execution.planned_executor import PlannedExecutor
from src.agent.execution.react_executor import ReactExecutor
from src.agent.execution.workflow_executor import WorkflowExecutor
from src.agent.latency import LatencyTracker
from src.agent.models.capability_decision import CapabilityDecision
from src.agent.models.execution_context import ExecutionContext
from src.agent.models.execution_decision import ExecutionDecision
from src.agent.models.execution_plan import ExecutionPlan
from src.agent.models.final_response import FinalResponse
from src.agent.models.request_context import RequestContext
from src.agent.models.skill_result import SkillResult
from src.agent.models.task_profile import TaskProfile
from src.agent.models.workflow_graph import WorkflowGraph
from src.agent.streaming_service import BaseStreamingGenerator


def _context(
    *,
    route: str = "direct_task",
    task_type: str = "smalltalk",
    capability_hints: list[str] | None = None,
    capability_decision: CapabilityDecision | None = None,
    query: str = "你好",
) -> ExecutionContext:
    profile = TaskProfile.from_legacy_route(
        route=route,
        task_type=task_type,
        confidence=0.9,
        capability_hints=list(capability_hints or []),
        expected_output_schema="generic_answer_v1",
    )
    return ExecutionContext(
        profile=profile,
        request=RequestContext(query=query, chat_history="", user_profile=""),
        capability_decision=capability_decision,
    )


class _DirectToolManager:
    def __init__(self) -> None:
        self.tool_calls: list[tuple[str, dict]] = []

    def call_mcp_tool(self, tool_name: str, **params) -> str:
        self.tool_calls.append((tool_name, params))
        return '{"ok": true}'


class _PlannedSkillManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call_skill(self, name: str, **params) -> SkillResult:
        self.calls.append((name, params))
        return SkillResult(
            skill_name=name,
            success=True,
            data={"params": params},
            summary=f"{name} ok",
        )


class _Synthesizer:
    prompt_version = "test_prompt"

    def synthesize_smalltalk(self, answer: str) -> FinalResponse:
        return FinalResponse(answer=answer, summary=answer)

    def synthesize_direct(self, **kwargs) -> FinalResponse:
        return FinalResponse(
            answer="direct ok",
            summary="direct ok",
            route="direct_task",
            task_type=kwargs.get("task_type", "single_tool_lookup"),
        )

    def synthesize(self, **kwargs) -> FinalResponse:
        return FinalResponse(
            answer="planned ok",
            summary="planned ok",
            route=kwargs.get("route", "planned_task"),
            task_type=kwargs.get("task_type", "unknown_complex_task"),
            execution_plan=kwargs.get("execution_plan"),
            execution_trace=kwargs.get("execution_trace", []),
            route_decision=kwargs.get("route_decision"),
            fallback_path=kwargs.get("fallback_path", []),
            budget_usage=kwargs.get("budget_usage"),
            versions=kwargs.get("versions"),
        )


@pytest.mark.asyncio
async def test_direct_executor_run_context_uses_capability_decision_without_route():
    manager = _DirectToolManager()
    executor = DirectExecutor(
        skill_manager=manager,
        rag_retriever=SimpleNamespace(),
        llm=SimpleNamespace(),
        synthesizer=_Synthesizer(),
    )
    capability = CapabilityDecision.for_tool(
        "web_search",
        confidence=0.9,
        reason="test",
        metadata={"params": {"query": "JWST latest", "max_results": 3}},
    )
    context = _context(
        task_type="single_tool_lookup",
        capability_decision=capability,
        query="JWST latest",
    )

    response = await executor.run_context(context)

    assert manager.tool_calls == [
        ("web_search", {"query": "JWST latest", "max_results": 3})
    ]
    assert response.audit_metadata["capability_kind"] == "tool"
    assert response.route_decision["capability_hints"] == []


@pytest.mark.asyncio
async def test_planned_executor_run_context_uses_profile_capability_hints():
    manager = _PlannedSkillManager()
    executor = PlannedExecutor(
        skill_manager=manager,
        llm=SimpleNamespace(),
        synthesizer=_Synthesizer(),
    )
    context = _context(
        route="planned_task",
        task_type="unknown_complex_task",
        capability_hints=["weather-lookup"],
        query="北京今晚天气怎么样",
    )
    context.profile.matched_skills = []

    response = await executor.run_context(context)

    assert manager.calls
    assert manager.calls[0][0] == "weather-lookup"
    assert response.route_decision["capability_hints"] == ["weather-lookup"]
    assert response.route_decision["matched_skills"] == []


@pytest.mark.asyncio
async def test_execution_engine_run_context_dispatches_without_legacy_route():
    context = _context()
    response = FinalResponse(answer="ok", summary="ok")
    direct = SimpleNamespace(run_context=AsyncMock(return_value=response))
    engine = ExecutionEngine.__new__(ExecutionEngine)
    engine._direct = direct
    engine._planned = SimpleNamespace()
    engine._react = SimpleNamespace()

    result = await engine.run_context(
        ExecutionDecision(mode="direct", reason="test"),
        context,
    )

    assert result is response
    direct.run_context.assert_awaited_once_with(context)
    assert result.execution_events[0]["type"] == "task_profile"
    assert result.execution_events[1]["type"] == "route_decided"


@pytest.mark.asyncio
async def test_streaming_unified_path_prefers_engine_run_context():
    context = _context()
    decision = context.profile.to_legacy_route_decision()
    exec_decision = ExecutionDecision(mode="direct", reason="test")
    response = FinalResponse(answer="stream ok", summary="stream ok")
    engine = SimpleNamespace(run_context=AsyncMock(return_value=response))

    service = BaseStreamingGenerator.__new__(BaseStreamingGenerator)
    service._execution_engine = engine
    service._task_orchestrator = None
    service._current_query = ""

    result = await service._run_orchestrated_path(
        "你好",
        decision,
        use_long_term_memory=False,
        latency=LatencyTracker(),
        chat_history="",
        user_profile="",
        execution_decision=exec_decision,
        exec_context=context,
    )

    assert result is response
    engine.run_context.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_without_execution_engine_fails_instead_of_legacy_orchestrator():
    context = _context()
    decision = context.profile.to_legacy_route_decision()
    service = BaseStreamingGenerator.__new__(BaseStreamingGenerator)
    service._execution_engine = None
    service._task_orchestrator = SimpleNamespace(run=AsyncMock())
    service._current_query = ""

    with pytest.raises(ValueError, match="execution engine is not configured"):
        await service._run_orchestrated_path(
            "你好",
            decision,
            use_long_term_memory=False,
            latency=LatencyTracker(),
            chat_history="",
            user_profile="",
        )

    service._task_orchestrator.run.assert_not_awaited()


def test_streaming_legacy_route_decision_uses_profile_without_router_route():
    profile = _context().profile
    service = BaseStreamingGenerator.__new__(BaseStreamingGenerator)
    service._request_router = SimpleNamespace(
        profile=lambda query: profile,
        route=lambda query: (_ for _ in ()).throw(
            AssertionError("streaming must not call router.route()")
        ),
    )

    decision = service._resolve_legacy_route_decision(
        "你好",
        precomputed_profile=profile,
    )

    assert decision.route == profile.legacy_route
    assert service._resolve_legacy_route_decision("你好") is None


@pytest.mark.asyncio
async def test_execution_engine_react_mode_uses_react_run_context():
    context = _context(
        route="fallback_react",
        task_type="open_domain_reasoning",
        query="写一篇宇宙随笔",
    )
    response = FinalResponse(answer="react ok", summary="react ok")

    class _React:
        def __init__(self) -> None:
            self.calls: list[ExecutionContext] = []

        async def run_context(self, ctx: ExecutionContext) -> FinalResponse:
            self.calls.append(ctx)
            return response

    react = _React()
    engine = ExecutionEngine.__new__(ExecutionEngine)
    engine._direct = SimpleNamespace()
    engine._planned = SimpleNamespace()
    engine._react = react

    result = await engine.run_context(
        ExecutionDecision(mode="react", reason="test"),
        context,
    )

    assert result is response
    assert react.calls == [context]


@pytest.mark.asyncio
async def test_react_executor_legacy_wrapper_matches_run_context_route_metadata():
    class _InvokeExecutor:
        def invoke(self, agent_input):
            return {"output": "Final Answer: ok"}

    context = _context(
        route="fallback_react",
        task_type="open_domain_reasoning",
        query="开放问题",
    )
    executor = ReactExecutor(agent_executor=_InvokeExecutor())

    context_response = await executor.run_context(context)

    engine = ExecutionEngine.__new__(ExecutionEngine)
    engine._direct = SimpleNamespace()
    engine._planned = SimpleNamespace()
    engine._react = executor
    legacy_response = await engine.run(
        ExecutionDecision(mode="react", reason="legacy_adapter_test"),
        context.profile.to_legacy_route_decision(),
        context.query,
        chat_history=context.chat_history,
        user_profile=context.user_profile,
    )

    assert legacy_response.route_decision == context_response.route_decision
    assert legacy_response.route == context_response.route
    assert legacy_response.task_type == context_response.task_type


def test_legacy_decision_to_context_adapter_mirrors_compat_fields():
    legacy_decision = SimpleNamespace(
        route="direct_task",
        task_type="single_tool_lookup",
        confidence=0.77,
        reason="legacy route",
        matched_skills=["weather-lookup"],
        expected_output_schema="tool_answer_v1",
        router_source="legacy",
        tool_necessity_action="allow",
        tool_necessity_missing_params=["city"],
    )

    context = ExecutionContext.from_legacy_decision(
        legacy_decision,
        "北京今晚天气",
        chat_history="history",
        user_profile="profile",
    )

    assert context.query == "北京今晚天气"
    assert context.chat_history == "history"
    assert context.user_profile == "profile"
    assert context.profile.legacy_route == "direct_task"
    assert context.profile.task_type == "single_tool_lookup"
    assert context.profile.matched_skills == ["weather-lookup"]
    assert context.profile.capability_hints == ["weather-lookup"]
    assert context.profile.tool_necessity_missing_params == ["city"]


def test_legacy_context_conversion_is_centralized_in_execution_context():
    root = Path(__file__).resolve().parents[2]
    production_sources = {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in (root / "src" / "agent").rglob("*.py")
    }

    for legacy_helper in (
        "_context_from_legacy_decision",
        "_context_from_legacy_route",
    ):
        offenders = [
            path
            for path, source in production_sources.items()
            if legacy_helper in source
        ]
        assert offenders == []


def test_legacy_route_api_usage_stays_on_allowlisted_boundaries():
    root = Path(__file__).resolve().parents[2]

    route_decision_allowed = {
        "src/agent/execution/engine.py",
        "src/agent/governance.py",
        "src/agent/models/execution_context.py",
        "src/agent/models/task_profile.py",
        "src/agent/request_router.py",
        "src/agent/streaming_service.py",
    }
    route_call_allowed = {
        "src/agent/governance.py",
    }
    choose_path_allowed = {
        "src/agent/governance.py",
        "src/agent/models/execution_decision.py",
    }

    route_decision_files: set[str] = set()
    route_call_files: set[str] = set()
    choose_path_files: set[str] = set()
    for path in (root / "src" / "agent").rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        if "RouteDecision" in source:
            route_decision_files.add(rel)
        if ".route(" in source:
            route_call_files.add(rel)
        if "choose_path(" in source:
            choose_path_files.add(rel)

    assert route_decision_files <= route_decision_allowed
    assert route_call_files <= route_call_allowed
    assert choose_path_files <= choose_path_allowed


def test_legacy_runtime_fallback_symbols_are_not_in_online_initialization():
    root = Path(__file__).resolve().parents[2]
    agent_init = (root / "src" / "agent" / "__init__.py").read_text(
        encoding="utf-8"
    )
    streaming = (root / "src" / "agent" / "streaming_service.py").read_text(
        encoding="utf-8"
    )

    assert "TaskOrchestrator" not in agent_init
    assert "StepExecutor" not in agent_init
    assert "TaskOrchestrator" not in streaming
    assert "StepExecutor" not in streaming


def test_unified_engine_flag_is_not_read_by_production_runtime():
    root = Path(__file__).resolve().parents[2]
    offenders: list[str] = []
    for path in (root / "src").rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        if "ENABLE_UNIFIED_EXECUTION_ENGINE" in source:
            offenders.append(rel)

    assert offenders == []


def test_core_selectors_and_planner_do_not_read_legacy_matched_skills():
    import src.agent.planner as planner_module
    import src.capabilities.selector as capability_selector_module
    import src.tools.selector as tool_selector_module

    for module in (
        planner_module,
        capability_selector_module,
        tool_selector_module,
    ):
        source = inspect.getsource(module)
        assert "matched_skills" not in source


def test_executor_main_paths_do_not_read_legacy_matched_skills():
    guarded_callables = (
        ExecutionEngine.run_context,
        DirectExecutor.run_context,
        DirectExecutor._run_tool_task,
        PlannedExecutor.run_context,
        PlannedExecutor._resolve_plan_and_graph_for_profile,
        BaseStreamingGenerator._run_orchestrated_path,
    )

    for guarded in guarded_callables:
        source = inspect.getsource(guarded)
        assert "matched_skills" not in source


def test_executable_resolution_does_not_use_legacy_skill_field():
    source = inspect.getsource(WorkflowExecutor._resolve_executable)

    assert "node.skill" not in source


@pytest.mark.asyncio
async def test_workflow_executor_runs_skill_only_legacy_plan_after_from_dict_adapter():
    class _Manager:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def call_skill(self, name: str, **params) -> SkillResult:
            self.calls.append(name)
            return SkillResult(skill_name=name, success=True, data={}, summary="ok")

    plan = ExecutionPlan.from_dict(
        {
            "task_type": "single_tool_lookup",
            "output_schema": "tool_answer_v1",
            "steps": [
                {
                    "id": "weather",
                    "kind": "tool",
                    "skill": "weather-lookup",
                }
            ],
        }
    )
    manager = _Manager()

    outcome = await WorkflowExecutor(manager).execute(
        WorkflowGraph.from_execution_plan(plan),
        plan,
        query="北京天气",
        param_builder=lambda name, query: {"query": query},
    )

    assert not outcome.halted
    assert manager.calls == ["weather-lookup"]
    assert outcome.step_results[0].capability_kind == "skill"
    assert outcome.step_results[0].capability_name == "weather-lookup"
