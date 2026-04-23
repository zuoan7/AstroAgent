import asyncio
import sys
from types import SimpleNamespace

import pytest

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()
sys.modules.pop("src.agent.streaming_service", None)

from src.agent.executor import StepExecutor
from src.agent.models.execution_plan import ExecutionPlan, PlanStep
from src.agent.models.final_response import FinalResponse
from src.agent.models.skill_result import SkillResult
from src.agent.planner import Planner
from src.agent.request_router import RouteDecision
from src.agent.task_orchestrator import TaskOrchestrator
from src.agent.streaming_service import StreamingService
from src.skills.router import AstronomySkillRouter
from src.skills.skill_handlers import (
    CelestialEventsForecastHandler,
    CelestialPositionCalculatorHandler,
    ObservationPlannerHandler,
)


class _MemoryStub:
    def __init__(self):
        self.messages = []
        self.session_id = "test_session"

    def build_context(self, request):
        return {"context_text": ""}

    def append_message(self, request):
        self.messages.append(
            {
                "role": request.role,
                "content": request.content,
                "timestamp": request.timestamp,
            }
        )


class _RagStub:
    def retrieve(self, query, fast_mode=True):
        return {"context": ""}


class _LLMStub:
    def invoke(self, prompt):
        return "stubbed"


def test_planner_builds_formal_observation_plan():
    planner = Planner()
    decision = RouteDecision(
        route="planned_task",
        task_type="observation_recommendation",
        confidence=0.8,
        reason="matched_multiple_skills",
        matched_skills=["weather-lookup", "observation-planner"],
        expected_output_schema="observation_answer_v1",
    )

    plan = planner.plan(
        query="帮我看下北京今晚适合观测什么",
        route_decision=decision,
        chat_history="",
        user_profile="",
    )

    assert isinstance(plan, ExecutionPlan)
    assert plan.task_type == "observation_recommendation"
    assert [step.id for step in plan.steps] == ["weather_context", "observation_plan"]
    assert plan.steps[0].skill == "weather-lookup"
    assert plan.steps[1].skill == "observation-planner"


@pytest.mark.asyncio
async def test_step_executor_runs_parallel_group_and_collects_trace():
    class _SkillManagerStub:
        def call_skill(self, name, **params):
            return SkillResult(
                skill_name=name,
                success=True,
                data={"params": params},
                summary=f"{name} ok",
                sources=[{"source_id": name, "kind": "tool_output", "title": name}],
            )

    executor = StepExecutor(skill_manager=_SkillManagerStub())
    plan = ExecutionPlan(
        task_type="astrophotography_advice",
        output_schema="astrophotography_answer_v1",
        steps=[
            PlanStep(
                id="photo_settings",
                kind="tool",
                title="计算摄影参数",
                skill="astrophotography-calculator",
                parallel_group="g1",
            ),
            PlanStep(
                id="photo_weather",
                kind="tool",
                title="查询天气",
                skill="weather-lookup",
                parallel_group="g1",
                required=False,
            ),
        ],
    )

    outcome = await executor.execute(
        plan,
        query="帮我做今晚M31摄影计划",
        param_builder=lambda skill, query: {"query": query, "skill": skill},
    )

    assert outcome.halted is False
    assert len(outcome.skill_results) == 2
    assert {step.step_id for step in outcome.step_results} == {
        "photo_settings",
        "photo_weather",
    }
    assert all(step.status == "success" for step in outcome.step_results)


@pytest.mark.asyncio
async def test_streaming_service_replays_real_plan_steps_for_planned_task():
    decision = RouteDecision(
        route="planned_task",
        task_type="observation_recommendation",
        confidence=0.8,
        reason="matched_multiple_skills",
        matched_skills=["weather-lookup", "observation-planner"],
        expected_output_schema="observation_answer_v1",
    )

    plan = ExecutionPlan(
        task_type="observation_recommendation",
        output_schema="observation_answer_v1",
        steps=[
            PlanStep(
                id="weather_context",
                kind="tool",
                title="查询天气条件",
                skill="weather-lookup",
            ),
            PlanStep(
                id="observation_plan",
                kind="tool",
                title="生成观测计划",
                skill="observation-planner",
            ),
        ],
    )

    async def fake_run(decision, query, **kwargs):
        return FinalResponse(
            answer="今晚适合观测猎户座和木星。",
            summary="今晚适合观测猎户座和木星。",
            tools_used=[],
            sources=[],
            confidence=0.88,
            route="planned_task",
            task_type="observation_recommendation",
            execution_plan=plan.to_dict(),
            execution_trace=[
                {
                    "step_id": "weather_context",
                    "title": "查询天气条件",
                    "status": "success",
                    "skill": "weather-lookup",
                    "summary": "天气良好",
                    "latency_ms": 10.0,
                    "sources": [{"source_id": "weather", "kind": "tool_output"}],
                },
                {
                    "step_id": "observation_plan",
                    "title": "生成观测计划",
                    "status": "success",
                    "skill": "observation-planner",
                    "summary": "适合观测猎户座和木星",
                    "latency_ms": 20.0,
                    "sources": [{"source_id": "planner", "kind": "tool_output"}],
                },
            ],
        )

    service = StreamingService(
        agent_executor=None,
        memory=_MemoryStub(),
        user_id="test_user",
        request_router=SimpleNamespace(route=lambda query: decision),
        task_orchestrator=SimpleNamespace(
            build_execution_plan=lambda *args, **kwargs: plan,
            run=fake_run,
        ),
    )

    events = []
    async for event in service.generate_events("帮我看下北京今晚适合观测什么"):
        events.append(event)

    step_start_ids = [
        event["step_id"] for event in events if event["type"] == "step_start"
    ]
    assert "weather_context" in step_start_ids
    assert "observation_plan" in step_start_ids

    final_answer = next(event for event in events if event["type"] == "final_answer")
    assert "猎户座" in final_answer["final_answer"]

    planned_updates = [
        event for event in events if event["type"] == "plan_update"
    ]
    assert any(
        any(step.get("id") == "weather_context" for step in update["steps"])
        for update in planned_updates
    )


def test_task_orchestrator_extracts_monthly_event_range():
    orchestrator = TaskOrchestrator(
        skill_manager=SimpleNamespace(),
        rag_retriever=_RagStub(),
        llm=_LLMStub(),
    )

    current_month_params = orchestrator._build_skill_params(
        "celestial-events-forecast",
        "请告诉我本月的天象",
    )
    explicit_month_params = orchestrator._build_skill_params(
        "celestial-events-forecast",
        "请告诉我2026年8月的天象",
    )

    assert "query" not in current_month_params
    assert current_month_params["start_date"].endswith("-01")
    assert current_month_params["end_date"] is not None
    assert explicit_month_params == {
        "start_date": "2026-08-01",
        "end_date": "2026-08-31",
    }


def test_skill_router_filters_unknown_handler_params():
    router = AstronomySkillRouter()

    def fake_handler(mcp, start_date=None, end_date=None, event_type=None):
        return SkillResult(
            skill_name="celestial-events-forecast",
            success=True,
            data={
                "start_date": start_date,
                "end_date": end_date,
                "event_type": event_type,
            },
            summary="ok",
        )

    router._handlers["celestial-events-forecast"] = fake_handler

    result = router.call(
        "celestial-events-forecast",
        query="请告诉我未来一周的天象",
        start_date="2026-04-23",
        end_date="2026-04-30",
    )

    assert result.success is True
    assert result.data == {
        "start_date": "2026-04-23",
        "end_date": "2026-04-30",
        "event_type": None,
    }


@pytest.mark.asyncio
async def test_direct_and_planned_celestial_event_paths_use_normalized_params():
    class _SkillManagerStub:
        def __init__(self):
            self.calls = []

        def call_skill(self, name, **params):
            self.calls.append((name, dict(params)))
            assert "query" not in params
            return SkillResult(
                skill_name=name,
                success=True,
                data={"params": params},
                summary=f"{name} ok",
            )

    skill_manager = _SkillManagerStub()
    orchestrator = TaskOrchestrator(
        skill_manager=skill_manager,
        rag_retriever=_RagStub(),
        llm=_LLMStub(),
    )

    direct_decision = RouteDecision(
        route="direct_task",
        task_type="single_tool_lookup",
        confidence=0.9,
        reason="matched_single_skill",
        matched_skills=["celestial-events-forecast"],
        expected_output_schema="tool_answer_v1",
    )
    await orchestrator.run(
        direct_decision,
        "请告诉我未来一周的天象",
        chat_history="",
        user_profile="",
    )

    planned_plan = ExecutionPlan(
        task_type="celestial_event_analysis",
        output_schema="event_analysis_answer_v1",
        steps=[
            PlanStep(
                id="event_forecast",
                kind="tool",
                title="查询天象事件",
                skill="celestial-events-forecast",
            )
        ],
    )
    await orchestrator._executor.execute(
        planned_plan,
        query="请告诉我2026年8月的天象",
        param_builder=orchestrator._build_skill_params,
    )

    assert skill_manager.calls[0][0] == "celestial-events-forecast"
    assert skill_manager.calls[0][1]["end_date"] is not None
    assert skill_manager.calls[1] == (
        "celestial-events-forecast",
        {
            "start_date": "2026-08-01",
            "end_date": "2026-08-31",
        },
    )


def test_observation_planner_summary_uses_clean_tool_text():
    handler = ObservationPlannerHandler()

    class _MCPStub:
        def call_tools_parallel(self, calls):
            return [
                '{"ok": true, "data": {"forecast": {"city": "苏州市", "casts": [{"date": "2026-04-23", "dayweather": "小雨", "nightweather": "晴", "daytemp": "15", "nighttemp": "10"}]}}, "meta": {"tool_name": "get_weather", "schema_version": "1.0"}}',
                '{"ok": true, "data": "🌌 本周没有特殊天象。", "meta": {"tool_name": "get_weekly_events", "schema_version": "1.0"}}',
                '{"ok": true, "data": "🌙 今晚适合先看月球。", "meta": {"tool_name": "get_tonight_best", "schema_version": "1.0"}}',
            ]

    result = handler(_MCPStub(), location="苏州", date="2026-04-23")

    assert '"ok"' not in result.summary
    assert '"meta"' not in result.summary
    assert "本周没有特殊天象" in result.summary
    assert "今晚适合先看月球" in result.summary
    assert result.data["weekly_events"] == "🌌 本周没有特殊天象。"
    assert result.data["tonight_best"] == "🌙 今晚适合先看月球。"


def test_celestial_events_forecast_summary_uses_clean_monthly_text():
    handler = CelestialEventsForecastHandler()

    class _MCPStub:
        def call_tools_parallel(self, calls):
            assert len(calls) == 2
            return [
                '{"ok": true, "data": "2026年8月上旬可见英仙座流星雨。", "meta": {"tool_name": "get_monthly_events", "schema_version": "1.0"}}',
                '{"ok": true, "data": "2026年9月初月相适合观测。", "meta": {"tool_name": "get_monthly_events", "schema_version": "1.0"}}',
            ]

    result = handler(
        _MCPStub(),
        start_date="2026-08-01",
        end_date="2026-09-05",
    )

    assert '"ok"' not in result.summary
    assert '"schema_version"' not in result.summary
    assert "英仙座流星雨" in result.summary
    assert "月相适合观测" in result.summary
    assert result.data["events_body"] == [
        "2026年8月上旬可见英仙座流星雨。",
        "2026年9月初月相适合观测。",
    ]


def test_celestial_position_summary_uses_clean_payload_text():
    handler = CelestialPositionCalculatorHandler()

    class _MCPStub:
        def call_tool(self, tool_name, **kwargs):
            assert tool_name == "get_planet_position"
            return '{"ok": true, "data": {"ra_hours": 12.34, "dec": -5.67, "altitude": 45.0}, "meta": {"tool_name": "get_planet_position", "schema_version": "1.0"}}'

    result = handler(_MCPStub(), target="mars", datetime="2026-04-23")

    assert '"ok"' not in result.summary
    assert '"meta"' not in result.summary
    assert "ra_hours" in result.summary
    assert result.data["position"] == {
        "ra_hours": 12.34,
        "dec": -5.67,
        "altitude": 45.0,
    }
