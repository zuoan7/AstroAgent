import sys
from types import SimpleNamespace

import pytest

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()
sys.modules.pop("src.agent.streaming_service", None)

from src.agent.governance import (
    AgentExecutionPolicy,
    GovernanceMetricsRegistry,
    RequestObservation,
    evaluate_router_benchmark,
    load_phase0_benchmark_cases,
)
from src.agent.request_router import RequestRouter
from src.agent.streaming_service import StreamingService


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


def test_phase0_benchmark_dataset_is_available():
    cases = load_phase0_benchmark_cases()

    assert len(cases) >= 40
    assert cases[0].case_id
    assert all(case.acceptable_latency_ms > 0 for case in cases)


def test_router_benchmark_evaluation_reports_mismatch_rate():
    cases = load_phase0_benchmark_cases()
    report = evaluate_router_benchmark(RequestRouter(), cases)

    assert report["evaluated_cases"] == len(cases)
    assert 0.0 <= report["route_mismatch_rate"] <= 1.0
    assert "smalltalk" in report["by_category"]


def test_governance_metrics_registry_computes_phase0_summary():
    registry = GovernanceMetricsRegistry()
    registry.record(
        RequestObservation(
            route="direct_task",
            request_total_ms=100.0,
            agent_mode="hybrid",
            execution_path="direct",
            fallback_used=False,
            output_schema_parse_success=True,
            route_expected="direct_task",
        )
    )
    registry.record(
        RequestObservation(
            route="planned_task",
            request_total_ms=500.0,
            agent_mode="react",
            execution_path="react",
            fallback_used=True,
            output_schema_parse_success=False,
            route_expected="direct_task",
        )
    )

    snapshot = registry.snapshot()

    assert snapshot["total_requests"] == 2
    assert snapshot["fallback_rate"] == 0.5
    assert snapshot["output_schema_parse_rate"] == 0.5
    assert snapshot["route_mismatch_rate"] == 0.5
    assert snapshot["latency_ms"]["p50"] == 300.0
    assert snapshot["latency_ms"]["p90"] == 460.0


def test_execution_policy_supports_phase0_flags():
    hybrid = AgentExecutionPolicy(mode="hybrid")
    react = AgentExecutionPolicy(mode="react")
    planned = AgentExecutionPolicy(mode="planned", enable_planner=False)

    assert hybrid.choose_path("direct_task") == "direct"
    assert hybrid.choose_path("planned_task") == "planned"
    assert react.choose_path("direct_task") == "react"
    assert planned.choose_path("fallback_react") == "react"


@pytest.mark.asyncio
async def test_streaming_service_records_governance_metrics():
    registry = GovernanceMetricsRegistry()
    service = StreamingService(
        agent_executor=None,
        memory=_MemoryStub(),
        user_id="test_user",
        request_router=RequestRouter(),
        task_orchestrator=SimpleNamespace(),
        execution_policy=AgentExecutionPolicy(mode="hybrid"),
        governance_metrics=registry,
    )

    async def fake_run(decision, query, **kwargs):
        assert decision.route == "direct_task"
        assert decision.task_type == "smalltalk"
        return {
            "answer": "你好，我可以帮你查询天象。",
            "tools_used": [],
            "sources": [],
        }

    service._task_orchestrator.run = fake_run

    events = []
    async for event in service.generate_events("你好"):
        events.append(event)

    snapshot = registry.snapshot()

    assert any(event["type"] == "final_answer" for event in events)
    assert snapshot["total_requests"] == 1
    assert snapshot["fallback_rate"] == 0.0
    assert snapshot["by_mode"]["hybrid"] == 1


@pytest.mark.asyncio
async def test_streaming_service_runs_planned_task_without_react():
    registry = GovernanceMetricsRegistry()
    service = StreamingService(
        agent_executor=None,
        memory=_MemoryStub(),
        user_id="test_user",
        request_router=RequestRouter(),
        task_orchestrator=SimpleNamespace(),
        execution_policy=AgentExecutionPolicy(mode="hybrid", enable_planner=True),
        governance_metrics=registry,
        agent_executor_factory=lambda: (_ for _ in ()).throw(
            AssertionError("react executor should not be created")
        ),
    )

    async def fake_run(decision, query, **kwargs):
        assert decision.route == "planned_task"
        return {
            "answer": "已根据天气和天象生成今晚观测建议。",
            "tools_used": [{"tool": "weather-lookup", "status": "success"}],
            "sources": [{"source_id": "weather-lookup", "kind": "tool_output"}],
        }

    service._task_orchestrator.run = fake_run

    events = []
    async for event in service.generate_events("请比较今晚用双筒和赤道仪观测方案并给出步骤"):
        events.append(event)

    final_answer = next(event for event in events if event["type"] == "final_answer")
    assert "观测建议" in final_answer["final_answer"]
    assert registry.snapshot()["by_route"]["planned_task"] == 1


@pytest.mark.asyncio
async def test_streaming_service_only_builds_react_on_fallback():
    class _FallbackRouter:
        def route(self, query):
            return SimpleNamespace(
                route="fallback_react",
                task_type="open_domain_reasoning",
                matched_skills=[],
                expected_output_schema="react_answer_v1",
                to_meta=lambda: {
                    "route": "fallback_react",
                    "task_type": "open_domain_reasoning",
                    "matched_skills": [],
                    "expected_output_schema": "react_answer_v1",
                },
            )

    class _AgentExecutorStub:
        async def astream_events(self, agent_input, version="v1"):
            yield {
                "event": "on_llm_stream",
                "data": {"chunk": SimpleNamespace(content="Final Answer: fallback ok")},
                "run_id": "react-1",
            }

    created = {"count": 0}

    def factory():
        created["count"] += 1
        return _AgentExecutorStub()

    service = StreamingService(
        agent_executor=None,
        memory=_MemoryStub(),
        user_id="test_user",
        request_router=_FallbackRouter(),
        task_orchestrator=SimpleNamespace(),
        execution_policy=AgentExecutionPolicy(mode="hybrid", enable_react_fallback=True),
        agent_executor_factory=factory,
    )

    events = []
    async for event in service.generate_events("写一篇关于宇宙意义的开放式散文"):
        events.append(event)

    final_answer = next(event for event in events if event["type"] == "final_answer")
    assert final_answer["final_answer"] == "fallback ok"
    assert created["count"] == 1
