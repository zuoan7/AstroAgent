import asyncio
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()
sys.modules.pop("src.agent.streaming_service", None)

from src.agent.models.final_response import FinalResponse
from src.agent.models.execution_event import ExecutionEvent
from src.agent.models.task_profile import TaskProfile
from src.agent.request_router import RequestRouter
from src.agent.streaming_service import StreamingService
from src.skills.mcp_client import MCPClient


class _MemoryStub:
    def __init__(self):
        self.messages = []
        self.session_id = "test_session"

    def build_context(self, request):
        formatted = [
            f"{'用户' if msg['role'] == 'user' else '助手'}: {msg['content']}"
            for msg in self.messages
        ]
        return {"context_text": "\n".join(formatted) or "无历史对话"}

    def append_message(self, request):
        self.messages.append(
            {
                "role": request.role,
                "content": request.content,
                "timestamp": request.timestamp,
            }
        )


@pytest.mark.parametrize(
    ("query", "expected_route", "expected_task_type"),
    [
        ("你好", "direct_task", "smalltalk"),
        ("北京天气怎么样", "direct_task", "single_tool_lookup"),
        ("赤经是什么", "direct_task", "simple_qa"),
        (
            "请比较双筒和赤道仪观测方案并给出步骤",
            "planned_task",
            "observation_recommendation",
        ),
    ],
)
def test_request_router_routes_expected_queries(query, expected_route, expected_task_type):
    router = RequestRouter()
    decision = router.route(query)
    assert decision.route == expected_route
    assert decision.task_type == expected_task_type


def test_mcp_parallel_calls_are_truly_concurrent(monkeypatch):
    client = MCPClient()

    async def fake_parallel_dispatch(tool_name, **kwargs):
        await asyncio.sleep(0.2)
        return f"{tool_name}:{kwargs.get('value')}"

    monkeypatch.setattr(client, "_dispatch_parallel_tool_call", fake_parallel_dispatch)

    started = time.perf_counter()
    results = client.call_tools_parallel(
        [
            {"tool_name": "tool_a", "kwargs": {"value": 1}},
            {"tool_name": "tool_b", "kwargs": {"value": 2}},
            {"tool_name": "tool_c", "kwargs": {"value": 3}},
        ]
    )
    elapsed = time.perf_counter() - started

    assert results == ["tool_a:1", "tool_b:2", "tool_c:3"]
    assert elapsed < 0.4, f"parallel execution too slow: {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_streaming_service_smalltalk_uses_direct_route():
    service = StreamingService(
        agent_executor=None,
        memory=_MemoryStub(),
        user_id="test_user",
        request_router=RequestRouter(),
        task_orchestrator=SimpleNamespace(),
    )

    async def fake_run(decision, query, **kwargs):
        assert decision.route == "direct_task"
        assert decision.task_type == "smalltalk"
        return FinalResponse(
            answer="你好，我可以帮你查询天象、观测条件、天体位置和天文知识。",
            summary="你好，我可以帮你查询天象、观测条件、天体位置和天文知识。",
            tools_used=[],
            sources=[],
            confidence=0.98,
            route="direct_task",
            task_type="smalltalk",
        )

    service._task_orchestrator.run = fake_run

    events = []
    async for event in service.generate_events("你好"):
        events.append(event)

    event_types = [event["type"] for event in events]
    assert "route_decision" in event_types
    assert "latency_metrics" in event_types
    final_answer = next(event for event in events if event["type"] == "final_answer")
    assert "你好" in final_answer["final_answer"]
    assert "route_decision_ms" in final_answer["latency_metrics"]["stages_ms"]


@pytest.mark.asyncio
async def test_streaming_service_generate_events_uses_policy_decide():
    route_decision = SimpleNamespace(
        route="direct_task",
        task_type="smalltalk",
        confidence=0.98,
        reason="matched_smalltalk_pattern",
        matched_skills=[],
        expected_output_schema="chat_answer_v1",
        to_meta=lambda: {
            "route": "direct_task",
            "task_type": "smalltalk",
            "route_confidence": 0.98,
            "route_reason": "matched_smalltalk_pattern",
            "matched_skills": [],
            "expected_output_schema": "chat_answer_v1",
        },
    )
    profile = TaskProfile.from_legacy_route(
        route="direct_task",
        task_type="smalltalk",
        confidence=0.98,
        reason="matched_smalltalk_pattern",
        expected_output_schema="chat_answer_v1",
    )
    router = SimpleNamespace(route=lambda query: route_decision, profile=lambda query: profile)
    service = StreamingService(
        agent_executor=None,
        memory=_MemoryStub(),
        user_id="test_user",
        request_router=router,
        task_orchestrator=SimpleNamespace(),
    )

    called = {"count": 0, "mode": None}

    def decide(p, context=None):
        called["count"] += 1
        called["mode"] = "direct"
        return SimpleNamespace(
            mode="direct",
            reason="test_decide",
            fallback_modes=[],
            legacy_execution_path="direct",
            to_dict=lambda: {
                "mode": "direct",
                "reason": "test_decide",
                "fallback_modes": [],
                "legacy_execution_path": "direct",
            },
        )

    service._execution_policy = SimpleNamespace(
        mode="hybrid",
        decide=decide,
        choose_path=lambda route: "react" if route is None else "direct",
        to_dict=lambda: {"mode": "hybrid"},
    )

    async def fake_run(decision, query, **kwargs):
        return FinalResponse(
            answer="你好，我可以帮你查询天象。",
            summary="你好，我可以帮你查询天象。",
            tools_used=[],
            sources=[],
            confidence=0.98,
            route="direct_task",
            task_type="smalltalk",
        )

    service._task_orchestrator.run = fake_run

    events = []
    async for event in service.generate_events("你好"):
        events.append(event)

    assert called["count"] >= 1
    final_answer = next(event for event in events if event["type"] == "final_answer")
    assert "你好" in final_answer["final_answer"]


def test_save_to_memory_schedules_ltm_async(monkeypatch):
    memory = _MemoryStub()
    ltm = MagicMock()
    service = StreamingService(
        agent_executor=None,
        memory=memory,
        long_term_memory=ltm,
        user_id="test_user",
    )

    called = {"started": False}

    def fake_schedule(query, response):
        called["started"] = True

    monkeypatch.setattr(service, "_schedule_long_term_memory_update", fake_schedule)
    service._save_to_memory("u", "a", use_long_term_memory=True)

    assert len(memory.messages) == 2
    assert called["started"] is True


def test_streaming_service_generate_response_uses_policy_decide():
    route_decision = SimpleNamespace(
        route="direct_task",
        task_type="smalltalk",
        confidence=0.98,
        reason="matched_smalltalk_pattern",
        matched_skills=[],
        expected_output_schema="chat_answer_v1",
    )
    profile = TaskProfile.from_legacy_route(
        route="direct_task",
        task_type="smalltalk",
        confidence=0.98,
        reason="matched_smalltalk_pattern",
        expected_output_schema="chat_answer_v1",
    )
    router = SimpleNamespace(route=lambda query: route_decision, profile=lambda query: profile)
    service = StreamingService(
        agent_executor=None,
        memory=_MemoryStub(),
        user_id="test_user",
        request_router=router,
        task_orchestrator=SimpleNamespace(),
    )

    called = {"count": 0}

    def decide(p, context=None):
        called["count"] += 1
        return SimpleNamespace(
            mode="direct",
            reason="test_decide",
            fallback_modes=[],
            legacy_execution_path="direct",
            to_dict=lambda: {
                "mode": "direct",
                "reason": "test_decide",
                "fallback_modes": [],
                "legacy_execution_path": "direct",
            },
        )

    service._execution_policy = SimpleNamespace(
        mode="hybrid",
        decide=decide,
        choose_path=lambda route: "react" if route is None else "direct",
        to_dict=lambda: {"mode": "hybrid"},
    )

    async def fake_run(decision, query, **kwargs):
        return FinalResponse(
            answer="你好，我在。",
            summary="你好，我在。",
            tools_used=[],
            sources=[],
            confidence=0.98,
            route="direct_task",
            task_type="smalltalk",
        )

    service._task_orchestrator.run = fake_run

    chunks = list(service.generate_response("你好"))

    assert called["count"] >= 1
    assert chunks == ["你好，我在。"]


def test_streaming_service_generate_response_uses_engine_for_react():
    route_decision = SimpleNamespace(
        route="fallback_react",
        task_type="open_domain_reasoning",
        confidence=0.88,
        reason="open_ended",
        matched_skills=[],
        expected_output_schema="react_answer_v1",
    )
    profile = TaskProfile.from_legacy_route(
        route="fallback_react",
        task_type="open_domain_reasoning",
        confidence=0.88,
        reason="open_ended",
        expected_output_schema="react_answer_v1",
    )
    router = SimpleNamespace(route=lambda query: route_decision, profile=lambda query: profile)
    service = StreamingService(
        agent_executor=SimpleNamespace(
            invoke=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("legacy react invoke should not be called")
            )
        ),
        memory=_MemoryStub(),
        user_id="test_user",
        request_router=router,
        task_orchestrator=SimpleNamespace(),
    )

    called = {"count": 0}

    def decide(p, context=None):
        return SimpleNamespace(
            mode="react",
            reason="test_decide",
            fallback_modes=[],
            legacy_execution_path="react",
            to_dict=lambda: {
                "mode": "react",
                "reason": "test_decide",
                "fallback_modes": [],
                "legacy_execution_path": "react",
            },
        )

    async def fake_engine_run(*args, **kwargs):
        called["count"] += 1
        return FinalResponse(
            answer="engine react 答案",
            summary="engine react 答案",
            tools_used=[],
            sources=[],
            confidence=0.8,
            route="fallback_react",
            task_type="open_domain_reasoning",
        )

    service._execution_policy = SimpleNamespace(
        mode="hybrid",
        decide=decide,
        choose_path=lambda route: "react",
        to_dict=lambda: {"mode": "hybrid"},
    )
    service._execution_engine = SimpleNamespace(run=fake_engine_run)
    service._task_orchestrator.run = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("legacy orchestrator should not be called")
    )

    chunks = list(service.generate_response("写一段开放式宇宙随笔"))

    assert called["count"] == 1
    assert chunks == ["engine react 答案"]


@pytest.mark.asyncio
async def test_streaming_service_generate_events_uses_engine_stream_for_react():
    route_decision = SimpleNamespace(
        route="fallback_react",
        task_type="open_domain_reasoning",
        confidence=0.88,
        reason="open_ended",
        matched_skills=[],
        expected_output_schema="react_answer_v1",
        to_meta=lambda: {
            "route": "fallback_react",
            "task_type": "open_domain_reasoning",
            "route_confidence": 0.88,
            "route_reason": "open_ended",
            "matched_skills": [],
            "expected_output_schema": "react_answer_v1",
        },
    )
    profile = TaskProfile.from_legacy_route(
        route="fallback_react",
        task_type="open_domain_reasoning",
        confidence=0.88,
        reason="open_ended",
        expected_output_schema="react_answer_v1",
    )
    router = SimpleNamespace(route=lambda query: route_decision, profile=lambda query: profile)
    service = StreamingService(
        agent_executor=None,
        memory=_MemoryStub(),
        user_id="test_user",
        request_router=router,
        task_orchestrator=SimpleNamespace(),
        agent_executor_factory=lambda: (_ for _ in ()).throw(
            AssertionError("legacy react executor factory should not be called")
        ),
    )

    streamed = {"count": 0}

    def decide(p, context=None):
        return SimpleNamespace(
            mode="react",
            reason="test_decide",
            fallback_modes=[],
            legacy_execution_path="react",
            to_dict=lambda: {
                "mode": "react",
                "reason": "test_decide",
                "fallback_modes": [],
                "legacy_execution_path": "react",
            },
        )

    async def fake_astream_events(*args, **kwargs):
        streamed["count"] += 1
        yield {
            "event": "on_llm_stream",
            "data": {"chunk": SimpleNamespace(content="Final Answer: engine stream 答案")},
            "run_id": "react-engine-1",
        }

    service._execution_policy = SimpleNamespace(
        mode="hybrid",
        decide=decide,
        choose_path=lambda route: "react",
        to_dict=lambda: {"mode": "hybrid"},
    )
    service._execution_engine = SimpleNamespace(astream_events=fake_astream_events)

    events = []
    async for event in service.generate_events("写一段开放式宇宙随笔"):
        events.append(event)

    assert streamed["count"] == 1
    final_answer = next(event for event in events if event["type"] == "final_answer")
    assert final_answer["final_answer"] == "engine stream 答案"


@pytest.mark.asyncio
async def test_streaming_service_planned_events_use_engine_preview_plan():
    route_decision = SimpleNamespace(
        route="planned_task",
        task_type="observation_recommendation",
        confidence=0.9,
        reason="complex_task",
        matched_skills=["weather-lookup", "observation-planner"],
        expected_output_schema="observation_answer_v1",
        to_meta=lambda: {
            "route": "planned_task",
            "task_type": "observation_recommendation",
            "route_confidence": 0.9,
            "route_reason": "complex_task",
            "matched_skills": ["weather-lookup", "observation-planner"],
            "expected_output_schema": "observation_answer_v1",
        },
    )
    profile = TaskProfile.from_legacy_route(
        route="planned_task",
        task_type="observation_recommendation",
        confidence=0.9,
        reason="complex_task",
        matched_skills=["weather-lookup", "observation-planner"],
        expected_output_schema="observation_answer_v1",
    )
    router = SimpleNamespace(route=lambda query: route_decision, profile=lambda query: profile)
    service = StreamingService(
        agent_executor=None,
        memory=_MemoryStub(),
        user_id="test_user",
        request_router=router,
        task_orchestrator=SimpleNamespace(
            build_execution_plan=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("legacy build_execution_plan should not be called")
            )
        ),
    )

    def decide(p, context=None):
        return SimpleNamespace(
            mode="planned",
            reason="test_decide",
            fallback_modes=["react"],
            legacy_execution_path="planned",
            to_dict=lambda: {
                "mode": "planned",
                "reason": "test_decide",
                "fallback_modes": ["react"],
                "legacy_execution_path": "planned",
            },
        )

    from src.agent.models.execution_plan import ExecutionPlan, PlanStep
    plan = ExecutionPlan(
        task_type="observation_recommendation",
        output_schema="observation_answer_v1",
        steps=[
            PlanStep(id="weather_context", kind="tool", title="查询天气", skill="weather-lookup"),
            PlanStep(id="observation_plan", kind="tool", title="生成观测计划", skill="observation-planner"),
        ],
    )

    preview_called = {"count": 0}
    run_called = {"count": 0}

    def preview_plan(*args, **kwargs):
        preview_called["count"] += 1
        return plan

    async def fake_run(*args, **kwargs):
        run_called["count"] += 1
        return FinalResponse(
            answer="今晚适合先看猎户座。",
            summary="今晚适合先看猎户座。",
            tools_used=[],
            sources=[],
            confidence=0.9,
            route="planned_task",
            task_type="observation_recommendation",
            execution_plan=plan.to_dict(),
            execution_trace=[],
            execution_events=[
                ExecutionEvent(
                    type="plan_created",
                    payload={"plan": plan.to_dict()},
                    source="planned",
                ).to_dict(),
                ExecutionEvent(
                    type="step_started",
                    payload={"step_id": "weather_context", "title": "查询天气"},
                    source="planned",
                ).to_dict(),
                ExecutionEvent(
                    type="step_finished",
                    payload={"step_id": "weather_context", "status": "success"},
                    source="planned",
                ).to_dict(),
                ExecutionEvent(
                    type="step_started",
                    payload={"step_id": "observation_plan", "title": "生成观测计划"},
                    source="planned",
                ).to_dict(),
                ExecutionEvent(
                    type="step_finished",
                    payload={"step_id": "observation_plan", "status": "success"},
                    source="planned",
                ).to_dict(),
                ExecutionEvent(
                    type="answer_ready",
                    payload={"answer": "今晚适合先看猎户座。"},
                    source="planned",
                ).to_dict(),
            ],
        )

    service._execution_policy = SimpleNamespace(
        mode="hybrid",
        decide=decide,
        choose_path=lambda route: "planned",
        to_dict=lambda: {"mode": "hybrid"},
    )
    service._execution_engine = SimpleNamespace(
        preview_plan=preview_plan,
        run=fake_run,
    )

    events = []
    async for event in service.generate_events("帮我看下北京今晚适合观测什么"):
        events.append(event)

    assert preview_called["count"] == 1
    assert run_called["count"] == 1
    assert any(event["type"] == "route_decision" for event in events)
    assert any(event["type"] == "plan_update" for event in events)
    assert any(event["type"] == "step_start" and event["step_id"] == "weather_context" for event in events)
    assert any(event["type"] == "step_end" and event["step_id"] == "observation_plan" for event in events)
    final_answer = next(event for event in events if event["type"] == "final_answer")
    assert "猎户座" in final_answer["final_answer"]
