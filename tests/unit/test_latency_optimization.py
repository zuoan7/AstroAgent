import asyncio
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()
sys.modules.pop("src.agent.streaming_service", None)

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

    async def fake_ensure_session():
        return True

    async def fake_call(tool_name, _skip_session_check=False, **kwargs):
        await asyncio.sleep(0.2)
        return f"{tool_name}:{kwargs.get('value')}"

    monkeypatch.setattr(client, "_ensure_session", fake_ensure_session)
    monkeypatch.setattr(client, "_async_call_tool", fake_call)

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
        return {
            "answer": "你好，我可以帮你查询天象、观测条件、天体位置和天文知识。",
            "tools_used": [],
            "sources": [],
        }

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
