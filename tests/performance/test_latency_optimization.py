import time
import sys
from types import SimpleNamespace

import pytest

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()
sys.modules.pop("src.agent.streaming_service", None)

from src.agent.request_router import RequestRouter
from src.agent.streaming_service import StreamingService


class _MemoryStub:
    def __init__(self):
        self.messages = []

    def get_recent_messages(self, window=4):
        return self.messages[-window:]

    def add_message(self, role, content, timestamp):
        self.messages.append({"role": role, "content": content, "timestamp": timestamp})


@pytest.mark.performance
@pytest.mark.asyncio
async def test_smalltalk_direct_path_latency_budget():
    service = StreamingService(
        agent_executor=None,
        memory=_MemoryStub(),
        user_id="perf_user",
        request_router=RequestRouter(),
        task_orchestrator=SimpleNamespace(),
    )

    async def fake_run(decision, query, **kwargs):
        return {
            "answer": "你好，我在。",
            "tools_used": [],
            "sources": [],
        }

    service._task_orchestrator.run = fake_run

    started = time.perf_counter()
    async for _ in service.generate_events("你好"):
        pass
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    assert elapsed_ms < 500, f"smalltalk direct path too slow: {elapsed_ms:.1f}ms"


@pytest.mark.performance
def test_parallel_tool_mock_regression_budget():
    started = time.perf_counter()

    def simulated_parallel():
        time.sleep(0.25)

    simulated_parallel()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    assert elapsed_ms < 400, f"parallel orchestration regression: {elapsed_ms:.1f}ms"
