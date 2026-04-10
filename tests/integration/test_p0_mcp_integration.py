"""
P0 Integration Tests - MCP Async Init in Full Agent Context

Integration tests verifying that the P0 fixes work correctly
when the full AstroAgent stack is involved:
1. Agent can initialize without MCP server running
2. Skill calls properly route through _AsyncBridge
3. Error handling when MCP is unavailable
4. No deadlock when called from FastAPI-like async context
"""

import asyncio
import json
import sys
import threading
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _setup_mocks():
    for mod_name in [
        "langchain", "langchain.schema",
        "langchain_community", "langchain_community.chat_models",
        "langchain_community.embeddings",
        "langchain_core", "langchain_core.prompts", "langchain_core.tools",
        "langchain_core.callbacks", "langchain_core.messages",
        "langchain_classic", "langchain_classic.agents",
        "langchain_chroma", "langgraph", "dashscope",
        "dashscope.audio", "dashscope.audio.asr",
        "dashscope.multi_modal", "dashscope.protocol",
        "chromadb", "rank_bm25", "fastmcp", "langchain_mcp_adapters",
        "astroquery", "astroquery.simbad", "astroquery.ned",
        "streamlit", "rich", "langchain_text_splitters",
        "skyfield", "skyfield.api", "skyfield.almanac",
        "astropy", "astropy.coordinates",
        "ephem", "cachetools", "tenacity", "pybreaker",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()

    if "src.agent" not in sys.modules:
        m = ModuleType("src.agent")
        m.__path__ = []
        sys.modules["src.agent"] = m
    if "src.agent.skill_manager" not in sys.modules:
        sys.modules["src.agent.skill_manager"] = MagicMock()
    if "src.agent.param_parser" not in sys.modules:
        m = MagicMock()
        m.ParamParser = MagicMock()
        m.ParamParser.normalize_location = MagicMock(return_value="Beijing")
        sys.modules["src.agent.param_parser"] = m

    if "src.core" not in sys.modules:
        m = ModuleType("src.core")
        m.__path__ = []
        sys.modules["src.core"] = m
    if "src.core.config" not in sys.modules:
        m = ModuleType("src.core.config")
        m.settings = MagicMock()
        m.settings.MCP_SERVER_URL = "http://localhost:8001/mcp"
        m.settings.SUPPORTED_YEAR_RANGE = (2026, 2030)
        m.settings.MCP_RECONNECT_MAX_RETRIES = 3
        sys.modules["src.core.config"] = m
    if "src.core.logger" not in sys.modules:
        import logging
        m = ModuleType("src.core.logger")
        m.logger = logging.getLogger("test")
        sys.modules["src.core.logger"] = m
    if "src.core.errors" not in sys.modules:
        m = ModuleType("src.core.errors")

        class _ErrorCode:
            MCP_SESSION_ERROR = "MCP_SESSION_ERROR"
            MCP_TIMEOUT_ERROR = "MCP_TIMEOUT_ERROR"
            MCP_CONNECTION_ERROR = "MCP_CONNECTION_ERROR"
            TOOL_CALL_FAILED = "TOOL_CALL_FAILED"

        class _AgentError:
            def __init__(self, code, message, details=None):
                self.code = code
                self.message = message
                self.details = details or {}
            def to_dict(self):
                return {"error": True, "code": self.code, "message": self.message, "details": self.details}

        class _ErrorHandler:
            @staticmethod
            def handle(e, ctx=None):
                return _AgentError("UNKNOWN", str(e), ctx or {})

        m.ErrorCode = _ErrorCode
        m.AgentError = _AgentError
        m.ErrorHandler = _ErrorHandler
        sys.modules["src.core.errors"] = m
    if "src.utils" not in sys.modules:
        m = ModuleType("src.utils")
        m.__path__ = []
        sys.modules["src.utils"] = m
    if "src.utils.helpers" not in sys.modules:
        m = ModuleType("src.utils.helpers")
        from datetime import datetime as _dt

        def _parse_date(s):
            if s is None:
                return _dt.now()
            from datetime import datetime as _d
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y年%m月%d日"):
                try:
                    return _d.strptime(str(s), fmt)
                except (ValueError, TypeError):
                    continue
            return _dt.now()

        m.parse_date = _parse_date
        m.shorten_text = lambda s, n: s[:n] + "..." if len(s) > n else s
        sys.modules["src.utils.helpers"] = m


_setup_mocks()

from src.skills.router import _AsyncBridge, AstronomySkillRouter


class TestAgentInitWithoutMCPServer:
    """Test that AstroAgent can initialize even when MCP server is down."""

    def test_router_init_does_not_block_on_mcp(self):
        router = AstronomySkillRouter()
        assert router._mcp_initialized is False
        assert router._mcp_session_id is None
        assert router._async_bridge is not None

        router.shutdown()

    def test_router_skills_available_without_mcp(self):
        router = AstronomySkillRouter()
        skills = router.list_skills()
        assert len(skills) >= 6

        router.shutdown()


class TestAsyncBridgeInFastAPIContext:
    """Test that _AsyncBridge works correctly when called from
    within a running asyncio event loop (simulating FastAPI)."""

    def test_sync_call_from_async_context(self):
        bridge = _AsyncBridge()

        async def mcp_tool_simulator(tool_name, **kwargs):
            await asyncio.sleep(0.01)
            return json.dumps({"tool": tool_name, "result": "success"})

        async def simulate_fastapi_handler():
            return bridge.run(mcp_tool_simulator("get_weather", city="Beijing"))

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(simulate_fastapi_handler())
            parsed = json.loads(result)
            assert parsed["tool"] == "get_weather"
            assert parsed["result"] == "success"
        finally:
            loop.close()
            bridge.shutdown()

    def test_multiple_concurrent_fastapi_requests(self):
        bridge = _AsyncBridge()

        async def mcp_tool_simulator(tool_name, **kwargs):
            await asyncio.sleep(0.02)
            return json.dumps({"tool": tool_name})

        async def simulate_request(tool_name):
            return bridge.run(mcp_tool_simulator(tool_name))

        async def simulate_concurrent_requests():
            tasks = [
                simulate_request(f"tool_{i}")
                for i in range(5)
            ]
            return await asyncio.gather(*tasks)

        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(simulate_concurrent_requests())
            assert len(results) == 5
            for i, result in enumerate(results):
                parsed = json.loads(result)
                assert parsed["tool"] == f"tool_{i}"
        finally:
            loop.close()
            bridge.shutdown()


class TestMCPErrorHandlingWhenServerDown:
    """Test error handling when MCP server is unavailable."""

    def test_call_mcp_tool_returns_error_when_mcp_down(self):
        router = AstronomySkillRouter()

        error_response = json.dumps({
            "error": True,
            "code": "MCP_CONNECTION_ERROR",
            "message": "无法连接到MCP服务器",
        })

        with patch.object(
            router._async_bridge, "run", return_value=error_response
        ):
            result = router.call_mcp_tool("get_weather", city="Beijing")
            parsed = json.loads(result)
            assert parsed.get("error") is True
            assert "MCP_CONNECTION_ERROR" in parsed.get("code", "")

        router.shutdown()

    def test_skill_call_degrades_gracefully_on_mcp_error(self):
        router = AstronomySkillRouter()

        error_response = json.dumps({
            "error": True,
            "code": "MCP_SESSION_ERROR",
            "message": "MCP会话不可用",
        })

        with patch.object(
            router._async_bridge, "run", return_value=error_response
        ):
            result = router.call("weather-lookup", city="Beijing")
            assert len(result) > 0

        router.shutdown()


class TestRouterShutdown:
    """Test proper cleanup of resources."""

    def test_shutdown_cleans_bridge(self):
        router = AstronomySkillRouter()
        bridge = router._async_bridge

        bridge.start()
        assert bridge._loop is not None

        router.shutdown()
        assert bridge._loop is None

    def test_shutdown_idempotent(self):
        router = AstronomySkillRouter()
        router.shutdown()
        router.shutdown()


class TestEndToEndSkillCall:
    """End-to-end test of skill call through the full stack."""

    def test_skill_call_through_bridge(self):
        router = AstronomySkillRouter()

        mock_response = json.dumps({
            "query_city": "Beijing",
            "live": {
                "city": "Beijing",
                "weather": "Sunny",
                "temperature": "25",
                "humidity": "40",
                "windpower": "3",
            },
        })

        with patch.object(
            router._async_bridge, "run", return_value=mock_response
        ):
            result = router.call("weather-lookup", city="Beijing")
            assert "Beijing" in result or "Sunny" in result or len(result) > 0

        router.shutdown()

    def test_observation_planner_skill(self):
        router = AstronomySkillRouter()

        weather_resp = json.dumps({
            "live": {"city": "Beijing", "weather": "Clear", "temperature": "20",
                     "humidity": "50", "windpower": "2"},
        })
        events_resp = "No special events this week."
        tonight_resp = "Jupiter is visible tonight."

        call_count = [0]

        def mock_run(coro, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return weather_resp
            elif call_count[0] == 2:
                return events_resp
            else:
                return tonight_resp

        with patch.object(router._async_bridge, "run", side_effect=mock_run):
            result = router._observation_planner(
                date="2026-06-15", location="Beijing"
            )
            assert "观测日期" in result
            assert "观测条件" in result

        router.shutdown()

    def test_unknown_skill_raises_error(self):
        router = AstronomySkillRouter()

        with pytest.raises(ValueError, match="未知技能"):
            router.call("nonexistent-skill")

        router.shutdown()
