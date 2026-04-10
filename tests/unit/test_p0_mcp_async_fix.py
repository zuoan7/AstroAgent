"""
P0 Fix Tests - MCP Async Initialization & AsyncBridge

Tests for the critical P0 fixes:
1. AstronomySkillRouter no longer blocks on __init__ with sync MCP init
2. _AsyncBridge safely runs async operations from sync context
3. No more dangerous asyncio.run() inside running event loops
4. Lazy MCP session initialization (connect on first call, not on init)
5. Year replacement uses SUPPORTED_YEAR_RANGE instead of hardcoded 2026
"""

import asyncio
import json
import sys
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch
from types import ModuleType

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


class TestAsyncBridge:
    """Test the _AsyncBridge utility class."""

    def test_bridge_creates_background_loop(self):
        bridge = _AsyncBridge()
        assert bridge._loop is None
        assert bridge._thread is None

        bridge.start()
        assert bridge._loop is not None
        assert bridge._thread is not None
        assert bridge._thread.daemon is True
        assert not bridge._loop.is_closed()

        bridge.shutdown()

    def test_bridge_run_async_coroutine(self):
        bridge = _AsyncBridge()

        async def sample_coro():
            return 42

        result = bridge.run(sample_coro())
        assert result == 42

        bridge.shutdown()

    def test_bridge_run_with_exception(self):
        bridge = _AsyncBridge()

        async def failing_coro():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            bridge.run(failing_coro())

        bridge.shutdown()

    def test_bridge_idempotent_start(self):
        bridge = _AsyncBridge()
        bridge.start()
        loop1 = bridge._loop
        bridge.start()
        assert bridge._loop is loop1

        bridge.shutdown()

    def test_bridge_shutdown_cleans_up(self):
        bridge = _AsyncBridge()
        bridge.start()
        assert bridge._loop is not None

        bridge.shutdown()
        assert bridge._loop is None
        assert bridge._thread is None

    def test_bridge_run_multiple_coroutines(self):
        bridge = _AsyncBridge()

        async def add(a, b):
            return a + b

        results = [bridge.run(add(i, i * 2)) for i in range(5)]
        assert results == [0, 3, 6, 9, 12]

        bridge.shutdown()

    def test_bridge_timeout(self):
        bridge = _AsyncBridge()

        async def slow_coro():
            await asyncio.sleep(10)
            return "done"

        with pytest.raises(Exception):
            bridge.run(slow_coro(), timeout=0.5)

        bridge.shutdown()


class TestRouterLazyInit:
    """Test that AstronomySkillRouter initializes without blocking on MCP."""

    def test_router_init_does_not_call_mcp(self):
        with patch("src.skills.router._AsyncBridge") as MockBridge:
            mock_bridge_instance = MagicMock()
            MockBridge.return_value = mock_bridge_instance

            router = AstronomySkillRouter()

            assert router._mcp_initialized is False
            assert router._mcp_session_id is None

    def test_router_init_completes_without_mcp_server(self):
        router = AstronomySkillRouter()
        assert router._mcp_initialized is False
        assert router._async_bridge is not None

        router.shutdown()

    def test_router_has_shutdown_method(self):
        router = AstronomySkillRouter()
        assert hasattr(router, "shutdown")
        assert callable(router.shutdown)

        router.shutdown()

    def test_router_list_skills_without_mcp(self):
        router = AstronomySkillRouter()
        skills = router.list_skills()

        assert isinstance(skills, dict)
        assert "observation-planner" in skills
        assert "weather-lookup" in skills
        assert "neo-tracker" in skills

        router.shutdown()


class TestAsyncBridgeNoDeadlock:
    """Test that _AsyncBridge does not deadlock when called from within
    an already-running event loop (the core P0 fix)."""

    def test_call_from_running_event_loop(self):
        bridge = _AsyncBridge()

        async def inner_coro():
            return "from_inner"

        async def main():
            return bridge.run(inner_coro())

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(main())
            assert result == "from_inner"
        finally:
            loop.close()
            bridge.shutdown()

    def test_concurrent_calls_from_threads(self):
        bridge = _AsyncBridge()
        results = []
        errors = []

        async def compute(n):
            await asyncio.sleep(0.01)
            return n * 2

        def worker(n):
            try:
                r = bridge.run(compute(n))
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert sorted(results) == [i * 2 for i in range(10)]

        bridge.shutdown()


class TestMCPReconnectUsesAsyncInit:
    """Test that _reconnect delegates to _init_mcp_session (async)."""

    @pytest.mark.asyncio
    async def test_reconnect_calls_init_mcp_session(self):
        router = AstronomySkillRouter()

        with patch.object(
            router, "_init_mcp_session", new_callable=AsyncMock, return_value=True
        ) as mock_init:
            result = await router._reconnect()
            assert result is True
            mock_init.assert_called_once()

        router.shutdown()

    @pytest.mark.asyncio
    async def test_reconnect_retries_on_failure(self):
        router = AstronomySkillRouter()

        with patch.object(
            router,
            "_init_mcp_session",
            new_callable=AsyncMock,
            side_effect=[False, False, True],
        ):
            with patch("src.skills.router.asyncio.sleep", new_callable=AsyncMock):
                result = await router._reconnect()
                assert result is True

        router.shutdown()

    @pytest.mark.asyncio
    async def test_reconnect_fails_after_max_retries(self):
        router = AstronomySkillRouter()

        with patch.object(
            router,
            "_init_mcp_session",
            new_callable=AsyncMock,
            return_value=False,
        ):
            with patch("src.skills.router.asyncio.sleep", new_callable=AsyncMock):
                result = await router._reconnect()
                assert result is False

        router.shutdown()


class TestYearRangeFix:
    """Test that celestial-events-forecast uses SUPPORTED_YEAR_RANGE
    instead of hardcoded 2026."""

    def test_year_within_range_not_replaced(self):
        router = AstronomySkillRouter()

        with patch.object(router, "call_mcp_tool", return_value='{"events": []}'):
            with patch("src.skills.router.settings") as mock_settings:
                mock_settings.SUPPORTED_YEAR_RANGE = (2026, 2030)
                mock_settings.MCP_SERVER_URL = "http://localhost:8001/mcp"

                result = router._celestial_events_forecast(
                    start_date="2027-06-15"
                )
                assert "2027-06-15" in result

        router.shutdown()

    def test_year_outside_range_replaced(self):
        router = AstronomySkillRouter()

        with patch.object(router, "call_mcp_tool", return_value='{"events": []}'):
            with patch("src.skills.router.settings") as mock_settings:
                mock_settings.SUPPORTED_YEAR_RANGE = (2026, 2030)
                mock_settings.MCP_SERVER_URL = "http://localhost:8001/mcp"

                result = router._celestial_events_forecast(
                    start_date="2024-06-15"
                )
                assert "2026-06-15" in result
                assert "2024" not in result

        router.shutdown()


class TestCallMcpToolUsesAsyncBridge:
    """Test that call_mcp_tool delegates to _AsyncBridge.run."""

    def test_call_mcp_tool_uses_bridge(self):
        router = AstronomySkillRouter()

        with patch.object(
            router._async_bridge, "run", return_value='{"result": "ok"}'
        ) as mock_run:
            result = router.call_mcp_tool("test_tool", param1="value1")
            assert result == '{"result": "ok"}'
            mock_run.assert_called_once()

        router.shutdown()

    def test_call_mcp_tool_on_simple_skill(self):
        router = AstronomySkillRouter()

        with patch.object(
            router._async_bridge, "run", return_value='{"weather": "sunny"}'
        ) as mock_run:
            result = router.call("weather-lookup", city="Beijing")
            mock_run.assert_called_once()

        router.shutdown()


class TestNoAsyncioRunInRunningLoop:
    """Verify that the old dangerous pattern (asyncio.run inside running loop)
    has been removed from the codebase."""

    def test_no_asyncio_run_in_router(self):
        with open("src/skills/router.py", "r") as f:
            content = f.read()

        import re
        import ast
        try:
            tree = ast.parse(content)
        except SyntaxError:
            pytest.skip("router.py has syntax issues, skipping AST check")

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (isinstance(func, ast.Attribute)
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "asyncio"
                        and func.attr == "run"):
                    pytest.fail(
                        f"asyncio.run() found at line {node.lineno} in router.py - "
                        "it causes RuntimeError/deadlock when called from a running event loop"
                    )

    def test_no_thread_pool_executor_in_router(self):
        with open("src/skills/router.py", "r") as f:
            content = f.read()

        assert "ThreadPoolExecutor" not in content, (
            "ThreadPoolExecutor should not be used in router.py - "
            "use _AsyncBridge instead for safe async/sync bridging"
        )

    def test_no_init_mcp_session_sync(self):
        with open("src/skills/router.py", "r") as f:
            content = f.read()

        assert "_init_mcp_session_sync" not in content, (
            "_init_mcp_session_sync should be removed - "
            "MCP initialization is now async and lazy"
        )

    def test_no_call_mcp_tool_internal(self):
        with open("src/skills/router.py", "r") as f:
            content = f.read()

        assert "_call_mcp_tool_internal" not in content, (
            "_call_mcp_tool_internal should be removed - "
            "use _AsyncBridge.run() + _async_call_mcp_tool() instead"
        )
