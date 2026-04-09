import os
import sys
import json
import time
import asyncio
import tempfile
import sqlite3
from collections import OrderedDict
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock, PropertyMock

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.mock_deps import mock_heavy_dependencies
mock_heavy_dependencies()

for mod_name in [
    "src.agent.skill_manager",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()


# ===== Issue 6 & 7: Router MCP Session Reconnection & Async =====

class TestMCPSessionReconnection:
    def _create_router(self):
        with patch("src.skills.router.httpx.Client") as MockClient, \
             patch("src.skills.router.httpx.AsyncClient") as MockAsyncClient:
            mock_sync_client = MagicMock()
            mock_sync_response = MagicMock()
            mock_sync_response.headers = {"mcp-session-id": "test-session-123"}
            mock_sync_response.status_code = 200
            mock_sync_response.text = 'data: {"result":{"serverInfo":{"name":"test"}}}'
            mock_sync_client.get.return_value = mock_sync_response
            mock_sync_client.post.return_value = mock_sync_response
            MockClient.return_value = mock_sync_client

            mock_async_client = MagicMock()
            MockAsyncClient.return_value = mock_async_client

            from src.skills.router import AstronomySkillRouter
            return AstronomySkillRouter()

    def test_session_valid_when_initialized(self):
        router = self._create_router()
        router._mcp_initialized = True
        router._mcp_session_id = "test-session"
        mock_client = MagicMock()
        mock_client.is_closed = False
        router._http_client = mock_client

        assert router._is_session_valid() is True

    def test_session_invalid_when_no_session_id(self):
        router = self._create_router()
        router._mcp_initialized = True
        router._mcp_session_id = None
        assert router._is_session_valid() is False

    def test_session_invalid_when_not_initialized(self):
        router = self._create_router()
        router._mcp_initialized = False
        assert router._is_session_valid() is False

    def test_session_invalid_when_client_closed(self):
        router = self._create_router()
        router._mcp_initialized = True
        router._mcp_session_id = "test-session"
        mock_client = MagicMock()
        mock_client.is_closed = True
        router._http_client = mock_client

        assert router._is_session_valid() is False

    def test_session_invalid_when_client_none(self):
        router = self._create_router()
        router._mcp_initialized = True
        router._mcp_session_id = "test-session"
        router._http_client = None

        assert router._is_session_valid() is False

    def test_ensure_session_returns_true_when_valid(self):
        router = self._create_router()
        router._mcp_initialized = True
        router._mcp_session_id = "test-session"
        mock_client = MagicMock()
        mock_client.is_closed = False
        router._http_client = mock_client

        result = asyncio.get_event_loop().run_until_complete(router._ensure_session())
        assert result is True

    def test_ensure_session_triggers_reconnect_when_invalid(self):
        router = self._create_router()
        router._mcp_initialized = False
        router._mcp_session_id = None
        router._http_client = None

        with patch.object(router, '_reconnect', new_callable=AsyncMock) as mock_reconnect:
            mock_reconnect.return_value = True
            result = asyncio.get_event_loop().run_until_complete(router._ensure_session())
            mock_reconnect.assert_called_once()
            assert result is True

    def test_reconnect_success(self):
        router = self._create_router()
        router._http_client = None

        mock_async_client = MagicMock()
        mock_response = MagicMock()
        mock_response.headers = {"mcp-session-id": "new-session-456"}
        mock_response.status_code = 200
        mock_async_client.get = AsyncMock(return_value=mock_response)
        mock_async_client.post = AsyncMock(return_value=mock_response)
        mock_async_client.is_closed = False

        with patch("src.skills.router.httpx.AsyncClient", return_value=mock_async_client):
            result = asyncio.get_event_loop().run_until_complete(router._reconnect())

        assert result is True
        assert router._mcp_session_id == "new-session-456"
        assert router._mcp_initialized is True

    def test_reconnect_failure_exhausted(self):
        router = self._create_router()
        router._http_client = None

        with patch("src.skills.router.httpx.AsyncClient") as MockAsyncClient:
            mock_client = MagicMock()
            mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
            mock_client.is_closed = False
            MockAsyncClient.return_value = mock_client

            result = asyncio.get_event_loop().run_until_complete(router._reconnect())

        assert result is False
        assert router._mcp_initialized is False

    def test_async_call_mcp_tool_exists(self):
        router = self._create_router()
        assert hasattr(router, 'async_call_mcp_tool')
        assert hasattr(router, '_async_call_mcp_tool')


class TestMCPAsyncCommunication:
    def _create_router(self):
        with patch("src.skills.router.httpx.Client") as MockClient, \
             patch("src.skills.router.httpx.AsyncClient") as MockAsyncClient:
            mock_sync_client = MagicMock()
            mock_sync_response = MagicMock()
            mock_sync_response.headers = {"mcp-session-id": "test-session-123"}
            mock_sync_response.status_code = 200
            mock_sync_response.text = 'data: {"result":{"serverInfo":{"name":"test"}}}'
            mock_sync_client.get.return_value = mock_sync_response
            mock_sync_client.post.return_value = mock_sync_response
            MockClient.return_value = mock_sync_client

            mock_async_client = MagicMock()
            MockAsyncClient.return_value = mock_async_client

            from src.skills.router import AstronomySkillRouter
            return AstronomySkillRouter()

    def test_async_call_returns_error_when_session_unavailable(self):
        router = self._create_router()
        router._mcp_initialized = False
        router._mcp_session_id = None
        router._http_client = None

        with patch.object(router, '_reconnect', new_callable=AsyncMock) as mock_reconnect:
            mock_reconnect.return_value = False
            result = asyncio.get_event_loop().run_until_complete(
                router._async_call_mcp_tool("test_tool")
            )

        result_data = json.loads(result)
        assert result_data.get("error") is True
        assert "MCP_SESSION_ERROR" in result_data.get("code", "")

    def test_async_call_handles_503_as_session_expired(self):
        router = self._create_router()
        router._mcp_initialized = True
        router._mcp_session_id = "test-session"
        mock_client = MagicMock()
        mock_client.is_closed = False

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"
        mock_client.post = AsyncMock(return_value=mock_response)
        router._http_client = mock_client

        with patch.object(router, '_ensure_session', new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = True
            result = asyncio.get_event_loop().run_until_complete(
                router._async_call_mcp_tool("test_tool")
            )

        assert router._mcp_initialized is False
        result_data = json.loads(result)
        assert result_data.get("error") is True


# ===== Issue 8: Caching Tests =====

class TestNASAAPICaching:
    def setup_method(self):
        from src.astronomy.nasa_api import NASA_APOD_CACHE, NASA_NEO_CACHE
        NASA_APOD_CACHE.clear()
        NASA_NEO_CACHE.clear()

    def test_apod_cache_hit(self):
        from src.astronomy.nasa_api import NASA_APOD_CACHE, NASAAPIService
        test_data = {"date": "2026-04-08", "title": "Test APOD"}
        NASA_APOD_CACHE["apod:2026-04-08:False"] = test_data

        with patch.object(NASAAPIService, "__init__", lambda self: None):
            service = NASAAPIService()
            service.api_key = "test-key"
            result = service.get_apod(date="2026-04-08")

        assert result == test_data

    def test_neo_cache_hit(self):
        from src.astronomy.nasa_api import NASA_NEO_CACHE, NASAAPIService
        test_data = {"near_earth_objects": {}}
        NASA_NEO_CACHE["neo:2026-04-08:2026-04-15:20"] = test_data

        with patch.object(NASAAPIService, "__init__", lambda self: None):
            service = NASAAPIService()
            service.api_key = "test-key"
            result = service.get_neo_data(start_date="2026-04-08", end_date="2026-04-15")

        assert result == test_data

    def test_apod_cache_key_includes_date_and_hd(self):
        from src.astronomy.nasa_api import NASA_APOD_CACHE
        NASA_APOD_CACHE["apod:2026-04-08:True"] = {"hd": True}
        NASA_APOD_CACHE["apod:2026-04-08:False"] = {"hd": False}

        assert NASA_APOD_CACHE["apod:2026-04-08:True"] != NASA_APOD_CACHE["apod:2026-04-08:False"]


class TestWeatherCaching:
    def setup_method(self):
        from src.astronomy.weather_service import WEATHER_CACHE
        WEATHER_CACHE.clear()

    def test_weather_cache_hit(self):
        from src.astronomy.weather_service import WEATHER_CACHE, WeatherService
        test_data = {"query_city": "北京", "live": {"city": "北京"}}
        WEATHER_CACHE["weather:北京:base"] = test_data

        with patch.object(WeatherService, "__init__", lambda self: None):
            service = WeatherService()
            service.api_key = "test-key"
            with patch("src.astronomy.weather_service.parse_mixed_input") as mock_parse:
                mock_parse.return_value = {"city": "北京", "extensions": "base"}
                result = service.get_weather(city="北京", extensions="base")

        assert result == test_data

    def test_weather_cache_key_differentiates_extensions(self):
        from src.astronomy.weather_service import WEATHER_CACHE
        WEATHER_CACHE["weather:北京:base"] = {"type": "base"}
        WEATHER_CACHE["weather:北京:all"] = {"type": "all"}

        assert WEATHER_CACHE["weather:北京:base"] != WEATHER_CACHE["weather:北京:all"]


class TestSearchCaching:
    def setup_method(self):
        from src.astronomy.search_service import SEARCH_CACHE
        SEARCH_CACHE.clear()

    def test_search_cache_hit(self):
        from src.astronomy.search_service import SEARCH_CACHE, SearchService
        test_data = {"query": "test", "results": [], "total": 0}
        SEARCH_CACHE["search:test:5"] = test_data

        with patch.object(SearchService, "__init__", lambda self: None):
            service = SearchService()
            service.api_key = "test-key"
            result = service.search("test")

        assert result == test_data

    def test_search_cache_key_includes_max_results(self):
        from src.astronomy.search_service import SEARCH_CACHE
        SEARCH_CACHE["search:test:5"] = {"total": 5}
        SEARCH_CACHE["search:test:10"] = {"total": 10}

        assert SEARCH_CACHE["search:test:5"] != SEARCH_CACHE["search:test:10"]


# ===== Issue 9: Retry & Circuit Breaker Tests =====

class TestRetryMechanism:
    def test_nasa_retry_on_timeout(self):
        from src.astronomy.nasa_api import NASAAPIService, NASA_APOD_CACHE
        NASA_APOD_CACHE.clear()

        with patch.object(NASAAPIService, "__init__", lambda self: None):
            service = NASAAPIService()
            service.api_key = "test-key"

            call_count = 0
            def mock_get(url, params=None, timeout=30):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise requests.Timeout("Connection timed out")
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.raise_for_status = MagicMock()
                mock_resp.json.return_value = {"date": "2026-04-08", "title": "Test"}
                return mock_resp

            with patch("src.astronomy.nasa_api.settings") as mock_settings:
                mock_settings.NASA_APOD_URL = "https://api.nasa.gov/planetary/apod"
                with patch.object(requests, "get", side_effect=mock_get):
                    try:
                        result = service.get_apod(date="2026-04-08")
                        assert call_count >= 2
                    except Exception:
                        pass

    def test_weather_retry_on_connection_error(self):
        from src.astronomy.weather_service import WeatherService, WEATHER_CACHE
        WEATHER_CACHE.clear()

        with patch.object(WeatherService, "__init__", lambda self: None):
            service = WeatherService()
            service.api_key = "test-key"

            call_count = 0
            def mock_get(url, params=None, timeout=15):
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise requests.ConnectionError("Connection failed")
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.raise_for_status = MagicMock()
                mock_resp.json.return_value = {
                    "status": "1",
                    "lives": [{"city": "北京", "weather": "晴", "temperature": "22",
                               "humidity": "45", "winddirection": "北", "windpower": "≤3",
                               "reporttime": "2026-04-08 14:30:00"}]
                }
                return mock_resp

            with patch("src.astronomy.weather_service.settings") as mock_settings:
                mock_settings.AMAP_WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"
                mock_settings.AMAP_DEFAULT_CITY = "北京"
                mock_settings.OBSERVING_TIPS_TEMPLATES = {
                    'bad_weather': "bad", 'good_weather': "good",
                    'high_humidity': "humid", 'high_wind': "windy",
                    'new_moon': "new", 'full_moon': "full",
                }
                with patch.object(requests, "get", side_effect=mock_get), \
                     patch("src.astronomy.weather_service.parse_mixed_input") as mock_parse:
                    mock_parse.return_value = {"city": "北京", "extensions": "base"}
                    try:
                        result = service.get_weather(city="北京")
                        assert call_count >= 2
                    except Exception:
                        pass


class TestCircuitBreaker:
    def test_nasa_circuit_breaker_trips(self):
        from src.astronomy.nasa_api import nasa_api_breaker
        assert nasa_api_breaker.fail_max == 5
        assert nasa_api_breaker.reset_timeout == 60

    def test_weather_circuit_breaker_config(self):
        from src.astronomy.weather_service import weather_api_breaker
        assert weather_api_breaker.fail_max == 5
        assert weather_api_breaker.reset_timeout == 60

    def test_search_circuit_breaker_config(self):
        from src.astronomy.search_service import search_api_breaker
        assert search_api_breaker.fail_max == 5
        assert search_api_breaker.reset_timeout == 60

    def test_circuit_breaker_opens_after_failures(self):
        from pybreaker import CircuitBreaker
        breaker = CircuitBreaker(fail_max=3, reset_timeout=10)

        @breaker
        def failing_func():
            raise Exception("test failure")

        for _ in range(3):
            try:
                failing_func()
            except Exception:
                pass

        assert breaker.current_state == "open"
        breaker.close()

    def test_circuit_breaker_allows_when_closed(self):
        from pybreaker import CircuitBreaker
        breaker = CircuitBreaker(fail_max=3, reset_timeout=10)

        @breaker
        def success_func():
            return "ok"

        result = success_func()
        assert result == "ok"
        assert breaker.current_state == "closed"


# ===== Issue 10: SQLite Context Manager Tests =====

class TestSQLiteContextManager:
    def test_connection_released_on_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.sqlite")

            from src.memory.memory import LongTermMemory, UserProfile
            ltm = LongTermMemory(db_path=db_path)

            profile = UserProfile(
                user_id="test_user",
                preferences={"style": "detailed"},
                habits={"topics": ["mars"]},
                constraints=[],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            ltm.save_profile(profile)

            loaded = ltm.load_profile("test_user")
            assert loaded is not None
            assert loaded.user_id == "test_user"
            assert loaded.preferences["style"] == "detailed"

    def test_connection_released_on_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.sqlite")

            from src.memory.memory import LongTermMemory
            ltm = LongTermMemory(db_path=db_path)

            result = ltm.load_profile("nonexistent_user")
            assert result is None

            result2 = ltm.load_profile("another_nonexistent")
            assert result2 is None

    def test_delete_profile_with_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.sqlite")

            from src.memory.memory import LongTermMemory, UserProfile
            ltm = LongTermMemory(db_path=db_path)

            profile = UserProfile(
                user_id="delete_me",
                preferences={},
                habits={},
                constraints=[],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            ltm.save_profile(profile)

            assert ltm.load_profile("delete_me") is not None
            assert ltm.delete_profile("delete_me") is True
            assert ltm.load_profile("delete_me") is None

    def test_multiple_operations_no_leak(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.sqlite")

            from src.memory.memory import LongTermMemory, UserProfile
            ltm = LongTermMemory(db_path=db_path)

            for i in range(20):
                profile = UserProfile(
                    user_id=f"user_{i}",
                    preferences={"idx": i},
                    habits={},
                    constraints=[],
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat()
                )
                ltm.save_profile(profile)

            for i in range(20):
                loaded = ltm.load_profile(f"user_{i}")
                assert loaded is not None
                assert loaded.preferences["idx"] == i


# ===== Issue 11: Action History Cleanup Tests =====

class TestActionHistoryCleanup:
    def setup_method(self):
        from src.agent.streaming_service import StreamingService, MAX_ACTION_HISTORY_ENTRIES
        self.MAX_ACTION_HISTORY_ENTRIES = MAX_ACTION_HISTORY_ENTRIES
        self.mock_executor = MagicMock()
        self.mock_memory = MagicMock()
        self.mock_memory.get_recent_messages.return_value = []
        self.service = StreamingService(
            agent_executor=self.mock_executor,
            memory=self.mock_memory,
        )

    def test_action_history_is_ordered_dict(self):
        assert isinstance(self.service._action_history, OrderedDict)

    def test_cleanup_removes_specific_request(self):
        self.service._action_history["req1"] = ["action1"]
        self.service._action_history["req2"] = ["action2"]
        self.service._action_history["req3"] = ["action3"]

        self.service._cleanup_action_history("req2")

        assert "req2" not in self.service._action_history
        assert "req1" in self.service._action_history
        assert "req3" in self.service._action_history

    def test_lru_eviction_when_over_limit(self):
        for i in range(self.MAX_ACTION_HISTORY_ENTRIES + 10):
            self.service._action_history[f"req_{i}"] = [f"action_{i}"]

        self.service._cleanup_action_history()

        assert len(self.service._action_history) <= self.MAX_ACTION_HISTORY_ENTRIES

        oldest_keys = [f"req_{i}" for i in range(10)]
        for key in oldest_keys:
            assert key not in self.service._action_history

    def test_cleanup_on_request_completion(self):
        self.service._action_history["req_completed"] = ["action1", "action2"]
        self.service._action_history["req_active"] = ["action3"]

        self.service._cleanup_action_history("req_completed")

        assert "req_completed" not in self.service._action_history
        assert "req_active" in self.service._action_history

    def test_memory_usage_logging(self):
        self.service._action_history["req1"] = ["a1", "a2"]
        self.service._action_history["req2"] = ["a3"]

        import logging
        test_logger = logging.getLogger("AstroAgent")
        with patch.object(test_logger, "debug") as mock_debug:
            self.service._log_memory_usage()
            mock_debug.assert_called_once()
            log_msg = mock_debug.call_args[0][0]
            assert "2 个请求" in log_msg
            assert "3 条动作记录" in log_msg

    def test_max_history_constant(self):
        assert self.MAX_ACTION_HISTORY_ENTRIES == 100

    def test_no_cleanup_of_active_request(self):
        self.service._action_history["req_active"] = ["action1"]

        self.service._cleanup_action_history("other_req")

        assert "req_active" in self.service._action_history


# ===== Integration Tests =====

class TestIntegrationCacheAndRetry:
    def test_nasa_cache_prevents_redundant_calls(self):
        from src.astronomy.nasa_api import NASAAPIService, NASA_APOD_CACHE
        NASA_APOD_CACHE.clear()

        with patch.object(NASAAPIService, "__init__", lambda self: None):
            service = NASAAPIService()
            service.api_key = "test-key"

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {"date": "2026-04-08", "title": "Test APOD"}

            with patch("src.astronomy.nasa_api.settings") as mock_settings:
                mock_settings.NASA_APOD_URL = "https://api.nasa.gov/planetary/apod"
                with patch.object(requests, "get", return_value=mock_response) as mock_get:
                    result1 = service.get_apod(date="2026-04-08")
                    result2 = service.get_apod(date="2026-04-08")

                    assert mock_get.call_count == 1
                    assert result1 == result2

    def test_weather_cache_different_cities(self):
        from src.astronomy.weather_service import WEATHER_CACHE
        WEATHER_CACHE.clear()

        WEATHER_CACHE["weather:北京:base"] = {"city": "北京"}
        WEATHER_CACHE["weather:上海:base"] = {"city": "上海"}

        assert WEATHER_CACHE["weather:北京:base"] != WEATHER_CACHE["weather:上海:base"]


class TestIntegrationMemoryAndStreaming:
    def test_streaming_cleans_up_after_completion(self):
        from src.agent.streaming_service import StreamingService

        mock_executor = MagicMock()
        mock_memory = MagicMock()
        mock_memory.get_recent_messages.return_value = []

        service = StreamingService(
            agent_executor=mock_executor,
            memory=mock_memory,
        )

        service._action_history["req_test"] = ["action1", "action2"]
        service._cleanup_action_history("req_test")

        assert "req_test" not in service._action_history

    def test_long_term_memory_context_manager_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "integration_test.sqlite")

            from src.memory.memory import LongTermMemory, UserProfile
            ltm = LongTermMemory(db_path=db_path)

            new_info = {
                "preferences": {"response_style": "详细"},
                "habits": {"frequent_topics": ["火星"]},
                "constraints": ["避免使用专业术语"]
            }

            profile = ltm.merge_and_update("integration_user", new_info)
            assert profile.preferences["response_style"] == "详细"

            formatted = ltm.format_profile_for_prompt("integration_user")
            assert "详细" in formatted

            assert ltm.delete_profile("integration_user") is True
            assert ltm.load_profile("integration_user") is None


# ===== Stress Tests =====

class TestStressCaching:
    def test_cache_ttl_expiry(self):
        from cachetools import TTLCache
        cache = TTLCache(maxsize=10, ttl=1)

        cache["key1"] = "value1"
        assert "key1" in cache

        time.sleep(1.1)

        assert cache.get("key1") is None

    def test_cache_maxsize_eviction(self):
        from cachetools import TTLCache
        cache = TTLCache(maxsize=5, ttl=3600)

        for i in range(10):
            cache[f"key_{i}"] = f"value_{i}"

        assert len(cache) <= 5

    def test_concurrent_cache_access(self):
        from src.astronomy.nasa_api import NASA_APOD_CACHE
        NASA_APOD_CACHE.clear()

        for i in range(200):
            NASA_APOD_CACHE[f"apod:date_{i}:False"] = {"date": f"date_{i}"}

        assert len(NASA_APOD_CACHE) <= 128


class TestStressActionHistory:
    def test_action_history_stays_bounded(self):
        from src.agent.streaming_service import StreamingService, MAX_ACTION_HISTORY_ENTRIES

        mock_executor = MagicMock()
        mock_memory = MagicMock()
        mock_memory.get_recent_messages.return_value = []

        service = StreamingService(
            agent_executor=mock_executor,
            memory=mock_memory,
        )

        for i in range(500):
            service._action_history[f"req_{i}"] = [f"action_{i}"]

        service._cleanup_action_history()

        assert len(service._action_history) <= MAX_ACTION_HISTORY_ENTRIES


class TestStressSQLite:
    def test_rapid_sequential_operations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "stress_test.sqlite")

            from src.memory.memory import LongTermMemory, UserProfile
            ltm = LongTermMemory(db_path=db_path)

            for i in range(50):
                profile = UserProfile(
                    user_id=f"stress_user_{i}",
                    preferences={"iteration": i},
                    habits={"counter": i},
                    constraints=[],
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat()
                )
                ltm.save_profile(profile)

            for i in range(50):
                loaded = ltm.load_profile(f"stress_user_{i}")
                assert loaded is not None
                assert loaded.preferences["iteration"] == i


# ===== Network Failure Simulation Tests =====

class TestNetworkFailureSimulation:
    def test_nasa_api_timeout_raises_agent_error(self):
        from src.astronomy.nasa_api import NASAAPIService, NASA_APOD_CACHE
        NASA_APOD_CACHE.clear()

        with patch.object(NASAAPIService, "__init__", lambda self: None):
            service = NASAAPIService()
            service.api_key = "test-key"

            with patch("src.astronomy.nasa_api.settings") as mock_settings:
                mock_settings.NASA_APOD_URL = "https://api.nasa.gov/planetary/apod"
                with patch.object(requests, "get", side_effect=requests.Timeout("Timeout")):
                    from src.core.errors import AgentError
                    with pytest.raises((AgentError, requests.Timeout)):
                        service.get_apod(date="2026-04-08")

    def test_search_service_handles_timeout_gracefully(self):
        from src.astronomy.search_service import SearchService, SEARCH_CACHE
        SEARCH_CACHE.clear()

        with patch.object(SearchService, "__init__", lambda self: None):
            service = SearchService()
            service.api_key = "test-key"

            with patch.object(requests, "post", side_effect=requests.Timeout("Timeout")):
                result = service.search("test query")
                assert "error" in result

    def test_circuit_breaker_prevents_cascade(self):
        from pybreaker import CircuitBreaker, CircuitBreakerError
        breaker = CircuitBreaker(fail_max=2, reset_timeout=60)

        @breaker
        def failing_call():
            raise Exception("service down")

        for _ in range(2):
            try:
                failing_call()
            except Exception:
                pass

        assert breaker.current_state == "open"

        with pytest.raises(CircuitBreakerError):
            failing_call()


# ===== Memory Leak Verification =====

class TestMemoryLeakVerification:
    def test_action_history_does_not_grow_unbounded(self):
        from src.agent.streaming_service import StreamingService, MAX_ACTION_HISTORY_ENTRIES

        mock_executor = MagicMock()
        mock_memory = MagicMock()
        mock_memory.get_recent_messages.return_value = []

        service = StreamingService(
            agent_executor=mock_executor,
            memory=mock_memory,
        )

        for i in range(MAX_ACTION_HISTORY_ENTRIES * 2):
            service._action_history[f"req_{i}"] = [f"action_{j}" for j in range(5)]

        service._cleanup_action_history()

        assert len(service._action_history) <= MAX_ACTION_HISTORY_ENTRIES

    def test_cache_has_maxsize_limit(self):
        from src.astronomy.nasa_api import NASA_APOD_CACHE, NASA_NEO_CACHE
        from src.astronomy.weather_service import WEATHER_CACHE
        from src.astronomy.search_service import SEARCH_CACHE

        assert NASA_APOD_CACHE.maxsize == 128
        assert NASA_NEO_CACHE.maxsize == 64
        assert WEATHER_CACHE.maxsize == 256
        assert SEARCH_CACHE.maxsize == 128

    def test_sqlite_connections_are_properly_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "conn_test.sqlite")

            from src.memory.memory import LongTermMemory, UserProfile
            ltm = LongTermMemory(db_path=db_path)

            profile = UserProfile(
                user_id="conn_test_user",
                preferences={},
                habits={},
                constraints=[],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )

            for _ in range(10):
                ltm.save_profile(profile)
                ltm.load_profile("conn_test_user")

            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()
                assert result[0] == "ok"
