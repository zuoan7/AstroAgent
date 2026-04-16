import importlib
import json
import sys
import types

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()

from src.core.errors import ErrorCode, safe_tool_call
from src.core.mcp_protocol import (
    TOOL_INPUT_MODELS,
    TOOL_OUTPUT_MODELS,
    extract_tool_data,
    is_tool_error,
    parse_tool_response,
)


EXPECTED_MCP_TOOLS = {
    "get_planet_position",
    "get_altaz",
    "coordinate_transformation",
    "get_rise_set_times",
    "get_current_sky_objects",
    "get_astrophysical_object_info",
    "get_galaxy_data",
    "get_nasa_apod",
    "get_neo_data",
    "get_weather",
    "web_search",
    "get_tonight_best",
    "get_weekly_events",
    "get_monthly_events",
}


class _FakePlanetary:
    def get_planet_position(self, *args, **kwargs):
        return {"ra_hours": 10.5, "ra_degrees": 157.5, "dec": 20.3, "distance_au": 1.5}

    def get_altaz(self, *args, **kwargs):
        return {"planet": "mars", "altitude": 35.2, "azimuth": 180.1, "distance_au": 1.2}

    def coordinate_transformation(self, *args, **kwargs):
        return {"ra_hours": 10.5, "ra_degrees": 157.5, "dec": 20.3}

    def get_rise_set_times(self, *args, **kwargs):
        return {"rise_time": "2026-04-16T06:00:00+08:00", "set_time": "2026-04-16T18:00:00+08:00"}

    def get_current_sky_objects(self, *args, **kwargs):
        return {"mars": {"altitude": 35.2}, "sun": {"rise_time": "06:00"}}


class _FakeCelestialDB:
    def get_object_info(self, *args, **kwargs):
        return {"name": "M31", "type": "galaxy"}

    def get_galaxy_data(self, *args, **kwargs):
        return {"name": "M31", "distance_ly": 2537000}


class _FakeNasa:
    def get_apod(self, *args, **kwargs):
        return {"title": "Test APOD", "date": "2026-04-16"}

    def get_neo_data(self, *args, **kwargs):
        return {"near_earth_objects": {"2026-04-16": []}}


class _FakeWeather:
    def get_weather(self, *args, **kwargs):
        return {"live": {"city": "北京", "weather": "晴"}}


class _FakeSearch:
    def search(self, *args, **kwargs):
        return {"answer": "test", "results": [{"title": "A", "url": "https://example.com"}]}


class _FakeEventsPredictor:
    def get_tonight_best(self):
        return "今晚适合观测火星。"

    def get_weekly_events(self, *args, **kwargs):
        return "本周有流星雨。"

    def get_monthly_events(self, *args, **kwargs):
        return "本月有一次满月。"


def _load_mcp_server():
    class _FakeFastMCP:
        def __init__(self, *args, **kwargs):
            pass

        def tool(self):
            def decorator(fn):
                return fn
            return decorator

        def resource(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

        def prompt(self):
            def decorator(fn):
                return fn
            return decorator

    sys.modules["fastmcp"] = types.SimpleNamespace(FastMCP=_FakeFastMCP)
    sys.modules.pop("src.services.mcp_server", None)
    module = importlib.import_module("src.services.mcp_server")
    module.planetary = _FakePlanetary()
    module.celestial_db = _FakeCelestialDB()
    module.nasa_api = _FakeNasa()
    module.weather = _FakeWeather()
    module.search = _FakeSearch()
    module.events_predictor = _FakeEventsPredictor()
    return module


def test_protocol_registry_covers_all_tools():
    assert EXPECTED_MCP_TOOLS <= set(TOOL_INPUT_MODELS.keys())
    assert EXPECTED_MCP_TOOLS <= set(TOOL_OUTPUT_MODELS.keys())


def test_safe_tool_call_wraps_success_and_failure_into_envelopes():
    @safe_tool_call
    def ok_tool():
        return {"value": 1}

    @safe_tool_call(error_code=ErrorCode.VALIDATION_ERROR)
    def bad_tool():
        raise ValueError("bad input")

    ok_payload = json.loads(ok_tool())
    bad_payload = json.loads(bad_tool())

    assert ok_payload["ok"] is True
    assert ok_payload["data"]["value"] == 1
    assert ok_payload["meta"]["tool_name"] == "ok_tool"

    assert bad_payload["ok"] is False
    assert bad_payload["error"]["code"] == "VALIDATION_ERROR"
    assert bad_payload["meta"]["tool_name"] == "bad_tool"


def test_extract_tool_data_and_error_detection():
    payload = json.dumps(
        {
            "ok": True,
            "data": {"live": {"city": "北京"}},
            "meta": {"tool_name": "get_weather", "schema_version": "1.0"},
        },
        ensure_ascii=False,
    )
    assert is_tool_error(payload) is False
    assert extract_tool_data(payload)["live"]["city"] == "北京"


def test_mcp_server_tools_return_success_envelopes():
    mcp_server = _load_mcp_server()

    calls = {
        "get_planet_position": lambda: mcp_server.get_planet_position("mars"),
        "get_altaz": lambda: mcp_server.get_altaz("mars", latitude=39.9, longitude=116.4),
        "coordinate_transformation": lambda: mcp_server.coordinate_transformation(10.5, 20.3),
        "get_rise_set_times": lambda: mcp_server.get_rise_set_times("sun", 39.9, 116.4, "2026-04-16"),
        "get_current_sky_objects": lambda: mcp_server.get_current_sky_objects(39.9, 116.4, "2026-04-16"),
        "get_astrophysical_object_info": lambda: mcp_server.get_astrophysical_object_info("M31"),
        "get_galaxy_data": lambda: mcp_server.get_galaxy_data("M31"),
        "get_nasa_apod": lambda: mcp_server.get_nasa_apod("2026-04-16"),
        "get_neo_data": lambda: mcp_server.get_neo_data("2026-04-16", "2026-04-17", 5),
        "get_weather": lambda: mcp_server.get_weather("北京", "all"),
        "web_search": lambda: mcp_server.web_search("火星", 3),
        "get_tonight_best": mcp_server.get_tonight_best,
        "get_weekly_events": lambda: mcp_server.get_weekly_events("2026-04-16"),
        "get_monthly_events": lambda: mcp_server.get_monthly_events(2026, 4),
    }

    for tool_name, call in calls.items():
        envelope = parse_tool_response(call())
        assert envelope is not None
        assert envelope.ok is True
        assert envelope.meta.tool_name == tool_name


def test_mcp_server_tools_return_error_envelopes():
    mcp_server = _load_mcp_server()

    class _BrokenWeather:
        def get_weather(self, *args, **kwargs):
            return {
                "error": True,
                "code": "WEATHER_API_ERROR",
                "message": "weather failed",
                "details": {"city": "北京"},
            }

    mcp_server.weather = _BrokenWeather()

    error_payload = parse_tool_response(mcp_server.get_weather("北京", "all"))
    validation_payload = parse_tool_response(mcp_server.get_altaz("mars"))

    assert error_payload is not None
    assert error_payload.ok is False
    assert error_payload.error.code == "WEATHER_API_ERROR"
    assert error_payload.meta.tool_name == "get_weather"

    assert validation_payload is not None
    assert validation_payload.ok is False
    assert validation_payload.error.code == "VALIDATION_ERROR"
    assert validation_payload.meta.tool_name == "get_altaz"
