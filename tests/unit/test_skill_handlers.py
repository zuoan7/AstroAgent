from __future__ import annotations

import json

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()

from src.skills.context import SkillContext  # noqa: E402
from src.skills.handlers.celestial_events_forecast import (  # noqa: E402
    celestial_events_forecast_handler,
)
from src.skills.handlers.celestial_position_calculator import (  # noqa: E402
    celestial_position_calculator_handler,
)
from src.skills.handlers.deep_sky_observing_guide import (  # noqa: E402
    deep_sky_observing_guide_handler,
)
from src.skills.handlers.observation_planner import (  # noqa: E402
    observation_planner_handler,
)
from src.skills.inputs import (  # noqa: E402
    CelestialEventsForecastInput,
    CelestialPositionCalculatorInput,
    DeepSkyObservingGuideInput,
    ObservationPlannerInput,
)
from src.tools.results import ToolResult  # noqa: E402


class _FakeMCP:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.parallel_calls: list[list[dict]] = []

    def _raw_tool(self, tool_name: str, **kwargs):
        self.calls.append((tool_name, kwargs))
        return json.dumps(
            {
                "summary": tool_name,
                "altitude": 35.0,
                "azimuth": 120.0,
                "live": {
                    "city": kwargs.get("city"),
                    "weather": "晴",
                    "temperature": "20",
                },
            },
            ensure_ascii=False,
        )

    def _raw_tools_parallel(self, calls: list[dict]):
        self.parallel_calls.append(calls)
        results = []
        for call in calls:
            results.append(
                json.dumps(
                    {
                        "summary": call["tool_name"],
                        "live": {
                            "city": call.get("kwargs", {}).get("city"),
                            "weather": "晴",
                            "temperature": "20",
                        },
                    },
                    ensure_ascii=False,
                )
            )
        return results

    def with_policy(self, **_: object) -> "_FakeMCP":
        return self

    def invoke(self, tool_name: str, **kwargs):
        return ToolResult.from_raw(tool_name, self._raw_tool(tool_name, **kwargs))

    def invoke_parallel(self, calls: list[dict]):
        return [
            ToolResult.from_raw(call["tool_name"], raw)
            for call, raw in zip(calls, self._raw_tools_parallel(calls))
        ]


def _call_handler(handler, input_model, skill_name: str, tool_kit: _FakeMCP, **params):
    payload = input_model.model_validate(params)
    ctx = SkillContext(tool_kit=tool_kit, skill_name=skill_name)
    return handler(ctx, payload)


def _call_position(tool_kit: _FakeMCP, **params):
    return _call_handler(
        celestial_position_calculator_handler,
        CelestialPositionCalculatorInput,
        "celestial-position-calculator",
        tool_kit,
        **params,
    )


def _call_events(tool_kit: _FakeMCP, **params):
    return _call_handler(
        celestial_events_forecast_handler,
        CelestialEventsForecastInput,
        "celestial-events-forecast",
        tool_kit,
        **params,
    )


def test_celestial_position_altaz_branch_uses_get_altaz():
    mcp = _FakeMCP()

    result = _call_position(
        mcp,
        target="木星",
        datetime="今晚",
        location="北京",
        output_format="altaz",
    )

    assert result.success is True
    assert mcp.calls[0][0] == "get_altaz"
    assert mcp.calls[0][1]["planet_name"] == "jupiter"
    assert mcp.calls[0][1]["latitude"] == 39.9
    assert result.sources[0]["tool"] == "get_altaz"
    assert result.logical_skill == "celestial-position-calculator"
    assert result.operation == "altaz"
    assert result.expected_mcp_tools == ["get_altaz"]


def test_celestial_position_operation_overrides_output_format():
    mcp = _FakeMCP()

    result = _call_position(
        mcp,
        target="火星",
        datetime="明晚",
        location="广州",
        output_format="radec",
        operation="altaz",
    )

    assert result.success is True
    assert mcp.calls[0][0] == "get_altaz"
    assert result.operation == "altaz"


def test_celestial_position_radec_branch_uses_get_planet_position():
    mcp = _FakeMCP()

    result = _call_position(
        mcp,
        target="木星",
        datetime="今晚",
        location="北京",
        output_format="radec",
    )

    assert result.success is True
    assert mcp.calls[0][0] == "get_planet_position"


def test_celestial_position_rise_set_branch_uses_get_rise_set_times():
    mcp = _FakeMCP()

    result = _call_position(
        mcp,
        target="木星",
        datetime="今晚",
        location="北京",
        output_format="rise_set",
    )

    assert result.success is True
    assert mcp.calls[0][0] == "get_rise_set_times"
    assert mcp.calls[0][1]["body_name"] == "jupiter"
    assert mcp.calls[0][1]["latitude"] == 39.9
    assert result.operation == "rise_set"


def test_celestial_position_coordinate_operation_uses_coordinate_transformation():
    mcp = _FakeMCP()

    result = _call_position(
        mcp,
        operation="coordinate_transformation",
        ra=5.5,
        dec=-5.25,
    )

    assert result.success is True
    assert mcp.calls[0][0] == "coordinate_transformation"
    assert mcp.calls[0][1]["ra"] == 5.5
    assert mcp.calls[0][1]["dec"] == -5.25
    assert result.operation == "coordinate_transformation"
    assert result.expected_mcp_tools == ["coordinate_transformation"]


def test_celestial_events_monthly_operation_uses_get_monthly_events():
    mcp = _FakeMCP()

    result = _call_events(
        mcp,
        start_date="2026-05-01",
        end_date="2026-05-31",
        operation="monthly",
    )

    assert result.success is True
    tool_names = [call["tool_name"] for call in mcp.parallel_calls[0]]
    assert tool_names == ["get_monthly_events"]
    assert result.logical_skill == "celestial-events-forecast"
    assert result.operation == "monthly"
    assert result.expected_mcp_tools == ["get_monthly_events"]


def test_celestial_events_weekly_operation_uses_get_weekly_events():
    mcp = _FakeMCP()

    result = _call_events(
        mcp,
        start_date="2026-05-19",
        operation="weekly",
    )

    assert result.success is True
    assert mcp.calls[0][0] == "get_weekly_events"
    assert result.operation == "weekly"
    assert result.expected_mcp_tools == ["get_weekly_events"]


def test_deep_sky_m31_branch_fetches_object_info_and_galaxy_data():
    mcp = _FakeMCP()

    result = _call_handler(
        deep_sky_observing_guide_handler,
        DeepSkyObservingGuideInput,
        "deep-sky-observing-guide",
        mcp,
        target="M31",
    )

    assert result.success is True
    tool_names = [call["tool_name"] for call in mcp.parallel_calls[0]]
    assert tool_names == ["get_astrophysical_object_info", "get_galaxy_data"]
    assert [source["tool"] for source in result.sources] == [
        "get_astrophysical_object_info",
        "get_galaxy_data",
    ]


def test_deep_sky_does_not_call_weather_even_with_observer_location():
    mcp = _FakeMCP()

    result = _call_handler(
        deep_sky_observing_guide_handler,
        DeepSkyObservingGuideInput,
        "deep-sky-observing-guide",
        mcp,
        target="M31",
        observer_location="上海",
    )

    assert result.success is True
    tool_names = [call["tool_name"] for call in mcp.parallel_calls[0]]
    assert "get_weather" not in tool_names


def test_observation_planner_defaults_to_beijing_and_calls_expected_mcp_tools():
    mcp = _FakeMCP()

    result = _call_handler(
        observation_planner_handler,
        ObservationPlannerInput,
        "observation-planner",
        mcp,
        date="今晚",
        location=None,
    )

    assert result.success is True
    tool_names = [call["tool_name"] for call in mcp.parallel_calls[0]]
    assert tool_names == ["get_weather", "get_weekly_events", "get_tonight_best"]
    weather_call = mcp.parallel_calls[0][0]
    assert weather_call["kwargs"]["city"] == "北京"
