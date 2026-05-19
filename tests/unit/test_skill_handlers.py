from __future__ import annotations

import json

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()

from src.skills.skill_handlers import (  # noqa: E402
    CelestialEventsForecastHandler,
    CelestialPositionCalculatorHandler,
    DeepSkyObservingGuideHandler,
    ObservationPlannerHandler,
)


class _FakeMCP:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.parallel_calls: list[list[dict]] = []

    def call_tool(self, tool_name: str, **kwargs):
        self.calls.append((tool_name, kwargs))
        return json.dumps(
            {
                "summary": tool_name,
                "altitude": 35.0,
                "azimuth": 120.0,
                "live": {"city": kwargs.get("city"), "weather": "晴", "temperature": "20"},
            },
            ensure_ascii=False,
        )

    def call_tools_parallel(self, calls: list[dict]):
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


def test_celestial_position_altaz_branch_uses_get_altaz():
    mcp = _FakeMCP()

    result = CelestialPositionCalculatorHandler()(
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

    result = CelestialPositionCalculatorHandler()(
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

    result = CelestialPositionCalculatorHandler()(
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

    result = CelestialPositionCalculatorHandler()(
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

    result = CelestialPositionCalculatorHandler()(
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

    result = CelestialEventsForecastHandler()(
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

    result = CelestialEventsForecastHandler()(
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

    result = DeepSkyObservingGuideHandler()(mcp, target="M31")

    assert result.success is True
    tool_names = [call["tool_name"] for call in mcp.parallel_calls[0]]
    assert tool_names == ["get_astrophysical_object_info", "get_galaxy_data"]
    assert [source["tool"] for source in result.sources] == [
        "get_astrophysical_object_info",
        "get_galaxy_data",
    ]


def test_deep_sky_does_not_call_weather_even_with_observer_location():
    mcp = _FakeMCP()

    result = DeepSkyObservingGuideHandler()(mcp, target="M31", observer_location="上海")

    assert result.success is True
    tool_names = [call["tool_name"] for call in mcp.parallel_calls[0]]
    assert "get_weather" not in tool_names


def test_observation_planner_defaults_to_beijing_and_calls_expected_mcp_tools():
    mcp = _FakeMCP()

    result = ObservationPlannerHandler()(mcp, date="今晚", location=None)

    assert result.success is True
    tool_names = [call["tool_name"] for call in mcp.parallel_calls[0]]
    assert tool_names == ["get_weather", "get_weekly_events", "get_tonight_best"]
    weather_call = mcp.parallel_calls[0][0]
    assert weather_call["kwargs"]["city"] == "北京"
