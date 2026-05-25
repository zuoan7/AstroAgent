from __future__ import annotations

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()

from src.core.config import settings  # noqa: E402
from src.skills.context import SkillContext  # noqa: E402
from src.skills.inputs import (  # noqa: E402
    CelestialEventsForecastInput,
    CelestialPositionCalculatorInput,
    NeoTrackerInput,
    ObservationPlannerInput,
)
from src.skills.services.celestial_events import forecast_celestial_events  # noqa: E402
from src.skills.services.celestial_position import (  # noqa: E402
    calculate_celestial_position,
)
from src.skills.services.near_earth_objects import (  # noqa: E402
    track_near_earth_objects,
)
from src.skills.services.observation_plan import build_observation_plan  # noqa: E402
from src.tools.results import ToolError, ToolResult  # noqa: E402


class _FakeToolKit:
    def __init__(
        self,
        *,
        invoke_result: ToolResult | None = None,
        parallel_results: list[ToolResult] | None = None,
    ) -> None:
        self.invoke_result = invoke_result
        self.parallel_results = parallel_results or []
        self.calls: list[tuple[str, dict]] = []
        self.parallel_calls: list[list[dict]] = []
        self.policies: list[dict] = []

    def with_policy(self, **kwargs):
        self.policies.append(kwargs)
        return self

    def invoke(self, tool_name: str, **kwargs):
        self.calls.append((tool_name, kwargs))
        if self.invoke_result is not None:
            return self.invoke_result
        return ToolResult(ok=True, tool_name=tool_name, data={"summary": tool_name})

    def invoke_parallel(self, calls: list[dict]):
        self.parallel_calls.append(calls)
        if self.parallel_results:
            return self.parallel_results
        return [
            ToolResult(
                ok=True,
                tool_name=call["tool_name"],
                data={"summary": call.get("_key", call["tool_name"])},
            )
            for call in calls
        ]


def _ctx(tool_kit: _FakeToolKit, skill_name: str) -> SkillContext:
    return SkillContext(tool_kit=tool_kit, skill_name=skill_name)


def _neo_payload() -> dict:
    return {
        "near_earth_objects": {
            "2026-05-25": [
                {
                    "name": "Close Bright",
                    "estimated_diameter": {"meters": {"estimated_diameter_max": 150.0}},
                    "close_approach_data": [{"miss_distance": {"lunar": "3.0"}}],
                    "absolute_magnitude_h": 20.0,
                    "is_potentially_hazardous_asteroid": True,
                },
                {
                    "name": "Too Small",
                    "estimated_diameter": {"meters": {"estimated_diameter_max": 20.0}},
                    "close_approach_data": [{"miss_distance": {"lunar": "2.0"}}],
                    "absolute_magnitude_h": 20.0,
                    "is_potentially_hazardous_asteroid": False,
                },
                {
                    "name": "Too Far",
                    "estimated_diameter": {"meters": {"estimated_diameter_max": 200.0}},
                    "close_approach_data": [{"miss_distance": {"lunar": "9.0"}}],
                    "absolute_magnitude_h": 20.0,
                    "is_potentially_hazardous_asteroid": False,
                },
                {
                    "name": "Too Faint",
                    "estimated_diameter": {"meters": {"estimated_diameter_max": 180.0}},
                    "close_approach_data": [{"miss_distance": {"lunar": "2.0"}}],
                    "absolute_magnitude_h": 26.0,
                    "is_potentially_hazardous_asteroid": False,
                },
            ]
        }
    }


def test_neo_service_filters_size_distance_and_observable_targets():
    tool_kit = _FakeToolKit(
        invoke_result=ToolResult(
            ok=True,
            tool_name="get_neo_data",
            data=_neo_payload(),
        )
    )

    result = track_near_earth_objects(
        _ctx(tool_kit, "neo-tracker"),
        NeoTrackerInput(
            time_range="未来30天",
            min_size=100,
            max_distance=5,
            observable_only=True,
        ),
    )

    assert result.success is True
    assert tool_kit.calls[0][0] == "get_neo_data"
    assert tool_kit.calls[0][1]["limit"] == 50
    assert result.data["total_raw"] == 4
    assert [item["name"] for item in result.data["filtered_neos"]] == ["Close Bright"]
    assert "NASA NEO API 最多只支持查询7天的数据" in result.summary


def test_neo_service_maps_tool_errors_to_skill_result():
    tool_kit = _FakeToolKit(
        invoke_result=ToolResult(
            ok=False,
            tool_name="get_neo_data",
            error=ToolError(code="NASA_ERROR", message="neo failed"),
        )
    )

    result = track_near_earth_objects(
        _ctx(tool_kit, "neo-tracker"),
        NeoTrackerInput(time_range="未来7天"),
    )

    assert result.success is False
    assert result.error_code == "NASA_ERROR"
    assert result.error_message == "neo failed"


def test_neo_service_keeps_non_dict_payload_as_raw_data():
    tool_kit = _FakeToolKit(
        invoke_result=ToolResult(
            ok=True,
            tool_name="get_neo_data",
            data="raw neo text",
        )
    )

    result = track_near_earth_objects(
        _ctx(tool_kit, "neo-tracker"),
        NeoTrackerInput(),
    )

    assert result.success is True
    assert result.data == {"raw": "raw neo text"}
    assert result.summary == "raw neo text"


def test_celestial_position_service_supports_current_sky_without_target():
    tool_kit = _FakeToolKit()

    result = calculate_celestial_position(
        _ctx(tool_kit, "celestial-position-calculator"),
        CelestialPositionCalculatorInput(
            operation="current_sky",
            datetime="2026-05-25",
            location="31.23,121.47",
        ),
    )

    assert result.success is True
    assert result.operation == "current_sky"
    assert result.expected_mcp_tools == ["get_current_sky_objects"]
    assert tool_kit.calls == [
        (
            "get_current_sky_objects",
            {"latitude": 31.23, "longitude": 121.47, "date": "2026-05-25"},
        )
    ]


def test_celestial_position_service_validates_coordinate_transform_inputs():
    tool_kit = _FakeToolKit()

    result = calculate_celestial_position(
        _ctx(tool_kit, "celestial-position-calculator"),
        CelestialPositionCalculatorInput(operation="coordinate_transformation", ra=5.5),
    )

    assert result.success is False
    assert result.error_code == "VALIDATION_ERROR"
    assert tool_kit.calls == []


def test_celestial_events_service_splits_cross_month_forecast_calls():
    tool_kit = _FakeToolKit()

    result = forecast_celestial_events(
        _ctx(tool_kit, "celestial-events-forecast"),
        CelestialEventsForecastInput(
            start_date="2026-05-01",
            end_date="2026-07-10",
        ),
    )

    assert result.success is True
    assert result.operation == "monthly"
    assert [call["kwargs"] for call in tool_kit.parallel_calls[0]] == [
        {"year": 2026, "month": 5},
        {"year": 2026, "month": 6},
        {"year": 2026, "month": 7},
    ]


def test_celestial_events_service_uses_weekly_for_short_ranges():
    tool_kit = _FakeToolKit()

    result = forecast_celestial_events(
        _ctx(tool_kit, "celestial-events-forecast"),
        CelestialEventsForecastInput(
            start_date="2026-05-01",
            end_date="2026-05-05",
        ),
    )

    assert result.success is True
    assert result.operation == "weekly"
    assert tool_kit.calls == [("get_weekly_events", {"start_date": "2026-05-01"})]


def test_celestial_events_service_clamps_unsupported_start_year():
    tool_kit = _FakeToolKit()
    supported_min = settings.SUPPORTED_YEAR_RANGE[0]

    result = forecast_celestial_events(
        _ctx(tool_kit, "celestial-events-forecast"),
        CelestialEventsForecastInput(
            start_date="1800-05-01",
            operation="monthly",
        ),
    )

    assert result.success is True
    assert result.data["start_date"] == f"{supported_min}-05-01"
    assert tool_kit.calls == [
        ("get_monthly_events", {"year": supported_min, "month": 5})
    ]


def test_observation_plan_service_treats_date_text_as_location_when_needed():
    tool_kit = _FakeToolKit()

    result = build_observation_plan(
        _ctx(tool_kit, "observation-planner"),
        ObservationPlannerInput(date="上海", location=None),
    )

    assert result.success is True
    assert tool_kit.parallel_calls[0][0]["tool_name"] == "get_weather"
    assert tool_kit.parallel_calls[0][0]["kwargs"]["city"] == "上海"
    assert result.data["location"] == "上海"


def test_observation_plan_service_uses_dict_location_city_and_skips_tonight_best():
    tool_kit = _FakeToolKit()

    result = build_observation_plan(
        _ctx(tool_kit, "observation-planner"),
        ObservationPlannerInput(
            date="2099-01-01",
            location={"city": "广州", "latitude": 23.13, "longitude": 113.26},
        ),
    )

    assert result.success is True
    assert [call["tool_name"] for call in tool_kit.parallel_calls[0]] == [
        "get_weather",
        "get_weekly_events",
    ]
    assert tool_kit.parallel_calls[0][0]["kwargs"]["city"] == "广州"
