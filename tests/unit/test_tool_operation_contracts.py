from __future__ import annotations

from src.skills import registry


def test_position_operation_contracts_map_to_atomic_mcp_tools():
    expected = {
        "altaz": "get_altaz",
        "rise_set": "get_rise_set_times",
        "planet_position": "get_planet_position",
        "current_sky": "get_current_sky_objects",
    }

    for operation, mcp_tool in expected.items():
        spec = registry.get_operation_spec(
            "celestial-position-calculator",
            operation,
        )

        assert spec.atomic_tool_name == mcp_tool
        assert spec.allowed_child_tools == [mcp_tool]


def test_event_operation_contracts_map_to_weekly_and_monthly_tools():
    weekly = registry.get_operation_spec("celestial-events-forecast", "weekly")
    monthly = registry.get_operation_spec("celestial-events-forecast", "monthly")

    assert weekly.atomic_tool_name == "get_weekly_events"
    assert monthly.atomic_tool_name == "get_monthly_events"
    assert "get_monthly_events" in weekly.forbidden_child_tools
    assert "get_weekly_events" in monthly.forbidden_child_tools
