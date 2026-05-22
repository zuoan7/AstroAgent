from __future__ import annotations

from src.agent.executor import _extract_mcp_tools_from_sources
from src.core.mcp_protocol import serialize_envelope, success_envelope
from src.skills import registry
from src.skills.router import AstronomySkillRouter


def test_position_operation_contracts_map_to_atomic_mcp_tools():
    expected = {
        "altaz": "get_altaz",
        "rise_set": "get_rise_set_times",
        "planet_position": "get_planet_position",
        "current_sky": "get_current_sky_objects",
        "coordinate_transformation": "coordinate_transformation",
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


def test_mcp_tool_extraction_uses_registered_tool_names_not_prefixes():
    tools = _extract_mcp_tools_from_sources(
        [
            {"kind": "tool_output", "tool": "get_weather"},
            {"kind": "tool_output", "tool": "web_search"},
            {"kind": "tool_output", "tool": "coordinate_transformation"},
            {"kind": "tool_output", "tool": "weather-lookup"},
            {"kind": "tool_output", "tool": "RAGRetrieve"},
        ]
    )

    assert tools == ["get_weather", "web_search", "coordinate_transformation"]


def test_simple_skill_result_declares_logical_skill_and_mcp_contract(monkeypatch):
    router = AstronomySkillRouter()

    def fake_call_mcp_tool(tool_name: str, **kwargs):
        return serialize_envelope(
            success_envelope(
                tool_name,
                {"results": [], "query": kwargs.get("query", "")},
            )
        )

    monkeypatch.setattr(router, "call_mcp_tool", fake_call_mcp_tool)

    result = router.call("web_search", query="最近 JWST 新结果", max_results=2)

    assert result.success is True
    assert result.skill_name == "web_search"
    assert result.logical_skill == "web_search"
    assert result.expected_mcp_tools == ["web_search"]
    assert result.allowed_child_tools == ["web_search"]
    assert result.sources[0]["tool"] == "web_search"
