from __future__ import annotations

from pydantic import BaseModel

from src.agent.executor import _extract_mcp_tools_from_sources
from src.agent.tool_result_adapter import tool_result_to_skill_result
from src.skills import registry
from src.skills.definition import SkillDefinition
from src.skills.kit import SkillKit
from src.skills.registry import SkillRegistry
from src.tools.protocol import serialize_envelope, success_envelope
from src.tools.kit import ToolKit


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


class _Backend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def invoke(self, tool_name: str, **kwargs):
        self.calls.append((tool_name, kwargs))
        return serialize_envelope(
            success_envelope(
                tool_name,
                {"results": [], "query": kwargs.get("query", "")},
            )
        )


def test_atomic_tool_wrapper_declares_logical_skill_and_mcp_contract():
    tool_kit = ToolKit(_Backend())

    result = tool_result_to_skill_result(
        tool_kit.invoke("web_search", query="最近 JWST 新结果", max_results=2)
    )

    assert result.success is True
    assert result.skill_name == "web_search"
    assert result.logical_skill == "web_search"
    assert result.expected_mcp_tools == ["web_search"]
    assert result.allowed_child_tools == ["web_search"]
    assert result.sources[0]["tool"] == "web_search"


def test_weather_lookup_uses_explicit_skill_handler():
    backend = _Backend()
    skill_kit = SkillKit(tool_kit=ToolKit(backend))

    result = skill_kit.invoke("weather-lookup", {"location": "北京"})

    assert result.success is True
    assert backend.calls == [("get_weather", {"city": "北京", "extensions": "all"})]
    assert result.skill_name == "weather-lookup"
    assert result.logical_skill == "weather-lookup"
    assert result.expected_mcp_tools == ["get_weather"]
    assert result.allowed_child_tools == ["get_weather"]
    assert result.sources[0]["tool"] == "get_weather"


class _EmptySkillInput(BaseModel):
    pass


def test_skill_kit_rejects_handler_output_that_does_not_match_contract():
    def bad_handler(ctx, payload):
        return {"summary": "not a SkillResult"}

    skill_kit = SkillKit(
        tool_kit=ToolKit(_Backend()),
        registry=SkillRegistry(
            [
                SkillDefinition(
                    name="bad-skill",
                    display_name="BadSkill",
                    summary="bad",
                    description="bad",
                    input_model=_EmptySkillInput,
                    handler=bad_handler,
                )
            ]
        ),
    )

    result = skill_kit.invoke("bad-skill", {})

    assert result.success is False
    assert result.error_code == "HANDLER_ERROR"
    assert "expected SkillResult" in (result.error_message or "")
