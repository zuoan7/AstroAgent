from __future__ import annotations

import pytest

from src.agent.capability_kit import CapabilityKit
from src.skills.result import SkillResult
from src.tools.protocol import serialize_envelope, success_envelope
from src.tools.results import ToolResult
from src.tools.kit import ToolKit


class _Backend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def invoke(self, tool_name: str, **kwargs):
        self.calls.append((tool_name, kwargs))
        data = {
            "live": {
                "city": kwargs.get("city"),
                "weather": "晴",
                "temperature": "20",
                "humidity": "40",
                "windpower": "2",
            }
        }
        return serialize_envelope(success_envelope(tool_name, data))


def _kit() -> CapabilityKit:
    return CapabilityKit(tool_kit=ToolKit(_Backend()))


def test_call_skill_returns_skill_result():
    kit = _kit()

    result = kit.call_skill("weather-lookup", city="北京", extensions="all")

    assert isinstance(result, SkillResult)
    assert result.success is True
    assert result.skill_name == "weather-lookup"
    assert result.expected_mcp_tools == ["get_weather"]


def test_call_tool_returns_tool_result():
    kit = _kit()

    result = kit.call_tool("get_weather", city="北京", extensions="all")

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.tool_name == "get_weather"


def test_call_skill_does_not_fallback_to_atomic_tool():
    kit = _kit()

    with pytest.raises(KeyError):
        kit.call_skill("get_weather", city="北京")


def test_call_tool_does_not_fallback_to_skill():
    kit = _kit()

    result = kit.call_tool("weather-lookup", city="北京")

    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error is not None
    assert result.error.details["tool_name"] == "weather-lookup"
