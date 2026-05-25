from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.tools.protocol import serialize_envelope, success_envelope
from src.tools.registry import get_default_tool_registry
from src.tools.results import ToolResult
from src.tools.kit import ToolKit


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def invoke(self, tool_name: str, **kwargs):
        self.calls.append((tool_name, kwargs))
        return serialize_envelope(
            success_envelope(
                tool_name,
                {"live": {"city": kwargs.get("city", "北京"), "weather": "晴"}},
            )
        )

    def invoke_parallel(self, calls: list[dict]):
        return [
            self.invoke(call["tool_name"], **call.get("kwargs", {})) for call in calls
        ]


def test_tool_registry_is_canonical_schema_source():
    registry = get_default_tool_registry()

    definition = registry.get_tool("get_weather")
    assert "get_weather" in registry.list_names()
    assert list(definition.input_model.model_fields.keys()) == ["city", "extensions"]
    assert definition.output_model is registry.get_tool("get_weather").output_model


def test_toolkit_invoke_returns_structured_tool_result():
    backend = _FakeBackend()
    result = ToolKit(backend).invoke("get_weather", city="北京", extensions="base")

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.tool_name == "get_weather"
    assert result.data["live"]["city"] == "北京"
    assert backend.calls == [("get_weather", {"city": "北京", "extensions": "base"})]


def test_toolkit_invoke_returns_structured_validation_error():
    backend = _FakeBackend()
    result = ToolKit(backend).invoke("get_altaz", planet_name="mars")

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "TOOL_INPUT_VALIDATION_ERROR"
    assert backend.calls == []


def test_toolkit_invoke_is_the_public_sync_tool_api():
    result = ToolKit(_FakeBackend()).invoke(
        "get_weather",
        city="北京",
        extensions="base",
    )

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.data["live"]["city"] == "北京"


def test_toolkit_with_policy_blocks_tools_outside_allowed_set():
    backend = _FakeBackend()
    result = (
        ToolKit(backend)
        .with_policy(
            logical_skill="weather-lookup",
            allowed_tools=["get_weather"],
            enforce_allowed_tools=True,
        )
        .invoke("web_search", query="JWST")
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "TOOL_GUARD_REJECTED"
    assert backend.calls == []


@pytest.mark.asyncio
async def test_toolkit_ainvoke_parallel_returns_structured_results():
    backend = _FakeBackend()

    results = await ToolKit(backend).ainvoke_parallel(
        [
            {"tool_name": "get_weather", "kwargs": {"city": "北京"}},
            {"tool_name": "get_weather", "kwargs": {"city": "上海"}},
        ]
    )

    assert [result.ok for result in results] == [True, True]
    assert [result.data["live"]["city"] for result in results] == ["北京", "上海"]


def test_tools_layer_does_not_import_skills_or_agent():
    forbidden_prefixes = ("src.skills", "src.agent")
    tools_root = Path(__file__).parents[2] / "src" / "tools"

    for path in tools_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
            elif isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)

        assert not [
            module
            for module in imported_modules
            if module.startswith(forbidden_prefixes)
        ], f"{path} imports forbidden upper-layer modules"
