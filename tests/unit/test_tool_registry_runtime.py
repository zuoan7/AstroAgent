from __future__ import annotations

import ast
from pathlib import Path

from src.core.mcp_protocol import (
    TOOL_INPUT_MODELS,
    TOOL_OUTPUT_MODELS,
    serialize_envelope,
    success_envelope,
)
from src.tools.catalog import get_default_tool_catalog
from src.tools.registry import get_default_tool_registry
from src.tools.results import ToolResult
from src.tools.runtime import ToolKit, ToolRuntime


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, tool_name: str, **kwargs):
        self.calls.append((tool_name, kwargs))
        return serialize_envelope(
            success_envelope(
                tool_name,
                {"live": {"city": kwargs.get("city", "北京"), "weather": "晴"}},
            )
        )


def test_tool_registry_is_canonical_for_legacy_schema_maps_and_catalog():
    registry = get_default_tool_registry()
    catalog = get_default_tool_catalog()

    assert set(registry.list_names()) == set(TOOL_INPUT_MODELS)
    assert set(registry.list_names()) == set(TOOL_OUTPUT_MODELS)
    assert set(catalog.list_names()) == set(registry.list_names())
    assert catalog.get_tool("get_weather").param_names == ["city", "extensions"]
    assert registry.get_tool("get_weather").input_model is TOOL_INPUT_MODELS["get_weather"]


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


def test_toolruntime_legacy_api_still_returns_raw_envelope_string():
    raw = ToolRuntime(_FakeBackend()).call_tool(
        "get_weather",
        city="北京",
        extensions="base",
    )

    assert isinstance(raw, str)
    assert '"ok": true' in raw


def test_toolkit_with_policy_blocks_tools_outside_allowed_set():
    backend = _FakeBackend()
    result = ToolKit(backend).with_policy(
        logical_skill="weather-lookup",
        allowed_tools=["get_weather"],
        enforce_allowed_tools=True,
    ).invoke("web_search", query="JWST")

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "TOOL_GUARD_REJECTED"
    assert backend.calls == []


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

