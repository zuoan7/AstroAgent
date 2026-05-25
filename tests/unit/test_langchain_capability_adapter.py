from __future__ import annotations

from src.agent.adapters.langchain_adapter import to_langchain_tools
from src.agent.capability_kit import CapabilityKit
from src.skills.result import SkillResult
from src.skills.inputs import WeatherLookupInput
from src.tools.protocol import serialize_envelope, success_envelope
from src.tools.results import ToolResult
from src.tools.kit import ToolKit
from src.tools.schemas.weather import WeatherInput


class _Backend:
    def invoke(self, tool_name: str, **kwargs):
        return serialize_envelope(
            success_envelope(
                tool_name,
                {
                    "live": {
                        "city": kwargs.get("city"),
                        "weather": "晴",
                        "temperature": "20",
                    }
                },
            )
        )


class _RAG:
    def get_relevant_context(self, query: str) -> str:
        return f"rag:{query}"


def test_adapter_generates_rag_skills_and_react_exposed_tools():
    kit = CapabilityKit(tool_kit=ToolKit(_Backend()), rag_retriever=_RAG())

    tools = to_langchain_tools(kit)
    names = {tool.name for tool in tools}

    assert "RAGRetrieve" in names
    assert "WeatherLookup" in names
    assert "ObservationPlanner" in names
    assert {"get_nasa_apod", "get_weather", "web_search"} <= names
    assert "get_neo_data" not in names


def test_adapter_uses_pydantic_input_schemas():
    kit = CapabilityKit(tool_kit=ToolKit(_Backend()), rag_retriever=_RAG())
    tools = {tool.name: tool for tool in to_langchain_tools(kit)}

    assert tools["WeatherLookup"].args_schema is WeatherLookupInput
    assert tools["get_weather"].args_schema is WeatherInput


def test_skill_wrapper_calls_capability_kit_call_skill(monkeypatch):
    kit = CapabilityKit(tool_kit=ToolKit(_Backend()), rag_retriever=_RAG())
    calls = []

    def fake_call_skill(name: str, **payload):
        calls.append((name, payload))
        return SkillResult(skill_name=name, success=True, data={}, summary="ok")

    monkeypatch.setattr(kit, "call_skill", fake_call_skill)
    tool = {tool.name: tool for tool in to_langchain_tools(kit)}["WeatherLookup"]

    output = tool.func({"city": "北京", "extensions": "all"})

    assert calls == [
        ("weather-lookup", {"city": "北京", "location": None, "extensions": "all"})
    ]
    assert "ok" in output


def test_atomic_tool_wrapper_calls_capability_kit_call_tool(monkeypatch):
    kit = CapabilityKit(tool_kit=ToolKit(_Backend()), rag_retriever=_RAG())
    calls = []

    def fake_call_tool(name: str, **payload):
        calls.append((name, payload))
        return ToolResult(ok=True, tool_name=name, data={"status": "ok"})

    monkeypatch.setattr(kit, "call_tool", fake_call_tool)
    tool = {tool.name: tool for tool in to_langchain_tools(kit)}["get_weather"]

    output = tool.func({"city": "北京", "extensions": "all"})

    assert calls == [("get_weather", {"city": "北京", "extensions": "all"})]
    assert "status" in output
