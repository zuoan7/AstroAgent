from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agent.execution.direct_executor import DirectExecutor
from src.agent.models.capability_decision import CapabilityDecision
from src.agent.models.execution_context import ExecutionContext
from src.agent.models.final_response import FinalResponse
from src.agent.models.request_context import RequestContext
from src.agent.models.task_profile import TaskProfile
from src.capabilities.param_builder import CapabilityParamBuilder
from src.capabilities.selector import CapabilitySelector
from src.tools.selector import ToolSelector


def _profile(
    *,
    tool_need: str = "single",
    matched_skills: list[str] | None = None,
    capability_hints: list[str] | None = None,
) -> TaskProfile:
    return TaskProfile(
        task_type="single_tool_lookup",
        complexity="low",
        openness="low",
        tool_need=tool_need,
        matched_skills=list(matched_skills or []),
        capability_hints=list(capability_hints or []),
        confidence=0.8,
        legacy_route="direct_task",
    )


def test_tool_selector_routes_stable_atomic_tools():
    selector = ToolSelector()

    search = selector.select("帮我查一下最近韦布望远镜有什么新结果")
    apod = selector.select("今天的 NASA 每日天文图是什么")
    weather = selector.select("北京今晚天气怎么样")

    assert search is not None
    assert search.tool_name == "web_search"
    assert search.params == {"query": "最近韦布望远镜有什么新结果", "max_results": 5}
    assert apod is not None
    assert apod.tool_name == "get_nasa_apod"
    assert apod.params["date"]
    assert apod.params["hd"] is False
    assert weather is not None
    assert weather.tool_name == "get_weather"
    assert weather.params == {"city": "北京", "extensions": "all"}


def test_tool_selector_ignores_non_tool_questions():
    assert ToolSelector().select("为什么会有流星雨？") is None


def test_capability_selector_keeps_skill_capability_hint_priority():
    selected = CapabilitySelector().select(
        profile=_profile(capability_hints=["weather-lookup"]),
        query="北京今晚天气怎么样",
    )

    assert selected.kind == "skill"
    assert selected.name == "weather-lookup"


def test_capability_selector_ignores_unmirrored_legacy_matched_skills():
    selected = CapabilitySelector().select(
        profile=_profile(matched_skills=["weather-lookup"]),
        query="普通问题",
    )

    assert selected.kind == "none"


def test_capability_selector_prefers_capability_hints_over_legacy_matched_skills():
    selected = CapabilitySelector().select(
        profile=_profile(
            matched_skills=["weather-lookup"],
            capability_hints=["web_search"],
        ),
        query="最近 JWST 有什么结果",
    )

    assert selected.kind == "tool"
    assert selected.name == "web_search"


def test_tool_selector_does_not_read_legacy_matched_skills_as_hints():
    selected = ToolSelector().select(
        "普通问题",
        profile=SimpleNamespace(matched_skills=["web_search"], capability_hints=[]),
    )

    assert selected is None


def test_capability_selector_falls_back_to_atomic_tool_rules():
    selected = CapabilitySelector().select(
        profile=_profile(),
        query="北京今晚天气怎么样",
    )

    assert selected.kind == "tool"
    assert selected.name == "get_weather"
    assert selected.metadata["params"] == {"city": "北京", "extensions": "all"}


def test_capability_selector_keeps_none_fallback_without_tool_match():
    selected = CapabilitySelector().select(
        profile=_profile(),
        query="为什么会有流星雨？",
    )

    assert selected.kind == "none"


def test_capability_param_builder_uses_decision_metadata_params():
    builder = CapabilityParamBuilder(skill_param_builder=SimpleNamespace())
    decision = CapabilityDecision.for_tool(
        "web_search",
        confidence=0.8,
        reason="test",
        metadata={"params": {"query": "JWST latest", "max_results": 3}},
    )

    assert builder.build_for_decision(decision, "ignored") == {
        "query": "JWST latest",
        "max_results": 3,
    }


class _DirectToolManager:
    def __init__(self) -> None:
        self.tool_calls: list[tuple[str, dict]] = []

    def call_mcp_tool(self, tool_name: str, **params) -> str:
        self.tool_calls.append((tool_name, params))
        return '{"ok": true}'


class _DirectSynthesizer:
    def synthesize_direct(self, **kwargs):
        return FinalResponse(
            answer="direct tool ok",
            summary="direct tool ok",
            route="direct_task",
            task_type=kwargs.get("task_type", "single_tool_lookup"),
        )


@pytest.mark.asyncio
async def test_direct_executor_tool_decision_calls_mcp_tool_with_capability_audit():
    manager = _DirectToolManager()
    executor = DirectExecutor(
        skill_manager=manager,
        rag_retriever=SimpleNamespace(),
        llm=SimpleNamespace(),
        synthesizer=_DirectSynthesizer(),
    )
    capability = CapabilityDecision.for_tool(
        "web_search",
        confidence=0.88,
        reason="matched_fresh_external_search_rule",
        metadata={
            "params": {
                "query": "最近韦布望远镜有什么新结果",
                "max_results": 5,
            }
        },
    )
    context = ExecutionContext(
        profile=_profile(),
        request=RequestContext(query="帮我查一下最近韦布望远镜有什么新结果"),
        capability_decision=capability,
    )
    response = await executor.run_context(context)

    assert manager.tool_calls == [
        ("web_search", {"query": "最近韦布望远镜有什么新结果", "max_results": 5})
    ]
    assert response.audit_metadata["capability_kind"] == "tool"
    assert response.audit_metadata["capability_name"] == "web_search"
    assert response.audit_metadata["expected_mcp_tools"] == ["web_search"]
    assert response.execution_events[0]["payload"]["capability_kind"] == "tool"
