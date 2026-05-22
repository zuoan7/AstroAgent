from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agent.execution.direct_executor import DirectExecutor
from src.agent.governance import AgentExecutionPolicy
from src.agent.models.capability_decision import CapabilityDecision
from src.agent.models.execution_decision import ExecutionDecision
from src.agent.models.execution_context import ExecutionContext
from src.agent.models.final_response import FinalResponse
from src.agent.models.request_context import RequestContext
from src.agent.models.skill_result import SkillResult
from src.agent.models.task_profile import TaskProfile
from src.agent.request_router import RouteDecision
from src.agent.streaming_service import StreamingService
from src.capabilities.registry import CapabilityRegistry
from src.capabilities.selector import CapabilitySelector
from src.core.mcp_protocol import is_tool_error, parse_tool_response
from src.tools.runtime import ToolRuntime


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, tool_name: str, **kwargs):
        self.calls.append((tool_name, kwargs))
        return '{"ok": true, "data": {"status": "ok"}, "meta": {"tool_name": "%s", "schema_version": "1.0"}}' % tool_name


def test_capability_registry_projects_skill_allowed_tools():
    registry = CapabilityRegistry()

    observation = registry.get_skill("observation-planner")
    position = registry.get_skill("celestial-position-calculator")

    assert observation.allowed_tools == [
        "get_weather",
        "get_weekly_events",
        "get_tonight_best",
    ]
    assert "get_altaz" in position.allowed_tools
    assert "rise_set" in position.operations


def test_capability_selector_uses_task_profile_hints():
    registry = CapabilityRegistry()
    selector = CapabilitySelector(registry)
    profile = TaskProfile.from_legacy_route(
        route="direct_task",
        task_type="single_tool_lookup",
        confidence=0.9,
        matched_skills=["weather-lookup"],
    )
    decision = ExecutionDecision(
        mode="direct",
        reason="single_tool_low_openness",
        legacy_execution_path="direct",
    )

    selected = selector.select(
        profile=profile,
        execution_decision=decision,
        query="北京今晚天气怎么样",
    )

    assert selected.kind == "skill"
    assert selected.name == "weather-lookup"
    assert selected.allowed_tools == ["get_weather"]


def test_streaming_service_resolution_attaches_capability_decision():
    profile = TaskProfile.from_legacy_route(
        route="direct_task",
        task_type="single_tool_lookup",
        confidence=0.9,
        matched_skills=["weather-lookup"],
    )
    service = StreamingService(
        agent_executor=None,
        memory=SimpleNamespace(),
        request_router=SimpleNamespace(profile=lambda query: profile),
        execution_policy=AgentExecutionPolicy(mode="hybrid"),
    )

    execution_decision, resolved_profile, context = service._resolve_execution_decision(
        "北京今晚天气怎么样",
        None,
        use_long_term_memory=True,
    )

    assert execution_decision.mode == "direct"
    assert resolved_profile is profile
    assert context.capability_decision is not None
    assert context.capability_decision.name == "weather-lookup"


def test_tool_runtime_rejects_tools_outside_skill_policy():
    backend = _FakeBackend()
    runtime = ToolRuntime(backend).with_context(
        logical_skill="weather-lookup",
        allowed_tools=["get_weather"],
        enforce_allowed_tools=True,
    )

    blocked = runtime.call_tool("web_search", query="JWST")

    assert is_tool_error(blocked)
    envelope = parse_tool_response(blocked)
    assert envelope is not None
    assert envelope.error.code == "TOOL_GUARD_REJECTED"
    assert backend.calls == []


class _DirectSkillManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call_skill(self, name: str, **params):
        self.calls.append((name, params))
        return SkillResult(
            skill_name=name,
            success=True,
            data={"params": params},
            summary=f"{name} ok",
        )


class _DirectSynthesizer:
    def synthesize_direct(self, **kwargs):
        return FinalResponse(
            answer="direct ok",
            summary="direct ok",
            route="direct_task",
            task_type=kwargs.get("task_type", "single_tool_lookup"),
        )


@pytest.mark.asyncio
async def test_direct_executor_prefers_capability_decision_over_matched_skills():
    manager = _DirectSkillManager()
    executor = DirectExecutor(
        skill_manager=manager,
        rag_retriever=SimpleNamespace(),
        llm=SimpleNamespace(),
        synthesizer=_DirectSynthesizer(),
    )
    profile = TaskProfile.from_legacy_route(
        route="direct_task",
        task_type="single_tool_lookup",
        confidence=0.9,
        matched_skills=["weather-lookup"],
    )
    context = ExecutionContext(
        profile=profile,
        request=RequestContext(query="最近 JWST 有什么结果"),
        capability_decision=CapabilityDecision.for_skill(
            "web_search",
            confidence=0.8,
            reason="test_override",
            allowed_tools=["web_search"],
        ),
    )
    decision = RouteDecision(
        route="direct_task",
        task_type="single_tool_lookup",
        confidence=0.9,
        reason="legacy",
        matched_skills=["weather-lookup"],
    )

    response = await executor.run(
        decision,
        "最近 JWST 有什么结果",
        context=context,
    )

    assert manager.calls[0][0] == "web_search"
    assert response.audit_metadata["capability_name"] == "web_search"
    assert response.execution_events[0]["payload"]["capability_name"] == "web_search"
