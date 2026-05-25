from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agent.execution.direct_executor import DirectExecutor
from src.agent.execution.planned_executor import PlannedExecutor
from src.agent.governance import AgentExecutionPolicy
from src.capabilities.decision import CapabilityDecision
from src.agent.models.execution_context import ExecutionContext
from src.agent.models.execution_decision import ExecutionDecision
from src.agent.models.execution_plan import ExecutionPlan, PlanStep
from src.agent.models.final_response import FinalResponse
from src.agent.models.request_context import RequestContext
from src.skills.result import SkillResult
from src.agent.models.task_profile import TaskProfile
from src.agent.request_router import RouteDecision
from src.agent.streaming_service import StreamingService
from src.capabilities.selector import CapabilitySelector
from src.skills.registry import get_default_skill_registry
from src.tools.registry import get_default_tool_registry
from src.tools.kit import ToolKit


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def invoke(self, tool_name: str, **kwargs):
        self.calls.append((tool_name, kwargs))
        return (
            '{"ok": true, "data": {"status": "ok"}, "meta": {"tool_name": "%s", "schema_version": "1.0"}}'
            % tool_name
        )


def test_skill_and_tool_registries_project_capability_boundaries():
    skill_registry = get_default_skill_registry()
    tool_registry = get_default_tool_registry()

    observation = skill_registry.get("observation-planner")
    position = skill_registry.get("celestial-position-calculator")

    assert list(observation.allowed_tools) == [
        "get_weather",
        "get_weekly_events",
        "get_tonight_best",
    ]
    assert "get_altaz" in position.allowed_tools
    assert "rise_set" in [operation.operation for operation in position.operations]
    assert skill_registry.has_skill("web_search") is False
    assert skill_registry.has_skill("get_nasa_apod") is False
    assert tool_registry.has_tool("web_search") is True
    assert tool_registry.has_tool("get_nasa_apod") is True


def test_capability_selector_uses_task_profile_hints():
    selector = CapabilitySelector()
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


def test_toolkit_rejects_tools_outside_skill_policy():
    backend = _FakeBackend()
    runtime = ToolKit(backend).with_policy(
        logical_skill="weather-lookup",
        allowed_tools=["get_weather"],
        enforce_allowed_tools=True,
    )

    blocked = runtime.invoke("web_search", query="JWST")

    assert blocked.ok is False
    assert blocked.error is not None
    assert blocked.error.code == "TOOL_GUARD_REJECTED"
    assert backend.calls == []


class _DirectCapabilityKit:
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
    manager = _DirectCapabilityKit()
    executor = DirectExecutor(
        capability_kit=manager,
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
    response = await executor.run_context(context)

    assert manager.calls[0][0] == "web_search"
    assert response.audit_metadata["capability_name"] == "web_search"
    assert response.execution_events[0]["payload"]["capability_name"] == "web_search"


def test_planned_observability_metadata_contains_capability_fields():
    executor = PlannedExecutor(
        capability_kit=SimpleNamespace(),
        llm=SimpleNamespace(),
        synthesizer=SimpleNamespace(),
    )
    plan = ExecutionPlan(
        task_type="observation_recommendation",
        output_schema="observation_answer_v1",
        steps=[
            PlanStep(
                id="observation_plan",
                kind="tool",
                skill="observation-planner",
                capability_kind="skill",
                capability_name="observation-planner",
                params={"location": "北京"},
            )
        ],
    )
    decision = RouteDecision(
        route="planned_task",
        task_type="observation_recommendation",
        confidence=0.8,
        reason="test",
    )

    metadata = executor._build_observability_metadata(
        decision=decision,
        plan=plan,
        execution_trace=[
            {
                "step_id": "observation_plan",
                "skill": "observation-planner",
                "capability_kind": "skill",
                "capability_name": "observation-planner",
                "status": "success",
            }
        ],
        evidence_by_key={},
        skipped_step_ids=[],
    )

    assert metadata["plan_steps_with_params"][0]["capability_kind"] == "skill"
    assert (
        metadata["plan_steps_with_params"][0]["capability_name"]
        == "observation-planner"
    )
