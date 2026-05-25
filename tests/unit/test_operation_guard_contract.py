from __future__ import annotations

import json

from src.agent.capability_kit import CapabilityKit
from src.skills.policies.operation_policy import OperationPolicyResolver
from src.skills.policies.skill_policy import SkillPolicy
from src.skills.registry import get_default_skill_registry
from src.tools.kit import ToolKit


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def invoke(self, tool_name: str, **kwargs):
        self.calls.append((tool_name, kwargs))
        if tool_name == "get_altaz":
            data = {
                "planet": kwargs.get("planet_name", "mars"),
                "altitude": 42.0,
                "azimuth": 120.0,
                "distance_au": 1.5,
            }
        elif tool_name in {
            "get_tonight_best",
            "get_weekly_events",
            "get_monthly_events",
        }:
            data = f"{tool_name} ok"
        else:
            data = {"ok": True}
        return json.dumps(
            {
                "ok": True,
                "data": data,
                "meta": {"tool_name": tool_name, "schema_version": "1.0"},
            },
            ensure_ascii=False,
        )

    def invoke_parallel(self, calls: list[dict]):
        results = []
        for call in calls:
            results.append(
                self.invoke(
                    call["tool_name"],
                    **call.get("kwargs", {}),
                )
            )
        return results


def test_operation_policy_resolves_position_altaz():
    policy = OperationPolicyResolver().resolve(
        "celestial-position-calculator",
        {
            "target": "火星",
            "operation": "altaz",
            "location": "北京",
            "datetime": "今晚",
        },
    )

    assert policy is not None
    assert policy.operation == "altaz"
    assert policy.allowed_tools == ["get_altaz"]
    assert "get_planet_position" in policy.forbidden_tools


def test_operation_policy_resolves_events_weekly_and_monthly():
    resolver = OperationPolicyResolver()

    weekly = resolver.resolve(
        "celestial-events-forecast",
        {"start_date": "2026-05-01", "end_date": "2026-05-07"},
    )
    monthly = resolver.resolve(
        "celestial-events-forecast",
        {"start_date": "2026-05-01", "end_date": "2026-05-31"},
    )

    assert weekly is not None
    assert weekly.operation == "weekly"
    assert weekly.allowed_tools == ["get_weekly_events"]
    assert monthly is not None
    assert monthly.operation == "monthly"
    assert monthly.allowed_tools == ["get_monthly_events"]


def test_skill_policy_keeps_skill_required_params_out_of_tool_guard_by_default():
    definition = get_default_skill_registry().get("observation-planner")

    policy = SkillPolicy.from_definition(definition)
    kwargs = policy.to_tool_policy_kwargs()

    assert policy.skill_name == "observation-planner"
    assert policy.required_params == ("location",)
    assert kwargs["logical_skill"] == "observation-planner"
    assert kwargs["allowed_tools"] == [
        "get_weather",
        "get_weekly_events",
        "get_tonight_best",
    ]
    assert "required_params" not in kwargs
    assert policy.to_tool_policy_kwargs(include_required_params=True)[
        "required_params"
    ] == ["location"]


def test_toolkit_rejects_forbidden_operation_child_tool():
    backend = _FakeBackend()
    runtime = ToolKit(backend).with_policy(
        logical_skill="celestial-position-calculator",
        operation="altaz",
        allowed_tools=["get_altaz"],
        forbidden_tools=["get_planet_position"],
        enforce_allowed_tools=True,
    )

    result = runtime.invoke(
        "get_planet_position",
        planet_name="mars",
        observation_time="2026-05-22T00:00:00",
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "TOOL_GUARD_REJECTED"
    assert result.error.details["operation"] == "altaz"
    assert backend.calls == []


def test_toolkit_validates_mcp_input_schema():
    backend = _FakeBackend()
    runtime = ToolKit(backend)

    result = runtime.invoke("get_altaz", planet_name="mars")

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "TOOL_INPUT_VALIDATION_ERROR"
    assert backend.calls == []


def test_parallel_toolkit_returns_tool_result_for_rejected_call():
    backend = _FakeBackend()
    runtime = ToolKit(backend).with_policy(
        logical_skill="celestial-events-forecast",
        operation="weekly",
        allowed_tools=["get_weekly_events"],
        forbidden_tools=["get_monthly_events"],
        enforce_allowed_tools=True,
    )

    results = runtime.invoke_parallel(
        [
            {
                "tool_name": "get_weekly_events",
                "kwargs": {"start_date": "2026-05-22"},
            },
            {
                "tool_name": "get_monthly_events",
                "kwargs": {"year": 2026, "month": 5},
            },
        ]
    )

    assert len(results) == 2
    assert results[0].ok is True
    assert results[1].ok is False
    assert results[1].error is not None
    assert results[1].error.code == "TOOL_GUARD_REJECTED"
    assert backend.calls == [("get_weekly_events", {"start_date": "2026-05-22"})]


def test_capability_kit_applies_operation_policy_to_position_handler():
    backend = _FakeBackend()
    kit = CapabilityKit(tool_kit=ToolKit(backend))

    result = kit.call_skill(
        "celestial-position-calculator",
        target="火星",
        operation="altaz",
        location="北京",
        datetime="今晚",
    )

    assert result.success is True
    assert result.operation == "altaz"
    assert result.expected_mcp_tools == ["get_altaz"]
    assert result.allowed_child_tools == ["get_altaz"]
    assert "get_planet_position" in result.forbidden_child_tools
    assert backend.calls == [
        (
            "get_altaz",
            {
                "planet_name": "mars",
                "observation_time": result.data["observation_time"],
                "latitude": 39.9,
                "longitude": 116.4,
            },
        )
    ]
