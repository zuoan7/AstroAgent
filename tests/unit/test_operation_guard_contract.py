from __future__ import annotations

import json

from src.agent.models.skill_result import SkillResult
from src.core.mcp_protocol import (
    is_tool_error,
    parse_tool_response,
)
from src.skills.executor import SkillExecutor
from src.skills.policies.operation_policy import OperationPolicyResolver
from src.tools.runtime import ToolRuntime


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, tool_name: str, **kwargs):
        self.calls.append((tool_name, kwargs))
        return json.dumps(
            {
                "ok": True,
                "data": {"ok": True},
                "meta": {"tool_name": tool_name, "schema_version": "1.0"},
            },
            ensure_ascii=False,
        )

    def call_tools_parallel(self, calls: list[dict]):
        results = []
        for call in calls:
            results.append(
                self.call_tool(
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


def test_tool_runtime_rejects_forbidden_operation_child_tool():
    backend = _FakeBackend()
    runtime = ToolRuntime(backend).with_context(
        logical_skill="celestial-position-calculator",
        operation="altaz",
        allowed_tools=["get_altaz"],
        forbidden_tools=["get_planet_position"],
        enforce_allowed_tools=True,
    )

    raw = runtime.call_tool(
        "get_planet_position",
        planet_name="mars",
        observation_time="2026-05-22T00:00:00",
    )

    assert is_tool_error(raw)
    envelope = parse_tool_response(raw)
    assert envelope is not None
    assert envelope.error.code == "TOOL_GUARD_REJECTED"
    assert envelope.error.details["operation"] == "altaz"
    assert backend.calls == []


def test_tool_runtime_validates_mcp_input_schema():
    backend = _FakeBackend()
    runtime = ToolRuntime(backend)

    raw = runtime.call_tool("get_altaz", planet_name="mars")

    assert is_tool_error(raw)
    envelope = parse_tool_response(raw)
    assert envelope is not None
    assert envelope.error.code == "TOOL_INPUT_VALIDATION_ERROR"
    assert backend.calls == []


def test_parallel_runtime_returns_error_envelope_for_rejected_call():
    backend = _FakeBackend()
    runtime = ToolRuntime(backend).with_context(
        logical_skill="celestial-events-forecast",
        operation="weekly",
        allowed_tools=["get_weekly_events"],
        forbidden_tools=["get_monthly_events"],
        enforce_allowed_tools=True,
    )

    results = runtime.call_tools_parallel(
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
    assert not is_tool_error(results[0])
    assert is_tool_error(results[1])
    assert backend.calls == [("get_weekly_events", {"start_date": "2026-05-22"})]


class _BadPositionHandler:
    def __call__(self, runtime: ToolRuntime, **params):
        raw = runtime.call_tool(
            "get_planet_position",
            planet_name="mars",
            observation_time="2026-05-22T00:00:00",
        )
        envelope = parse_tool_response(raw)
        return SkillResult(
            skill_name="celestial-position-calculator",
            success=not is_tool_error(raw),
            data={
                "error_code": getattr(getattr(envelope, "error", None), "code", "")
            },
            summary=raw,
        )


def test_skill_executor_applies_operation_policy_to_handler_runtime():
    backend = _FakeBackend()
    executor = SkillExecutor(
        tool_runtime=ToolRuntime(backend),
        handlers={"celestial-position-calculator": _BadPositionHandler()},
    )

    result = executor.call(
        "celestial-position-calculator",
        target="火星",
        operation="altaz",
        location="北京",
        datetime="今晚",
    )

    assert result.success is False
    assert result.operation == "altaz"
    assert result.data["error_code"] == "TOOL_GUARD_REJECTED"
    assert backend.calls == []
