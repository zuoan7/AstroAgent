import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()
sys.modules.pop("src.agent.streaming_service", None)

from src.agent.audit import RequestAuditLogger
from src.agent.execution.planned_executor import PlannedExecutor
from src.agent.models.execution_context import ExecutionContext
from src.agent.models.final_response import FinalResponse
from src.skills.result import SkillResult
from src.agent.models.task_profile import TaskProfile
from src.agent.policies import (
    BudgetExceededError,
    ModelPolicy,
    RequestBudget,
    RequestBudgetTracker,
)
from src.agent.request_router import RouteDecision
from src.agent.streaming_service import StreamingService


class _MemoryStub:
    def __init__(self):
        self.messages = []
        self.session_id = "test_session"

    def build_context(self, request):
        return {"context_text": ""}

    def append_message(self, request):
        self.messages.append(
            {
                "role": request.role,
                "content": request.content,
                "timestamp": request.timestamp,
            }
        )


def test_budget_tracker_enforces_limits():
    tracker = RequestBudgetTracker(
        RequestBudget(
            max_llm_calls=1,
            max_tool_calls=1,
            max_total_time_ms=10000,
            max_parallelism=1,
            max_context_chars=10,
        )
    )

    tracker.register_context_chars(8)
    tracker.register_llm_call()

    with pytest.raises(BudgetExceededError):
        tracker.register_llm_call()


def test_model_policy_selects_small_model_for_router(monkeypatch):
    monkeypatch.setattr(
        "src.agent.policies.model_policy.settings.SMALL_MODEL_PROVIDER", "dashscope"
    )
    monkeypatch.setattr(
        "src.agent.policies.model_policy.settings.SMALL_MODEL_NAME", "qwen-plus"
    )
    policy = ModelPolicy()

    selected = policy.select("router")

    assert selected.tier == "small"
    assert selected.model_name == "qwen-plus"


@pytest.mark.asyncio
async def test_planned_executor_response_contains_enterprise_metadata():
    class _CapabilityKitStub:
        def call_skill(self, name, **params):
            return SkillResult(
                skill_name=name,
                success=True,
                data={"params": params},
                summary="observation ok",
                sources=[{"source_id": name, "kind": "tool_output"}],
            )

    class _LLMStub:
        def invoke(self, prompt):
            return SimpleNamespace(content="综合后建议今晚仍可先进行目标筛选。")

    class _SynthesizerStub:
        prompt_version = "test_synth"

        def synthesize(self, **kwargs):
            return FinalResponse(
                answer="综合后建议今晚仍可先进行目标筛选。",
                summary="综合后建议今晚仍可先进行目标筛选。",
                route=kwargs.get("route", ""),
                task_type=kwargs.get("task_type", ""),
                execution_plan=kwargs.get("execution_plan"),
                execution_trace=kwargs.get("execution_trace", []),
                route_decision=kwargs.get("route_decision"),
                fallback_path=kwargs.get("fallback_path", []),
                budget_usage=kwargs.get("budget_usage"),
                versions=kwargs.get("versions"),
            )

    executor = PlannedExecutor(
        capability_kit=_CapabilityKitStub(),
        llm=_LLMStub(),
        synthesizer=_SynthesizerStub(),
    )
    decision = RouteDecision(
        route="planned_task",
        task_type="observation_recommendation",
        confidence=0.9,
        reason="matched_multiple_skills",
        matched_skills=["weather-lookup", "observation-planner"],
        capability_hints=["weather-lookup", "observation-planner"],
        expected_output_schema="observation_answer_v1",
    )

    result = await executor.run_context(
        ExecutionContext.from_legacy_decision(
            decision,
            "帮我看下今晚适合观测什么",
            chat_history="",
            user_profile="",
        )
    )

    assert result.execution_plan is not None
    assert result.budget_usage is not None
    assert result.versions["planner_version"]
    assert result.versions["synth_prompt_version"]
    assert result.route_decision["route"] == "planned_task"
    assert result.fallback_path == []


@pytest.mark.asyncio
async def test_streaming_service_writes_audit_log(tmp_path):
    audit_path = tmp_path / "requests.jsonl"
    logger = RequestAuditLogger(enabled=True, output_path=str(audit_path))
    decision = RouteDecision(
        route="planned_task",
        task_type="observation_recommendation",
        confidence=0.8,
        reason="matched_multiple_skills",
        matched_skills=["weather-lookup", "observation-planner"],
        capability_hints=["weather-lookup", "observation-planner"],
        expected_output_schema="observation_answer_v1",
    )

    class _PlanStub:
        def to_dict(self):
            return {"task_type": "observation_recommendation", "steps": []}

        def to_frontend_steps(self):
            return []

    profile = TaskProfile.from_legacy_route(
        route=decision.route,
        task_type=decision.task_type,
        confidence=decision.confidence,
        matched_skills=decision.matched_skills,
        capability_hints=decision.capability_hints,
        reason=decision.reason,
        expected_output_schema=decision.expected_output_schema,
    )

    async def fake_run_context(exec_decision, context, **kwargs):
        from src.agent.models.final_response import FinalResponse

        return FinalResponse(
            answer="企业级链路已返回观测建议。",
            summary="企业级链路已返回观测建议。",
            tools_used=[],
            sources=[],
            confidence=0.8,
            route="planned_task",
            task_type="observation_recommendation",
            execution_plan={"task_type": "observation_recommendation", "steps": []},
            execution_trace=[],
            route_decision=context.profile.to_legacy_route_meta(),
            fallback_path=[],
            budget_usage={"usage": {"llm_calls": 1}},
            versions={"planner_version": "planner_v2", "schema_version": "schema_v2"},
        )

    service = StreamingService(
        agent_executor=None,
        memory=_MemoryStub(),
        user_id="test_user",
        request_router=SimpleNamespace(
            profile=lambda query: profile,
            route=lambda query: (_ for _ in ()).throw(
                AssertionError("streaming should not call route()")
            ),
        ),
        execution_engine=SimpleNamespace(
            preview_plan_context=lambda *args, **kwargs: _PlanStub(),
            run_context=fake_run_context,
        ),
        audit_logger=logger,
    )

    events = []
    async for event in service.generate_events("帮我规划今晚观测"):
        events.append(event)

    assert any(event["type"] == "final_answer" for event in events)
    records = [
        json.loads(line)
        for line in Path(audit_path).read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["route_decision"]["route"] == "planned_task"
    assert "final_response" in records[0]
