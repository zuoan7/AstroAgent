import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()
sys.modules.pop("src.agent.streaming_service", None)

from src.agent.audit import RequestAuditLogger
from src.agent.models.skill_result import SkillResult
from src.agent.policies import BudgetExceededError, ModelPolicy, RequestBudget, RequestBudgetTracker
from src.agent.request_router import RouteDecision
from src.agent.streaming_service import StreamingService
from src.agent.task_orchestrator import TaskOrchestrator


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
    monkeypatch.setattr("src.agent.policies.model_policy.settings.SMALL_MODEL_PROVIDER", "dashscope")
    monkeypatch.setattr("src.agent.policies.model_policy.settings.SMALL_MODEL_NAME", "qwen-plus")
    policy = ModelPolicy()

    selected = policy.select("router")

    assert selected.tier == "small"
    assert selected.model_name == "qwen-plus"


@pytest.mark.asyncio
async def test_task_orchestrator_planned_response_contains_enterprise_metadata():
    class _SkillManagerStub:
        def call_skill(self, name, **params):
            if name == "weather-lookup":
                return SkillResult(
                    skill_name=name,
                    success=False,
                    data={},
                    summary="[错误] weather unavailable",
                    error_code="TOOL_FAIL",
                    error_message="weather unavailable",
                )
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

    orchestrator = TaskOrchestrator(
        skill_manager=_SkillManagerStub(),
        rag_retriever=SimpleNamespace(retrieve=lambda *args, **kwargs: {"context": ""}),
        llm=_LLMStub(),
    )
    decision = RouteDecision(
        route="planned_task",
        task_type="observation_recommendation",
        confidence=0.9,
        reason="matched_multiple_skills",
        matched_skills=["weather-lookup", "observation-planner"],
        expected_output_schema="observation_answer_v1",
    )

    result = await orchestrator.run(
        decision,
        "帮我看下今晚适合观测什么",
        chat_history="",
        user_profile="",
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
        expected_output_schema="observation_answer_v1",
    )

    class _PlanStub:
        def to_dict(self):
            return {"task_type": "observation_recommendation", "steps": []}

        def to_frontend_steps(self):
            return []

    async def fake_run(decision, query, **kwargs):
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
            route_decision=decision.to_meta(),
            fallback_path=[],
            budget_usage={"usage": {"llm_calls": 1}},
            versions={"planner_version": "planner_v2", "schema_version": "schema_v2"},
        )

    service = StreamingService(
        agent_executor=None,
        memory=_MemoryStub(),
        user_id="test_user",
        request_router=SimpleNamespace(route=lambda query: decision),
        task_orchestrator=SimpleNamespace(
            build_execution_plan=lambda *args, **kwargs: _PlanStub(),
            run=fake_run,
        ),
        audit_logger=logger,
    )

    events = []
    async for event in service.generate_events("帮我规划今晚观测"):
        events.append(event)

    assert any(event["type"] == "final_answer" for event in events)
    records = [json.loads(line) for line in Path(audit_path).read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["route_decision"]["route"] == "planned_task"
    assert "final_response" in records[0]
