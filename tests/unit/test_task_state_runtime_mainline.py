import asyncio
import os
from types import SimpleNamespace

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()

from src.agent.models.execution_decision import ExecutionDecision
from src.agent.models.execution_plan import ExecutionPlan, PlanStep
from src.agent.models.final_response import FinalResponse
from src.agent.models.task_profile import TaskProfile
from src.agent.request_router import RouteDecision
from src.agent.streaming_service import StreamingService
from src.core.config import settings
from src.memory.api.dto import BuildContextRequest
from src.memory.api.memory_service import MemoryService
from src.memory.application.task_state_runtime_service import TaskStateRuntimeService
from src.memory.domain.events import MemoryEventType


def _memory(tmp_path) -> MemoryService:
    return MemoryService(
        db_path=os.path.join(tmp_path, "memory.sqlite"),
        tenant_id="tenant",
        session_id="session",
        user_id="user",
    )


def _profile(
    *,
    route: str = "planned_task",
    task_type: str = "observation_recommendation",
    confidence: float = 0.82,
    skills: list[str] | None = None,
) -> TaskProfile:
    return TaskProfile.from_legacy_route(
        route=route,
        task_type=task_type,
        confidence=confidence,
        matched_skills=skills or ["weather-lookup"],
        reason="test",
        expected_output_schema="observation_answer_v1",
    )


def test_runtime_start_patch_uses_plan_goal_and_next_action():
    runtime = TaskStateRuntimeService(memory=SimpleNamespace())
    plan = ExecutionPlan(
        task_type="observation_recommendation",
        output_schema="observation_answer_v1",
        steps=[
            PlanStep(id="weather", kind="tool", title="查询天气", skill="weather-lookup"),
            PlanStep(id="plan", kind="tool", title="生成观测计划", skill="observation-planner"),
        ],
    )

    patch = runtime.build_turn_started_patch(
        "今晚北京适合观测 M42 吗？",
        profile=_profile(),
        execution_decision=ExecutionDecision(mode="planned", reason="test"),
        execution_plan=plan,
    )

    assert patch["status"] == "running"
    assert patch["current_goal"] == "今晚北京适合观测 M42 吗？"
    assert patch["pending_steps"] == ["查询天气", "生成观测计划"]
    assert patch["next_action"] == "查询天气"
    assert patch["confidence"] == 0.82


def test_runtime_completion_patch_handles_completed_blocked_and_awaiting_user():
    runtime = TaskStateRuntimeService(memory=SimpleNamespace())

    completed = runtime.build_turn_completed_patch(
        response=FinalResponse(
            answer="可以观测",
            summary="可以观测",
            confidence=0.8,
            task_type="observation_recommendation",
            execution_trace=[
                {"step_id": "weather", "title": "查询天气", "status": "success"},
            ],
        ),
        profile=_profile(),
    )
    assert completed["status"] == "completed"
    assert completed["completed_steps"] == ["查询天气"]
    assert completed["blockers"] == []

    blocked = runtime.build_turn_completed_patch(
        response=FinalResponse(
            answer="失败",
            summary="失败",
            confidence=0.3,
            task_type="observation_recommendation",
            execution_trace=[
                {
                    "step_id": "weather",
                    "title": "查询天气",
                    "status": "error",
                    "error": "天气服务超时",
                },
            ],
        ),
        profile=_profile(),
    )
    assert blocked["status"] == "blocked"
    assert blocked["pending_steps"] == ["查询天气"]
    assert "天气服务超时" in blocked["blockers"][0]

    awaiting = runtime.build_turn_completed_patch(
        response=FinalResponse(
            answer="请补充城市？",
            summary="请补充城市？",
            confidence=0.7,
            task_type="clarification",
        ),
        profile=_profile(route="direct_task", task_type="clarification"),
    )
    assert awaiting["status"] == "awaiting_user"
    assert awaiting["pending_steps"] == ["等待用户补充信息"]
    assert awaiting["open_questions"]


def test_runtime_smalltalk_does_not_build_task_state_patch():
    runtime = TaskStateRuntimeService(memory=SimpleNamespace())
    smalltalk = _profile(route="direct_task", task_type="smalltalk", confidence=0.98, skills=[])

    assert runtime.build_turn_started_patch("你好", profile=smalltalk) == {}
    assert runtime.build_turn_completed_patch(
        response=FinalResponse(answer="你好", summary="你好", task_type="smalltalk"),
        profile=smalltalk,
    ) == {}


def test_runtime_apply_retries_once_on_optimistic_lock_conflict(tmp_path):
    memory = _memory(tmp_path)
    runtime = TaskStateRuntimeService(memory)
    first = memory.update_task_state(
        "session",
        {"current_goal": "初始目标"},
        tenant_id="tenant",
    )

    updated = runtime.apply_patch_with_retry(
        session_id="session",
        tenant_id="tenant",
        patch={"next_action": "继续执行"},
        expected_version=first.version - 1,
        turn_id="turn_retry",
    )

    assert updated is not None
    assert updated.next_action == "继续执行"
    assert updated.version == first.version + 1


def test_update_task_state_event_records_turn_id_and_normalizes_lists(tmp_path):
    memory = _memory(tmp_path)

    state = memory.update_task_state(
        "session",
        {
            "status": "running",
            "pending_steps": [" 查询天气 ", "", "查询天气", "生成计划"],
            "confidence": 1.5,
        },
        tenant_id="tenant",
        turn_id="turn_1",
    )

    assert state.pending_steps == ["查询天气", "生成计划"]
    assert state.confidence == 1.0
    events = memory.event_store.list_by_session(
        "session",
        event_type=MemoryEventType.TASK_STATE_UPDATED.value,
    )
    assert events[-1].turn_id == "turn_1"


def test_llm_enrichment_patch_allows_only_enrichment_fields():
    class FakeLLM:
        def invoke(self, prompt):
            return SimpleNamespace(
                content=(
                    '{"current_goal":"细化 M42 观测",'
                    '"active_constraints":["北京","","北京"],'
                    '"open_questions":["是否带相机？"],'
                    '"assumptions":["默认今晚"],'
                    '"next_action":"补充相机参数",'
                    '"confidence":0.9,'
                    '"status":"blocked",'
                    '"blockers":["不应写入"]}'
                )
            )

    runtime = TaskStateRuntimeService(memory=SimpleNamespace(), llm=FakeLLM())

    patch = runtime._extract_enrichment_patch(
        user_message="继续",
        assistant_message="请补充相机参数",
        current_state={},
    )

    assert patch == {
        "current_goal": "细化 M42 观测",
        "active_constraints": ["北京"],
        "open_questions": ["是否带相机？"],
        "assumptions": ["默认今晚"],
        "next_action": "补充相机参数",
        "confidence": 0.9,
    }


def test_stale_enrichment_turn_is_discarded_by_turn_id_check(tmp_path):
    memory = _memory(tmp_path)
    runtime = TaskStateRuntimeService(memory)
    memory.update_task_state(
        "session",
        {"current_goal": "第一轮"},
        tenant_id="tenant",
        turn_id="turn_1",
    )
    memory.update_task_state(
        "session",
        {"current_goal": "第二轮"},
        tenant_id="tenant",
        turn_id="turn_2",
    )

    assert runtime._is_latest_task_state_turn("session", "turn_1") is False
    assert runtime._is_latest_task_state_turn("session", "turn_2") is True


def test_streaming_updates_task_state_before_assistant_message(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_TASK_STATE_LLM_ENRICH_ENABLED", False)
    memory = _memory(tmp_path)
    decision = RouteDecision(
        route="planned_task",
        task_type="observation_recommendation",
        confidence=0.82,
        reason="test",
        matched_skills=["weather-lookup"],
        capability_hints=["weather-lookup"],
        expected_output_schema="observation_answer_v1",
    )
    plan = ExecutionPlan(
        task_type="observation_recommendation",
        output_schema="observation_answer_v1",
        steps=[PlanStep(id="weather", kind="tool", title="查询天气", skill="weather-lookup")],
    )
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
        return FinalResponse(
            answer="今晚适合观测。",
            summary="今晚适合观测。",
            confidence=0.88,
            route=context.profile.legacy_route,
            task_type=context.profile.task_type,
            execution_plan=plan.to_dict(),
            execution_trace=[
                {"step_id": "weather", "title": "查询天气", "status": "success"},
            ],
        )

    service = StreamingService(
        agent_executor=None,
        memory=memory,
        user_id="user",
        request_router=SimpleNamespace(
            profile=lambda q: profile,
            route=lambda q: (_ for _ in ()).throw(
                AssertionError("streaming should not call route()")
            ),
        ),
        execution_engine=SimpleNamespace(
            preview_plan_context=lambda *a, **kw: plan,
            run_context=fake_run_context,
        ),
    )

    events = asyncio.run(_collect_events(service.generate_events("今晚北京适合观测吗？")))
    final_events = [event for event in events if event.get("type") == "final_answer"]
    assert final_events and final_events[-1].get("task_state")

    memory_events = memory.event_store.list_by_session("session", limit=20)
    event_types = [event.event_type for event in memory_events]
    completion_index = max(
        index
        for index, event in enumerate(memory_events)
        if event.event_type == MemoryEventType.TASK_STATE_UPDATED.value
    )
    assistant_index = next(
        index
        for index, event in enumerate(memory_events)
        if event.event_type == MemoryEventType.MESSAGE_CREATED.value
        and event.payload.get("role") == "assistant"
    )
    assert completion_index < assistant_index
    assert MemoryEventType.TASK_STATE_UPDATED.value in event_types
    assert memory_events[completion_index].turn_id

    context = memory.build_context(
        BuildContextRequest(
            tenant_id="tenant",
            session_id="session",
            query="下一步",
        )
    )
    assert "current_goal:" in context["context_text"]
    assert "next_action:" in context["context_text"]


def test_streaming_smalltalk_does_not_update_task_state(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_TASK_STATE_LLM_ENRICH_ENABLED", False)
    memory = _memory(tmp_path)
    decision = RouteDecision(
        route="direct_task",
        task_type="smalltalk",
        confidence=0.98,
        reason="matched_smalltalk_pattern",
    )
    profile = TaskProfile.from_legacy_route(
        route=decision.route,
        task_type=decision.task_type,
        confidence=decision.confidence,
        reason=decision.reason,
    )

    async def fake_run_context(exec_decision, context, **kwargs):
        return FinalResponse(
            answer="你好。",
            summary="你好。",
            confidence=0.98,
            route=context.profile.legacy_route,
            task_type=context.profile.task_type,
        )

    service = StreamingService(
        agent_executor=None,
        memory=memory,
        user_id="user",
        request_router=SimpleNamespace(
            profile=lambda q: profile,
            route=lambda q: (_ for _ in ()).throw(
                AssertionError("streaming should not call route()")
            ),
        ),
        execution_engine=SimpleNamespace(run_context=fake_run_context),
    )

    asyncio.run(_collect_events(service.generate_events("你好")))

    events = memory.event_store.list_by_session(
        "session",
        event_type=MemoryEventType.TASK_STATE_UPDATED.value,
    )
    assert events == []


def test_follow_up_query_is_augmented_for_router(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_TASK_STATE_LLM_ENRICH_ENABLED", False)
    memory = _memory(tmp_path)
    memory.update_task_state(
        "session",
        {
            "current_goal": "判断北京 M42 是否适合观测",
            "next_action": "查询 M42 高度",
            "active_constraints": ["地点是北京"],
        },
        tenant_id="tenant",
    )
    captured_queries: list[str] = []

    def profile(query):
        captured_queries.append(query)
        return TaskProfile.from_legacy_route(
            route="direct_task",
            task_type="simple_qa",
            confidence=0.8,
            reason="test",
        )

    async def fake_run_context(exec_decision, context, **kwargs):
        return FinalResponse(
            answer="继续处理 M42。",
            summary="继续处理 M42。",
            confidence=0.7,
            route=context.profile.legacy_route,
            task_type=context.profile.task_type,
        )

    service = StreamingService(
        agent_executor=None,
        memory=memory,
        user_id="user",
        request_router=SimpleNamespace(
            profile=profile,
            route=lambda q: (_ for _ in ()).throw(
                AssertionError("streaming should not call route()")
            ),
        ),
        execution_engine=SimpleNamespace(run_context=fake_run_context),
    )

    asyncio.run(_collect_events(service.generate_events("下一步")))

    assert captured_queries
    assert "判断北京 M42 是否适合观测" in captured_queries[0]
    assert "查询 M42 高度" in captured_queries[0]


async def _collect_events(generator):
    events = []
    async for event in generator:
        events.append(event)
    return events
