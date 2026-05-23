import os
from types import SimpleNamespace

import pytest

from src.memory.api.dto import (
    AppendMessageRequest,
    AppendToolCallRequest,
    BuildContextRequest,
    DeleteMemoryRequest,
)
from src.memory.api.memory_service import MemoryService
from src.memory.domain.events import MemoryEvent, MemoryEventType
from src.memory.domain.task_state import TaskStateConflictError
from src.memory.infrastructure.repositories.artifact_store import ArtifactStore
from src.memory.infrastructure.repositories.event_store import EventStore


def _settings(db_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        DEFAULT_USER_ID="test_user",
        MEMORY_SIZE=15,
        MEMORY_WINDOW=8,
        MEMORY_CONTEXT_MAX_TOKENS=4000,
        MEMORY_CONTEXT_BUDGET=4000,
        MEMORY_MAX_RECENT_MESSAGES=6,
        MEMORY_MAX_TOOL_RECORDS=5,
        MEMORY_MAX_SALIENT_FACTS=32,
        MEMORY_SUMMARY_MAX_TOKENS=500,
        MEMORY_SUMMARY_TRIGGER_MESSAGES=100,
        MEMORY_SUMMARY_TRIGGER_TOKENS=100000,
        MEMORY_SUMMARY_KEEP_LAST_N=3,
        MEMORY_ENABLE_SUMMARY=True,
        MEMORY_PERSISTENCE_ENABLED=True,
        MEMORY_PERSISTENCE_PATH=db_path,
        MEMORY_IMPORTANCE_HIGH_ROLES={"user", "system"},
        MEMORY_TOOL_RESULT_MAX_LENGTH=120,
        DASHSCOPE_API_KEY="",
        MODEL_NAME="test-model",
    )


@pytest.fixture
def memory_db(tmp_path, monkeypatch):
    db_path = os.path.join(tmp_path, "memory.sqlite")
    from src.memory import config as memory_module

    monkeypatch.setattr(memory_module, "settings", _settings(db_path))
    return db_path


def test_event_store_append_is_idempotent(memory_db):
    store = EventStore(memory_db)
    event = MemoryEvent(
        event_id="evt_fixed",
        tenant_id="tenant",
        session_id="session",
        event_type=MemoryEventType.MESSAGE_CREATED.value,
        payload={"content": "hello"},
    )

    store.append(event)
    store.append(event)

    events = store.list_by_session("session")
    assert len(events) == 1
    assert events[0].event_id == "evt_fixed"
    assert events[0].payload["content"] == "hello"


def test_event_store_can_read_incremental_events(memory_db):
    store = EventStore(memory_db)
    first = store.append(
        MemoryEvent(
            tenant_id="tenant",
            session_id="session",
            event_type=MemoryEventType.MESSAGE_CREATED.value,
            payload={"content": "first"},
        )
    )
    store.append(
        MemoryEvent(
            tenant_id="tenant",
            session_id="session",
            event_type=MemoryEventType.MESSAGE_CREATED.value,
            payload={"content": "second"},
        )
    )
    store.append(
        MemoryEvent(
            tenant_id="tenant",
            session_id="session",
            event_type=MemoryEventType.TOOL_CALL_FINISHED.value,
            payload={"tool_name": "demo"},
        )
    )

    events = store.list_by_session("session", after_event_id=first.event_id, limit=10)

    assert len(events) == 2
    assert [event.event_type for event in events] == [
        MemoryEventType.MESSAGE_CREATED.value,
        MemoryEventType.TOOL_CALL_FINISHED.value,
    ]


def test_artifact_store_round_trips_raw_tool_output(memory_db):
    store = ArtifactStore(memory_db)

    artifact = store.put(
        tenant_id="tenant",
        session_id="session",
        tool_call_id="tool_1",
        raw_content='{"large": "payload"}',
        content_type="application/json",
    )

    assert artifact.artifact_id.startswith("art_")
    assert artifact.size_bytes == len('{"large": "payload"}'.encode("utf-8"))
    assert store.get_content(artifact.artifact_id) == '{"large": "payload"}'


def test_memory_service_stores_tool_artifact_and_event(memory_db):
    service = MemoryService(db_path=memory_db, tenant_id="tenant")

    record = service.append_tool_call(
        AppendToolCallRequest(
            tenant_id="tenant",
            session_id="session",
            user_id="user",
            tool_name="search",
            tool_input="mars opposition",
            raw_output="raw result with details",
            success=True,
        )
    )

    assert record.raw_artifact_id
    assert service.get_raw_artifact(record.raw_artifact_id) == "raw result with details"
    events = service.event_store.list_by_session(
        "session", event_type=MemoryEventType.TOOL_CALL_FINISHED.value
    )
    assert len(events) == 1
    assert events[0].payload["raw_artifact_id"] == record.raw_artifact_id


def test_task_state_patch_uses_optimistic_lock(memory_db):
    service = MemoryService(db_path=memory_db, tenant_id="tenant")

    state = service.update_task_state(
        "session",
        {
            "current_goal": "完成记忆层 P0 改造",
            "pending_steps": ["写测试", "跑测试"],
            "next_action": "写测试",
        },
    )

    assert state.version == 2
    assert state.current_goal == "完成记忆层 P0 改造"
    with pytest.raises(TaskStateConflictError):
        service.update_task_state(
            "session", {"next_action": "过期更新"}, expected_version=1
        )


def test_memory_service_append_message_updates_event_view(memory_db):
    service = MemoryService(
        db_path=memory_db, tenant_id="tenant", session_id="session", user_id="user"
    )

    message = service.append_message(
        AppendMessageRequest(
            tenant_id="tenant",
            session_id="session",
            user_id="user",
            role="user",
            content="请记录这个目标",
        )
    )

    assert message.content == "请记录这个目标"
    messages = service.get_all_messages()
    assert len(messages) == 1
    assert messages[0]["content"] == "请记录这个目标"


def test_memory_service_append_tool_call_exposes_raw_artifact(memory_db):
    service = MemoryService(
        db_path=memory_db, tenant_id="tenant", session_id="session", user_id="user"
    )

    record = service.append_tool_call(
        AppendToolCallRequest(
            tenant_id="tenant",
            session_id="session",
            user_id="user",
            tool_name="search",
            tool_input="query",
            raw_output="full raw tool output",
        )
    )

    assert record.raw_artifact_id
    assert service.get_raw_artifact(record.raw_artifact_id) == "full raw tool output"
    tool_calls = service.get_tool_calls()
    assert tool_calls[0]["raw_artifact_id"] == record.raw_artifact_id


def test_memory_service_adds_tool_evidence_metadata(memory_db):
    service = MemoryService(
        db_path=memory_db, tenant_id="tenant", session_id="session", user_id="user"
    )

    record = service.append_tool_call(
        AppendToolCallRequest(
            tenant_id="tenant",
            session_id="session",
            user_id="user",
            tool_name="weather-lookup",
            tool_input='{"time":"22:00","city":"北京"}',
            raw_output="北京 22:00 天气晴朗",
            timestamp=1000.0,
            metadata={"source": "test"},
        )
    )

    assert record.metadata["source"] == "test"
    assert record.metadata["params_hash"]
    assert record.metadata["produced_at"] == 1000.0
    assert record.metadata["effective_until"] > record.metadata["produced_at"]

    stored = service.get_tool_calls("session")[0]
    assert stored["metadata"]["params_hash"] == record.metadata["params_hash"]
    assert stored["metadata"]["produced_at"] == 1000.0


def test_compression_creates_summary_snapshot_from_events(memory_db):
    service = MemoryService(db_path=memory_db, tenant_id="tenant")
    service.append_message(
        AppendMessageRequest(
            tenant_id="tenant",
            session_id="session",
            user_id="user",
            role="user",
            content="目标是完成 SummarySnapshot 改造",
        )
    )
    service.append_tool_call(
        AppendToolCallRequest(
            tenant_id="tenant",
            session_id="session",
            tool_name="search",
            tool_input="SummarySnapshot",
            raw_output='{"answer": "snapshot created", "items": [1, 2, 3]}',
        )
    )

    snapshot = service.create_summary_snapshot("session", tenant_id="tenant")

    assert snapshot.snapshot_id.startswith("snap_")
    assert snapshot.source_count == 2
    assert "SummarySnapshot" in snapshot.summary_text
    latest = service.summary_snapshot_manager.get_latest("session")
    assert latest and latest.snapshot_id == snapshot.snapshot_id


def test_rebase_summary_snapshot_only_uses_new_events(memory_db):
    service = MemoryService(db_path=memory_db, tenant_id="tenant")
    service.append_message(
        AppendMessageRequest(
            tenant_id="tenant",
            session_id="session",
            role="user",
            content="first",
        )
    )
    base = service.create_summary_snapshot("session", tenant_id="tenant")
    service.append_message(
        AppendMessageRequest(
            tenant_id="tenant",
            session_id="session",
            role="assistant",
            content="second",
        )
    )

    rebased = service.rebase_summary_snapshot("session", tenant_id="tenant")

    assert rebased.snapshot_id != base.snapshot_id
    assert rebased.source_count == 1
    assert rebased.covered_from_event_id != base.covered_from_event_id
    assert "新增事件" in rebased.summary_text


def test_create_summary_snapshot_uses_recent_batch_without_latest_snapshot(memory_db):
    service = MemoryService(db_path=memory_db, tenant_id="tenant")
    contents = ["first", "second", "third"]
    for content in contents:
        service.append_message(
            AppendMessageRequest(
                tenant_id="tenant",
                session_id="session",
                role="user",
                content=content,
            )
        )

    snapshot = service.create_summary_snapshot(
        "session",
        tenant_id="tenant",
        snapshot_batch_size=2,
    )

    assert snapshot.source_count == 2
    assert "first" not in snapshot.summary_text
    assert "second" in snapshot.summary_text
    assert "third" in snapshot.summary_text


def test_create_summary_snapshot_uses_uncovered_batch_after_latest_snapshot(memory_db):
    service = MemoryService(db_path=memory_db, tenant_id="tenant")
    for content in ["m1", "m2"]:
        service.append_message(
            AppendMessageRequest(
                tenant_id="tenant",
                session_id="session",
                role="user",
                content=content,
            )
        )
    first_snapshot = service.create_summary_snapshot(
        "session",
        tenant_id="tenant",
        snapshot_batch_size=10,
    )
    for content in ["m3", "m4", "m5"]:
        service.append_message(
            AppendMessageRequest(
                tenant_id="tenant",
                session_id="session",
                role="assistant",
                content=content,
            )
        )

    next_snapshot = service.create_summary_snapshot(
        "session",
        tenant_id="tenant",
        snapshot_batch_size=2,
    )

    assert next_snapshot.snapshot_id != first_snapshot.snapshot_id
    assert next_snapshot.source_count == 2
    assert "m3" in next_snapshot.summary_text
    assert "m4" in next_snapshot.summary_text
    assert "m5" not in next_snapshot.summary_text


def test_retrieval_planner_includes_task_state_and_relevant_tool_evidence(memory_db):
    service = MemoryService(db_path=memory_db, tenant_id="tenant")
    service.append_message(
        AppendMessageRequest(
            tenant_id="tenant",
            session_id="session",
            role="user",
            content="请分析火星冲日观测计划",
        )
    )
    service.append_tool_call(
        AppendToolCallRequest(
            tenant_id="tenant",
            session_id="session",
            tool_name="ephemeris",
            tool_input="mars opposition",
            raw_output="火星冲日亮度和高度数据",
        )
    )
    service.update_task_state(
        "session",
        {
            "current_goal": "制定火星冲日观测计划",
            "active_constraints": ["保留工具证据"],
            "next_action": "核查天气",
        },
        tenant_id="tenant",
    )
    service.create_summary_snapshot("session", tenant_id="tenant")

    context = service.build_context(
        BuildContextRequest(
            tenant_id="tenant",
            session_id="session",
            query="下一步如何使用火星工具证据？",
            max_tokens=1200,
        )
    )

    assert context["query_type"] in {"task_progress", "evidence"}
    assert "current_goal: 制定火星冲日观测计划" in context["context_text"]
    assert "ephemeris" in context["context_text"]
    assert context["retrieval_plan"]["selected_task_state_version"] >= 2


def test_delete_tool_call_tombstones_event_and_artifact(memory_db):
    service = MemoryService(db_path=memory_db, tenant_id="tenant")
    record = service.append_tool_call(
        AppendToolCallRequest(
            tenant_id="tenant",
            session_id="session",
            tool_name="search",
            tool_input="delete me",
            raw_output="sensitive raw output",
        )
    )

    job = service.delete_memory(
        DeleteMemoryRequest(
            tenant_id="tenant",
            scope="tool_call",
            selector={"session_id": "session", "tool_call_id": record.tool_call_id},
            requested_by="tester",
        )
    )

    assert job.status == "completed"
    assert job.result["events_marked"] == 1
    assert service.get_raw_artifact(record.raw_artifact_id) is None
    assert service.get_tool_calls("session") == []
    deleted_events = service.event_store.list_by_source(
        "session", "tool_call", record.tool_call_id, include_deleted=True
    )
    assert deleted_events and deleted_events[0].is_deleted is True


def test_delete_session_marks_task_state_and_snapshot_deleted(memory_db):
    service = MemoryService(db_path=memory_db, tenant_id="tenant")
    service.append_message(
        AppendMessageRequest(
            tenant_id="tenant",
            session_id="session",
            role="user",
            content="delete session",
        )
    )
    service.update_task_state(
        "session", {"current_goal": "temporary"}, tenant_id="tenant"
    )
    service.create_summary_snapshot("session", tenant_id="tenant")

    job = service.delete_memory(
        DeleteMemoryRequest(
            tenant_id="tenant",
            scope="session",
            selector={"session_id": "session"},
            requested_by="tester",
        )
    )

    assert job.status == "completed"
    assert service.task_state_repository.get("session") is None
    assert service.summary_snapshot_manager.get_latest("session") is None
    active_events = service.event_store.list_by_session("session")
    assert len(active_events) == 1
    assert active_events[0].event_type == "memory_deleted"
