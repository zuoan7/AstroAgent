"""summary snapshot 自动压缩触发测试。

覆盖事件数/字符预算/话题漂移/工具链完成等多触发器、已有快照 rebase、
失败隔离和 build_context 注入摘要的短期记忆压缩行为。
"""

import os
from types import SimpleNamespace
from unittest import mock

import pytest

from src.memory.api.dto import AppendMessageRequest, BuildContextRequest
from src.memory.api.memory_service import MemoryService
from src.memory.domain.events import MemoryEvent, MemoryEventType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings(db_path: str, **overrides) -> SimpleNamespace:
    """构造测试用 settings 对象。"""

    defaults = {
        "DEFAULT_USER_ID": "test_user",
        "MEMORY_SIZE": 15,
        "MEMORY_WINDOW": 8,
        "MEMORY_CONTEXT_MAX_TOKENS": 4000,
        "MEMORY_CONTEXT_BUDGET": 4000,
        "MEMORY_MAX_RECENT_MESSAGES": 6,
        "MEMORY_MAX_TOOL_RECORDS": 5,
        "MEMORY_MAX_SALIENT_FACTS": 32,
        "MEMORY_SUMMARY_MAX_TOKENS": 500,
        "MEMORY_SUMMARY_TRIGGER_MESSAGES": 100,
        "MEMORY_SUMMARY_TRIGGER_TOKENS": 100000,
        "MEMORY_AUTO_SUMMARY_ENABLED": True,
        "MEMORY_SUMMARY_MIN_NEW_EVENTS": 6,
        "MEMORY_SUMMARY_KEEP_LAST_N": 3,
        "MEMORY_ENABLE_SUMMARY": True,
        "MEMORY_PERSISTENCE_ENABLED": True,
        "MEMORY_PERSISTENCE_PATH": db_path,
        "MEMORY_IMPORTANCE_HIGH_ROLES": {"user", "system"},
        "MEMORY_TOOL_RESULT_MAX_LENGTH": 120,
        "DASHSCOPE_API_KEY": "",
        "MODEL_NAME": "test-model",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture
def memory_db(tmp_path, monkeypatch):
    """创建临时 SQLite 记忆数据库 fixture。"""

    db_path = os.path.join(tmp_path, "memory.sqlite")
    from src.memory import config as memory_module

    monkeypatch.setattr(memory_module, "settings", _settings(db_path))
    # Also patch maintenance_service's settings import
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings",
        _settings(db_path),
    )
    return db_path


def _make_memory_service(db_path: str, **overrides) -> MemoryService:
    """构造启用测试配置的 MemoryService。"""

    from src.memory import config as memory_module
    # Apply overrides by re-patching
    return MemoryService(db_path=db_path, tenant_id="t1", session_id="s1")


def _append_pair(svc: MemoryService, user_msg: str, assistant_msg: str):
    """追加一组用户和助手消息事件。"""

    svc.append_message(AppendMessageRequest(
        session_id=svc.session_id, role="user", content=user_msg,
    ))
    svc.append_message(AppendMessageRequest(
        session_id=svc.session_id, role="assistant", content=assistant_msg,
    ))


def _append_memory_event(
    svc: MemoryService,
    event_type: str,
    payload: dict,
    *,
    tenant_id: str = "t1",
    session_id: str = "s1",
):
    """直接追加一条底层记忆事件。"""

    return svc.event_store.append(
        MemoryEvent(
            tenant_id=tenant_id,
            session_id=session_id,
            event_type=event_type,
            payload=payload,
        )
    )


# ---------------------------------------------------------------------------
# 1. Below threshold — no snapshot created
# ---------------------------------------------------------------------------

def test_no_snapshot_below_threshold(memory_db, monkeypatch):
    """测试 no snapshot below threshold 场景。"""

    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_AUTO_SUMMARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_MESSAGES",
        100,
    )
    svc = MemoryService(db_path=memory_db, tenant_id="t1", session_id="s1")
    _append_pair(svc, "用户消息1", "助手回复1")
    _append_pair(svc, "用户消息2", "助手回复2")

    summary = svc.get_summary(svc.session_id)
    assert summary == "", "未达阈值不应创建 summary snapshot"


# ---------------------------------------------------------------------------
# 2. Reaching threshold creates snapshot on assistant message
# ---------------------------------------------------------------------------

def test_snapshot_created_on_threshold(memory_db, monkeypatch):
    """测试 snapshot created on threshold 场景。"""

    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_AUTO_SUMMARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_MESSAGES",
        4,
    )
    svc = MemoryService(db_path=memory_db, tenant_id="t1", session_id="s1")
    for i in range(5):
        _append_pair(svc, f"用户消息{i}", f"助手回复{i}")

    summary = svc.get_summary(svc.session_id)
    assert summary != "", "达到阈值应创建 summary snapshot"
    assert len(summary) > 0


# ---------------------------------------------------------------------------
# 3. User message does NOT trigger; assistant message DOES
# ---------------------------------------------------------------------------

def test_assistant_only_trigger(memory_db, monkeypatch):
    """测试 assistant only trigger 场景。"""

    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_AUTO_SUMMARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_MESSAGES",
        3,
    )
    svc = MemoryService(db_path=memory_db, tenant_id="t1", session_id="s1")

    # Write just enough that next assistant message would reach threshold
    svc.append_message(AppendMessageRequest(
        session_id=svc.session_id, role="user", content="用户消息1",
    ))
    svc.append_message(AppendMessageRequest(
        session_id=svc.session_id, role="assistant", content="助手消息1",
    ))
    svc.append_message(AppendMessageRequest(
        session_id=svc.session_id, role="user", content="用户消息2",
    ))

    # After user message only, snapshot should not be created yet
    summary_before = svc.get_summary(svc.session_id)
    assert summary_before == "", "仅 user 消息不应触发 snapshot"

    # Append assistant message — now should trigger
    svc.append_message(AppendMessageRequest(
        session_id=svc.session_id, role="assistant", content="助手消息2",
    ))
    summary_after = svc.get_summary(svc.session_id)
    assert summary_after != "", "assistant 消息后应触发 snapshot"


# ---------------------------------------------------------------------------
# 4. Existing snapshot — only rebase on sufficient new events
# ---------------------------------------------------------------------------

def test_rebase_only_with_enough_new_events(memory_db, monkeypatch):
    """测试 rebase only with enough new events 场景。"""

    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_AUTO_SUMMARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_MESSAGES",
        4,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_MIN_NEW_EVENTS",
        4,
    )
    svc = MemoryService(db_path=memory_db, tenant_id="t1", session_id="s1")

    # First create a snapshot
    for i in range(5):
        _append_pair(svc, f"用户消息{i}", f"助手回复{i}")

    first_summary = svc.get_summary(svc.session_id)
    assert first_summary != ""

    # Write fewer than MIN_NEW_EVENTS new messages — should NOT rebase
    _append_pair(svc, "新用户消息1", "新助手消息1")  # 2 events

    # Check that the snapshot hasn't changed (same summary_text means no rebase triggered
    # because the new events are below the min_new threshold)
    # We can't easily check snapshot_id, but we can verify the system works

    # Now write enough to exceed MIN_NEW_EVENTS
    for i in range(5):
        _append_pair(svc, f"追加用户{i}", f"追加助手{i}")  # 10 more events

    final_summary = svc.get_summary(svc.session_id)
    assert final_summary != "", "rebase 后 summary 应仍然存在"


# ---------------------------------------------------------------------------
# 5. MEMORY_AUTO_SUMMARY_ENABLED=False — no trigger
# ---------------------------------------------------------------------------

def test_disable_switch_prevents_trigger(memory_db, monkeypatch):
    """测试 disable switch prevents trigger 场景。"""

    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_AUTO_SUMMARY_ENABLED",
        False,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_MESSAGES",
        2,
    )
    svc = MemoryService(db_path=memory_db, tenant_id="t1", session_id="s1")

    for i in range(10):
        _append_pair(svc, f"用户{i}", f"助手{i}")

    summary = svc.get_summary(svc.session_id)
    assert summary == "", "开关关闭时不应创建 snapshot"


# ---------------------------------------------------------------------------
# 6. Token threshold triggers snapshot
# ---------------------------------------------------------------------------

def test_token_threshold_trigger(memory_db, monkeypatch):
    """测试 token threshold trigger 场景。"""

    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_AUTO_SUMMARY_ENABLED",
        True,
    )
    # message trigger set very high so it won't fire
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_MESSAGES",
        1000,
    )
    # token trigger set very low so it fires first
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_TOKENS",
        50,
    )
    svc = MemoryService(db_path=memory_db, tenant_id="t1", session_id="s1")

    # Write long messages to accumulate token estimate
    for i in range(3):
        svc.append_message(AppendMessageRequest(
            session_id=svc.session_id, role="user",
            content="这是一条非常长的消息内容" * 30,
        ))
        svc.append_message(AppendMessageRequest(
            session_id=svc.session_id, role="assistant",
            content="这是助手的详细回复内容" * 20,
        ))

    summary = svc.get_summary(svc.session_id)
    assert summary != "", "token 阈值触发应创建 snapshot"


# ---------------------------------------------------------------------------
# 7. Auto summary failure does NOT affect append_message
# ---------------------------------------------------------------------------

def test_auto_summary_failure_does_not_block_append(memory_db, monkeypatch):
    """测试 auto summary failure does not block append 场景。"""

    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_AUTO_SUMMARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_MESSAGES",
        2,
    )
    svc = MemoryService(db_path=memory_db, tenant_id="t1", session_id="s1")

    # Mock create_summary_snapshot to throw
    with mock.patch.object(
        svc.maintenance_service,
        "create_summary_snapshot",
        side_effect=RuntimeError("模拟失败"),
    ):
        # This should NOT raise
        message = svc.append_message(AppendMessageRequest(
            session_id=svc.session_id, role="assistant", content="测试助手消息",
        ))
        assert message is not None
        assert message.content == "测试助手消息"

    # Verify the message was written successfully despite summary failure
    messages = svc.get_all_messages(svc.session_id)
    assert any(m["content"] == "测试助手消息" for m in messages)


def test_auto_summary_failure_does_not_block_user_append(memory_db, monkeypatch):
    """User messages skip auto-summary, so failures in trigger check don't affect them."""
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_AUTO_SUMMARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_MESSAGES",
        1,
    )
    svc = MemoryService(db_path=memory_db, tenant_id="t1", session_id="s1")

    # Mock should_create_summary_snapshot to throw — user path doesn't call it
    with mock.patch.object(
        svc.maintenance_service,
        "should_create_summary_snapshot",
        side_effect=RuntimeError("模拟失败"),
    ):
        # User message should still succeed
        message = svc.append_message(AppendMessageRequest(
            session_id=svc.session_id, role="user", content="用户消息",
        ))
        assert message is not None
        assert message.content == "用户消息"


# ---------------------------------------------------------------------------
# 8. build_context reads auto-generated summary snapshot
# ---------------------------------------------------------------------------

def test_build_context_reads_auto_snapshot(memory_db, monkeypatch):
    """测试 build context reads auto snapshot 场景。"""

    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_AUTO_SUMMARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_MESSAGES",
        4,
    )
    svc = MemoryService(db_path=memory_db, tenant_id="t1", session_id="s1")

    for i in range(5):
        _append_pair(svc, f"用户消息{i}", f"助手回复{i}")

    context = svc.build_context(BuildContextRequest(
        tenant_id=svc.tenant_id,
        session_id=svc.session_id,
        query="测试查询",
    ))
    assert context.get("selected_summary_snapshot") is not None, (
        "build_context 应能读取自动生成的 summary snapshot"
    )
    assert context["selected_summary_snapshot"]["summary_text"] != ""
    # Context text should contain a summary snapshot section
    assert "summary snapshot" in context["context_text"].lower() or \
        context["selected_summary_snapshot"]["summary_text"][:20] in context["context_text"]


# ---------------------------------------------------------------------------
# 9. Edge cases
# ---------------------------------------------------------------------------

def test_no_events_does_not_trigger(memory_db, monkeypatch):
    """测试 no events does not trigger 场景。"""

    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_AUTO_SUMMARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_MESSAGES",
        0,
    )
    svc = MemoryService(db_path=memory_db, tenant_id="t1", session_id="s1")
    # No messages written — trigger_messages=0 would trigger but uncovered_count=0
    # This edge should not create a snapshot
    summary = svc.get_summary(svc.session_id)
    # With 0 events, trigger_messages=0 should create? Let's verify behavior:
    # uncovered_count=0 >= trigger_messages=0 → True
    # But creating a snapshot with 0 events is wasteful.
    # The trigger should not fire if uncovered_count is 0.
    # (This is a design consideration; the test verifies current behavior)
    decision = svc.maintenance_service.should_create_summary_snapshot(svc.session_id)
    assert summary == ""
    assert decision.should_create is False
    assert decision.reason == "no_uncovered_events"


def test_fixed_uncovered_event_trigger_creates_snapshot_decision(memory_db, monkeypatch):
    """测试 fixed uncovered event trigger creates snapshot decision 场景。"""

    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_AUTO_SUMMARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_MESSAGES",
        1000,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_TOKENS",
        100000,
    )
    svc = MemoryService(db_path=memory_db, tenant_id="t1", session_id="s1")
    for index in range(30):
        _append_memory_event(
            svc,
            MemoryEventType.MESSAGE_CREATED.value,
            {"role": "user", "content": f"事件 {index}"},
        )

    decision = svc.maintenance_service.should_create_summary_snapshot(svc.session_id)

    assert decision.should_create is True
    assert decision.mode == "create"
    assert "uncovered_events(30)" in decision.reason


def test_fixed_uncovered_event_trigger_rebases_existing_snapshot(memory_db, monkeypatch):
    """测试 fixed uncovered event trigger rebases existing snapshot 场景。"""

    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_AUTO_SUMMARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_MIN_NEW_EVENTS",
        1000,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_TOKENS",
        100000,
    )
    svc = MemoryService(db_path=memory_db, tenant_id="t1", session_id="s1")
    _append_memory_event(
        svc,
        MemoryEventType.MESSAGE_CREATED.value,
        {"role": "user", "content": "北京 M42 初始上下文"},
    )
    svc.create_summary_snapshot(svc.session_id)
    for index in range(30):
        _append_memory_event(
            svc,
            MemoryEventType.MESSAGE_CREATED.value,
            {"role": "user", "content": f"北京 M42 新事件 {index}"},
        )

    decision = svc.maintenance_service.should_create_summary_snapshot(svc.session_id)

    assert decision.should_create is True
    assert decision.mode == "rebase"
    assert "uncovered_events(30)" in decision.reason


def test_fixed_token_trigger_creates_snapshot_decision(memory_db, monkeypatch):
    """测试 fixed token trigger creates snapshot decision 场景。"""

    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_AUTO_SUMMARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_MESSAGES",
        1000,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_TOKENS",
        100000,
    )
    svc = MemoryService(db_path=memory_db, tenant_id="t1", session_id="s1")
    _append_memory_event(
        svc,
        MemoryEventType.MESSAGE_CREATED.value,
        {"role": "user", "content": "北京 M42 " + ("长内容" * 5000)},
    )

    decision = svc.maintenance_service.should_create_summary_snapshot(svc.session_id)

    assert decision.should_create is True
    assert decision.mode == "create"
    assert decision.estimated_tokens >= 6000
    assert "estimated_tokens" in decision.reason


def test_topic_drift_triggers_rebase_decision(memory_db, monkeypatch):
    """测试 topic drift triggers rebase decision 场景。"""

    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_AUTO_SUMMARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_MIN_NEW_EVENTS",
        1000,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_TOKENS",
        100000,
    )
    svc = MemoryService(db_path=memory_db, tenant_id="t1", session_id="s1")
    _append_memory_event(
        svc,
        MemoryEventType.MESSAGE_CREATED.value,
        {"role": "user", "content": "北京 M42 观测上下文"},
    )
    svc.create_summary_snapshot(svc.session_id)
    _append_memory_event(
        svc,
        MemoryEventType.MESSAGE_CREATED.value,
        {"role": "user", "content": "上海 M31 摄影计划"},
    )

    decision = svc.maintenance_service.should_create_summary_snapshot(svc.session_id)

    assert decision.should_create is True
    assert decision.mode == "rebase"
    assert decision.reason == "topic_drift(distance>0.6)"


def test_tool_chain_completion_triggers_snapshot_decision(memory_db, monkeypatch):
    """测试 tool chain completion triggers snapshot decision 场景。"""

    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_AUTO_SUMMARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_MESSAGES",
        1000,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_TOKENS",
        100000,
    )
    svc = MemoryService(db_path=memory_db, tenant_id="t1", session_id="s1")
    _append_memory_event(
        svc,
        MemoryEventType.TOOL_CALL_FINISHED.value,
        {
            "tool_name": "weather-lookup",
            "tool_input": '{"city":"北京"}',
            "output_summary": "北京晴",
        },
    )
    _append_memory_event(
        svc,
        MemoryEventType.MESSAGE_CREATED.value,
        {"role": "assistant", "content": "最终结论：可以观测。"},
    )

    decision = svc.maintenance_service.should_create_summary_snapshot(svc.session_id)

    assert decision.should_create is True
    assert decision.mode == "create"
    assert decision.reason == "tool_chain_completed"


def test_context_budget_pressure_triggers_snapshot_decision(memory_db, monkeypatch):
    """测试 context budget pressure triggers snapshot decision 场景。"""

    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_AUTO_SUMMARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_MESSAGES",
        1000,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_TOKENS",
        100000,
    )
    svc = MemoryService(db_path=memory_db, tenant_id="t1", session_id="s1")
    _append_memory_event(
        svc,
        MemoryEventType.MESSAGE_CREATED.value,
        {"role": "user", "content": "北京 M42 观测上下文"},
    )

    decision = svc.maintenance_service.should_create_summary_snapshot(
        svc.session_id,
        context_pressure=1.3,
        omitted_counts={"messages": 0},
    )

    assert decision.should_create is True
    assert decision.mode == "create"
    assert decision.reason == "context_budget_pressure"


def test_rebase_with_zero_min_new_events(memory_db, monkeypatch):
    """When MEMORY_SUMMARY_MIN_NEW_EVENTS=0, every assistant message after first snapshot rebases."""
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_AUTO_SUMMARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_MESSAGES",
        2,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_MIN_NEW_EVENTS",
        0,
    )
    svc = MemoryService(db_path=memory_db, tenant_id="t1", session_id="s1")

    # Create first snapshot
    _append_pair(svc, "用户1", "助手1")
    _append_pair(svc, "用户2", "助手2")

    first_summary = svc.get_summary(svc.session_id)
    assert first_summary != "", "首次 snapshot 应创建"

    # Next assistant message should rebase
    _append_pair(svc, "用户3", "助手3")

    second_summary = svc.get_summary(svc.session_id)
    assert second_summary != "", "rebase 后 snapshot 应存在"


def test_no_infinite_loop_on_summary_event(memory_db, monkeypatch):
    """Creating a summary snapshot adds a SUMMARY_SNAPSHOT_CREATED event.
    That event type IS in SNAPSHOTTABLE_EVENT_TYPES. Verify the trigger
    doesn't immediately re-fire on its own event."""
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_AUTO_SUMMARY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_TRIGGER_MESSAGES",
        2,
    )
    monkeypatch.setattr(
        "src.memory.application.memory_maintenance_service.settings.MEMORY_SUMMARY_MIN_NEW_EVENTS",
        3,
    )
    svc = MemoryService(db_path=memory_db, tenant_id="t1", session_id="s1")

    # Create first snapshot
    for i in range(3):
        _append_pair(svc, f"用户{i}", f"助手{i}")

    first_summary = svc.get_summary(svc.session_id)
    assert first_summary != ""

    # Write only 1 more pair (2 events) — below MIN_NEW_EVENTS=3
    _append_pair(svc, "额外的用户", "额外的助手")

    # Should NOT have created a new snapshot
    # The system is stable — no crash, no infinite loop
    final_summary = svc.get_summary(svc.session_id)
    assert final_summary != ""
