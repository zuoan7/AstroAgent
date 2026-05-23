"""Phase 7 统一 ExecutionTrace / ExecutionEvent 测试

目标：
1. ExecutionTraceEntry 能从 StepExecutionResult / dict / react_tool 构造
2. StepExecutionResult.to_trace_entry() 可用
3. ExecutionEvent 内部类型 <-> 前端事件名映射正确
4. FrontendExecutionEventAdapter 对 planned 路径正确产出旧前端事件序列
5. 旧前端事件名（plan_update / step_start / step_end / evidence_found）不变
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()

from src.agent.frontend_event_adapter import FrontendExecutionEventAdapter
from src.agent.models.execution_trace_entry import ExecutionTraceEntry
from src.agent.models.execution_event import ExecutionEvent, EXECUTION_EVENT_TYPES
from src.agent.streaming_events import StreamEvent
from src.agent.executor import StepExecutionResult

# ─────────────────────────────────────────────────────────────────
# ExecutionTraceEntry 测试
# ─────────────────────────────────────────────────────────────────


class TestExecutionTraceEntry:

    def _make_step_result(self, status: str = "success") -> StepExecutionResult:
        return StepExecutionResult(
            step_id="s1",
            title="天气查询",
            kind="tool",
            status=status,
            skill="weather-lookup",
            input_params={"query": "北京"},
            attempts=1,
            required=True,
            latency_ms=123.4,
            error=None if status == "success" else "mock error",
            summary="天气晴",
            sources=[
                {"source_id": "x", "kind": "tool_output", "title": "t", "snippet": "s"}
            ],
        )

    def test_from_step_result_fields(self):
        sr = self._make_step_result()
        entry = ExecutionTraceEntry.from_step_result(sr)
        assert entry.step_id == "s1"
        assert entry.title == "天气查询"
        assert entry.skill == "weather-lookup"
        assert entry.status == "success"
        assert entry.latency_ms == 123.4
        assert len(entry.sources) == 1

    def test_from_step_result_error(self):
        sr = self._make_step_result(status="error")
        entry = ExecutionTraceEntry.from_step_result(sr)
        assert entry.status == "error"
        assert entry.error == "mock error"

    def test_to_trace_entry_on_step_result(self):
        sr = self._make_step_result()
        entry = sr.to_trace_entry()
        assert isinstance(entry, ExecutionTraceEntry)
        assert entry.step_id == sr.step_id

    def test_from_dict_roundtrip(self):
        sr = self._make_step_result()
        entry = ExecutionTraceEntry.from_step_result(sr)
        d = entry.to_dict()
        recovered = ExecutionTraceEntry.from_dict(d)
        assert recovered.step_id == entry.step_id
        assert recovered.skill == entry.skill
        assert recovered.latency_ms == entry.latency_ms

    def test_from_dict_partial(self):
        """旧 execution_trace dict 字段不全也能恢复。"""
        d = {"step_id": "x", "title": "X", "kind": "tool", "status": "success"}
        entry = ExecutionTraceEntry.from_dict(d)
        assert entry.step_id == "x"
        assert entry.sources == []
        assert entry.required is True  # 默认值

    def test_from_react_tool(self):
        entry = ExecutionTraceEntry.from_react_tool(
            step_id="run_abc",
            tool_name="weather-lookup",
            tool_input='{"location": "北京"}',
            output_summary="晴天",
            status="success",
            duration_sec=0.5,
        )
        assert entry.kind == "tool"
        assert entry.tool_name == "weather-lookup"
        assert entry.duration_sec == 0.5
        assert entry.status == "success"

    def test_to_dict_has_all_keys(self):
        sr = self._make_step_result()
        entry = ExecutionTraceEntry.from_step_result(sr)
        d = entry.to_dict()
        assert "step_id" in d
        assert "title" in d
        assert "status" in d
        assert "skill" in d
        assert "sources" in d
        assert "latency_ms" in d

    def test_step_trace_to_execution_events(self):
        entry = ExecutionTraceEntry.from_step_result(self._make_step_result())
        events = entry.to_execution_events(source="planned")
        assert [event.type for event in events] == ["step_started", "step_finished"]
        assert events[0].payload["step_id"] == "s1"
        assert events[0].payload["logical_skill"] == "weather-lookup"
        assert events[1].payload["status"] == "success"
        assert events[1].payload["logical_skill"] == "weather-lookup"

    def test_react_trace_to_execution_events(self):
        entry = ExecutionTraceEntry.from_react_tool(
            step_id="run_abc",
            tool_name="weather-lookup",
            tool_input="北京",
            output_summary="晴天",
            status="success",
            duration_sec=0.5,
        )
        events = entry.to_execution_events(source="react")
        assert [event.type for event in events] == ["tool_called", "tool_result"]
        assert events[0].payload["tool"] == "weather-lookup"
        assert events[1].payload["output_summary"] == "晴天"


# ─────────────────────────────────────────────────────────────────
# ExecutionEvent 测试
# ─────────────────────────────────────────────────────────────────


class TestExecutionEvent:

    def test_known_type_constants(self):
        for t in (
            "task_profile",
            "route_decided",
            "execution_decision",
            "plan_built",
            "plan_created",
            "step_started",
            "step_finished",
            "tool_called",
            "tool_result",
            "fallback_triggered",
            "answer_ready",
            "final_answer",
            "tool_returned",
        ):
            assert t in EXECUTION_EVENT_TYPES

    def test_frontend_type_mapping_all(self):
        mappings = {
            "route_decided": "route_decision",
            "plan_built": "plan_update",
            "plan_created": "plan_update",
            "step_started": "step_start",
            "step_finished": "step_end",
            "answer_ready": "final_answer",
            "final_answer": "final_answer",
            "tool_called": "tool_start",
            "tool_result": "tool_end",
            "tool_returned": "tool_end",
        }
        for internal, expected_frontend in mappings.items():
            ev = ExecutionEvent(type=internal, payload={}, source="planned")
            assert ev.to_frontend_type() == expected_frontend

    def test_unknown_type_returns_none(self):
        ev = ExecutionEvent(type="custom_event", payload={})
        assert ev.to_frontend_type() is None

    def test_to_dict(self):
        ev = ExecutionEvent(
            type="step_started", payload={"step_id": "s1"}, source="planned"
        )
        d = ev.to_dict()
        assert d["type"] == "step_started"
        assert d["payload"]["step_id"] == "s1"
        assert d["source"] == "planned"


# ─────────────────────────────────────────────────────────────────
# FrontendExecutionEventAdapter trace 映射测试
# ─────────────────────────────────────────────────────────────────


class TestEmitTraceEvents:
    """验证 adapter 产出事件序列与旧前端事件名兼容。"""

    def _make_adapter(self):
        return FrontendExecutionEventAdapter()

    def _make_trace_dict(self, status: str = "success") -> dict:
        return {
            "step_id": "s1",
            "title": "天气查询",
            "kind": "tool",
            "status": status,
            "skill": "weather-lookup",
            "input_params": {"query": "北京"},
            "latency_ms": 100.0,
            "error": None if status == "success" else "mock err",
            "summary": "晴天",
            "sources": [
                {"source_id": "x", "kind": "tool_output", "title": "t", "snippet": "s"}
            ],
        }

    def _collect_events(self, trace_dict: dict, plan_steps: list) -> list:
        adapter = self._make_adapter()
        evidence_items = []
        tool_timeline = []
        collected = []

        sequence_counter = [0]

        def next_event_fn(event_type, *, content=None, meta=None, modality="text"):
            sequence_counter[0] += 1
            return StreamEvent(
                type=event_type,
                content=content,
                meta=meta or {"request_id": "test"},
                sequence=sequence_counter[0],
            )

        async def emit_fn(event):
            yield event

        async def run():
            async for ev in adapter.emit_trace_events(
                trace_dict,
                plan_steps=plan_steps,
                evidence_items=evidence_items,
                tool_timeline=tool_timeline,
                next_event_fn=next_event_fn,
                emit_fn=emit_fn,
                preview_text_fn=lambda value, max_len: str(value)[:max_len],
            ):
                collected.append(ev)

        asyncio.run(run())
        return collected, evidence_items, tool_timeline

    def test_event_types_order(self):
        """事件顺序：plan_update -> step_start -> evidence_found -> step_end -> plan_update。"""
        plan_steps = [{"id": "s1", "title": "天气查询", "status": "pending"}]
        trace = self._make_trace_dict()
        events, _, _ = self._collect_events(trace, plan_steps)

        types = [e.type for e in events]
        assert types[0] == "plan_update"
        assert types[1] == "step_start"
        assert "evidence_found" in types
        assert "step_end" in types
        assert types[-1] == "plan_update"

    def test_step_start_content(self):
        plan_steps = [{"id": "s1", "title": "天气查询", "status": "pending"}]
        trace = self._make_trace_dict()
        events, _, _ = self._collect_events(trace, plan_steps)

        step_start = next(e for e in events if e.type == "step_start")
        assert step_start.content["step_id"] == "s1"
        assert step_start.content["title"] == "天气查询"

    def test_step_end_content_success(self):
        plan_steps = [{"id": "s1", "title": "天气查询", "status": "pending"}]
        trace = self._make_trace_dict(status="success")
        events, _, _ = self._collect_events(trace, plan_steps)

        step_end = next(e for e in events if e.type == "step_end")
        assert step_end.content["step_id"] == "s1"
        assert step_end.content["status"] == "success"

    def test_step_end_content_error(self):
        plan_steps = [{"id": "s1", "title": "天气查询", "status": "pending"}]
        trace = self._make_trace_dict(status="error")
        events, _, _ = self._collect_events(trace, plan_steps)

        step_end = next(e for e in events if e.type == "step_end")
        assert step_end.content["status"] == "error"
        assert step_end.content["error"] == "mock err"

    def test_evidence_found_emitted(self):
        plan_steps = [{"id": "s1", "title": "天气查询", "status": "pending"}]
        trace = self._make_trace_dict()
        events, evidence_items, _ = self._collect_events(trace, plan_steps)

        ev_found = [e for e in events if e.type == "evidence_found"]
        assert len(ev_found) == 1
        assert len(evidence_items) == 1

    def test_tool_timeline_populated(self):
        plan_steps = [{"id": "s1", "title": "天气查询", "status": "pending"}]
        trace = self._make_trace_dict()
        _, _, tool_timeline = self._collect_events(trace, plan_steps)

        assert len(tool_timeline) == 1
        assert tool_timeline[0]["tool"] == "weather-lookup"
        assert tool_timeline[0]["display_tool"] == "weather-lookup"
        assert tool_timeline[0]["logical_skill"] == "weather-lookup"
        assert tool_timeline[0]["status"] == "success"

    def test_plan_steps_status_updated(self):
        plan_steps = [{"id": "s1", "title": "天气查询", "status": "pending"}]
        trace = self._make_trace_dict(status="success")
        self._collect_events(trace, plan_steps)

        assert plan_steps[0]["status"] == "done"

    def test_plan_steps_status_error(self):
        plan_steps = [{"id": "s1", "title": "天气查询", "status": "pending"}]
        trace = self._make_trace_dict(status="error")
        self._collect_events(trace, plan_steps)

        assert plan_steps[0]["status"] == "error"

    def test_accepts_trace_entry_object(self):
        """也接受 ExecutionTraceEntry 对象（非 dict）。"""
        plan_steps = [{"id": "s1", "title": "天气查询", "status": "pending"}]
        sr = StepExecutionResult(
            step_id="s1",
            title="天气查询",
            kind="tool",
            status="success",
            skill="weather-lookup",
            input_params={},
            attempts=1,
        )
        entry = ExecutionTraceEntry.from_step_result(sr)
        adapter = self._make_adapter()
        collected = []
        sequence_counter = [0]

        def next_event_fn(event_type, *, content=None, meta=None, modality="text"):
            sequence_counter[0] += 1
            return StreamEvent(
                type=event_type,
                content=content,
                meta=meta or {"request_id": "test"},
                sequence=sequence_counter[0],
            )

        async def emit_fn(event):
            yield event

        async def run():
            async for ev in adapter.emit_trace_events(
                entry,
                plan_steps=plan_steps,
                evidence_items=[],
                tool_timeline=[],
                next_event_fn=next_event_fn,
                emit_fn=emit_fn,
                preview_text_fn=lambda value, max_len: str(value)[:max_len],
            ):
                collected.append(ev)

        asyncio.run(run())
        types = [e.type for e in collected]
        assert "step_start" in types
        assert "step_end" in types


class TestEmitExecutionEvent:
    def _make_adapter(self):
        return FrontendExecutionEventAdapter()

    def test_route_decided_maps_to_route_decision(self):
        adapter = self._make_adapter()
        emitted = []

        def next_event_fn(event_type, *, content=None, meta=None, modality="text"):
            return StreamEvent(
                type=event_type,
                content=content,
                meta=meta or {"request_id": "test"},
                sequence=1,
            )

        async def emit_fn(event):
            yield event

        async def run():
            async for event in adapter.emit_execution_event(
                ExecutionEvent(
                    type="route_decided",
                    payload={"route": "direct_task", "task_type": "smalltalk"},
                    source="router",
                ),
                next_event_fn=next_event_fn,
                emit_fn=emit_fn,
            ):
                emitted.append(event)

        asyncio.run(run())
        assert len(emitted) == 1
        assert emitted[0].type == "route_decision"

    def test_plan_built_maps_to_plan_update(self):
        adapter = self._make_adapter()
        emitted = []

        def next_event_fn(event_type, *, content=None, meta=None, modality="text"):
            return StreamEvent(
                type=event_type,
                content=content,
                meta=meta or {"request_id": "test"},
                sequence=1,
            )

        async def emit_fn(event):
            yield event

        async def run():
            async for event in adapter.emit_execution_event(
                ExecutionEvent(
                    type="plan_built",
                    payload={"steps": [{"id": "s1", "status": "pending"}]},
                    source="planned",
                ),
                next_event_fn=next_event_fn,
                emit_fn=emit_fn,
            ):
                emitted.append(event)

        asyncio.run(run())
        assert len(emitted) == 1
        assert emitted[0].type == "plan_update"

    def test_plan_created_with_plan_payload_maps_to_plan_update(self):
        adapter = self._make_adapter()
        emitted = []

        def next_event_fn(event_type, *, content=None, meta=None, modality="text"):
            return StreamEvent(
                type=event_type,
                content=content,
                meta=meta or {"request_id": "test"},
                sequence=1,
            )

        async def emit_fn(event):
            yield event

        async def run():
            async for event in adapter.emit_response_execution_events(
                MagicMock(
                    execution_events=[
                        ExecutionEvent(
                            type="plan_created",
                            payload={
                                "plan": {"steps": [{"id": "s1", "status": "pending"}]}
                            },
                            source="planned",
                        ).to_dict()
                    ]
                ),
                plan_steps=[],
                evidence_items=[],
                tool_timeline=[],
                next_event_fn=next_event_fn,
                emit_fn=emit_fn,
            ):
                emitted.append(event)

        asyncio.run(run())
        assert len(emitted) == 1
        assert emitted[0].type == "plan_update"
