"""Phase 7 统一 ExecutionTrace / ExecutionEvent 测试

目标：
1. ExecutionTraceEntry 能从 StepExecutionResult / dict / react_tool 构造
2. StepExecutionResult.to_trace_entry() 可用
3. ExecutionEvent 内部类型 <-> 前端事件名映射正确
4. StreamingService._emit_trace_events() 对 planned 路径正确产出旧前端事件序列
5. 旧前端事件名（plan_update / step_start / step_end / evidence_found）不变
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock

from tests.mock_deps import mock_heavy_dependencies

mock_heavy_dependencies()

from src.agent.models.execution_trace_entry import ExecutionTraceEntry
from src.agent.models.execution_event import ExecutionEvent, EXECUTION_EVENT_TYPES
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
            sources=[{"source_id": "x", "kind": "tool_output", "title": "t", "snippet": "s"}],
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


# ─────────────────────────────────────────────────────────────────
# ExecutionEvent 测试
# ─────────────────────────────────────────────────────────────────

class TestExecutionEvent:

    def test_known_type_constants(self):
        for t in ("route_decided", "plan_built", "step_started", "step_finished",
                  "answer_ready", "tool_called", "tool_returned"):
            assert t in EXECUTION_EVENT_TYPES

    def test_frontend_type_mapping_all(self):
        mappings = {
            "route_decided": "route_decision",
            "plan_built": "plan_update",
            "step_started": "step_start",
            "step_finished": "step_end",
            "answer_ready": "final_answer",
            "tool_called": "tool_start",
            "tool_returned": "tool_end",
        }
        for internal, expected_frontend in mappings.items():
            ev = ExecutionEvent(type=internal, payload={}, source="planned")
            assert ev.to_frontend_type() == expected_frontend

    def test_unknown_type_returns_none(self):
        ev = ExecutionEvent(type="custom_event", payload={})
        assert ev.to_frontend_type() is None

    def test_to_dict(self):
        ev = ExecutionEvent(type="step_started", payload={"step_id": "s1"}, source="planned")
        d = ev.to_dict()
        assert d["type"] == "step_started"
        assert d["payload"]["step_id"] == "s1"
        assert d["source"] == "planned"


# ─────────────────────────────────────────────────────────────────
# StreamingService._emit_trace_events() 测试
# ─────────────────────────────────────────────────────────────────

class TestEmitTraceEvents:
    """验证 _emit_trace_events 产出事件序列与旧前端事件名兼容。"""

    def _make_service(self):
        from src.agent.streaming_service import BaseStreamingGenerator
        svc = BaseStreamingGenerator.__new__(BaseStreamingGenerator)
        svc._event_processors = []
        return svc

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
            "sources": [{"source_id": "x", "kind": "tool_output", "title": "t", "snippet": "s"}],
        }

    def _collect_events(self, trace_dict: dict, plan_steps: list) -> list:
        svc = self._make_service()
        evidence_items = []
        tool_timeline = []
        collected = []

        sequence_counter = [0]

        def next_event_fn(event_type, *, content=None, meta=None, modality="text"):
            from src.agent.streaming_events import StreamEvent
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
            async for ev in svc._emit_trace_events(
                trace_dict,
                plan_steps=plan_steps,
                evidence_items=evidence_items,
                tool_timeline=tool_timeline,
                next_event_fn=next_event_fn,
                emit_fn=emit_fn,
            ):
                collected.append(ev)

        asyncio.get_event_loop().run_until_complete(run())
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
            step_id="s1", title="天气查询", kind="tool", status="success",
            skill="weather-lookup", input_params={}, attempts=1,
        )
        entry = ExecutionTraceEntry.from_step_result(sr)
        svc = self._make_service()
        collected = []
        sequence_counter = [0]

        def next_event_fn(event_type, *, content=None, meta=None, modality="text"):
            from src.agent.streaming_events import StreamEvent
            sequence_counter[0] += 1
            return StreamEvent(
                type=event_type, content=content,
                meta=meta or {"request_id": "test"}, sequence=sequence_counter[0]
            )

        async def emit_fn(event):
            yield event

        async def run():
            async for ev in svc._emit_trace_events(
                entry,
                plan_steps=plan_steps,
                evidence_items=[],
                tool_timeline=[],
                next_event_fn=next_event_fn,
                emit_fn=emit_fn,
            ):
                collected.append(ev)

        asyncio.get_event_loop().run_until_complete(run())
        types = [e.type for e in collected]
        assert "step_start" in types
        assert "step_end" in types
