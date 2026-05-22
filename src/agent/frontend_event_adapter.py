from __future__ import annotations

from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from src.agent.models.execution_event import ExecutionEvent
from src.agent.models.execution_trace_entry import ExecutionTraceEntry
from src.agent.models.final_response import FinalResponse


def _update_status(steps: List[Dict[str, Any]], step_id: str, status: str) -> None:
    for step in steps:
        if step.get("id") == step_id:
            step["status"] = status
            return


class FrontendExecutionEventAdapter:
    """将统一执行事件/trace 适配为旧前端流事件。"""

    FRONTEND_EVENT_TYPE_MAP = {
        "route_decided": "route_decision",
        "plan_built": "plan_update",
        "plan_created": "plan_update",
        "plan_repaired": "plan_update",
        "step_started": "step_start",
        "step_finished": "step_end",
        "answer_ready": "final_answer",
        "final_answer": "final_answer",
        "tool_called": "tool_start",
        "tool_result": "tool_end",
        "tool_returned": "tool_end",
    }

    def to_execution_event(self, event: Any, *, source: str = "") -> ExecutionEvent:
        if isinstance(event, ExecutionEvent):
            return event
        if isinstance(event, dict):
            return ExecutionEvent(
                type=str(event.get("type", "")),
                payload=dict(event.get("payload", {}) or {}),
                source=source or str(event.get("source", "") or ""),
            )
        raise TypeError(f"unsupported execution event payload: {type(event)!r}")

    def to_frontend_event_type(self, event: ExecutionEvent) -> Optional[str]:
        return self.FRONTEND_EVENT_TYPE_MAP.get(event.type)

    async def emit_execution_event(
        self,
        event: Any,
        *,
        next_event_fn: Callable[..., Any],
        emit_fn: Callable[[Any], AsyncGenerator[Any, None]],
        source: str = "",
    ) -> AsyncGenerator[Any, None]:
        execution_event = self.to_execution_event(event, source=source)
        frontend_type = self.to_frontend_event_type(execution_event)
        if not frontend_type:
            return
        async for processed in emit_fn(
            next_event_fn(frontend_type, content=execution_event.payload)
        ):
            yield processed

    def iter_response_execution_events(
        self,
        response: FinalResponse,
    ) -> List[ExecutionEvent]:
        return [
            self.to_execution_event(event)
            for event in (getattr(response, "execution_events", []) or [])
        ]

    async def emit_response_execution_events(
        self,
        response: FinalResponse,
        *,
        plan_steps: List[Dict[str, Any]],
        evidence_items: List[Dict[str, Any]],
        tool_timeline: List[Dict[str, Any]],
        next_event_fn: Callable[..., Any],
        emit_fn: Callable[[Any], AsyncGenerator[Any, None]],
        include_answer_ready: bool = False,
    ) -> AsyncGenerator[Any, None]:
        for event in self.iter_response_execution_events(response):
            if event.type in {
                "task_profile",
                "execution_decision",
                "fallback_triggered",
            }:
                continue
            if event.type == "route_decided":
                continue
            if (
                event.type in {"answer_ready", "final_answer"}
                and not include_answer_ready
            ):
                continue
            if event.type in {"plan_built", "plan_created", "plan_repaired"}:
                plan_steps_payload = self._extract_plan_steps_payload(event)
                if isinstance(plan_steps_payload, list):
                    async for processed in self.emit_execution_event(
                        ExecutionEvent(
                            type="plan_built",
                            payload={"steps": plan_steps_payload},
                            source=event.source,
                        ),
                        next_event_fn=next_event_fn,
                        emit_fn=emit_fn,
                    ):
                        yield processed
                continue
            if event.type == "step_started":
                step_id = str(event.payload.get("step_id", ""))
                if step_id:
                    _update_status(plan_steps, step_id, "running")
                async for processed in self.emit_execution_event(
                    event,
                    next_event_fn=next_event_fn,
                    emit_fn=emit_fn,
                ):
                    yield processed
                continue
            if event.type == "step_finished":
                step_id = str(event.payload.get("step_id", ""))
                status = str(event.payload.get("status", ""))
                if step_id:
                    _update_status(
                        plan_steps,
                        step_id,
                        "done" if status == "success" else "error",
                    )
                skill_name = event.payload.get("skill")
                if skill_name:
                    logical_skill = event.payload.get("logical_skill") or skill_name
                    tool_timeline.append(
                        {
                            "run_id": step_id,
                            "tool": skill_name,
                            "display_tool": skill_name,
                            "logical_skill": logical_skill,
                            "input": event.payload.get("input", {}),
                            "output_summary": event.payload.get("output_summary", ""),
                            "latency_ms": event.payload.get("latency_ms"),
                            "status": status,
                            "param_builder_source": event.payload.get(
                                "param_builder_source", ""
                            ),
                            "mcp_tools_used": event.payload.get("mcp_tools_used", []),
                            "expected_mcp_tools": event.payload.get(
                                "expected_mcp_tools", []
                            ),
                            "operation": event.payload.get("operation"),
                        }
                    )
                async for processed in self.emit_execution_event(
                    event,
                    next_event_fn=next_event_fn,
                    emit_fn=emit_fn,
                ):
                    yield processed
                continue
            if event.type in {"tool_called", "tool_result", "tool_returned"}:
                payload = dict(event.payload)
                tool_name = payload.get("tool")
                display_tool = payload.get("display_tool") or tool_name
                logical_skill = payload.get("logical_skill")
                if event.type == "tool_called":
                    tool_timeline.append(
                        {
                            "run_id": payload.get("run_id"),
                            "tool": tool_name,
                            "display_tool": display_tool,
                            "logical_skill": logical_skill,
                            "input": payload.get("input", ""),
                            "status": "running",
                            "mcp_tools_used": payload.get("mcp_tools_used", []),
                            "expected_mcp_tools": payload.get("expected_mcp_tools", []),
                            "operation": payload.get("operation"),
                        }
                    )
                else:
                    tool_timeline.append(
                        {
                            "run_id": payload.get("run_id"),
                            "tool": tool_name,
                            "display_tool": display_tool,
                            "logical_skill": logical_skill,
                            "output_summary": payload.get("output_summary", ""),
                            "duration_sec": payload.get("duration_sec"),
                            "status": payload.get("status"),
                            "mcp_tools_used": payload.get("mcp_tools_used", []),
                            "expected_mcp_tools": payload.get("expected_mcp_tools", []),
                            "operation": payload.get("operation"),
                        }
                    )
                async for processed in self.emit_execution_event(
                    event,
                    next_event_fn=next_event_fn,
                    emit_fn=emit_fn,
                ):
                    yield processed
                continue
            async for processed in self.emit_execution_event(
                event,
                next_event_fn=next_event_fn,
                emit_fn=emit_fn,
            ):
                yield processed

    async def emit_trace_events(
        self,
        trace_entry: Any,
        *,
        plan_steps: List[Dict[str, Any]],
        evidence_items: List[Dict[str, Any]],
        tool_timeline: List[Dict[str, Any]],
        next_event_fn: Callable[..., Any],
        emit_fn: Callable[[Any], AsyncGenerator[Any, None]],
        preview_text_fn: Callable[[Any, int], str],
    ) -> AsyncGenerator[Any, None]:
        if isinstance(trace_entry, dict):
            entry = ExecutionTraceEntry.from_dict(trace_entry)
        else:
            entry = trace_entry

        step_id = entry.step_id

        _update_status(plan_steps, step_id, "running")
        async for processed in self.emit_execution_event(
            ExecutionEvent(
                type="plan_built",
                payload={"steps": list(plan_steps)},
                source="planned",
            ),
            next_event_fn=next_event_fn,
            emit_fn=emit_fn,
        ):
            yield processed

        trace_events = entry.to_execution_events(source="planned")

        async for processed in self.emit_execution_event(
            trace_events[0],
            next_event_fn=next_event_fn,
            emit_fn=emit_fn,
        ):
            yield processed

        tool_timeline.append(
            {
                "run_id": step_id,
                "tool": entry.skill,
                "display_tool": entry.skill,
                "logical_skill": entry.logical_skill or entry.skill,
                "input": entry.input_params,
                "output_summary": preview_text_fn(entry.summary, 240),
                "latency_ms": entry.latency_ms,
                "status": entry.status,
                "param_builder_source": entry.param_builder_source,
                "mcp_tools_used": list(entry.mcp_tools_used),
                "expected_mcp_tools": list(entry.expected_mcp_tools),
                "operation": entry.operation,
            }
        )

        for source in entry.sources:
            if source not in evidence_items:
                evidence_items.append(source)
                async for processed in emit_fn(
                    next_event_fn("evidence_found", content=source)
                ):
                    yield processed

        mapped_status = "done" if entry.status == "success" else "error"
        _update_status(plan_steps, step_id, mapped_status)

        async for processed in self.emit_execution_event(
            trace_events[-1],
            next_event_fn=next_event_fn,
            emit_fn=emit_fn,
        ):
            yield processed

        async for processed in self.emit_execution_event(
            ExecutionEvent(
                type="plan_built",
                payload={"steps": list(plan_steps)},
                source="planned",
            ),
            next_event_fn=next_event_fn,
            emit_fn=emit_fn,
        ):
            yield processed

    @staticmethod
    def _extract_plan_steps_payload(
        event: ExecutionEvent,
    ) -> Optional[List[Dict[str, Any]]]:
        if isinstance(event.payload.get("steps"), list):
            return event.payload["steps"]
        plan = event.payload.get("plan")
        if isinstance(plan, dict) and isinstance(plan.get("steps"), list):
            return plan["steps"]
        return None
