"""ReactExecutor — context-first ReAct executor."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from src.agent.models.execution_event import ExecutionEvent
from src.agent.models.execution_trace_entry import ExecutionTraceEntry
from src.agent.models.final_response import FinalResponse
from src.agent.output_parser import extract_final_answer_text
from src.agent.execution.react_trace_adapter import ReactToolTraceAdapter


class ReactExecutor:
    """React 执行器。"""

    def __init__(
        self,
        agent_executor: Optional[Any] = None,
        agent_executor_factory: Optional[Callable[[], Any]] = None,
        trace_adapter: Optional[ReactToolTraceAdapter] = None,
    ) -> None:
        self._agent_executor = agent_executor
        self._agent_executor_factory = agent_executor_factory
        self._trace_adapter = trace_adapter or ReactToolTraceAdapter()

    def ensure_executor(self) -> Any:
        if self._agent_executor is not None:
            return self._agent_executor
        if self._agent_executor_factory is None:
            raise ValueError("react agent executor is not configured")
        self._agent_executor = self._agent_executor_factory()
        return self._agent_executor

    @staticmethod
    def build_agent_input(
        query: str,
        *,
        chat_history: str = "",
        user_profile: str = "",
    ) -> Dict[str, str]:
        return {
            "input": query,
            "chat_history": chat_history,
            "user_profile": user_profile,
        }

    async def run_context(self, context: Any) -> FinalResponse:
        """Context-first react non-streaming execution entry.

        优先复用 executor.invoke()；若底层只支持流式，则退化为聚合 astream_events()。
        """
        agent_input = self.build_agent_input(
            context.query,
            chat_history=context.chat_history,
            user_profile=context.user_profile,
        )
        executor = self.ensure_executor()
        route_meta = context.profile.to_legacy_route_meta()

        if hasattr(executor, "invoke"):
            result = await asyncio.to_thread(executor.invoke, agent_input)
            return self._final_response_from_invoke(route_meta, result)

        return await self._final_response_from_stream(route_meta, agent_input)

    async def astream_events(
        self,
        agent_input: Dict[str, Any],
        *,
        version: str = "v1",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """代理 agent_executor.astream_events()，使 react 拥有独立入口。"""
        executor = self.ensure_executor()
        async for event in executor.astream_events(agent_input, version=version):
            yield event

    def _final_response_from_invoke(
        self,
        route_meta: Dict[str, Any],
        result: Any,
    ) -> FinalResponse:
        payload = result if isinstance(result, dict) else {}
        output = ""
        if isinstance(payload, dict):
            output = str(payload.get("output", "") or "")
        elif result is not None:
            output = str(result)

        answer = self._recover_answer_text(output)
        if not answer and output:
            answer = output.strip()

        execution_trace = self._trace_from_intermediate_steps(
            payload.get("intermediate_steps") if isinstance(payload, dict) else None
        )
        tools_used = [
            self._trace_adapter.tool_usage_from_entry(entry)
            for entry in execution_trace
        ]
        execution_events = self._events_from_trace(execution_trace)
        execution_events.append(
            ExecutionEvent(
                type="answer_ready",
                payload={
                    "answer": answer,
                    "summary": answer,
                    "route": route_meta.get("route", ""),
                    "task_type": route_meta.get("task_type", ""),
                },
                source="react",
            ).to_dict()
        )

        trace_payload = [entry.to_dict() for entry in execution_trace]
        return FinalResponse(
            answer=answer,
            summary=answer,
            tools_used=tools_used,
            execution_trace=trace_payload,
            route=str(route_meta.get("route", "")),
            task_type=str(route_meta.get("task_type", "")),
            route_decision=route_meta,
            execution_events=execution_events,
            audit_metadata=self._build_react_audit_metadata(trace_payload),
        )

    async def _final_response_from_stream(
        self,
        route_meta: Dict[str, Any],
        agent_input: Dict[str, Any],
    ) -> FinalResponse:
        chunks: List[str] = []
        tool_runs: Dict[str, Dict[str, Any]] = {}
        execution_trace: List[Dict[str, Any]] = []
        tools_used: List[Dict[str, Any]] = []

        execution_events: List[Dict[str, Any]] = []

        async for event in self.astream_events(agent_input, version="v1"):
            event_type = event.get("event")
            data = event.get("data", {}) or {}
            run_id = event.get("run_id") or f"react_{len(execution_trace) + 1}"

            if event_type == "on_tool_start":
                tool_runs[run_id] = {
                    "name": data.get("name") or data.get("tool") or "unknown_tool",
                    "input": data.get("input"),
                    "start_time": time.perf_counter(),
                }
                continue

            if event_type == "on_tool_end":
                meta = tool_runs.pop(run_id, {})
                tool_name = (
                    meta.get("name")
                    or data.get("name")
                    or data.get("tool")
                    or "unknown_tool"
                )
                tool_input = meta.get("input", "")
                tool_output = data.get("output")
                duration_sec = None
                if meta.get("start_time") is not None:
                    duration_sec = time.perf_counter() - meta["start_time"]
                trace_entry = self._trace_adapter.build_entry(
                    step_id=run_id,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output=tool_output,
                    duration_sec=duration_sec,
                )
                execution_trace.append(trace_entry.to_dict())
                tools_used.append(
                    self._trace_adapter.tool_usage_from_entry(trace_entry)
                )
                execution_events.extend(
                    event.to_dict()
                    for event in trace_entry.to_execution_events(source="react")
                )
                continue

            if event_type not in ("on_chat_model_stream", "on_llm_stream"):
                continue

            chunk = data.get("chunk")
            text = getattr(chunk, "content", None) or getattr(chunk, "text", None)
            if text:
                chunks.append(str(text))

        raw_text = "".join(chunks)
        answer = self._recover_answer_text(raw_text)

        execution_events.append(
            ExecutionEvent(
                type="answer_ready",
                payload={
                    "answer": answer,
                    "summary": answer,
                    "route": route_meta.get("route", ""),
                    "task_type": route_meta.get("task_type", ""),
                },
                source="react",
            ).to_dict()
        )

        return FinalResponse(
            answer=answer,
            summary=answer,
            tools_used=tools_used,
            execution_trace=execution_trace,
            execution_events=execution_events,
            route=str(route_meta.get("route", "")),
            task_type=str(route_meta.get("task_type", "")),
            route_decision=route_meta,
            audit_metadata=self._build_react_audit_metadata(execution_trace),
        )

    def _trace_from_intermediate_steps(
        self,
        intermediate_steps: Any,
    ) -> List[ExecutionTraceEntry]:
        entries: List[ExecutionTraceEntry] = []
        if not isinstance(intermediate_steps, list):
            return entries

        for index, item in enumerate(intermediate_steps, start=1):
            action, observation = self._split_intermediate_step(item)
            if action is None:
                continue
            tool_name = self._action_tool_name(action)
            entries.append(
                self._trace_adapter.build_entry(
                    step_id=f"react_{index}",
                    tool_name=tool_name,
                    tool_input=self._action_tool_input(action),
                    tool_output=observation,
                )
            )
        return entries

    @staticmethod
    def _split_intermediate_step(item: Any) -> tuple[Any, Any]:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            return item[0], item[1]
        if isinstance(item, dict):
            return item.get("action") or item.get("agent_action"), item.get(
                "observation"
            )
        return None, None

    @staticmethod
    def _action_tool_name(action: Any) -> str:
        if isinstance(action, dict):
            return str(action.get("tool") or action.get("name") or "unknown_tool")
        return str(
            getattr(action, "tool", None)
            or getattr(action, "name", None)
            or "unknown_tool"
        )

    @staticmethod
    def _action_tool_input(action: Any) -> Any:
        if isinstance(action, dict):
            return action.get("tool_input", action.get("input", ""))
        return getattr(action, "tool_input", getattr(action, "input", ""))

    @staticmethod
    def _events_from_trace(
        execution_trace: List[ExecutionTraceEntry],
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for entry in execution_trace:
            events.extend(
                event.to_dict() for event in entry.to_execution_events(source="react")
            )
        return events

    @staticmethod
    def _build_react_audit_metadata(
        execution_trace: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return ReactToolTraceAdapter.summarize_audit(execution_trace)

    @staticmethod
    def _preview_text(value: Any, max_len: int) -> str:
        text = "" if value is None else str(value)
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    @staticmethod
    def _recover_answer_text(text: str) -> str:
        combined = (text or "").strip()
        if not combined:
            return ""

        extracted = extract_final_answer_text(combined)
        if extracted:
            return extracted.strip()

        # 兼容未显式携带 Final Answer 标签、但最后一行已是最终答案的 react 输出。
        stripped = re.sub(r"(?is)^thought:\s*", "", combined).strip()
        if "\n" in stripped:
            candidate = stripped.splitlines()[-1].strip()
            if candidate:
                return candidate
        return stripped
