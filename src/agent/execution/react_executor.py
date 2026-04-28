"""ReactExecutor — ReAct 任务执行器（Phase 4 引入）。

React 执行逻辑的独立入口，解耦其与 StreamingService 的强绑定。
当前同时提供：
1. 非流式 run()：返回统一 FinalResponse，供 ExecutionEngine.run() 主路径使用
2. 流式 astream_events()：保留给前端事件适配层消费
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from src.agent.models.execution_event import ExecutionEvent
from src.agent.models.execution_trace_entry import ExecutionTraceEntry
from src.agent.models.final_response import FinalResponse
from src.agent.output_parser import extract_final_answer_text
from src.agent.request_router import RouteDecision


class ReactExecutor:
    """React 执行器。

    非流式 run() 用于统一主执行入口；
    astream_events() 继续承担原始流式事件代理职责。
    """

    def __init__(
        self,
        agent_executor: Optional[Any] = None,
        agent_executor_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._agent_executor = agent_executor
        self._agent_executor_factory = agent_executor_factory

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

    async def run(
        self,
        decision: RouteDecision,
        query: str,
        *,
        chat_history: str = "",
        user_profile: str = "",
    ) -> FinalResponse:
        """统一的 react 非流式执行入口。

        优先复用 executor.invoke()；若底层只支持流式，则退化为聚合 astream_events()。
        """
        agent_input = self.build_agent_input(
            query,
            chat_history=chat_history,
            user_profile=user_profile,
        )
        executor = self.ensure_executor()

        if hasattr(executor, "invoke"):
            result = await asyncio.to_thread(executor.invoke, agent_input)
            return self._final_response_from_invoke(decision, result)

        return await self._final_response_from_stream(decision, agent_input)

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
        decision: RouteDecision,
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

        return FinalResponse(
            answer=answer,
            summary=answer,
            route=decision.route,
            task_type=decision.task_type,
            route_decision=decision.to_meta() if hasattr(decision, "to_meta") else None,
            execution_events=[
                ExecutionEvent(
                    type="answer_ready",
                    payload={
                        "answer": answer,
                        "summary": answer,
                        "route": decision.route,
                        "task_type": decision.task_type,
                    },
                    source="react",
                ).to_dict()
            ],
        )

    async def _final_response_from_stream(
        self,
        decision: RouteDecision,
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
                    "input": "" if data.get("input") is None else str(data.get("input")),
                    "start_time": time.perf_counter(),
                }
                execution_events.append(
                    ExecutionEvent(
                        type="tool_called",
                        payload={
                            "run_id": run_id,
                            "tool": tool_runs[run_id]["name"],
                            "input": tool_runs[run_id]["input"],
                            "status": "running",
                        },
                        source="react",
                    ).to_dict()
                )
                continue

            if event_type == "on_tool_end":
                meta = tool_runs.pop(run_id, {})
                tool_name = meta.get("name") or data.get("name") or data.get("tool") or "unknown_tool"
                tool_input = meta.get("input", "")
                tool_output = "" if data.get("output") is None else str(data.get("output"))
                duration_sec = None
                if meta.get("start_time") is not None:
                    duration_sec = time.perf_counter() - meta["start_time"]
                status = "error" if "error" in tool_output.lower() else "success"
                tools_used.append(
                    {
                        "run_id": run_id,
                        "tool": tool_name,
                        "input": tool_input,
                        "output_summary": self._preview_text(tool_output, 240),
                        "duration_sec": duration_sec,
                        "status": status,
                    }
                )
                execution_trace.append(
                    ExecutionTraceEntry.from_react_tool(
                        step_id=run_id,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        output_summary=self._preview_text(tool_output, 240),
                        status=status,
                        duration_sec=duration_sec,
                    ).to_dict()
                )
                execution_events.append(
                    ExecutionEvent(
                        type="tool_result",
                        payload={
                            "run_id": run_id,
                            "tool": tool_name,
                            "output_summary": self._preview_text(tool_output, 240),
                            "status": status,
                            "duration_sec": duration_sec,
                        },
                        source="react",
                    ).to_dict()
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
                    "route": decision.route,
                    "task_type": decision.task_type,
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
            route=decision.route,
            task_type=decision.task_type,
            route_decision=decision.to_meta() if hasattr(decision, "to_meta") else None,
        )

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
