#!/usr/bin/env python
"""Run a small live E2E latency probe and save per-turn stage timings.

This probe is intentionally narrower than the full benchmark runner. It is for
performance triage after routing/tool-layer changes: selected cases only,
sequential execution, raw SSE events saved for every setup and final turn.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import json
import math
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import httpx


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_DATASET = REPO_ROOT / "config/benchmarks/astro_agent_eval_dataset.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports/evaluation/astro_agent_latency"
DEFAULT_BASE_URL = "http://localhost:8002"
DEFAULT_CASE_IDS = [
    "control_001",
    "knowledge_001",
    "position_001",
    "event_003",
    "plan_001",
    "plan_023",
    "external_005",
    "memory_001",
    "memory_010",
]

_ACTIVE_RECORDER: StageRecorder | None = None


class StageRecorder:
    """Collects probe-only timings without changing production code."""

    def __init__(self) -> None:
        self._stages_ms: dict[str, float] = {}
        self._calls: list[dict[str, Any]] = []

    @contextlib.contextmanager
    def measure(
        self,
        stage_name: str,
        *,
        meta: dict[str, Any] | None = None,
    ):
        started = time.perf_counter()
        error = ""
        try:
            yield
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.record(stage_name, elapsed_ms, meta=meta, error=error)

    def record(
        self,
        stage_name: str,
        elapsed_ms: float,
        *,
        meta: dict[str, Any] | None = None,
        error: str = "",
    ) -> None:
        elapsed_ms = max(float(elapsed_ms), 0.0)
        self._stages_ms[stage_name] = self._stages_ms.get(stage_name, 0.0) + elapsed_ms
        call: dict[str, Any] = {
            "stage": stage_name,
            "elapsed_ms": round(elapsed_ms, 2),
        }
        if meta:
            call["meta"] = dict(meta)
        if error:
            call["error"] = error
        self._calls.append(call)

    def snapshot(self) -> dict[str, Any]:
        return {
            "stages_ms": {
                key: round(value, 2)
                for key, value in sorted(self._stages_ms.items())
            },
            "calls": list(self._calls),
        }


@contextlib.contextmanager
def active_recorder(recorder: StageRecorder):
    global _ACTIVE_RECORDER
    previous = _ACTIVE_RECORDER
    _ACTIVE_RECORDER = recorder
    try:
        yield
    finally:
        _ACTIVE_RECORDER = previous


def _current_recorder() -> StageRecorder | None:
    return _ACTIVE_RECORDER


def _short_value(value: Any, limit: int = 120) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _meta_skill(args: tuple[Any, ...], _kwargs: dict[str, Any]) -> dict[str, Any]:
    return {"skill": _short_value(args[0])} if args else {}


def _meta_tool(args: tuple[Any, ...], _kwargs: dict[str, Any]) -> dict[str, Any]:
    return {"tool": _short_value(args[0])} if args else {}


def _meta_query(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    query = kwargs.get("query")
    if query is None and args:
        query = args[0]
    return {"query": _short_value(query)}


def _meta_node(args: tuple[Any, ...], _kwargs: dict[str, Any]) -> dict[str, Any]:
    if not args:
        return {}
    node = args[0]
    return {
        "node_id": _short_value(getattr(node, "id", "")),
        "skill": _short_value(getattr(node, "skill", "")),
        "kind": _short_value(getattr(node, "kind", "")),
    }


class RuntimeInstrumentation:
    """Runtime monkey patches used by in-process latency probes only."""

    def __init__(self) -> None:
        self._patches: list[tuple[Any, str, Any]] = []

    def install(self) -> None:
        from src.agent.execution.direct_executor import DirectExecutor
        from src.agent.execution.engine import ExecutionEngine
        from src.agent.execution.planned_executor import PlannedExecutor
        from src.agent.execution.react_executor import ReactExecutor
        from src.agent.execution.workflow_executor import WorkflowExecutor
        from src.agent.planner import Planner
        from src.agent.response_synthesizer import ResponseSynthesizer
        from src.agent.skill_param_builder import SkillParamBuilder
        from src.agent.streaming_service import BaseStreamingGenerator
        from src.rag.online_retriever import OnlineRetriever
        from src.skills.router import AstronomySkillRouter

        self._wrap_sync(
            BaseStreamingGenerator,
            "_build_request_context",
            "request_context_ms",
        )
        self._wrap_sync(
            BaseStreamingGenerator,
            "_resolve_execution_decision",
            "execution_decision_ms",
        )
        self._wrap_sync(
            BaseStreamingGenerator,
            "_preview_execution_plan_for_streaming",
            "plan_preview_ms",
        )
        self._wrap_async(
            BaseStreamingGenerator,
            "_run_orchestrated_path",
            "orchestrated_path_ms",
        )
        self._wrap_async(ExecutionEngine, "run", "execution_engine_run_ms")
        self._wrap_async_generator(
            ExecutionEngine,
            "astream_events",
            "execution_engine_react_stream_ms",
        )
        self._wrap_async(DirectExecutor, "run", "direct_executor_ms")
        self._wrap_async(DirectExecutor, "_run_tool_task", "direct_tool_task_ms")
        self._wrap_async(DirectExecutor, "_run_simple_qa", "direct_simple_qa_ms")
        self._wrap_sync(DirectExecutor, "_invoke_llm", "direct_llm_invoke_ms")
        self._wrap_async(PlannedExecutor, "run", "planned_executor_ms")
        self._wrap_sync(
            PlannedExecutor,
            "preview_plan",
            "planned_preview_plan_ms",
        )
        self._wrap_sync(
            PlannedExecutor,
            "_resolve_plan_and_graph",
            "planned_resolve_plan_ms",
        )
        self._wrap_async(WorkflowExecutor, "execute", "workflow_execute_ms")
        self._wrap_async(
            WorkflowExecutor,
            "_execute_node",
            "workflow_node_ms",
            meta_fn=_meta_node,
        )
        self._wrap_sync(
            SkillParamBuilder,
            "build",
            "param_builder_ms",
            meta_fn=_meta_skill,
        )
        self._wrap_sync(
            ResponseSynthesizer,
            "synthesize",
            "synthesis_total_ms",
        )
        self._wrap_sync(
            ResponseSynthesizer,
            "synthesize_direct",
            "synthesis_direct_ms",
        )
        self._wrap_sync(ResponseSynthesizer, "synthesize_qa", "synthesis_qa_ms")
        self._wrap_sync(
            ResponseSynthesizer,
            "_invoke_llm",
            "synthesis_llm_ms",
        )
        self._wrap_sync(
            Planner,
            "plan_graph",
            "planner_plan_graph_ms",
            meta_fn=_meta_query,
        )
        self._wrap_sync(Planner, "plan", "planner_plan_ms", meta_fn=_meta_query)
        self._wrap_sync(
            OnlineRetriever,
            "retrieve",
            "rag_retrieve_ms",
            meta_fn=_meta_query,
        )
        self._wrap_sync(
            OnlineRetriever,
            "get_relevant_context",
            "rag_relevant_context_ms",
            meta_fn=_meta_query,
        )
        self._wrap_sync(
            AstronomySkillRouter,
            "call",
            "skill_router_call_ms",
            meta_fn=_meta_skill,
        )
        self._wrap_sync(
            AstronomySkillRouter,
            "call_mcp_tool",
            "skill_router_mcp_tool_ms",
            meta_fn=_meta_tool,
        )
        self._wrap_sync(
            AstronomySkillRouter,
            "call_mcp_tools_parallel",
            "skill_router_mcp_parallel_ms",
        )
        self._wrap_async_generator(
            ReactExecutor,
            "astream_events",
            "react_executor_stream_ms",
        )

    def uninstall(self) -> None:
        while self._patches:
            owner, name, original = self._patches.pop()
            setattr(owner, name, original)

    def _patch(self, owner: Any, name: str, replacement: Any) -> None:
        original = getattr(owner, name)
        self._patches.append((owner, name, original))
        setattr(owner, name, replacement)

    def _wrap_sync(
        self,
        owner: Any,
        name: str,
        stage_name: str,
        *,
        meta_fn: Any | None = None,
    ) -> None:
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapped(instance, *args, **kwargs):
            recorder = _current_recorder()
            if recorder is None:
                return original(instance, *args, **kwargs)
            meta = meta_fn(args, kwargs) if meta_fn else None
            with recorder.measure(stage_name, meta=meta):
                return original(instance, *args, **kwargs)

        self._patch(owner, name, wrapped)

    def _wrap_async(
        self,
        owner: Any,
        name: str,
        stage_name: str,
        *,
        meta_fn: Any | None = None,
    ) -> None:
        original = getattr(owner, name)

        @functools.wraps(original)
        async def wrapped(instance, *args, **kwargs):
            recorder = _current_recorder()
            if recorder is None:
                return await original(instance, *args, **kwargs)
            meta = meta_fn(args, kwargs) if meta_fn else None
            with recorder.measure(stage_name, meta=meta):
                return await original(instance, *args, **kwargs)

        self._patch(owner, name, wrapped)

    def _wrap_async_generator(
        self,
        owner: Any,
        name: str,
        stage_name: str,
    ) -> None:
        original = getattr(owner, name)

        @functools.wraps(original)
        async def wrapped(instance, *args, **kwargs):
            recorder = _current_recorder()
            if recorder is None:
                async for item in original(instance, *args, **kwargs):
                    yield item
                return
            started = time.perf_counter()
            count = 0
            error = ""
            try:
                async for item in original(instance, *args, **kwargs):
                    count += 1
                    yield item
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                recorder.record(
                    stage_name,
                    (time.perf_counter() - started) * 1000.0,
                    meta={"event_count": count},
                    error=error,
                )

        self._patch(owner, name, wrapped)


def load_dataset(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError(f"invalid dataset: {path}")
    return payload


def make_report_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{timestamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    (candidate / "events").mkdir(parents=True, exist_ok=False)
    return candidate


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil((pct / 100.0) * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def final_user_turn(case: dict[str, Any]) -> str:
    turns = case.get("turns") or []
    for turn in reversed(turns):
        if isinstance(turn, dict) and turn.get("role") == "user":
            return str(turn.get("content") or "")
    return str(case.get("prompt") or "")


def setup_user_turns(case: dict[str, Any]) -> list[str]:
    turns = case.get("turns") or []
    user_turns = [
        str(turn.get("content") or "")
        for turn in turns
        if isinstance(turn, dict) and turn.get("role") == "user"
    ]
    if len(user_turns) <= 1:
        return []
    return user_turns[:-1]


def normalize_base_url(value: str) -> str:
    return value.rstrip("/")


def preview(text: str, limit: int = 90) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


async def run_turn(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    query: str,
    user_id: str,
    session_id: str,
    disable_long_term_memory: bool,
) -> dict[str, Any]:
    payload = {
        "query": query,
        "user_id": user_id,
        "session_id": session_id,
        "disable_long_term_memory": disable_long_term_memory,
    }
    started = time.perf_counter()
    first_event_ms: float | None = None
    events: list[dict[str, Any]] = []
    final_event: dict[str, Any] | None = None
    status_code: int | None = None
    response_body = ""
    exception = ""

    try:
        async with client.stream("POST", f"{base_url}/query", json=payload) as response:
            status_code = response.status_code
            if response.status_code >= 400:
                raw_body = await response.aread()
                response_body = raw_body.decode("utf-8", errors="replace")
            else:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    if first_event_ms is None:
                        first_event_ms = (time.perf_counter() - started) * 1000.0
                    payload_text = line.removeprefix("data: ").strip()
                    if not payload_text:
                        continue
                    try:
                        event = json.loads(payload_text)
                    except json.JSONDecodeError:
                        events.append({"type": "parse_error", "raw": payload_text})
                        continue
                    if isinstance(event, dict):
                        event["_client_elapsed_ms"] = round(
                            (time.perf_counter() - started) * 1000.0, 2
                        )
                        events.append(event)
                        if event.get("type") == "final_answer":
                            final_event = event
    except Exception as exc:  # noqa: BLE001 - latency probe reports per-turn failures.
        exception = f"{type(exc).__name__}: {exc}"

    return {
        "status_code": status_code,
        "events": events,
        "final_event": final_event,
        "first_event_ms": first_event_ms,
        "e2e_ms": (time.perf_counter() - started) * 1000.0,
        "response_body": response_body,
        "exception": exception,
    }


def _parse_sse_chunk(chunk: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw_event in str(chunk).split("\n\n"):
        lines = [
            line.removeprefix("data: ").strip()
            for line in raw_event.splitlines()
            if line.startswith("data: ")
        ]
        if not lines:
            continue
        payload_text = "\n".join(lines).strip()
        if not payload_text:
            continue
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            events.append({"type": "parse_error", "raw": payload_text})
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


async def run_turn_inprocess(
    streaming_service: Any,
    *,
    query: str,
    use_long_term_memory: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    first_event_ms: float | None = None
    events: list[dict[str, Any]] = []
    final_event: dict[str, Any] | None = None
    response_body = ""
    exception = ""
    recorder = StageRecorder()

    try:
        with active_recorder(recorder):
            async for chunk in streaming_service.generate_sse(
                query,
                use_long_term_memory=use_long_term_memory,
            ):
                for event in _parse_sse_chunk(chunk):
                    if first_event_ms is None:
                        first_event_ms = (time.perf_counter() - started) * 1000.0
                    event["_client_elapsed_ms"] = round(
                        (time.perf_counter() - started) * 1000.0,
                        2,
                    )
                    events.append(event)
                    if event.get("type") == "final_answer":
                        final_event = event
    except Exception as exc:  # noqa: BLE001 - latency probe reports per-turn failures.
        exception = f"{type(exc).__name__}: {exc}"

    return {
        "status_code": 200 if not exception else None,
        "events": events,
        "final_event": final_event,
        "first_event_ms": first_event_ms,
        "e2e_ms": (time.perf_counter() - started) * 1000.0,
        "response_body": response_body,
        "exception": exception,
        "instrumentation": recorder.snapshot(),
    }


def latency_payload(turn_result: dict[str, Any]) -> dict[str, Any]:
    final_event = turn_result.get("final_event")
    if isinstance(final_event, dict):
        latency = final_event.get("latency_metrics")
        if isinstance(latency, dict):
            return latency
    for event in reversed(turn_result.get("events") or []):
        if event.get("type") == "latency_metrics":
            stages = event.get("stages_ms")
            meta = event.get("meta")
            if isinstance(stages, dict):
                return {
                    "stages_ms": stages,
                    "meta": meta if isinstance(meta, dict) else {},
                }
    return {"stages_ms": {}, "meta": {}}


def extract_tools(turn_result: dict[str, Any]) -> list[dict[str, Any]]:
    final_event = turn_result.get("final_event")
    tools: list[dict[str, Any]] = []
    if isinstance(final_event, dict):
        for item in final_event.get("tools_used") or []:
            if isinstance(item, dict):
                tools.append(item)
    if tools:
        return tools

    for event in turn_result.get("events") or []:
        if event.get("type") in {"tool_start", "tool_end"}:
            tools.append(
                {
                    "event_type": event.get("type"),
                    "tool": event.get("tool") or event.get("skill"),
                    "status": event.get("status"),
                    "latency_ms": event.get("latency_ms"),
                    "duration_sec": event.get("duration_sec"),
                }
            )
    return tools


def summarize_turn(
    *,
    case: dict[str, Any],
    turn_index: int,
    turn_kind: str,
    prompt: str,
    turn_result: dict[str, Any],
    events_path: Path,
) -> dict[str, Any]:
    latency = latency_payload(turn_result)
    stages = latency.get("stages_ms") if isinstance(latency, dict) else {}
    meta = latency.get("meta") if isinstance(latency, dict) else {}
    stages = stages if isinstance(stages, dict) else {}
    meta = meta if isinstance(meta, dict) else {}
    final_event = turn_result.get("final_event")
    audit = final_event.get("audit_metadata") if isinstance(final_event, dict) else {}
    audit = audit if isinstance(audit, dict) else {}
    tools = extract_tools(turn_result)
    instrumentation = turn_result.get("instrumentation") or {}
    detailed_stages = instrumentation.get("stages_ms") if isinstance(instrumentation, dict) else {}
    detailed_stages = detailed_stages if isinstance(detailed_stages, dict) else {}
    instrumentation_calls = instrumentation.get("calls") if isinstance(instrumentation, dict) else []
    instrumentation_calls = instrumentation_calls if isinstance(instrumentation_calls, list) else []

    agent_total = float(stages.get("agent_total_ms") or 0.0)
    tool_exec = float(stages.get("tool_exec_ms") or 0.0)
    rag_total = float(stages.get("rag_total_ms") or 0.0)
    non_tool_agent = max(agent_total - tool_exec - rag_total, 0.0)
    known_detailed = sum(
        float(detailed_stages.get(key) or 0.0)
        for key in [
            "planner_plan_graph_ms",
            "planner_plan_ms",
            "planned_preview_plan_ms",
            "planned_resolve_plan_ms",
            "workflow_execute_ms",
            "synthesis_llm_ms",
            "direct_llm_invoke_ms",
            "direct_tool_task_ms",
            "direct_simple_qa_ms",
            "react_executor_stream_ms",
            "execution_engine_react_stream_ms",
            "rag_retrieve_ms",
            "rag_relevant_context_ms",
        ]
    )

    tool_names: list[str] = []
    for item in tools:
        name = item.get("tool") or item.get("skill")
        if name and name not in tool_names:
            tool_names.append(str(name))

    return {
        "case_id": case.get("case_id"),
        "category": case.get("category"),
        "subcategory": case.get("subcategory"),
        "difficulty": case.get("difficulty"),
        "latency_bucket": case.get("latency_bucket"),
        "turn_index": turn_index,
        "turn_kind": turn_kind,
        "prompt": prompt,
        "status_code": turn_result.get("status_code"),
        "exception": turn_result.get("exception"),
        "first_event_ms": turn_result.get("first_event_ms"),
        "e2e_ms": turn_result.get("e2e_ms"),
        "event_count": len(turn_result.get("events") or []),
        "execution_path": meta.get("execution_path"),
        "exec_mode": meta.get("exec_mode"),
        "route_reason": (meta.get("execution_decision") or {}).get("reason")
        if isinstance(meta.get("execution_decision"), dict)
        else None,
        "router_source": audit.get("router_source"),
        "planner_source": audit.get("planner_source"),
        "tool_necessity_action": audit.get("tool_necessity_action"),
        "tool_necessity_reason": audit.get("tool_necessity_reason"),
        "tool_names": tool_names,
        "mcp_tools": audit.get("handler_mcp_tools_used", []),
        "stages_ms": stages,
        "detailed_stages_ms": detailed_stages,
        "derived_non_tool_agent_ms": round(non_tool_agent, 2),
        "derived_unattributed_agent_ms": round(max(agent_total - known_detailed, 0.0), 2),
        "instrumentation_calls": instrumentation_calls,
        "tools": tools,
        "events_path": str(events_path),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")


def write_report(report_dir: Path, summary: dict[str, Any], turns: list[dict[str, Any]]) -> None:
    with (report_dir / "latency_report.md").open("w", encoding="utf-8") as handle:
        handle.write("# AstroAgent E2E Latency Probe\n\n")
        handle.write(f"- report_dir: `{report_dir}`\n")
        handle.write(f"- base_url: `{summary['base_url']}`\n")
        handle.write(f"- mode: `{summary.get('mode', 'live')}`\n")
        handle.write(f"- run_started_at: `{summary['run_started_at']}`\n")
        handle.write(f"- run_finished_at: `{summary['run_finished_at']}`\n")
        handle.write(f"- selected_cases: `{', '.join(summary['case_ids'])}`\n")
        handle.write(f"- total_turns: `{summary['total_turns']}`\n")
        handle.write(f"- e2e_avg_ms: `{summary['e2e_avg_ms']}`\n")
        handle.write(f"- e2e_p95_ms: `{summary['e2e_p95_ms']}`\n\n")

        handle.write("## Per-turn timings\n\n")
        handle.write(
            "| case | turn | bucket | path | first_ms | e2e_ms | route_ms | "
            "prepare_ms | agent_ms | tool_exec_ms | non_tool_agent_ms | memory_save_ms | tools |\n"
        )
        handle.write(
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
        )
        for row in turns:
            stages = row.get("stages_ms") or {}
            handle.write(
                "| {case} | {turn} | {bucket} | {path} | {first:.2f} | {e2e:.2f} | "
                "{route:.2f} | {prepare:.2f} | {agent:.2f} | {tool:.2f} | "
                "{non_tool:.2f} | {memory:.2f} | {tools} |\n".format(
                    case=row["case_id"],
                    turn=f"{row['turn_index']}:{row['turn_kind']}",
                    bucket=row.get("latency_bucket") or "",
                    path=row.get("execution_path") or "",
                    first=float(row.get("first_event_ms") or 0.0),
                    e2e=float(row.get("e2e_ms") or 0.0),
                    route=float(stages.get("route_decision_ms") or 0.0),
                    prepare=float(stages.get("agent_prepare_ms") or 0.0),
                    agent=float(stages.get("agent_total_ms") or 0.0),
                    tool=float(stages.get("tool_exec_ms") or 0.0),
                    non_tool=float(row.get("derived_non_tool_agent_ms") or 0.0),
                    memory=float(stages.get("memory_save_ms") or 0.0),
                    tools=", ".join(row.get("tool_names") or []),
                )
            )

        handle.write("\n## Fine-grained timings\n\n")
        handle.write(
            "| case | turn | path | planner_ms | plan_preview_ms | workflow_ms | "
            "param_builder_ms | skill_router_ms | synth_total_ms | synth_llm_ms | "
            "direct_llm_ms | react_stream_ms | unattributed_agent_ms |\n"
        )
        handle.write(
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for row in turns:
            detail = row.get("detailed_stages_ms") or {}
            planner_ms = (
                float(detail.get("planner_plan_graph_ms") or 0.0)
                + float(detail.get("planner_plan_ms") or 0.0)
                + float(detail.get("planned_resolve_plan_ms") or 0.0)
            )
            react_stream_ms = max(
                float(detail.get("react_executor_stream_ms") or 0.0),
                float(detail.get("execution_engine_react_stream_ms") or 0.0),
            )
            handle.write(
                "| {case} | {turn} | {path} | {planner:.2f} | {preview:.2f} | "
                "{workflow:.2f} | {params:.2f} | {skills:.2f} | {synth_total:.2f} | "
                "{synth_llm:.2f} | {direct_llm:.2f} | {react:.2f} | {unattrib:.2f} |\n".format(
                    case=row["case_id"],
                    turn=f"{row['turn_index']}:{row['turn_kind']}",
                    path=row.get("execution_path") or "",
                    planner=planner_ms,
                    preview=float(detail.get("planned_preview_plan_ms") or 0.0),
                    workflow=float(detail.get("workflow_execute_ms") or 0.0),
                    params=float(detail.get("param_builder_ms") or 0.0),
                    skills=float(detail.get("skill_router_call_ms") or 0.0),
                    synth_total=float(detail.get("synthesis_total_ms") or 0.0),
                    synth_llm=float(detail.get("synthesis_llm_ms") or 0.0),
                    direct_llm=float(detail.get("direct_llm_invoke_ms") or 0.0),
                    react=react_stream_ms,
                    unattrib=float(row.get("derived_unattributed_agent_ms") or 0.0),
                )
            )

        handle.write("\n## Slowest turns\n\n")
        for row in sorted(turns, key=lambda item: float(item.get("e2e_ms") or 0), reverse=True)[:8]:
            stages = row.get("stages_ms") or {}
            detail = row.get("detailed_stages_ms") or {}
            handle.write(
                "- `{case}` {turn_kind} `{prompt}`: e2e={e2e:.2f} ms, "
                "agent={agent:.2f} ms, tool_exec={tool:.2f} ms, "
                "non_tool_agent={non_tool:.2f} ms, synth_llm={synth_llm:.2f} ms, "
                "workflow={workflow:.2f} ms, react_stream={react:.2f} ms, "
                "path={path}, tools={tools}\n".format(
                    case=row["case_id"],
                    turn_kind=row["turn_kind"],
                    prompt=preview(row.get("prompt") or "", 60),
                    e2e=float(row.get("e2e_ms") or 0.0),
                    agent=float(stages.get("agent_total_ms") or 0.0),
                    tool=float(stages.get("tool_exec_ms") or 0.0),
                    non_tool=float(row.get("derived_non_tool_agent_ms") or 0.0),
                    synth_llm=float(detail.get("synthesis_llm_ms") or 0.0),
                    workflow=float(detail.get("workflow_execute_ms") or 0.0),
                    react=max(
                        float(detail.get("react_executor_stream_ms") or 0.0),
                        float(detail.get("execution_engine_react_stream_ms") or 0.0),
                    ),
                    path=row.get("execution_path") or "",
                    tools=", ".join(row.get("tool_names") or []),
                )
            )


def select_cases(
    dataset: dict[str, Any],
    requested: list[str] | None,
) -> list[dict[str, Any]]:
    case_ids = requested or DEFAULT_CASE_IDS
    requested_set = set(case_ids)
    cases = [case for case in dataset["cases"] if case.get("case_id") in requested_set]
    found = {str(case.get("case_id")) for case in cases}
    missing = [case_id for case_id in case_ids if case_id not in found]
    if missing:
        raise ValueError(f"unknown case_id(s): {', '.join(missing)}")
    return cases


async def run_inprocess_probe(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from src.agent import AstroAgent
    from src.core.config import settings
    from src.memory.api.memory_service import MemoryService

    dataset = load_dataset(args.dataset)
    cases = select_cases(dataset, args.case_id)
    report_dir = make_report_dir(args.output)
    started_at = datetime.now().isoformat(timespec="seconds")
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    turns: list[dict[str, Any]] = []

    instrumentation = RuntimeInstrumentation()
    instrumentation.install()
    try:
        base_agent = AstroAgent(user_id=f"{args.user_id_prefix}_{run_id}")
        if getattr(base_agent, "skill_manager", None) is not None:
            base_agent.skill_manager.prewarm()

        for case in cases:
            case_id = str(case.get("case_id"))
            user_id = f"{args.user_id_prefix}_{run_id}"
            session_id = f"{case_id}_{uuid.uuid4().hex[:8]}"
            memory = MemoryService(
                db_path=settings.MEMORY_PERSISTENCE_PATH,
                session_id=f"mem_{user_id}::{session_id}",
                user_id=user_id,
            )
            runtime = base_agent.create_session_runtime(
                user_id=user_id,
                memory=memory,
            )
            streaming_service = runtime["streaming_service"]

            prompts = [
                (idx, "setup", prompt)
                for idx, prompt in enumerate(setup_user_turns(case), 1)
            ]
            prompts.append((len(prompts) + 1, "final", final_user_turn(case)))

            for turn_index, turn_kind, prompt in prompts:
                turn_result = await run_turn_inprocess(
                    streaming_service,
                    query=prompt,
                    use_long_term_memory=args.use_long_term_memory,
                )
                events_name = f"{case_id}__{turn_index:02d}_{turn_kind}.json"
                events_path = report_dir / "events" / events_name
                with events_path.open("w", encoding="utf-8") as handle:
                    json.dump(
                        turn_result.get("events") or [],
                        handle,
                        ensure_ascii=False,
                        indent=2,
                    )
                    handle.write("\n")
                row = summarize_turn(
                    case=case,
                    turn_index=turn_index,
                    turn_kind=turn_kind,
                    prompt=prompt,
                    turn_result=turn_result,
                    events_path=events_path.relative_to(report_dir),
                )
                turns.append(row)
                detail = row.get("detailed_stages_ms") or {}
                print(
                    f"{case_id} {turn_index}:{turn_kind} "
                    f"e2e_ms={row['e2e_ms']:.2f} "
                    f"agent_ms={(row['stages_ms'] or {}).get('agent_total_ms')} "
                    f"synth_llm_ms={detail.get('synthesis_llm_ms')} "
                    f"workflow_ms={detail.get('workflow_execute_ms')} "
                    f"react_ms={detail.get('react_executor_stream_ms')} "
                    f"path={row.get('execution_path')} tools={row.get('tool_names')}"
                )
    finally:
        instrumentation.uninstall()

    finished_at = datetime.now().isoformat(timespec="seconds")
    e2e_values = [
        float(row["e2e_ms"])
        for row in turns
        if row.get("e2e_ms") is not None
    ]
    summary = {
        "dataset_id": dataset.get("dataset_id"),
        "dataset_path": str(args.dataset),
        "base_url": "inprocess",
        "mode": "inprocess",
        "run_started_at": started_at,
        "run_finished_at": finished_at,
        "report_dir": str(report_dir),
        "case_ids": [str(case.get("case_id")) for case in cases],
        "total_cases": len(cases),
        "total_turns": len(turns),
        "use_long_term_memory": args.use_long_term_memory,
        "e2e_avg_ms": round(mean(e2e_values), 2) if e2e_values else None,
        "e2e_p95_ms": round(percentile(e2e_values, 95), 2) if e2e_values else None,
    }
    with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_jsonl(report_dir / "turns.jsonl", turns)
    write_report(report_dir, summary, turns)
    return summary, turns


async def run_probe(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if args.mode == "inprocess":
        return await run_inprocess_probe(args)

    dataset = load_dataset(args.dataset)
    cases = select_cases(dataset, args.case_id)

    report_dir = make_report_dir(args.output)
    started_at = datetime.now().isoformat(timespec="seconds")
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    timeout = httpx.Timeout(args.request_timeout_sec, connect=args.connect_timeout_sec)
    turns: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        if not args.skip_health_check:
            response = await client.get(f"{args.base_url}/health")
            response.raise_for_status()

        for case in cases:
            case_id = str(case.get("case_id"))
            user_id = f"{args.user_id_prefix}_{run_id}"
            session_id = f"{case_id}_{uuid.uuid4().hex[:8]}"
            prompts = [(idx, "setup", prompt) for idx, prompt in enumerate(setup_user_turns(case), 1)]
            prompts.append((len(prompts) + 1, "final", final_user_turn(case)))

            for turn_index, turn_kind, prompt in prompts:
                turn_result = await run_turn(
                    client,
                    base_url=args.base_url,
                    query=prompt,
                    user_id=user_id,
                    session_id=session_id,
                    disable_long_term_memory=not args.use_long_term_memory,
                )
                events_name = f"{case_id}__{turn_index:02d}_{turn_kind}.json"
                events_path = report_dir / "events" / events_name
                with events_path.open("w", encoding="utf-8") as handle:
                    json.dump(turn_result.get("events") or [], handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                row = summarize_turn(
                    case=case,
                    turn_index=turn_index,
                    turn_kind=turn_kind,
                    prompt=prompt,
                    turn_result=turn_result,
                    events_path=events_path.relative_to(report_dir),
                )
                turns.append(row)
                print(
                    f"{case_id} {turn_index}:{turn_kind} "
                    f"e2e_ms={row['e2e_ms']:.2f} "
                    f"agent_ms={(row['stages_ms'] or {}).get('agent_total_ms')} "
                    f"tool_exec_ms={(row['stages_ms'] or {}).get('tool_exec_ms')} "
                    f"path={row.get('execution_path')} tools={row.get('tool_names')}"
                )

    finished_at = datetime.now().isoformat(timespec="seconds")
    e2e_values = [float(row["e2e_ms"]) for row in turns if row.get("e2e_ms") is not None]
    summary = {
        "dataset_id": dataset.get("dataset_id"),
        "dataset_path": str(args.dataset),
        "base_url": args.base_url,
        "mode": "live",
        "run_started_at": started_at,
        "run_finished_at": finished_at,
        "report_dir": str(report_dir),
        "case_ids": [str(case.get("case_id")) for case in cases],
        "total_cases": len(cases),
        "total_turns": len(turns),
        "use_long_term_memory": args.use_long_term_memory,
        "e2e_avg_ms": round(mean(e2e_values), 2) if e2e_values else None,
        "e2e_p95_ms": round(percentile(e2e_values, 95), 2) if e2e_values else None,
    }
    with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_jsonl(report_dir / "turns.jsonl", turns)
    write_report(report_dir, summary, turns)
    return summary, turns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small live E2E latency probe.")
    parser.add_argument(
        "--mode",
        choices=["live", "inprocess"],
        default="live",
        help=(
            "live calls an existing /query API; inprocess runs the Agent in this "
            "probe process with runtime-only instrumentation. Default: live."
        ),
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case-id", action="append", help="Case id to run. Defaults to latency probe set.")
    parser.add_argument("--request-timeout-sec", type=float, default=180.0)
    parser.add_argument("--connect-timeout-sec", type=float, default=10.0)
    parser.add_argument("--user-id-prefix", default="astro_latency")
    parser.add_argument("--use-long-term-memory", action="store_true")
    parser.add_argument("--skip-health-check", action="store_true")
    args = parser.parse_args()
    args.base_url = normalize_base_url(args.base_url)
    return args


def main() -> int:
    args = parse_args()
    try:
        summary, _turns = asyncio.run(run_probe(args))
    except Exception as exc:  # noqa: BLE001 - CLI should report probe setup errors.
        print(f"latency probe failed: {type(exc).__name__}: {exc}")
        return 2
    print(f"report_dir: {summary['report_dir']}")
    print(f"total_turns: {summary['total_turns']}")
    print(f"e2e_avg_ms: {summary['e2e_avg_ms']}")
    print(f"e2e_p95_ms: {summary['e2e_p95_ms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
