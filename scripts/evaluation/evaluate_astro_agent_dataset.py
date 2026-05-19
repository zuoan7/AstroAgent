#!/usr/bin/env python
"""Run the AstroAgent benchmark dataset against the live /query SSE API."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import httpx


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "config/benchmarks/astro_agent_eval_dataset.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports/evaluation/astro_agent"
DEFAULT_BASE_URL = "http://localhost:8002"

LANGCHAIN_TOOL_TO_SKILL = {
    "RAGRetrieve": "RAGRetrieve",
    "WeatherLookup": "weather-lookup",
    "ObservationPlanner": "observation-planner",
    "CelestialEventsForecast": "celestial-events-forecast",
    "DeepSkyObservingGuide": "deep-sky-observing-guide",
    "NEOTracker": "neo-tracker",
    "AstrophotographyCalculator": "astrophotography-calculator",
    "CelestialPositionCalculator": "celestial-position-calculator",
}

CLARIFICATION_CUES = [
    "请",
    "需要",
    "告诉我",
    "补充",
    "提供",
    "确认",
    "澄清",
    "上传",
    "城市",
    "地点",
    "经纬度",
    "设备",
    "器材",
    "日期",
    "时间",
    "哪一个",
]


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
    return candidate


def safe_rate(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil((pct / 100.0) * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def preview(text: str, limit: int = 220) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def normalize_base_url(value: str) -> str:
    return value.rstrip("/")


def select_cases(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = list(cases)
    suites = set(args.suite or [])
    if suites:
        selected = [case for case in selected if case.get("suite") in suites]

    if args.category:
        categories = set(args.category)
        selected = [case for case in selected if case.get("category") in categories]

    if args.case_id:
        requested = set(args.case_id)
        selected = [case for case in selected if case.get("case_id") in requested]

    if args.sample_per_category:
        counts: Counter[str] = Counter()
        sampled: list[dict[str, Any]] = []
        for case in selected:
            category = str(case.get("category", ""))
            if counts[category] >= args.sample_per_category:
                continue
            sampled.append(case)
            counts[category] += 1
        selected = sampled

    if args.limit is not None:
        selected = selected[: args.limit]

    return selected


def event_tool_name(event: dict[str, Any]) -> str:
    for key in ("tool", "skill", "tool_name", "name"):
        value = event.get(key)
        if value:
            return str(value)
    meta = event.get("meta")
    if isinstance(meta, dict):
        for key in ("tool", "skill", "tool_name", "name"):
            value = meta.get(key)
            if value:
                return str(value)
    return ""


def item_tool_name(item: dict[str, Any]) -> str:
    for key in ("tool", "skill", "tool_name", "name"):
        value = item.get(key)
        if value:
            return str(value)
    return ""


def canonical_skill_name(raw_name: str, allowed_skills: set[str]) -> str | None:
    if not raw_name:
        return None
    name = raw_name.strip()
    if name in allowed_skills:
        return name
    if name in LANGCHAIN_TOOL_TO_SKILL:
        return LANGCHAIN_TOOL_TO_SKILL[name]
    lowered = name.lower()
    for skill in allowed_skills:
        if lowered == skill.lower():
            return skill
    for tool_name, skill in LANGCHAIN_TOOL_TO_SKILL.items():
        if lowered == tool_name.lower():
            return skill
    return None


def canonical_mcp_tool_name(raw_name: str, allowed_mcp_tools: set[str]) -> str | None:
    if not raw_name:
        return None
    name = raw_name.strip()
    if name in allowed_mcp_tools:
        return name
    lowered = name.lower()
    for tool in allowed_mcp_tools:
        if lowered == tool.lower():
            return tool
    return None


def extract_trace(
    events: list[dict[str, Any]],
    final_event: dict[str, Any] | None,
    *,
    allowed_skills: set[str],
    allowed_mcp_tools: set[str],
) -> dict[str, Any]:
    raw_tool_names: list[str] = []
    tool_records: list[dict[str, Any]] = []
    final_tools = []
    final_handler_mcp_tools = []
    if isinstance(final_event, dict):
        final_tools = list(final_event.get("tools_used") or [])
        final_handler_mcp_tools = list(final_event.get("handler_mcp_tools_used") or [])
        audit_metadata = final_event.get("audit_metadata")
        if isinstance(audit_metadata, dict):
            final_handler_mcp_tools.extend(
                list(audit_metadata.get("handler_mcp_tools_used") or [])
            )

    for event in events:
        if event.get("type") not in {"tool_start", "tool_end"}:
            continue
        raw_name = event_tool_name(event)
        if raw_name:
            raw_tool_names.append(raw_name)
        tool_records.append(
            {
                "event_type": event.get("type"),
                "tool": raw_name,
                "status": event.get("status"),
                "input": event.get("input"),
            }
        )

    for item in final_tools:
        if not isinstance(item, dict):
            continue
        raw_name = item_tool_name(item)
        if raw_name:
            raw_tool_names.append(raw_name)
        tool_records.append(
            {
                "event_type": "final_tools_used",
                "tool": raw_name,
                "status": item.get("status"),
                "input": item.get("input"),
            }
        )

    for raw_name in final_handler_mcp_tools:
        if not raw_name:
            continue
        raw_name = str(raw_name)
        raw_tool_names.append(raw_name)
        tool_records.append(
            {
                "event_type": "handler_mcp_tools_used",
                "tool": raw_name,
                "status": "success",
                "input": None,
            }
        )

    actual_skills = []
    actual_mcp_tools = []
    for raw_name in raw_tool_names:
        skill = canonical_skill_name(raw_name, allowed_skills)
        if skill:
            actual_skills.append(skill)
        mcp_tool = canonical_mcp_tool_name(raw_name, allowed_mcp_tools)
        if mcp_tool:
            actual_mcp_tools.append(mcp_tool)

    final_tool_count = None
    final_success_count = None
    final_error_count = None
    if isinstance(final_event, dict):
        for key, target in [
            ("tool_count", "tool_count"),
            ("tool_success_count", "success_count"),
            ("tool_error_count", "error_count"),
        ]:
            value = final_event.get(key)
            if isinstance(value, int):
                if target == "tool_count":
                    final_tool_count = value
                elif target == "success_count":
                    final_success_count = value
                else:
                    final_error_count = value

    inferred_success = 0
    inferred_error = 0
    for record in tool_records:
        status = str(record.get("status") or "").lower()
        if status == "success":
            inferred_success += 1
        elif status in {"error", "failed", "failure"}:
            inferred_error += 1

    success_count = final_success_count if final_success_count is not None else inferred_success
    error_count = final_error_count if final_error_count is not None else inferred_error
    completed_denominator = success_count + error_count
    if completed_denominator == 0 and raw_tool_names:
        completed_denominator = len(unique_preserve_order(raw_tool_names))

    return {
        "raw_tool_names": unique_preserve_order(raw_tool_names),
        "actual_skills": unique_preserve_order(actual_skills),
        "actual_mcp_tools": unique_preserve_order(actual_mcp_tools),
        "tool_records": tool_records,
        "tool_count": final_tool_count if final_tool_count is not None else len(raw_tool_names),
        "tool_success_count": success_count,
        "tool_error_count": error_count,
        "tool_call_denominator": completed_denominator,
    }


def contains_clarification_cue(answer: str) -> bool:
    return any(cue in answer for cue in CLARIFICATION_CUES)


def score_case(
    case: dict[str, Any],
    query_result: dict[str, Any],
    *,
    allowed_skills: set[str],
    allowed_mcp_tools: set[str],
    mcp_scoring: str,
    min_answer_chars: int,
    setup_failed: bool,
) -> dict[str, Any]:
    final_event = query_result.get("final_event")
    events = query_result.get("events", [])
    trace = extract_trace(
        events,
        final_event,
        allowed_skills=allowed_skills,
        allowed_mcp_tools=allowed_mcp_tools,
    )

    expected_skills = set(case.get("expected_skills") or [])
    forbidden_skills = set(case.get("forbidden_skills") or [])
    expected_mcp = set(case.get("expected_mcp_tools") or [])
    forbidden_mcp = set(case.get("forbidden_mcp_tools") or [])
    actual_skills = set(trace["actual_skills"])
    actual_mcp = set(trace["actual_mcp_tools"])

    missing_expected_skills = sorted(expected_skills - actual_skills)
    forbidden_skill_hits = sorted(forbidden_skills & actual_skills)
    skill_selection_pass = not missing_expected_skills and not forbidden_skill_hits

    forbidden_mcp_hits = sorted(forbidden_mcp & actual_mcp)
    missing_expected_mcp = sorted(expected_mcp - actual_mcp)
    mcp_observable = bool(actual_mcp) or bool(forbidden_mcp_hits)
    mcp_relevant = bool(expected_mcp or actual_mcp or forbidden_mcp_hits)
    mcp_selection_pass: bool | None
    mcp_unobserved_expected: list[str] = []
    if not mcp_relevant:
        missing_expected_mcp = []
        mcp_selection_pass = None
    elif mcp_scoring == "ignore":
        missing_expected_mcp = []
        mcp_selection_pass = not forbidden_mcp_hits
    elif mcp_scoring == "strict":
        mcp_selection_pass = not missing_expected_mcp and not forbidden_mcp_hits
    else:
        if expected_mcp and not mcp_observable:
            mcp_unobserved_expected = sorted(expected_mcp)
            missing_expected_mcp = []
            mcp_selection_pass = None
        else:
            mcp_selection_pass = not missing_expected_mcp and not forbidden_mcp_hits

    requires_tool_missing = bool(
        case.get("requires_tool")
        and (expected_skills or expected_mcp)
        and not trace["raw_tool_names"]
    )
    unknown_raw_tools = [
        raw_name
        for raw_name in trace["raw_tool_names"]
        if canonical_skill_name(raw_name, allowed_skills) is None
        and canonical_mcp_tool_name(raw_name, allowed_mcp_tools) is None
    ]
    unexpected_tool_for_no_tool_case = bool(
        not case.get("requires_tool")
        and (forbidden_skill_hits or forbidden_mcp_hits or unknown_raw_tools)
    )
    tool_selection_pass = (
        skill_selection_pass
        and not forbidden_mcp_hits
        and not requires_tool_missing
        and not unexpected_tool_for_no_tool_case
        and mcp_selection_pass is not False
    )

    final_answer = ""
    if isinstance(final_event, dict):
        final_answer = str(final_event.get("final_answer") or "")
    answer_available = len(final_answer.strip()) >= min_answer_chars
    error_events = [
        event
        for event in events
        if event.get("type") == "error" or str(event.get("status", "")).lower() == "error"
    ]

    tool_denominator = trace["tool_call_denominator"]
    if tool_denominator > 0:
        tool_call_success_pass = trace["tool_error_count"] == 0 and trace["tool_success_count"] > 0
        case_tool_call_success_rate = safe_rate(trace["tool_success_count"], tool_denominator)
    else:
        tool_call_success_pass = not case.get("requires_tool")
        case_tool_call_success_rate = None

    clarification_pass = True
    if case.get("should_clarify"):
        clarification_pass = contains_clarification_cue(final_answer)

    response_ok = query_result.get("status_code") is not None and query_result.get("status_code", 0) < 400
    e2e_pass = bool(
        response_ok
        and not setup_failed
        and answer_available
        and not error_events
        and tool_selection_pass
        and tool_call_success_pass
        and clarification_pass
    )

    failure_reasons: list[str] = []
    if setup_failed:
        failure_reasons.append("setup_turn_failed")
    if not response_ok:
        failure_reasons.append(f"http_error:{query_result.get('status_code')}")
    if query_result.get("exception"):
        failure_reasons.append(f"request_exception:{query_result['exception']}")
    if not answer_available:
        failure_reasons.append("missing_final_answer")
    if error_events:
        failure_reasons.append("error_event_emitted")
    if missing_expected_skills:
        failure_reasons.append("missing_expected_skills:" + ",".join(missing_expected_skills))
    if forbidden_skill_hits:
        failure_reasons.append("forbidden_skills:" + ",".join(forbidden_skill_hits))
    if missing_expected_mcp:
        failure_reasons.append("missing_expected_mcp:" + ",".join(missing_expected_mcp))
    if forbidden_mcp_hits:
        failure_reasons.append("forbidden_mcp:" + ",".join(forbidden_mcp_hits))
    if requires_tool_missing:
        failure_reasons.append("requires_tool_but_no_tool_observed")
    if unexpected_tool_for_no_tool_case:
        failure_reasons.append("no_tool_case_used_tool")
    if unknown_raw_tools:
        failure_reasons.append("unknown_tools:" + ",".join(unknown_raw_tools))
    if tool_denominator > 0 and not tool_call_success_pass:
        failure_reasons.append("tool_call_failed_or_unconfirmed")
    if case.get("should_clarify") and not clarification_pass:
        failure_reasons.append("clarification_not_detected")

    return {
        "case_id": case.get("case_id"),
        "suite": case.get("suite"),
        "category": case.get("category"),
        "subcategory": case.get("subcategory"),
        "difficulty": case.get("difficulty"),
        "requires_tool": case.get("requires_tool"),
        "should_clarify": case.get("should_clarify"),
        "prompt": case.get("prompt"),
        "expected_skills": sorted(expected_skills),
        "forbidden_skills": sorted(forbidden_skills),
        "expected_mcp_tools": sorted(expected_mcp),
        "forbidden_mcp_tools": sorted(forbidden_mcp),
        "actual_raw_tools": trace["raw_tool_names"],
        "actual_skills": trace["actual_skills"],
        "actual_mcp_tools": trace["actual_mcp_tools"],
        "skill_selection_pass": skill_selection_pass,
        "mcp_selection_pass": mcp_selection_pass,
        "mcp_unobserved_expected": mcp_unobserved_expected,
        "tool_selection_pass": tool_selection_pass,
        "unexpected_tool_for_no_tool_case": unexpected_tool_for_no_tool_case,
        "unknown_raw_tools": unknown_raw_tools,
        "tool_call_success_pass": tool_call_success_pass,
        "case_tool_call_success_rate": case_tool_call_success_rate,
        "tool_count": trace["tool_count"],
        "tool_success_count": trace["tool_success_count"],
        "tool_error_count": trace["tool_error_count"],
        "tool_call_denominator": tool_denominator,
        "answer_available": answer_available,
        "clarification_pass": clarification_pass,
        "e2e_pass": e2e_pass,
        "failure_reasons": failure_reasons,
        "status_code": query_result.get("status_code"),
        "first_event_ms": query_result.get("first_event_ms"),
        "e2e_ms": query_result.get("e2e_ms"),
        "event_count": len(events),
        "final_answer_preview": preview(final_answer),
        "latency_metrics": final_event.get("latency_metrics") if isinstance(final_event, dict) else None,
        "semantic_judge_status": "not_implemented",
    }


async def run_query(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    query: str,
    user_id: str,
    session_id: str,
    disable_long_term_memory: bool,
    model_provider: str | None,
    model_name: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "user_id": user_id,
        "session_id": session_id,
        "disable_long_term_memory": disable_long_term_memory,
    }
    if model_provider:
        payload["model_provider"] = model_provider
    if model_name:
        payload["model_name"] = model_name

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
                return {
                    "status_code": status_code,
                    "events": events,
                    "final_event": final_event,
                    "first_event_ms": first_event_ms,
                    "e2e_ms": (time.perf_counter() - started) * 1000.0,
                    "response_body": response_body,
                    "exception": exception,
                }

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
                    events.append(
                        {
                            "type": "parse_error",
                            "raw": payload_text,
                        }
                    )
                    continue
                if isinstance(event, dict):
                    events.append(event)
                    if event.get("type") == "final_answer":
                        final_event = event
    except Exception as exc:  # noqa: BLE001 - benchmark reports exceptions per case.
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


async def run_case(
    index: int,
    case: dict[str, Any],
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    report_dir: Path,
    allowed_skills: set[str],
    allowed_mcp_tools: set[str],
    run_id: str,
) -> dict[str, Any]:
    case_id = str(case.get("case_id"))
    user_id = f"{args.user_id_prefix}_{run_id}"
    session_id = f"{case_id}_{uuid.uuid4().hex[:8]}"
    setup_failed = False
    setup_results: list[dict[str, Any]] = []

    for turn_prompt in setup_user_turns(case):
        setup_result = await run_query(
            client,
            base_url=args.base_url,
            query=turn_prompt,
            user_id=user_id,
            session_id=session_id,
            disable_long_term_memory=not args.use_long_term_memory,
            model_provider=args.model_provider,
            model_name=args.model_name,
        )
        setup_ok = (
            setup_result.get("status_code") is not None
            and setup_result.get("status_code", 0) < 400
            and not setup_result.get("exception")
        )
        setup_failed = setup_failed or not setup_ok
        setup_results.append(
            {
                "prompt": turn_prompt,
                "status_code": setup_result.get("status_code"),
                "exception": setup_result.get("exception"),
                "event_count": len(setup_result.get("events") or []),
                "e2e_ms": setup_result.get("e2e_ms"),
            }
        )

    query_result = await run_query(
        client,
        base_url=args.base_url,
        query=final_user_turn(case),
        user_id=user_id,
        session_id=session_id,
        disable_long_term_memory=not args.use_long_term_memory,
        model_provider=args.model_provider,
        model_name=args.model_name,
    )

    score = score_case(
        case,
        query_result,
        allowed_skills=allowed_skills,
        allowed_mcp_tools=allowed_mcp_tools,
        mcp_scoring=args.mcp_scoring,
        min_answer_chars=args.min_answer_chars,
        setup_failed=setup_failed,
    )
    score["index"] = index
    score["user_id"] = user_id
    score["session_id"] = session_id
    score["setup_turn_count"] = len(setup_results)
    score["setup_results"] = setup_results

    if args.save_events:
        events_dir = report_dir / "events"
        events_dir.mkdir(parents=True, exist_ok=True)
        events_path = events_dir / f"{case_id}.json"
        with events_path.open("w", encoding="utf-8") as handle:
            json.dump(query_result.get("events") or [], handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        score["events_path"] = str(events_path.relative_to(report_dir))

    return score


def summarize_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    first_latencies = [
        float(result["first_event_ms"])
        for result in results
        if result.get("first_event_ms") is not None
    ]
    e2e_latencies = [
        float(result["e2e_ms"])
        for result in results
        if result.get("e2e_ms") is not None
    ]
    mcp_observed = [
        result for result in results if result.get("mcp_selection_pass") is not None
    ]
    tool_call_denominator = sum(int(result.get("tool_call_denominator") or 0) for result in results)
    tool_success_total = sum(int(result.get("tool_success_count") or 0) for result in results)
    return {
        "total_cases": total,
        "e2e_passed_cases": sum(1 for result in results if result.get("e2e_pass")),
        "end_to_end_task_success_rate": safe_rate(
            sum(1 for result in results if result.get("e2e_pass")),
            total,
        ),
        "tool_selection_accuracy": safe_rate(
            sum(1 for result in results if result.get("tool_selection_pass")),
            total,
        ),
        "skill_selection_accuracy": safe_rate(
            sum(1 for result in results if result.get("skill_selection_pass")),
            total,
        ),
        "mcp_selection_observed_cases": len(mcp_observed),
        "mcp_selection_accuracy_observed": safe_rate(
            sum(1 for result in mcp_observed if result.get("mcp_selection_pass")),
            len(mcp_observed),
        ),
        "tool_call_success_rate": safe_rate(tool_success_total, tool_call_denominator),
        "first_event_latency_avg_ms": mean(first_latencies) if first_latencies else None,
        "first_event_latency_p95_ms": percentile(first_latencies, 95),
        "end_to_end_latency_avg_ms": mean(e2e_latencies) if e2e_latencies else None,
        "end_to_end_p95_latency_ms": percentile(e2e_latencies, 95),
    }


def build_summary(
    data: dict[str, Any],
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    report_dir: Path,
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_category[str(result.get("category", ""))].append(result)

    return {
        "dataset_id": data.get("dataset_id"),
        "dataset_path": str(args.dataset),
        "run_started_at": started_at,
        "run_finished_at": finished_at,
        "base_url": args.base_url,
        "suite": args.suite,
        "category_filter": args.category,
        "case_id_filter": args.case_id,
        "mcp_scoring": args.mcp_scoring,
        "use_long_term_memory": args.use_long_term_memory,
        "report_dir": str(report_dir),
        "overall": summarize_group(results),
        "by_category": {
            category: summarize_group(items)
            for category, items in sorted(by_category.items())
        },
    }


def write_cases_jsonl(report_dir: Path, results: list[dict[str, Any]]) -> None:
    with (report_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for result in sorted(results, key=lambda item: item.get("index", 0)):
            json.dump(result, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")


def write_failures_md(report_dir: Path, results: list[dict[str, Any]]) -> None:
    failures = [result for result in results if not result.get("e2e_pass")]
    with (report_dir / "failures.md").open("w", encoding="utf-8") as handle:
        handle.write("# AstroAgent Evaluation Failures\n\n")
        handle.write(f"Total failures: {len(failures)}\n\n")
        for result in sorted(failures, key=lambda item: item.get("index", 0)):
            handle.write(f"## {result.get('case_id')}\n\n")
            handle.write(f"- category: `{result.get('category')}`\n")
            handle.write(f"- subcategory: `{result.get('subcategory')}`\n")
            handle.write(f"- prompt: {result.get('prompt')}\n")
            handle.write(f"- reasons: {', '.join(result.get('failure_reasons') or [])}\n")
            handle.write(f"- expected_skills: `{result.get('expected_skills')}`\n")
            handle.write(f"- actual_skills: `{result.get('actual_skills')}`\n")
            handle.write(f"- expected_mcp_tools: `{result.get('expected_mcp_tools')}`\n")
            handle.write(f"- actual_mcp_tools: `{result.get('actual_mcp_tools')}`\n")
            handle.write(f"- first_event_ms: `{result.get('first_event_ms')}`\n")
            handle.write(f"- e2e_ms: `{result.get('e2e_ms')}`\n")
            handle.write(f"- answer_preview: {result.get('final_answer_preview')}\n\n")


async def check_health(client: httpx.AsyncClient, base_url: str) -> None:
    response = await client.get(f"{base_url}/health")
    response.raise_for_status()


async def run_evaluation(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = load_dataset(args.dataset)
    cases = select_cases(data["cases"], args)
    if not cases:
        raise ValueError("no cases selected")

    report_dir = make_report_dir(args.output)
    started_at = datetime.now().isoformat(timespec="seconds")
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")

    allowed_values = data.get("allowed_values", {})
    allowed_skills = set(allowed_values.get("skills", []))
    allowed_mcp_tools = set(allowed_values.get("mcp_tools", []))

    timeout = httpx.Timeout(args.request_timeout_sec, connect=args.connect_timeout_sec)
    limits = httpx.Limits(max_connections=max(args.concurrency + 2, 4))
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        if not args.skip_health_check:
            await check_health(client, args.base_url)

        print(f"Selected cases: {len(cases)}")
        print(f"Report directory: {report_dir}")
        semaphore = asyncio.Semaphore(args.concurrency)

        async def bounded_run(index: int, case: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await run_case(
                    index,
                    case,
                    client,
                    args,
                    report_dir,
                    allowed_skills,
                    allowed_mcp_tools,
                    run_id,
                )

        tasks = [
            asyncio.create_task(bounded_run(index, case))
            for index, case in enumerate(cases)
        ]
        results: list[dict[str, Any]] = []
        completed = 0
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            completed += 1
            status = "PASS" if result.get("e2e_pass") else "FAIL"
            print(
                f"[{completed}/{len(cases)}] {status} "
                f"{result.get('case_id')} "
                f"e2e_ms={result.get('e2e_ms'):.2f} "
                f"tools={result.get('actual_raw_tools')}"
            )

    finished_at = datetime.now().isoformat(timespec="seconds")
    results = sorted(results, key=lambda item: item.get("index", 0))
    summary = build_summary(data, results, args, report_dir, started_at, finished_at)
    with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_cases_jsonl(report_dir, results)
    write_failures_md(report_dir, results)
    return summary, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AstroAgent benchmark dataset against the live /query SSE API."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Dataset JSON path. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"FastAPI base URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Report root directory. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=None,
        help="Suite to include. Can be repeated. Default: ability",
    )
    parser.add_argument(
        "--category",
        action="append",
        help="Category to include. Can be repeated.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        help="Specific case_id to include. Can be repeated.",
    )
    parser.add_argument("--limit", type=int, help="Limit selected cases after filtering.")
    parser.add_argument(
        "--sample-per-category",
        type=int,
        help="Take the first N selected cases per category.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent cases. Default: 1.",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=float,
        default=120.0,
        help="HTTP request timeout per turn. Default: 120.",
    )
    parser.add_argument(
        "--connect-timeout-sec",
        type=float,
        default=10.0,
        help="HTTP connect timeout. Default: 10.",
    )
    parser.add_argument(
        "--mcp-scoring",
        choices=["observed", "strict", "ignore"],
        default="observed",
        help=(
            "MCP scoring mode. observed does not fail missing expected MCP tools "
            "when the API trace exposes no MCP layer. Default: observed."
        ),
    )
    parser.add_argument(
        "--min-answer-chars",
        type=int,
        default=5,
        help="Minimum final answer length for deterministic E2E pass. Default: 5.",
    )
    parser.add_argument(
        "--user-id-prefix",
        default="astro_eval",
        help="Prefix for benchmark user_id. Default: astro_eval.",
    )
    parser.add_argument(
        "--model-provider",
        default=None,
        help="Optional model_provider passed to /query.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model_name passed to /query.",
    )
    parser.add_argument(
        "--use-long-term-memory",
        action="store_true",
        help="Enable long-term memory profile injection. Short-term session memory is always used.",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip GET /health before running cases.",
    )
    parser.add_argument(
        "--save-events",
        action="store_true",
        help="Save per-case raw SSE events under the report directory.",
    )
    parser.add_argument(
        "--fail-on-failed-cases",
        action="store_true",
        help="Exit non-zero when any case fails deterministic scoring.",
    )
    args = parser.parse_args()
    args.base_url = normalize_base_url(args.base_url)
    if args.suite is None:
        args.suite = ["ability"]
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    return args


def print_summary(summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    print("\nEvaluation summary")
    print(f"report_dir: {summary['report_dir']}")
    print(f"total_cases: {overall['total_cases']}")
    print(f"e2e_passed_cases: {overall['e2e_passed_cases']}")
    for key in [
        "end_to_end_task_success_rate",
        "tool_selection_accuracy",
        "skill_selection_accuracy",
        "mcp_selection_accuracy_observed",
        "tool_call_success_rate",
        "first_event_latency_p95_ms",
        "end_to_end_p95_latency_ms",
    ]:
        print(f"{key}: {overall.get(key)}")


def main() -> int:
    args = parse_args()
    try:
        summary, _results = asyncio.run(run_evaluation(args))
    except Exception as exc:  # noqa: BLE001 - CLI should report benchmark setup errors.
        print(f"evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print_summary(summary)
    if args.fail_on_failed_cases:
        overall = summary["overall"]
        if overall["e2e_passed_cases"] != overall["total_cases"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
