#!/usr/bin/env python
"""Synthetic memory context evaluation for AstroAgent.

This script exercises MemoryService directly with controlled messages, tool
outputs, and task state patches. It does not call any LLM, MCP server, or
external API.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.memory.api.dto import (  # noqa: E402
    AppendMessageRequest,
    AppendToolCallRequest,
    BuildContextRequest,
)
from src.memory.api.memory_service import MemoryService  # noqa: E402


DEFAULT_DATASET = REPO_ROOT / "data/eval/memory/memory_context_eval_v1.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports/evaluation/memory_context"
HIT_THRESHOLD = 0.6


def load_dataset(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {"dataset_id": path.stem, "scenarios": payload}
    if not isinstance(payload, dict) or not isinstance(payload.get("scenarios"), list):
        raise ValueError(f"dataset must be a JSON object with scenarios: {path}")
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


def contains_keyword(text: str, keyword: str) -> bool:
    if not keyword:
        return False
    return keyword.lower() in text.lower()


def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    return [keyword for keyword in keywords if contains_keyword(text, keyword)]


def estimate_tokens(service: MemoryService, text: str) -> int:
    return service._estimate_tokens(text)  # Reuse the same estimator as build_context.


def build_naive_full_context(scenario: dict[str, Any]) -> str:
    setup = scenario.get("setup", {})
    parts: list[str] = []
    for message in setup.get("messages", []):
        role = message.get("role", "message")
        parts.append(f"{role}: {message.get('content', '')}")
    for tool_call in setup.get("tool_calls", []):
        parts.append(
            "\n".join(
                [
                    f"tool_name: {tool_call.get('tool_name', '')}",
                    f"tool_input: {tool_call.get('tool_input', '')}",
                    f"tool_raw_output: {tool_call.get('raw_output', '')}",
                ]
            )
        )
    if setup.get("task_state_patch"):
        parts.append(
            "task_state_patch: "
            + json.dumps(setup["task_state_patch"], ensure_ascii=False, sort_keys=True)
        )
    return "\n\n".join(parts)


def safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil((pct / 100.0) * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def format_ms(value: float) -> str:
    return f"{value:.2f} ms"


def append_setup_to_memory(
    service: MemoryService,
    scenario: dict[str, Any],
    tenant_id: str,
    timestamp_seed: float,
) -> list[dict[str, Any]]:
    setup = scenario.get("setup", {})
    session_id = scenario["session_id"]
    tool_records: list[dict[str, Any]] = []
    timestamp = timestamp_seed

    for index, message in enumerate(setup.get("messages", [])):
        service.append_message(
            AppendMessageRequest(
                tenant_id=tenant_id,
                session_id=session_id,
                user_id="memory_eval",
                role=message["role"],
                content=message["content"],
                timestamp=timestamp + index,
            )
        )

    timestamp += len(setup.get("messages", [])) + 1
    for index, tool_call in enumerate(setup.get("tool_calls", [])):
        raw_output = tool_call.get("raw_output", "")
        record = service.append_tool_call(
            AppendToolCallRequest(
                tenant_id=tenant_id,
                session_id=session_id,
                user_id="memory_eval",
                tool_name=tool_call["tool_name"],
                tool_input=tool_call.get("tool_input", ""),
                raw_output=raw_output,
                success=bool(tool_call.get("success", True)),
                content_type=tool_call.get("content_type", "application/json"),
                timestamp=timestamp + index,
            )
        )
        summary = record.output_summary or record.output_digest or ""
        raw_chars = len(raw_output)
        summary_chars = len(summary)
        tool_records.append(
            {
                "tool_name": record.tool_name,
                "tool_call_id": record.tool_call_id,
                "raw_artifact_id": record.raw_artifact_id,
                "raw_chars": raw_chars,
                "summary_chars": summary_chars,
                "compression_rate": 1.0 - safe_rate(summary_chars, raw_chars),
                "output_summary": summary,
            }
        )

    task_state_patch = setup.get("task_state_patch")
    if task_state_patch:
        service.update_task_state(
            session_id=session_id,
            tenant_id=tenant_id,
            patch=task_state_patch,
            created_by="memory_eval",
        )

    return tool_records


def selected_tool_text(context: dict[str, Any]) -> str:
    return json.dumps(
        context.get("selected_tool_calls", []),
        ensure_ascii=False,
        sort_keys=True,
    )


def compute_tool_evidence_reused(
    scenario: dict[str, Any],
    context: dict[str, Any],
) -> bool:
    if not scenario.get("requires_tool_evidence", False):
        return False

    expected_tool_names = scenario.get("expected_tool_names", [])
    expected_keywords = scenario.get("expected_relevant_keywords", [])
    selected_tools = context.get("selected_tool_calls", [])
    selected_tool_names = {tool.get("tool_name", "") for tool in selected_tools}
    evidence_text = context.get("context_text", "") + "\n" + selected_tool_text(context)

    tool_name_hit = any(
        tool_name in selected_tool_names or contains_keyword(evidence_text, tool_name)
        for tool_name in expected_tool_names
    )
    output_keyword_hit = any(
        contains_keyword(evidence_text, keyword)
        for keyword in expected_keywords
        if keyword not in expected_tool_names
    )
    artifact_hit = any(tool.get("raw_artifact_id") for tool in selected_tools)
    return bool(tool_name_hit and (output_keyword_hit or artifact_hit))


def evaluate_scenario(
    service: MemoryService,
    scenario: dict[str, Any],
    tenant_id: str,
    max_tokens: int,
    scenario_index: int,
) -> dict[str, Any]:
    timestamp_seed = 1_700_000_000.0 + scenario_index * 1_000
    tool_records = append_setup_to_memory(
        service=service,
        scenario=scenario,
        tenant_id=tenant_id,
        timestamp_seed=timestamp_seed,
    )

    request = BuildContextRequest(
        tenant_id=tenant_id,
        session_id=scenario["session_id"],
        query=scenario.get("followup_query", ""),
        max_tokens=max_tokens,
    )
    started = time.perf_counter()
    context = service.build_context(request)
    latency_ms = (time.perf_counter() - started) * 1000

    context_text = context.get("context_text", "")
    expected_keywords = list(scenario.get("expected_relevant_keywords", []))
    hit_keywords = keyword_hits(context_text, expected_keywords)
    hit_ratio = safe_rate(len(hit_keywords), len(expected_keywords))
    memory_hit = hit_ratio >= HIT_THRESHOLD

    irrelevant_keywords = list(scenario.get("irrelevant_keywords", []))
    injected_irrelevant_keywords = keyword_hits(context_text, irrelevant_keywords)
    irrelevant_injection_rate = safe_rate(
        len(injected_irrelevant_keywords), len(irrelevant_keywords)
    )

    naive_full_context = build_naive_full_context(scenario)
    naive_tokens = estimate_tokens(service, naive_full_context)
    context_tokens = int(context.get("total_tokens") or estimate_tokens(service, context_text))
    context_token_saving = 1.0 - safe_rate(context_tokens, naive_tokens)

    total_raw_chars = sum(item["raw_chars"] for item in tool_records)
    total_summary_chars = sum(item["summary_chars"] for item in tool_records)
    compression_rate = (
        1.0 - safe_rate(total_summary_chars, total_raw_chars)
        if total_raw_chars
        else None
    )
    tool_evidence_reused = compute_tool_evidence_reused(scenario, context)

    return {
        "scenario_id": scenario["scenario_id"],
        "description": scenario.get("description", ""),
        "session_id": scenario["session_id"],
        "followup_query": scenario.get("followup_query", ""),
        "memory_hit": memory_hit,
        "memory_hit_ratio": hit_ratio,
        "memory_hit_keyword_count": len(hit_keywords),
        "expected_keyword_count": len(expected_keywords),
        "hit_keywords": hit_keywords,
        "missed_keywords": [
            keyword for keyword in expected_keywords if keyword not in hit_keywords
        ],
        "context_token_saving": context_token_saving,
        "context_tokens": context_tokens,
        "naive_full_context_tokens": naive_tokens,
        "tool_summary_compression": {
            "raw_chars": total_raw_chars,
            "summary_chars": total_summary_chars,
            "compression_rate": compression_rate,
            "tool_records": tool_records,
        },
        "irrelevant_memory_injection_rate": irrelevant_injection_rate,
        "injected_irrelevant_keyword_count": len(injected_irrelevant_keywords),
        "total_irrelevant_keyword_count": len(irrelevant_keywords),
        "injected_irrelevant_keywords": injected_irrelevant_keywords,
        "tool_evidence_reused": tool_evidence_reused,
        "requires_tool_evidence": bool(scenario.get("requires_tool_evidence", False)),
        "expected_tool_names": list(scenario.get("expected_tool_names", [])),
        "context_build_latency_ms": latency_ms,
        "context_text": context_text,
        "selected_recent_messages": context.get("selected_recent_messages", []),
        "selected_tool_calls": context.get("selected_tool_calls", []),
        "selected_salient_facts": context.get("selected_salient_facts", []),
        "selected_task_state": context.get("selected_task_state"),
        "selected_summary_snapshot": context.get("selected_summary_snapshot"),
        "retrieval_plan": context.get("retrieval_plan", {}),
        "total_tokens": context_tokens,
    }


def aggregate_results(per_scenario: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_count = len(per_scenario)
    latencies = [item["context_build_latency_ms"] for item in per_scenario]
    compression_rates = [
        item["tool_summary_compression"]["compression_rate"]
        for item in per_scenario
        if item["tool_summary_compression"]["compression_rate"] is not None
    ]
    required_tool_cases = [
        item for item in per_scenario if item.get("requires_tool_evidence", False)
    ]

    return {
        "scenario_count": scenario_count,
        "memory_hit_rate": safe_rate(
            sum(1 for item in per_scenario if item["memory_hit"]),
            scenario_count,
        ),
        "avg_context_token_saving": mean(
            [item["context_token_saving"] for item in per_scenario]
        )
        if per_scenario
        else 0.0,
        "avg_tool_summary_compression_rate": mean(compression_rates)
        if compression_rates
        else 0.0,
        "avg_irrelevant_memory_injection_rate": mean(
            [item["irrelevant_memory_injection_rate"] for item in per_scenario]
        )
        if per_scenario
        else 0.0,
        "context_build_latency_avg_ms": mean(latencies) if latencies else 0.0,
        "context_build_latency_p50_ms": percentile(latencies, 50),
        "context_build_latency_p95_ms": percentile(latencies, 95),
        "context_build_latency_max_ms": max(latencies) if latencies else 0.0,
        "tool_evidence_required_cases": len(required_tool_cases),
        "tool_evidence_reuse_rate": safe_rate(
            sum(1 for item in required_tool_cases if item["tool_evidence_reused"]),
            len(required_tool_cases),
        ),
    }


def build_findings(per_scenario: list[dict[str, Any]], overall: dict[str, Any]) -> list[str]:
    good = [
        item["scenario_id"]
        for item in per_scenario
        if item["memory_hit"]
        and (
            not item["requires_tool_evidence"] or item["tool_evidence_reused"]
        )
        and not item["injected_irrelevant_keywords"]
    ]
    missed = [
        f"{item['scenario_id']} missed {', '.join(item['missed_keywords'])}"
        for item in per_scenario
        if not item["memory_hit"]
    ]
    irrelevant = [
        f"{item['scenario_id']} injected {', '.join(item['injected_irrelevant_keywords'])}"
        for item in per_scenario
        if item["injected_irrelevant_keywords"]
    ]
    weak_compression = [
        (
            f"{item['scenario_id']} "
            f"({format_percent(item['tool_summary_compression']['compression_rate'])})"
        )
        for item in per_scenario
        if item["tool_summary_compression"]["compression_rate"] is not None
        and item["tool_summary_compression"]["compression_rate"] < 0.2
    ]

    findings = [
        "表现较好的 scenario: " + (", ".join(good) if good else "无"),
        "未达到 memory_hit 阈值的 scenario: " + ("; ".join(missed) if missed else "无"),
        "无关记忆注入: " + ("; ".join(irrelevant) if irrelevant else "未检测到"),
        (
            "平均工具摘要压缩率为 "
            f"{format_percent(overall['avg_tool_summary_compression_rate'])}，"
            "仅统计包含工具调用的 scenario。"
        ),
        "低/负压缩率 scenario: "
        + (
            ", ".join(weak_compression)
            if weak_compression
            else "无，当前工具摘要压缩率整体合理"
        ),
        (
            "context 构建延迟 avg/p95/max = "
            f"{overall['context_build_latency_avg_ms']:.2f}/"
            f"{overall['context_build_latency_p95_ms']:.2f}/"
            f"{overall['context_build_latency_max_ms']:.2f} ms。"
        ),
    ]
    return findings


def write_summary_markdown(
    path: Path,
    dataset: dict[str, Any],
    report: dict[str, Any],
) -> None:
    overall = report["overall_metrics"]
    per_scenario = report["per_scenario"]
    findings = build_findings(per_scenario, overall)

    rows = [
        ("scenario_count", str(overall["scenario_count"])),
        ("memory_hit_rate", format_percent(overall["memory_hit_rate"])),
        ("avg_context_token_saving", format_percent(overall["avg_context_token_saving"])),
        (
            "avg_tool_summary_compression_rate",
            format_percent(overall["avg_tool_summary_compression_rate"]),
        ),
        (
            "avg_irrelevant_memory_injection_rate",
            format_percent(overall["avg_irrelevant_memory_injection_rate"]),
        ),
        ("context_build_latency_avg_ms", format_ms(overall["context_build_latency_avg_ms"])),
        ("context_build_latency_p50_ms", format_ms(overall["context_build_latency_p50_ms"])),
        ("context_build_latency_p95_ms", format_ms(overall["context_build_latency_p95_ms"])),
        ("context_build_latency_max_ms", format_ms(overall["context_build_latency_max_ms"])),
        ("tool_evidence_reuse_rate", format_percent(overall["tool_evidence_reuse_rate"])),
    ]

    lines = [
        "# Memory Context Evaluation Summary",
        "",
        f"Dataset: `{dataset.get('dataset_id', 'unknown')}`",
        "",
        "## Overall Metrics",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(f"| {metric} | {value} |" for metric, value in rows)
    lines.extend(
        [
            "",
            "## Per Scenario Results",
            (
                "| Scenario | Memory Hit | Token Saving | Compression | "
                "Irrelevant Injection | Tool Evidence Reused | Latency |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in per_scenario:
        compression = item["tool_summary_compression"]["compression_rate"]
        lines.append(
            "| {scenario} | {hit} | {saving} | {compression} | {irrelevant} | "
            "{evidence} | {latency} |".format(
                scenario=item["scenario_id"],
                hit="yes" if item["memory_hit"] else "no",
                saving=format_percent(item["context_token_saving"]),
                compression=format_percent(compression),
                irrelevant=format_percent(item["irrelevant_memory_injection_rate"]),
                evidence=(
                    "n/a"
                    if not item["requires_tool_evidence"]
                    else ("yes" if item["tool_evidence_reused"] else "no")
                ),
                latency=format_ms(item["context_build_latency_ms"]),
            )
        )

    lines.extend(["", "## Findings"])
    lines.extend(f"- {finding}" for finding in findings)
    lines.extend(
        [
            "",
            "## Notes",
            "- 本评估为 synthetic eval，不调用真实 LLM/MCP。",
            "- 指标只评估 memory context construction，不评估最终回答质量。",
            "- 当前 MemoryService 支持通过 DTO 直接写入 message/tool_call；task_state 通过 update_task_state patch 写入。",
            "- naive_full_context 由原始 message、完整 tool raw_output 和 task_state_patch 拼接得到，并复用 MemoryService token estimator。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(
    dataset_path: Path,
    output_root: Path,
    tenant_id: str = "memory_eval",
    max_tokens: int = 4000,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    report_dir = make_report_dir(output_root)
    db_path = report_dir / "memory_context_eval.sqlite"
    service = MemoryService(db_path=str(db_path), tenant_id=tenant_id)

    per_scenario = [
        evaluate_scenario(
            service=service,
            scenario=scenario,
            tenant_id=tenant_id,
            max_tokens=max_tokens,
            scenario_index=index,
        )
        for index, scenario in enumerate(dataset["scenarios"], start=1)
    ]
    overall = aggregate_results(per_scenario)

    report = {
        "dataset": {
            "dataset_id": dataset.get("dataset_id", dataset_path.stem),
            "dataset_path": str(dataset_path),
            "description": dataset.get("description", ""),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "db_path": str(db_path),
        "report_dir": str(report_dir),
        "hit_threshold": HIT_THRESHOLD,
        "max_tokens": max_tokens,
        "overall_metrics": overall,
        "per_scenario": per_scenario,
    }

    metrics_path = report_dir / "metrics.json"
    summary_path = report_dir / "summary.md"
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_markdown(summary_path, dataset, report)
    report["metrics_path"] = str(metrics_path)
    report["summary_path"] = str(summary_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MemoryService context construction on synthetic scenarios."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to memory context eval dataset JSON.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for timestamped metrics and summary reports.",
    )
    parser.add_argument(
        "--tenant-id",
        default="memory_eval",
        help="Tenant id used for synthetic memory writes.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="Token budget passed to MemoryService.build_context.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_evaluation(
        dataset_path=args.dataset,
        output_root=args.output_root,
        tenant_id=args.tenant_id,
        max_tokens=args.max_tokens,
    )
    overall = report["overall_metrics"]
    print("Memory context evaluation complete")
    print(f"metrics_json: {report['metrics_path']}")
    print(f"summary_md: {report['summary_path']}")
    print(f"memory_hit_rate: {format_percent(overall['memory_hit_rate'])}")
    print(
        "avg_context_token_saving: "
        f"{format_percent(overall['avg_context_token_saving'])}"
    )
    print(
        "avg_tool_summary_compression_rate: "
        f"{format_percent(overall['avg_tool_summary_compression_rate'])}"
    )
    print(
        "avg_irrelevant_memory_injection_rate: "
        f"{format_percent(overall['avg_irrelevant_memory_injection_rate'])}"
    )
    print(
        "context_build_latency_avg/p50/p95/max_ms: "
        f"{overall['context_build_latency_avg_ms']:.2f}/"
        f"{overall['context_build_latency_p50_ms']:.2f}/"
        f"{overall['context_build_latency_p95_ms']:.2f}/"
        f"{overall['context_build_latency_max_ms']:.2f}"
    )
    print(f"tool_evidence_reuse_rate: {format_percent(overall['tool_evidence_reuse_rate'])}")


if __name__ == "__main__":
    main()
