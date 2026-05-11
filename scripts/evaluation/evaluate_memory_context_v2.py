#!/usr/bin/env python
"""Stress evaluation for AstroAgent memory context construction.

The v2 evaluation reuses the v1 MemoryService setup helpers, then adds:
- token budget sweep
- paraphrase robustness
- stale evidence avoidance
- noisy-history robustness
- expected tool rank and wrong tool injection checks

It does not call any LLM, MCP server, or external API.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.evaluate_memory_context import (  # noqa: E402
    HIT_THRESHOLD,
    append_setup_to_memory,
    build_naive_full_context,
    compute_tool_evidence_reused,
    estimate_tokens,
    format_ms,
    format_percent,
    keyword_hits,
    load_dataset,
    make_report_dir,
    percentile,
    safe_rate,
    selected_tool_text,
)
from src.memory.api.dto import BuildContextRequest  # noqa: E402
from src.memory.api.memory_service import MemoryService  # noqa: E402


DEFAULT_DATASET = REPO_ROOT / "data/eval/memory/memory_context_eval_v2.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports/evaluation/memory_context_v2"
DEFAULT_BUDGETS = [300, 600, 1000, 2000]
PRIMARY_BUDGET = 1000


def evidence_text(context: dict[str, Any]) -> str:
    return "\n".join(
        [
            context.get("context_text", ""),
            selected_tool_text(context),
            json.dumps(
                context.get("selected_recent_messages", []),
                ensure_ascii=False,
                sort_keys=True,
            ),
            json.dumps(
                context.get("selected_task_state", {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
        ]
    )


def build_context(
    service: MemoryService,
    scenario: dict[str, Any],
    tenant_id: str,
    query: str,
    max_tokens: int,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    context = service.build_context(
        BuildContextRequest(
            tenant_id=tenant_id,
            session_id=scenario["session_id"],
            query=query,
            max_tokens=max_tokens,
        )
    )
    return context, (time.perf_counter() - started) * 1000


def expected_tool_rank(
    scenario: dict[str, Any],
    context: dict[str, Any],
) -> int | None:
    expected_names = set(scenario.get("expected_tool_names", []))
    expected_keywords = list(scenario.get("expected_tool_keywords", []))
    if not expected_names:
        return None

    fallback_rank: int | None = None
    for index, tool in enumerate(context.get("selected_tool_calls", []), start=1):
        tool_name = tool.get("tool_name", "")
        tool_blob = json.dumps(tool, ensure_ascii=False, sort_keys=True)
        if tool_name not in expected_names:
            continue
        if fallback_rank is None:
            fallback_rank = index
        if not expected_keywords or keyword_hits(tool_blob, expected_keywords):
            return index
    return fallback_rank


def first_keyword_rank(
    context: dict[str, Any],
    keywords: list[str],
) -> int | None:
    if not keywords:
        return None
    for index, tool in enumerate(context.get("selected_tool_calls", []), start=1):
        tool_blob = json.dumps(tool, ensure_ascii=False, sort_keys=True)
        if keyword_hits(tool_blob, keywords):
            return index
    return None


def stale_avoidance_result(
    scenario: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any] | None:
    spec = scenario.get("stale_evidence")
    if not spec:
        return None

    fresh_keywords = list(spec.get("fresh_keywords", []))
    stale_keywords = list(spec.get("stale_keywords", []))
    text = evidence_text(context)
    fresh_hits = keyword_hits(text, fresh_keywords)
    stale_hits = keyword_hits(text, stale_keywords)
    fresh_rank = first_keyword_rank(context, fresh_keywords)
    stale_rank = first_keyword_rank(context, stale_keywords)
    fresh_is_primary = fresh_rank is not None and (
        stale_rank is None or fresh_rank < stale_rank
    )
    avoided = bool(fresh_hits and fresh_is_primary and not stale_hits)
    return {
        "fresh_hits": fresh_hits,
        "stale_hits": stale_hits,
        "fresh_rank": fresh_rank,
        "stale_rank": stale_rank,
        "fresh_is_primary": fresh_is_primary,
        "stale_evidence_avoided": avoided,
    }


def evaluate_context_result(
    service: MemoryService,
    scenario: dict[str, Any],
    context: dict[str, Any],
    latency_ms: float,
    query: str,
    max_tokens: int,
    include_context_text: bool = False,
) -> dict[str, Any]:
    text = context.get("context_text", "")
    all_text = evidence_text(context)
    expected_keywords = list(scenario.get("expected_relevant_keywords", []))
    hit_keywords = keyword_hits(text, expected_keywords)
    memory_hit_ratio = safe_rate(len(hit_keywords), len(expected_keywords))
    memory_hit = memory_hit_ratio >= HIT_THRESHOLD

    irrelevant_keywords = list(scenario.get("irrelevant_keywords", []))
    irrelevant_hits = keyword_hits(text, irrelevant_keywords)
    irrelevant_rate = safe_rate(len(irrelevant_hits), len(irrelevant_keywords))

    wrong_keywords = list(scenario.get("wrong_tool_keywords", []))
    wrong_hits = keyword_hits(all_text, wrong_keywords)
    wrong_injected = bool(wrong_hits)
    rank = expected_tool_rank(scenario, context)
    tool_evidence_reused = compute_tool_evidence_reused(scenario, context)
    stale_result = stale_avoidance_result(scenario, context)

    naive_context = build_naive_full_context(scenario)
    naive_tokens = estimate_tokens(service, naive_context)
    context_tokens = int(context.get("total_tokens") or estimate_tokens(service, text))
    result = {
        "query": query,
        "max_tokens": max_tokens,
        "memory_hit": memory_hit,
        "memory_hit_ratio": memory_hit_ratio,
        "hit_keywords": hit_keywords,
        "missed_keywords": [
            keyword for keyword in expected_keywords if keyword not in hit_keywords
        ],
        "tool_evidence_reused": tool_evidence_reused,
        "expected_tool_rank": rank,
        "irrelevant_memory_injection_rate": irrelevant_rate,
        "injected_irrelevant_keywords": irrelevant_hits,
        "wrong_tool_injected": wrong_injected,
        "wrong_tool_keywords_hit": wrong_hits,
        "stale_evidence": stale_result,
        "context_build_latency_ms": latency_ms,
        "context_tokens": context_tokens,
        "naive_full_context_tokens": naive_tokens,
        "context_token_saving": 1.0 - safe_rate(context_tokens, naive_tokens),
        "selected_tool_calls": context.get("selected_tool_calls", []),
        "selected_recent_messages": context.get("selected_recent_messages", []),
        "selected_task_state": context.get("selected_task_state"),
        "retrieval_plan": context.get("retrieval_plan", {}),
    }
    if include_context_text:
        result["context_text"] = text
    return result


def aggregate_budget_results(
    results: list[dict[str, Any]],
    budget: int,
) -> dict[str, Any]:
    required = [item for item in results if item["requires_tool_evidence"]]
    with_wrong_keywords = [item for item in results if item["wrong_tool_keyword_count"]]
    ranks = [
        item["expected_tool_rank"]
        for item in results
        if item["expected_tool_rank"] is not None
    ]
    return {
        "max_tokens": budget,
        "scenario_count": len(results),
        "memory_hit_rate": safe_rate(
            sum(1 for item in results if item["memory_hit"]),
            len(results),
        ),
        "tool_evidence_reuse_rate": safe_rate(
            sum(1 for item in required if item["tool_evidence_reused"]),
            len(required),
        ),
        "avg_irrelevant_memory_injection_rate": mean(
            [item["irrelevant_memory_injection_rate"] for item in results]
        )
        if results
        else 0.0,
        "wrong_tool_injection_rate": safe_rate(
            sum(1 for item in with_wrong_keywords if item["wrong_tool_injected"]),
            len(with_wrong_keywords),
        ),
        "avg_expected_tool_rank": mean(ranks) if ranks else None,
    }


def aggregate_overall(
    primary_results: list[dict[str, Any]],
    budget_sweep: dict[str, Any],
    paraphrase_results: list[dict[str, Any]],
) -> dict[str, Any]:
    required = [item for item in primary_results if item["requires_tool_evidence"]]
    stale_cases = [
        item
        for item in primary_results
        if item.get("stale_evidence") is not None
    ]
    noise_cases = [item for item in primary_results if item.get("noise_heavy")]
    wrong_cases = [
        item for item in primary_results if item["wrong_tool_keyword_count"] > 0
    ]
    ranks = [
        item["expected_tool_rank"]
        for item in primary_results
        if item["expected_tool_rank"] is not None
    ]
    latencies = [item["context_build_latency_ms"] for item in primary_results]

    paraphrase_passes = [
        item["memory_hit"]
        and (
            not item["requires_tool_evidence"] or item["tool_evidence_reused"]
        )
        for item in paraphrase_results
    ]
    noise_scores = [
        (
            (1.0 if item["memory_hit"] else 0.0)
            * (1.0 - item["irrelevant_memory_injection_rate"])
            * (0.0 if item["wrong_tool_injected"] else 1.0)
        )
        for item in noise_cases
    ]

    return {
        "scenario_count": len(primary_results),
        "primary_max_tokens": PRIMARY_BUDGET,
        "memory_hit_rate": safe_rate(
            sum(1 for item in primary_results if item["memory_hit"]),
            len(primary_results),
        ),
        "tool_evidence_reuse_rate": safe_rate(
            sum(1 for item in required if item["tool_evidence_reused"]),
            len(required),
        ),
        "avg_irrelevant_memory_injection_rate": mean(
            [item["irrelevant_memory_injection_rate"] for item in primary_results]
        )
        if primary_results
        else 0.0,
        "wrong_tool_injection_rate": safe_rate(
            sum(1 for item in wrong_cases if item["wrong_tool_injected"]),
            len(wrong_cases),
        ),
        "paraphrase_case_count": len(paraphrase_results),
        "paraphrase_hit_rate": safe_rate(sum(paraphrase_passes), len(paraphrase_passes)),
        "stale_evidence_case_count": len(stale_cases),
        "stale_evidence_avoidance_rate": safe_rate(
            sum(
                1
                for item in stale_cases
                if item["stale_evidence"]
                and item["stale_evidence"]["stale_evidence_avoided"]
            ),
            len(stale_cases),
        ),
        "noise_heavy_case_count": len(noise_cases),
        "noise_robustness_score": mean(noise_scores) if noise_scores else 0.0,
        "avg_expected_tool_rank": mean(ranks) if ranks else None,
        "avg_context_token_saving": mean(
            [item["context_token_saving"] for item in primary_results]
        )
        if primary_results
        else 0.0,
        "context_build_latency_avg_ms": mean(latencies) if latencies else 0.0,
        "context_build_latency_p50_ms": percentile(latencies, 50),
        "context_build_latency_p95_ms": percentile(latencies, 95),
        "context_build_latency_max_ms": max(latencies) if latencies else 0.0,
        "budget_sensitivity": budget_sweep,
    }


def evaluate_scenario(
    service: MemoryService,
    scenario: dict[str, Any],
    tenant_id: str,
    budgets: list[int],
    scenario_index: int,
) -> dict[str, Any]:
    tool_records = append_setup_to_memory(
        service=service,
        scenario=scenario,
        tenant_id=tenant_id,
        timestamp_seed=1_800_000_000.0 + scenario_index * 10_000,
    )
    raw_chars = sum(item["raw_chars"] for item in tool_records)
    summary_chars = sum(item["summary_chars"] for item in tool_records)
    compression_rate = (
        1.0 - safe_rate(summary_chars, raw_chars) if raw_chars else None
    )

    budget_results: dict[str, Any] = {}
    primary: dict[str, Any] | None = None
    for budget in budgets:
        context, latency_ms = build_context(
            service=service,
            scenario=scenario,
            tenant_id=tenant_id,
            query=scenario.get("followup_query", ""),
            max_tokens=budget,
        )
        result = evaluate_context_result(
            service=service,
            scenario=scenario,
            context=context,
            latency_ms=latency_ms,
            query=scenario.get("followup_query", ""),
            max_tokens=budget,
            include_context_text=(budget == PRIMARY_BUDGET),
        )
        budget_results[str(budget)] = result
        if budget == PRIMARY_BUDGET:
            primary = result

    if primary is None:
        primary = next(iter(budget_results.values()))

    paraphrase_results = []
    for query in scenario.get("paraphrase_queries", []):
        context, latency_ms = build_context(
            service=service,
            scenario=scenario,
            tenant_id=tenant_id,
            query=query,
            max_tokens=PRIMARY_BUDGET,
        )
        paraphrase_results.append(
            evaluate_context_result(
                service=service,
                scenario=scenario,
                context=context,
                latency_ms=latency_ms,
                query=query,
                max_tokens=PRIMARY_BUDGET,
            )
        )

    return {
        "scenario_id": scenario["scenario_id"],
        "description": scenario.get("description", ""),
        "session_id": scenario["session_id"],
        "message_count": len(scenario.get("setup", {}).get("messages", [])),
        "tool_call_count": len(scenario.get("setup", {}).get("tool_calls", [])),
        "requires_tool_evidence": bool(scenario.get("requires_tool_evidence", False)),
        "noise_heavy": bool(scenario.get("noise_heavy", False)),
        "tiny_budget_focus": bool(scenario.get("tiny_budget_focus", False)),
        "wrong_tool_keyword_count": len(scenario.get("wrong_tool_keywords", [])),
        "expected_tool_names": list(scenario.get("expected_tool_names", [])),
        "tool_summary_compression": {
            "raw_chars": raw_chars,
            "summary_chars": summary_chars,
            "compression_rate": compression_rate,
            "tool_records": tool_records,
        },
        "primary_result": primary,
        "budget_results": budget_results,
        "paraphrase_results": paraphrase_results,
    }


def flatten_primary(per_scenario: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in per_scenario:
        row = {
            **item["primary_result"],
            "scenario_id": item["scenario_id"],
            "description": item["description"],
            "requires_tool_evidence": item["requires_tool_evidence"],
            "noise_heavy": item["noise_heavy"],
            "tiny_budget_focus": item["tiny_budget_focus"],
            "wrong_tool_keyword_count": item["wrong_tool_keyword_count"],
            "message_count": item["message_count"],
            "tool_call_count": item["tool_call_count"],
        }
        rows.append(row)
    return rows


def flatten_budget(
    per_scenario: list[dict[str, Any]],
    budget: int,
) -> list[dict[str, Any]]:
    rows = []
    for item in per_scenario:
        result = item["budget_results"][str(budget)]
        rows.append(
            {
                **result,
                "scenario_id": item["scenario_id"],
                "requires_tool_evidence": item["requires_tool_evidence"],
                "wrong_tool_keyword_count": item["wrong_tool_keyword_count"],
            }
        )
    return rows


def flatten_paraphrases(per_scenario: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in per_scenario:
        for result in item["paraphrase_results"]:
            rows.append(
                {
                    **result,
                    "scenario_id": item["scenario_id"],
                    "requires_tool_evidence": item["requires_tool_evidence"],
                    "wrong_tool_keyword_count": item["wrong_tool_keyword_count"],
                }
            )
    return rows


def failure_cases(primary_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for item in primary_results:
        reasons = []
        if not item["memory_hit"]:
            reasons.append("memory_hit_failed")
        if item["requires_tool_evidence"] and not item["tool_evidence_reused"]:
            reasons.append("tool_evidence_not_reused")
        if item["wrong_tool_injected"]:
            reasons.append("wrong_tool_injected")
        if item.get("stale_evidence") and not item["stale_evidence"][
            "stale_evidence_avoided"
        ]:
            reasons.append("stale_evidence_not_avoided")
        if item["irrelevant_memory_injection_rate"] > 0:
            reasons.append("irrelevant_memory_injected")
        if reasons:
            failures.append(
                {
                    "scenario_id": item["scenario_id"],
                    "reasons": reasons,
                    "missed_keywords": item["missed_keywords"],
                    "wrong_tool_keywords_hit": item["wrong_tool_keywords_hit"],
                    "injected_irrelevant_keywords": item[
                        "injected_irrelevant_keywords"
                    ],
                    "expected_tool_rank": item["expected_tool_rank"],
                    "stale_evidence": item.get("stale_evidence"),
                }
            )
    return failures


def build_recommendations(overall: dict[str, Any], failures: list[dict[str, Any]]) -> list[str]:
    recommendations = []
    if overall["wrong_tool_injection_rate"] > 0:
        recommendations.append(
            "RetrievalPlanner 需要更强的实体约束或负向过滤，避免问北京时注入上海、问 M42 时注入 M31。"
        )
    if overall["stale_evidence_avoidance_rate"] < 1:
        recommendations.append(
            "对同一工具/实体的新旧结果应建立 freshness 规则，优先最新证据并压低旧结果。"
        )
    if overall["paraphrase_hit_rate"] < 0.8:
        recommendations.append(
            "模糊追问依赖简单词面 overlap 不稳，建议引入会话焦点、指代消解或轻量 query rewrite。"
        )
    if overall["noise_robustness_score"] < 0.8:
        recommendations.append(
            "长历史噪声下应限制 recent messages 的无差别注入，优先使用与当前焦点匹配的片段。"
        )
    if not recommendations:
        recommendations.append("当前压力集未暴露必须立即修复的问题，可继续扩大真实回放样本。")
    recommendations.append(
        f"本次共有 {len(failures)} 个 primary failure case，建议优先查看 metrics.json 中对应 context_text 和 selected_tool_calls。"
    )
    return recommendations


def write_summary_markdown(
    path: Path,
    dataset: dict[str, Any],
    report: dict[str, Any],
) -> None:
    overall = report["overall_metrics"]
    primary = report["primary_results"]
    failures = report["failure_cases"]
    recommendations = build_recommendations(overall, failures)

    lines = [
        "# Memory Context V2 Stress Evaluation Summary",
        "",
        f"Dataset: `{dataset.get('dataset_id', 'unknown')}`",
        "",
        "## Overall Metrics",
        "| Metric | Value |",
        "|---|---:|",
        f"| scenario_count | {overall['scenario_count']} |",
        f"| primary_max_tokens | {overall['primary_max_tokens']} |",
        f"| memory_hit_rate | {format_percent(overall['memory_hit_rate'])} |",
        f"| tool_evidence_reuse_rate | {format_percent(overall['tool_evidence_reuse_rate'])} |",
        f"| paraphrase_hit_rate | {format_percent(overall['paraphrase_hit_rate'])} |",
        f"| stale_evidence_avoidance_rate | {format_percent(overall['stale_evidence_avoidance_rate'])} |",
        f"| noise_robustness_score | {overall['noise_robustness_score']:.3f} |",
        f"| wrong_tool_injection_rate | {format_percent(overall['wrong_tool_injection_rate'])} |",
        f"| avg_expected_tool_rank | {overall['avg_expected_tool_rank'] if overall['avg_expected_tool_rank'] is not None else 'n/a'} |",
        f"| avg_context_token_saving | {format_percent(overall['avg_context_token_saving'])} |",
        f"| context_build_latency_avg_ms | {format_ms(overall['context_build_latency_avg_ms'])} |",
        f"| context_build_latency_p95_ms | {format_ms(overall['context_build_latency_p95_ms'])} |",
        "",
        "## Budget Sweep Results",
        "| max_tokens | Memory Hit | Tool Evidence Reuse | Irrelevant Injection | Wrong Tool Injection | Avg Expected Tool Rank |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in overall["budget_sensitivity"]:
        rank = item["avg_expected_tool_rank"]
        lines.append(
            "| {budget} | {hit} | {tool} | {irrelevant} | {wrong} | {rank} |".format(
                budget=item["max_tokens"],
                hit=format_percent(item["memory_hit_rate"]),
                tool=format_percent(item["tool_evidence_reuse_rate"]),
                irrelevant=format_percent(item["avg_irrelevant_memory_injection_rate"]),
                wrong=format_percent(item["wrong_tool_injection_rate"]),
                rank=f"{rank:.2f}" if rank is not None else "n/a",
            )
        )

    paraphrase_by_scenario = []
    for item in report["per_scenario"]:
        results = item["paraphrase_results"]
        if not results:
            continue
        passed = sum(
            1
            for result in results
            if result["memory_hit"]
            and (
                not item["requires_tool_evidence"]
                or result["tool_evidence_reused"]
            )
        )
        paraphrase_by_scenario.append((item["scenario_id"], passed, len(results)))

    lines.extend(
        [
            "",
            "## Paraphrase Robustness",
            "| Scenario | Passed | Total | Rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for scenario_id, passed, total in paraphrase_by_scenario:
        lines.append(
            f"| {scenario_id} | {passed} | {total} | {format_percent(safe_rate(passed, total))} |"
        )

    lines.extend(
        [
            "",
            "## Noise Robustness",
            "| Scenario | Memory Hit | Irrelevant Injection | Wrong Tool Injected |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in primary:
        if item["noise_heavy"]:
            lines.append(
                "| {scenario} | {hit} | {irrelevant} | {wrong} |".format(
                    scenario=item["scenario_id"],
                    hit="yes" if item["memory_hit"] else "no",
                    irrelevant=format_percent(item["irrelevant_memory_injection_rate"]),
                    wrong="yes" if item["wrong_tool_injected"] else "no",
                )
            )

    lines.extend(
        [
            "",
            "## Stale Evidence Avoidance",
            "| Scenario | Avoided | Fresh Rank | Stale Rank | Stale Hits |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for item in primary:
        stale = item.get("stale_evidence")
        if stale:
            lines.append(
                "| {scenario} | {avoided} | {fresh_rank} | {stale_rank} | {stale_hits} |".format(
                    scenario=item["scenario_id"],
                    avoided="yes" if stale["stale_evidence_avoided"] else "no",
                    fresh_rank=stale["fresh_rank"] or "n/a",
                    stale_rank=stale["stale_rank"] or "n/a",
                    stale_hits=", ".join(stale["stale_hits"]) or "none",
                )
            )

    lines.extend(
        [
            "",
            "## Failure Cases",
            "| Scenario | Reasons | Wrong Hits | Irrelevant Hits |",
            "|---|---|---|---|",
        ]
    )
    if failures:
        for item in failures:
            lines.append(
                "| {scenario} | {reasons} | {wrong} | {irrelevant} |".format(
                    scenario=item["scenario_id"],
                    reasons=", ".join(item["reasons"]),
                    wrong=", ".join(item["wrong_tool_keywords_hit"]) or "none",
                    irrelevant=", ".join(item["injected_irrelevant_keywords"]) or "none",
                )
            )
    else:
        lines.append("| none | none | none | none |")

    lines.extend(["", "## Recommendations"])
    lines.extend(f"- {item}" for item in recommendations)
    lines.extend(
        [
            "",
            "## Notes",
            "- 本评估为 synthetic stress eval，不调用真实 LLM/MCP。",
            "- 指标只评估 MemoryService.build_context 的上下文构造，不评估最终回答质量。",
            "- stale_evidence_avoidance 使用严格口径：必须命中新证据、新证据 rank 优于旧证据，并且上下文/selected_tool_calls 中不出现旧证据关键词。",
            "- noise_robustness_score 使用 primary budget 下的 memory_hit、irrelevant injection 和 wrong tool injection 合成分数。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(
    dataset_path: Path,
    output_root: Path,
    tenant_id: str = "memory_eval_v2",
    budgets: list[int] | None = None,
) -> dict[str, Any]:
    budgets = budgets or DEFAULT_BUDGETS
    if PRIMARY_BUDGET not in budgets:
        budgets = sorted([*budgets, PRIMARY_BUDGET])

    dataset = load_dataset(dataset_path)
    report_dir = make_report_dir(output_root)
    db_path = report_dir / "memory_context_eval_v2.sqlite"
    service = MemoryService(db_path=str(db_path), tenant_id=tenant_id)

    per_scenario = [
        evaluate_scenario(
            service=service,
            scenario=scenario,
            tenant_id=tenant_id,
            budgets=budgets,
            scenario_index=index,
        )
        for index, scenario in enumerate(dataset["scenarios"], start=1)
    ]
    primary_results = flatten_primary(per_scenario)
    paraphrase_results = flatten_paraphrases(per_scenario)
    budget_sweep = [
        aggregate_budget_results(flatten_budget(per_scenario, budget), budget)
        for budget in budgets
    ]
    failures = failure_cases(primary_results)
    overall = aggregate_overall(primary_results, budget_sweep, paraphrase_results)

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
        "budgets": budgets,
        "overall_metrics": overall,
        "primary_results": primary_results,
        "paraphrase_results": paraphrase_results,
        "failure_cases": failures,
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
        description="Run v2 stress evaluation for MemoryService context construction."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to v2 memory context eval dataset JSON.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for timestamped reports.",
    )
    parser.add_argument(
        "--tenant-id",
        default="memory_eval_v2",
        help="Tenant id used for synthetic memory writes.",
    )
    parser.add_argument(
        "--budgets",
        default=",".join(str(item) for item in DEFAULT_BUDGETS),
        help="Comma-separated max_tokens values for budget sweep.",
    )
    return parser.parse_args()


def parse_budgets(raw: str) -> list[int]:
    budgets = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            budgets.append(int(part))
    if not budgets:
        raise ValueError("--budgets must include at least one integer")
    return sorted(set(budgets))


def main() -> None:
    args = parse_args()
    report = run_evaluation(
        dataset_path=args.dataset,
        output_root=args.output_root,
        tenant_id=args.tenant_id,
        budgets=parse_budgets(args.budgets),
    )
    overall = report["overall_metrics"]
    print("Memory context v2 stress evaluation complete")
    print(f"metrics_json: {report['metrics_path']}")
    print(f"summary_md: {report['summary_path']}")
    print(f"memory_hit_rate: {format_percent(overall['memory_hit_rate'])}")
    print(f"tool_evidence_reuse_rate: {format_percent(overall['tool_evidence_reuse_rate'])}")
    print(f"paraphrase_hit_rate: {format_percent(overall['paraphrase_hit_rate'])}")
    print(
        "stale_evidence_avoidance_rate: "
        f"{format_percent(overall['stale_evidence_avoidance_rate'])}"
    )
    print(f"noise_robustness_score: {overall['noise_robustness_score']:.3f}")
    print(f"wrong_tool_injection_rate: {format_percent(overall['wrong_tool_injection_rate'])}")
    print(
        "context_build_latency_avg/p95/max_ms: "
        f"{overall['context_build_latency_avg_ms']:.2f}/"
        f"{overall['context_build_latency_p95_ms']:.2f}/"
        f"{overall['context_build_latency_max_ms']:.2f}"
    )


if __name__ == "__main__":
    main()
