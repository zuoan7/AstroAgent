#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.planner import Planner
from src.agent.request_router import RequestRouter
from src.agent.skill_param_builder import SkillParamBuilder
from src.skills import registry


DEFAULT_DATASET = Path("config/benchmarks/astro_agent_eval_dataset.json")
DEFAULT_OUTPUT = Path("reports/evaluation/astro_agent_static")


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def load_dataset(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError(f"invalid dataset: {path}")
    return data


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


def make_report_dir(root: Path) -> Path:
    path = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    path.mkdir(parents=True, exist_ok=False)
    return path


def infer_composite_mcp_tools(skill_name: str, params: dict[str, Any]) -> list[str]:
    if skill_name == "weather-lookup":
        return ["get_weather"]
    if skill_name == "observation-planner":
        tools = ["get_weekly_events"]
        if params.get("location"):
            tools.insert(0, "get_weather")
        date = str(params.get("date") or "")
        if not date or any(token in date for token in ("今晚", "今天", "今日")):
            tools.append("get_tonight_best")
        return tools
    if skill_name == "deep-sky-observing-guide":
        tools = ["get_astrophysical_object_info"]
        target = str(params.get("target") or "").lower().replace(" ", "")
        if "星系" in str(params.get("target") or "") or target in {
            "m31",
            "m33",
            "m51",
            "m81",
            "m82",
            "m87",
            "m101",
            "m104",
            "ngc224",
            "ngc598",
        }:
            tools.append("get_galaxy_data")
        if params.get("observer_location"):
            tools.append("get_weather")
        return tools
    if skill_name == "neo-tracker":
        return ["get_neo_data"]
    return []


def infer_operation_mcp_tools(skill_name: str, params: dict[str, Any]) -> list[str]:
    operation = str(params.get("operation") or "").strip()
    if operation:
        try:
            return [
                registry.get_operation_spec(
                    logical_skill=skill_name,
                    operation=operation,
                ).atomic_tool_name
            ]
        except KeyError:
            pass

    if skill_name == "celestial-position-calculator":
        output_format = str(params.get("output_format") or "").lower()
        if output_format == "altaz":
            return ["get_altaz"]
        if output_format in {"rise_set", "rise-set", "riseset"}:
            return ["get_rise_set_times"]
        return ["get_planet_position"]

    if skill_name == "celestial-events-forecast":
        if params.get("end_date"):
            return ["get_monthly_events"]
        return ["get_weekly_events"]

    return infer_composite_mcp_tools(skill_name, params)


def build_static_plan(
    *,
    query: str,
    profile: Any,
    planner: Planner,
    param_builder: SkillParamBuilder,
) -> tuple[list[str], list[str], list[dict[str, Any]], dict[str, Any] | None]:
    route_decision = profile.to_legacy_route_decision()
    plan_dict: dict[str, Any] | None = None
    plan_steps: list[dict[str, Any]] = []

    if profile.legacy_route == "planned_task":
        plan = planner.plan(query=query, route_decision=route_decision)
        plan_dict = plan.to_dict()
        for step in plan.steps:
            if step.kind != "tool" or not step.skill:
                continue
            params = dict(param_builder.build(step.skill, query))
            params.update(step.params or {})
            mcp_tools = infer_operation_mcp_tools(step.skill, params)
            plan_steps.append(
                {
                    "id": step.id,
                    "skill": step.skill,
                    "params": params,
                    "planner_source": step.planner_source or plan.planner_type,
                    "mcp_tools": mcp_tools,
                    "operation": params.get("operation"),
                }
            )
    elif profile.task_type == "single_tool_lookup" and profile.matched_skills:
        skill = profile.matched_skills[0]
        params = dict(param_builder.build(skill, query))
        mcp_tools = infer_operation_mcp_tools(skill, params)
        plan_steps.append(
            {
                "id": "direct_tool",
                "skill": skill,
                "params": params,
                "planner_source": "",
                "mcp_tools": mcp_tools,
                "operation": params.get("operation"),
            }
        )

    actual_skills = unique_preserve_order(
        [step["skill"] for step in plan_steps if step.get("skill")]
    )
    actual_mcp_tools = unique_preserve_order(
        [
            tool
            for step in plan_steps
            for tool in list(step.get("mcp_tools") or [])
        ]
    )
    return actual_skills, actual_mcp_tools, plan_steps, plan_dict


def score_case(
    index: int,
    case: dict[str, Any],
    *,
    router: RequestRouter,
    planner: Planner,
    param_builder: SkillParamBuilder,
    mcp_scoring: str,
) -> dict[str, Any]:
    query = str(case.get("prompt") or "")
    profile = router.profile(query)
    actual_skills, actual_mcp_tools, plan_steps, plan = build_static_plan(
        query=query,
        profile=profile,
        planner=planner,
        param_builder=param_builder,
    )

    expected_skills = set(case.get("expected_skills") or [])
    forbidden_skills = set(case.get("forbidden_skills") or [])
    expected_mcp = set(case.get("expected_mcp_tools") or [])
    forbidden_mcp = set(case.get("forbidden_mcp_tools") or [])
    actual_skill_set = set(actual_skills)
    actual_mcp_set = set(actual_mcp_tools)

    missing_expected_skills = sorted(expected_skills - actual_skill_set)
    forbidden_skill_hits = sorted(forbidden_skills & actual_skill_set)
    skill_selection_pass = not missing_expected_skills and not forbidden_skill_hits

    missing_expected_mcp = sorted(expected_mcp - actual_mcp_set)
    forbidden_mcp_hits = sorted(forbidden_mcp & actual_mcp_set)
    if mcp_scoring == "ignore":
        missing_expected_mcp = []
        mcp_selection_pass = not forbidden_mcp_hits
    elif mcp_scoring == "observed" and expected_mcp and not actual_mcp_tools:
        missing_expected_mcp = []
        mcp_selection_pass = None
    else:
        mcp_selection_pass = not missing_expected_mcp and not forbidden_mcp_hits

    requires_tool_missing = bool(
        case.get("requires_tool")
        and (expected_skills or expected_mcp)
        and not (actual_skills or actual_mcp_tools)
    )
    unexpected_tool_for_no_tool_case = bool(
        not case.get("requires_tool")
        and bool(actual_skills or actual_mcp_tools)
    )
    tool_selection_pass = (
        skill_selection_pass
        and mcp_selection_pass is not False
        and not forbidden_mcp_hits
        and not requires_tool_missing
        and not unexpected_tool_for_no_tool_case
    )

    route_pass = (
        not case.get("expected_route")
        or profile.legacy_route == case.get("expected_route")
    )
    static_pass = bool(tool_selection_pass)

    failure_reasons: list[str] = []
    route_diagnostics: list[str] = []
    if not route_pass:
        route_diagnostics.append(
            f"route_mismatch:{case.get('expected_route')}!={profile.legacy_route}"
        )
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

    return {
        "index": index,
        "case_id": case.get("case_id"),
        "suite": case.get("suite"),
        "category": case.get("category"),
        "subcategory": case.get("subcategory"),
        "requires_tool": case.get("requires_tool"),
        "should_clarify": case.get("should_clarify"),
        "prompt": query,
        "expected_route": case.get("expected_route"),
        "actual_route": profile.legacy_route,
        "task_type": profile.task_type,
        "router_source": profile.router_source,
        "route_reason": profile.reason,
        "tool_necessity_action": profile.tool_necessity_action,
        "tool_necessity_reason": profile.tool_necessity_reason,
        "expected_skills": sorted(expected_skills),
        "forbidden_skills": sorted(forbidden_skills),
        "expected_mcp_tools": sorted(expected_mcp),
        "forbidden_mcp_tools": sorted(forbidden_mcp),
        "actual_skills": actual_skills,
        "actual_mcp_tools": actual_mcp_tools,
        "plan_steps": plan_steps,
        "execution_plan": plan,
        "route_pass": route_pass,
        "route_diagnostics": route_diagnostics,
        "skill_selection_pass": skill_selection_pass,
        "mcp_selection_pass": mcp_selection_pass,
        "tool_selection_pass": tool_selection_pass,
        "static_pass": static_pass,
        "failure_reasons": failure_reasons,
    }


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def summarize_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    mcp_relevant = [
        result
        for result in results
        if result.get("expected_mcp_tools")
        or result.get("actual_mcp_tools")
        or result.get("forbidden_mcp_tools")
    ]
    mcp_scored = [
        result for result in mcp_relevant if result.get("mcp_selection_pass") is not None
    ]
    planned = [
        result
        for result in results
        if result.get("actual_route") == "planned_task"
        or result.get("execution_plan")
    ]
    return {
        "total_cases": total,
        "static_passed_cases": sum(1 for result in results if result.get("static_pass")),
        "static_success_rate": safe_rate(
            sum(1 for result in results if result.get("static_pass")),
            total,
        ),
        "route_accuracy": safe_rate(
            sum(1 for result in results if result.get("route_pass")),
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
        "mcp_selection_scored_cases": len(mcp_scored),
        "mcp_selection_accuracy_static": safe_rate(
            sum(1 for result in mcp_scored if result.get("mcp_selection_pass")),
            len(mcp_scored),
        ),
        "planned_cases": len(planned),
    }


def build_summary(
    *,
    data: dict[str, Any],
    args: argparse.Namespace,
    report_dir: Path,
    results: list[dict[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_category[str(result.get("category", ""))].append(result)
    return {
        "dataset_id": data.get("dataset_id"),
        "dataset_path": str(args.dataset),
        "run_started_at": started_at,
        "run_finished_at": datetime.now().isoformat(timespec="seconds"),
        "suite": args.suite,
        "category_filter": args.category,
        "case_id_filter": args.case_id,
        "mcp_scoring": args.mcp_scoring,
        "report_dir": str(report_dir),
        "overall": summarize_group(results),
        "by_category": {
            category: summarize_group(items)
            for category, items in sorted(by_category.items())
        },
    }


def write_outputs(report_dir: Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (report_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for result in sorted(results, key=lambda item: item.get("index", 0)):
            json.dump(result, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")

    failures = [result for result in results if not result.get("static_pass")]
    with (report_dir / "failures.md").open("w", encoding="utf-8") as handle:
        handle.write("# AstroAgent Static Routing/Planning Failures\n\n")
        handle.write(f"Total failures: {len(failures)}\n\n")
        for result in sorted(failures, key=lambda item: item.get("index", 0)):
            handle.write(f"## {result.get('case_id')}\n\n")
            handle.write(f"- category: `{result.get('category')}`\n")
            handle.write(f"- prompt: {result.get('prompt')}\n")
            handle.write(f"- reasons: {', '.join(result.get('failure_reasons') or [])}\n")
            if result.get("route_diagnostics"):
                handle.write(f"- route_diagnostics: `{result.get('route_diagnostics')}`\n")
            handle.write(f"- route: `{result.get('actual_route')}` expected `{result.get('expected_route')}`\n")
            handle.write(f"- actual_skills: `{result.get('actual_skills')}`\n")
            handle.write(f"- actual_mcp_tools: `{result.get('actual_mcp_tools')}`\n")
            handle.write(f"- plan_steps: `{result.get('plan_steps')}`\n\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fast offline router/planner checks without API, MCP, LLM synthesis, "
            "or tool execution."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--suite", action="append", default=None)
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-per-category", type=int, default=None)
    parser.add_argument(
        "--mcp-scoring",
        choices=["observed", "strict", "ignore"],
        default="strict",
    )
    parser.add_argument("--fail-on-failed-cases", action="store_true")
    args = parser.parse_args()
    if args.suite is None:
        args.suite = ["ability"]
    return args


def main() -> int:
    args = parse_args()
    data = load_dataset(args.dataset)
    cases = select_cases(data["cases"], args)
    if not cases:
        raise ValueError("no cases selected")

    report_dir = make_report_dir(args.output)
    started_at = datetime.now().isoformat(timespec="seconds")
    router = RequestRouter(enable_llm_fallback=False)
    planner = Planner(llm=None)
    param_builder = SkillParamBuilder(None)

    print(f"Selected cases: {len(cases)}")
    print(f"Report directory: {report_dir}")
    results = [
        score_case(
            index=index,
            case=case,
            router=router,
            planner=planner,
            param_builder=param_builder,
            mcp_scoring=args.mcp_scoring,
        )
        for index, case in enumerate(cases, start=1)
    ]

    for index, result in enumerate(results, start=1):
        status = "PASS" if result.get("static_pass") else "FAIL"
        print(
            f"[{index}/{len(results)}] {status} {result.get('case_id')} "
            f"route={result.get('actual_route')} "
            f"skills={result.get('actual_skills')} "
            f"mcp={result.get('actual_mcp_tools')}"
        )

    summary = build_summary(
        data=data,
        args=args,
        report_dir=report_dir,
        results=results,
        started_at=started_at,
    )
    write_outputs(report_dir, summary, results)

    overall = summary["overall"]
    print("\nStatic routing/planning summary")
    print(f"report_dir: {report_dir}")
    for key in (
        "total_cases",
        "static_passed_cases",
        "static_success_rate",
        "route_accuracy",
        "tool_selection_accuracy",
        "skill_selection_accuracy",
        "mcp_selection_accuracy_static",
        "planned_cases",
    ):
        print(f"{key}: {overall.get(key)}")

    if args.fail_on_failed_cases and overall.get("static_passed_cases") != overall.get("total_cases"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
