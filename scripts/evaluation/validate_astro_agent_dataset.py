#!/usr/bin/env python
"""Validate the AstroAgent benchmark dataset without calling the agent."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "config/benchmarks/astro_agent_eval_dataset.json"

CASE_REQUIRED_FIELDS = {
    "case_id",
    "suite",
    "prompt",
    "turns",
    "attachments",
    "category",
    "subcategory",
    "difficulty",
    "requires_tool",
    "should_clarify",
    "expected_route",
    "expected_skills",
    "forbidden_skills",
    "expected_skill_sequence",
    "expected_mcp_tools",
    "forbidden_mcp_tools",
    "expected_params",
    "param_match_rule",
    "time_context",
    "geo_context",
    "expected_answer_structure",
    "success_criteria",
    "judge_type",
    "latency_bucket",
    "timeout_ms",
    "weight",
    "tags",
    "notes",
}

LIST_FIELDS = {
    "turns",
    "attachments",
    "expected_skills",
    "forbidden_skills",
    "expected_skill_sequence",
    "expected_mcp_tools",
    "forbidden_mcp_tools",
    "success_criteria",
    "tags",
}


@dataclass(frozen=True)
class Issue:
    level: str
    code: str
    message: str
    case_id: str | None = None

    def format(self) -> str:
        prefix = f"[{self.level}] {self.code}"
        if self.case_id:
            prefix += f" {self.case_id}"
        return f"{prefix}: {self.message}"


def load_dataset(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("dataset root must be a JSON object")
    if not isinstance(payload.get("cases"), list):
        raise ValueError("dataset must contain a cases array")
    if not isinstance(payload.get("target_distribution"), list):
        raise ValueError("dataset must contain a target_distribution array")
    return payload


def add_issue(
    issues: list[Issue],
    level: str,
    code: str,
    message: str,
    case_id: str | None = None,
) -> None:
    issues.append(Issue(level=level, code=code, message=message, case_id=case_id))


def as_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value}


def validate_case(
    case: dict[str, Any],
    *,
    allowed: dict[str, Any],
    known_categories: set[str],
    known_subcategories: dict[str, set[str]],
    issues: list[Issue],
) -> None:
    case_id = str(case.get("case_id", ""))
    missing_fields = sorted(CASE_REQUIRED_FIELDS - set(case))
    if missing_fields:
        add_issue(
            issues,
            "error",
            "missing_fields",
            f"missing required fields: {', '.join(missing_fields)}",
            case_id or None,
        )

    if not case_id:
        add_issue(issues, "error", "empty_case_id", "case_id must be non-empty")

    prompt = case.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        add_issue(issues, "error", "empty_prompt", "prompt must be non-empty", case_id)

    for field in LIST_FIELDS:
        if field in case and not isinstance(case.get(field), list):
            add_issue(
                issues,
                "error",
                "invalid_list_field",
                f"{field} must be a list",
                case_id,
            )

    for field in ["expected_params", "time_context", "geo_context"]:
        if field in case and not isinstance(case.get(field), dict):
            add_issue(
                issues,
                "error",
                "invalid_object_field",
                f"{field} must be an object",
                case_id,
            )

    category = str(case.get("category", ""))
    if category not in known_categories:
        add_issue(
            issues,
            "error",
            "unknown_category",
            f"category is not in target_distribution: {category}",
            case_id,
        )
    else:
        subcategory = str(case.get("subcategory", ""))
        if subcategory not in known_subcategories.get(category, set()):
            add_issue(
                issues,
                "error",
                "unknown_subcategory",
                f"subcategory {subcategory!r} is not declared for {category}",
                case_id,
            )

    for field, allowed_key in [
        ("suite", "suite"),
        ("difficulty", "difficulty"),
        ("expected_route", "expected_route"),
        ("param_match_rule", "param_match_rule"),
        ("judge_type", "judge_type"),
        ("latency_bucket", "latency_bucket"),
    ]:
        value = case.get(field)
        if value not in set(allowed.get(allowed_key, [])):
            add_issue(
                issues,
                "error",
                "invalid_allowed_value",
                f"{field}={value!r} is not allowed",
                case_id,
            )

    allowed_skills = set(allowed.get("skills", []))
    allowed_mcp_tools = set(allowed.get("mcp_tools", []))
    expected_skills = as_set(case.get("expected_skills"))
    forbidden_skills = as_set(case.get("forbidden_skills"))
    expected_sequence = as_set(case.get("expected_skill_sequence"))
    expected_mcp = as_set(case.get("expected_mcp_tools"))
    forbidden_mcp = as_set(case.get("forbidden_mcp_tools"))

    for skill in sorted(expected_skills | forbidden_skills | expected_sequence):
        if skill not in allowed_skills:
            add_issue(
                issues,
                "error",
                "unknown_skill",
                f"skill is not allowed: {skill}",
                case_id,
            )
    for tool in sorted(expected_mcp | forbidden_mcp):
        if tool not in allowed_mcp_tools:
            add_issue(
                issues,
                "error",
                "unknown_mcp_tool",
                f"MCP tool is not allowed: {tool}",
                case_id,
            )

    skill_overlap = expected_skills & forbidden_skills
    if skill_overlap:
        add_issue(
            issues,
            "error",
            "skill_expectation_conflict",
            f"skills are both expected and forbidden: {sorted(skill_overlap)}",
            case_id,
        )

    mcp_overlap = expected_mcp & forbidden_mcp
    if mcp_overlap:
        add_issue(
            issues,
            "error",
            "mcp_expectation_conflict",
            f"MCP tools are both expected and forbidden: {sorted(mcp_overlap)}",
            case_id,
        )

    if not case.get("requires_tool") and (expected_skills or expected_mcp):
        add_issue(
            issues,
            "error",
            "no_tool_case_expects_tool",
            "requires_tool=false cases must not declare expected tools",
            case_id,
        )

    if case.get("requires_tool") and not (expected_skills or expected_mcp):
        add_issue(
            issues,
            "error",
            "tool_case_without_expected_tool",
            "requires_tool=true cases should declare expected skills or MCP tools",
            case_id,
        )

    turns = case.get("turns", [])
    if turns:
        if not all(isinstance(turn, dict) for turn in turns):
            add_issue(
                issues,
                "error",
                "invalid_turns",
                "turns must contain objects",
                case_id,
            )
        else:
            last_user_turn = next(
                (
                    str(turn.get("content", ""))
                    for turn in reversed(turns)
                    if turn.get("role") == "user"
                ),
                "",
            )
            if last_user_turn and last_user_turn != prompt:
                add_issue(
                    issues,
                    "warning",
                    "prompt_turn_mismatch",
                    "prompt does not match the final user turn",
                    case_id,
                )

    timeout_ms = case.get("timeout_ms")
    if not isinstance(timeout_ms, int) or timeout_ms <= 0:
        add_issue(
            issues,
            "error",
            "invalid_timeout",
            "timeout_ms must be a positive integer",
            case_id,
        )

    weight = case.get("weight")
    if not isinstance(weight, (int, float)) or weight < 0:
        add_issue(
            issues,
            "error",
            "invalid_weight",
            "weight must be a non-negative number",
            case_id,
        )


def validate_dataset(
    data: dict[str, Any],
    *,
    expected_total: int | None = None,
) -> list[Issue]:
    issues: list[Issue] = []
    cases = data["cases"]
    allowed = data.get("allowed_values", {})
    target_distribution = data["target_distribution"]

    targets: dict[str, int] = {}
    known_subcategories: dict[str, set[str]] = {}
    for item in target_distribution:
        if not isinstance(item, dict):
            add_issue(
                issues,
                "error",
                "invalid_target_distribution",
                "target_distribution entries must be objects",
            )
            continue
        category = str(item.get("category", ""))
        target_count = item.get("target_count")
        if not category:
            add_issue(
                issues,
                "error",
                "empty_target_category",
                "target_distribution category must be non-empty",
            )
            continue
        if category in targets:
            add_issue(
                issues,
                "error",
                "duplicate_target_category",
                f"duplicate category in target_distribution: {category}",
            )
        if not isinstance(target_count, int) or target_count < 0:
            add_issue(
                issues,
                "error",
                "invalid_target_count",
                f"target_count must be a non-negative integer for {category}",
            )
            target_count = 0
        targets[category] = target_count
        known_subcategories[category] = set(
            str(subcategory) for subcategory in item.get("subcategories", [])
        )

    known_categories = set(targets)
    case_ids = [str(case.get("case_id", "")) for case in cases if isinstance(case, dict)]
    duplicate_ids = sorted(
        case_id for case_id, count in Counter(case_ids).items() if case_id and count > 1
    )
    for case_id in duplicate_ids:
        add_issue(
            issues,
            "error",
            "duplicate_case_id",
            "case_id must be unique",
            case_id,
        )

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            add_issue(
                issues,
                "error",
                "invalid_case",
                f"case at index {index} must be an object",
            )
            continue
        validate_case(
            case,
            allowed=allowed,
            known_categories=known_categories,
            known_subcategories=known_subcategories,
            issues=issues,
        )

    category_counts = Counter(str(case.get("category", "")) for case in cases if isinstance(case, dict))
    for category, target_count in sorted(targets.items()):
        actual = category_counts.get(category, 0)
        if actual != target_count:
            add_issue(
                issues,
                "error",
                "target_count_mismatch",
                f"{category}: actual {actual}, target {target_count}",
            )

    unknown_count_categories = sorted(set(category_counts) - known_categories - {""})
    for category in unknown_count_categories:
        add_issue(
            issues,
            "error",
            "category_without_target",
            f"{category}: actual {category_counts[category]}, no target declared",
        )

    target_total = sum(targets.values())
    if len(cases) != target_total:
        add_issue(
            issues,
            "error",
            "dataset_total_mismatch",
            f"cases total {len(cases)} does not match target total {target_total}",
        )

    if expected_total is not None and len(cases) != expected_total:
        add_issue(
            issues,
            "error",
            "expected_total_mismatch",
            f"cases total {len(cases)} does not match expected total {expected_total}",
        )

    return issues


def print_report(data: dict[str, Any], issues: list[Issue]) -> None:
    cases = data["cases"]
    category_counts = Counter(case.get("category", "") for case in cases)
    suite_counts = Counter(case.get("suite", "") for case in cases)
    error_count = sum(1 for issue in issues if issue.level == "error")
    warning_count = sum(1 for issue in issues if issue.level == "warning")

    print("AstroAgent dataset validation")
    print(f"dataset_id: {data.get('dataset_id', '')}")
    print(f"cases: {len(cases)}")
    print(f"errors: {error_count}")
    print(f"warnings: {warning_count}")
    print(f"suites: {dict(sorted(suite_counts.items()))}")
    print("categories:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")

    if issues:
        print("\nissues:")
        for issue in issues:
            print(f"  - {issue.format()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the AstroAgent benchmark dataset without calling the agent."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Dataset JSON path. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=None,
        help="Optional expected total case count.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        data = load_dataset(args.dataset)
    except Exception as exc:
        print(f"failed to load dataset: {exc}", file=sys.stderr)
        return 2

    issues = validate_dataset(data, expected_total=args.expected_total)
    print_report(data, issues)

    has_errors = any(issue.level == "error" for issue in issues)
    has_warnings = any(issue.level == "warning" for issue in issues)
    if has_errors or (args.strict and has_warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
