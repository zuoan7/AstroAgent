"""Adapters between atomic ToolResult and legacy SkillResult evidence."""

from __future__ import annotations

import json
from typing import Any

from src.skills.result import SkillResult
from src.tools.results import ToolResult


def tool_result_to_skill_result(
    result: ToolResult,
    *,
    skill_name: str | None = None,
) -> SkillResult:
    """Wrap one atomic tool result as a legacy SkillResult for synthesis/audit."""
    name = skill_name or result.tool_name
    snippet = _summary_text(result.data)
    sources = [
        {
            "kind": "mcp_tool",
            "tool": result.tool_name,
            "snippet": str(snippet)[:240],
        }
    ]

    if not result.ok:
        failed = SkillResult.from_error(
            skill_name=name,
            error_code=result.error.code if result.error else "TOOL_CALL_FAILED",
            error_message=result.error.message if result.error else "工具调用失败",
            latency_ms=result.latency_ms,
        )
        failed.logical_skill = name
        failed.expected_mcp_tools = [result.tool_name]
        failed.allowed_child_tools = [result.tool_name]
        failed.sources = sources
        return failed

    payload = result.data
    data = payload if isinstance(payload, dict) else {"raw": payload}
    return SkillResult(
        skill_name=name,
        success=True,
        data=data,
        summary=_summary_text(payload),
        sources=sources,
        latency_ms=result.latency_ms,
        logical_skill=name,
        expected_mcp_tools=[result.tool_name],
        allowed_child_tools=[result.tool_name],
    )


def _summary_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False)
