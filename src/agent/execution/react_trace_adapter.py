from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from src.agent.models.execution_trace_entry import ExecutionTraceEntry
from src.skills import registry as skill_registry
from src.skills.definition import SkillDefinition
from src.tools.registry import ToolRegistry, get_default_tool_registry


@dataclass(frozen=True)
class ReactToolMapping:
    langchain_tool_name: str
    logical_skill: str
    capability_kind: str = ""
    capability_name: str = ""
    capability_reason: str = ""
    expected_mcp_tools: tuple[str, ...] = ()
    mcp_tools_used: tuple[str, ...] = ()


class ReactToolTraceAdapter:
    """Map legacy ReAct tool calls into the unified execution trace shape."""

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self._tool_registry = tool_registry or get_default_tool_registry()
        self._skill_definitions = list(skill_registry.get_skill_definitions())

    def build_entry(
        self,
        *,
        step_id: str,
        tool_name: str,
        tool_input: Any,
        tool_output: Any,
        status: str = "",
        duration_sec: Optional[float] = None,
    ) -> ExecutionTraceEntry:
        mapping = self.map_tool(tool_name)
        output_summary = self.preview_text(tool_output, 240)
        resolved_status = status or self.status_from_output(tool_output)
        return ExecutionTraceEntry.from_react_tool(
            step_id=step_id,
            tool_name=tool_name,
            tool_input=self.stringify_value(tool_input),
            output_summary=output_summary,
            status=resolved_status,
            duration_sec=duration_sec,
            logical_skill=mapping.logical_skill,
            capability_kind=mapping.capability_kind,
            capability_name=mapping.capability_name,
            capability_reason=mapping.capability_reason,
            expected_mcp_tools=list(mapping.expected_mcp_tools),
            mcp_tools_used=list(mapping.mcp_tools_used),
        )

    def tool_usage_from_entry(self, entry: ExecutionTraceEntry) -> Dict[str, Any]:
        return {
            "run_id": entry.step_id,
            "tool": entry.tool_name or entry.skill or entry.step_id,
            "display_tool": entry.logical_skill or entry.tool_name or entry.step_id,
            "langchain_tool_name": entry.tool_name,
            "logical_skill": entry.logical_skill or entry.skill or entry.tool_name,
            "input": entry.tool_input or "",
            "output_summary": entry.tool_output_summary or entry.summary,
            "duration_sec": entry.duration_sec,
            "status": entry.status,
            "capability_kind": entry.capability_kind,
            "capability_name": entry.capability_name,
            "mcp_tools_used": list(entry.mcp_tools_used),
            "expected_mcp_tools": list(entry.expected_mcp_tools),
        }

    def map_tool(self, tool_name: str) -> ReactToolMapping:
        name = str(tool_name or "").strip()
        if name == "RAGRetrieve":
            return ReactToolMapping(
                langchain_tool_name=name,
                logical_skill=name,
                capability_reason="react_rag_tool",
            )

        definition = self._skill_definition_for_tool_name(name)
        if definition is not None:
            return ReactToolMapping(
                langchain_tool_name=name,
                logical_skill=definition.name,
                capability_kind="skill",
                capability_name=definition.name,
                capability_reason="react_skill_mapping",
                expected_mcp_tools=tuple(definition.allowed_tools),
            )

        if self._tool_registry.has_tool(name):
            return ReactToolMapping(
                langchain_tool_name=name,
                logical_skill=name,
                capability_kind="tool",
                capability_name=name,
                capability_reason="react_atomic_tool_mapping",
                expected_mcp_tools=(name,),
                mcp_tools_used=(name,),
            )

        return ReactToolMapping(
            langchain_tool_name=name,
            logical_skill=name or "unknown_tool",
            capability_reason="react_unknown_tool",
        )

    @staticmethod
    def summarize_audit(trace: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        trace_items = [item for item in trace if isinstance(item, dict)]
        return {
            "react_tools_used": ReactToolTraceAdapter._unique(
                str(item.get("logical_skill") or item.get("tool_name") or "")
                for item in trace_items
            ),
            "react_mcp_tools_used": ReactToolTraceAdapter._unique(
                tool
                for item in trace_items
                for tool in list(item.get("mcp_tools_used") or [])
            ),
            "react_expected_mcp_tools": ReactToolTraceAdapter._unique(
                tool
                for item in trace_items
                for tool in list(item.get("expected_mcp_tools") or [])
            ),
            "react_trace_count": len(trace_items),
        }

    @staticmethod
    def status_from_output(value: Any) -> str:
        text = ReactToolTraceAdapter.stringify_value(value).lower()
        if text.startswith("[错误]") or "error" in text or "错误" in text:
            return "error"
        return "success"

    @staticmethod
    def preview_text(value: Any, max_len: int) -> str:
        text = ReactToolTraceAdapter.stringify_value(value)
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    @staticmethod
    def stringify_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list, tuple)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return str(value)
        return str(value)

    def _skill_definition_for_tool_name(
        self, tool_name: str
    ) -> Optional[SkillDefinition]:
        lowered = tool_name.lower()
        for definition in self._skill_definitions:
            if tool_name in {definition.display_name, definition.name}:
                return definition
            if lowered in {definition.display_name.lower(), definition.name.lower()}:
                return definition
        return None

    def _expected_mcp_tools_for_skill(self, skill_name: str) -> list[str]:
        try:
            if self._tool_registry.has_tool(skill_name):
                return [skill_name]
            return list(skill_registry.get_skill_definition(skill_name).allowed_tools)
        except Exception:
            return []

    @staticmethod
    def _unique(values: Iterable[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            if value and value not in result:
                result.append(value)
        return result
