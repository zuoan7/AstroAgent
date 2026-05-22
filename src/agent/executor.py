from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.agent.models.execution_plan import ExecutionPlan
from src.agent.models.skill_result import SkillResult
from src.core.mcp_protocol import TOOL_INPUT_MODELS

KNOWN_MCP_TOOL_NAMES = frozenset(TOOL_INPUT_MODELS)


EventCallback = Callable[[str, Dict[str, Any]], Awaitable[None] | None]
ParamBuilder = Callable[[str, str], Dict[str, Any]]


def _extract_mcp_tools_from_sources(sources: List[Dict[str, Any]]) -> List[str]:
    tools: List[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        tool_name = source.get("tool")
        if (
            isinstance(tool_name, str)
            and tool_name in KNOWN_MCP_TOOL_NAMES
            and tool_name not in tools
        ):
            tools.append(tool_name)
    return tools


@dataclass
class StepExecutionResult:
    step_id: str
    title: str
    kind: str
    status: str
    skill: Optional[str] = None
    capability_kind: str = ""
    capability_name: str = ""
    capability_reason: str = ""
    input_params: Dict[str, Any] = field(default_factory=dict)
    param_builder_source: str = ""
    mcp_tools_used: List[str] = field(default_factory=list)
    logical_skill: Optional[str] = None
    operation: Optional[str] = None
    expected_mcp_tools: List[str] = field(default_factory=list)
    attempts: int = 0
    required: bool = True
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    summary: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    evidence_key: str = ""
    fallback_strategy: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "kind": self.kind,
            "status": self.status,
            "skill": self.skill,
            "capability_kind": self.capability_kind,
            "capability_name": self.capability_name,
            "capability_reason": self.capability_reason,
            "input_params": dict(self.input_params),
            "param_builder_source": self.param_builder_source,
            "mcp_tools_used": list(self.mcp_tools_used),
            "logical_skill": self.logical_skill,
            "operation": self.operation,
            "expected_mcp_tools": list(self.expected_mcp_tools),
            "attempts": self.attempts,
            "required": self.required,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "summary": self.summary,
            "sources": list(self.sources),
            "depends_on": list(self.depends_on),
            "evidence_key": self.evidence_key,
            "fallback_strategy": self.fallback_strategy,
        }

    def to_trace_entry(self) -> "ExecutionTraceEntry":  # noqa: F821
        """转换为统一 trace 模型（Phase 7 引入）。"""
        from src.agent.models.execution_trace_entry import ExecutionTraceEntry

        return ExecutionTraceEntry.from_step_result(self)


@dataclass
class ExecutionOutcome:
    plan: ExecutionPlan
    skill_results: List[SkillResult] = field(default_factory=list)
    step_results: List[StepExecutionResult] = field(default_factory=list)
    evidence_by_key: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    evidence_items: List[Dict[str, Any]] = field(default_factory=list)
    skipped_step_ids: List[str] = field(default_factory=list)
    halted: bool = False
    halt_reason: Optional[str] = None


class EvidenceAggregator:
    """Collect stable evidence records from DAG step results."""

    def __init__(self) -> None:
        self._by_key: Dict[str, Dict[str, Any]] = {}
        self._items: List[Dict[str, Any]] = []
        self._seen_sources: set[tuple[str, str, str]] = set()

    def add(
        self,
        *,
        key: str,
        step_result: StepExecutionResult,
        skill_result: Optional[SkillResult],
    ) -> None:
        evidence_key = key or step_result.evidence_key or step_result.step_id
        record = {
            "key": evidence_key,
            "step_id": step_result.step_id,
            "skill": step_result.skill,
            "capability_kind": step_result.capability_kind,
            "capability_name": step_result.capability_name,
            "status": step_result.status,
            "summary": step_result.summary,
            "sources": list(step_result.sources),
            "error": step_result.error,
        }
        if skill_result is not None and skill_result.success and skill_result.data:
            record["data"] = dict(skill_result.data)
        self._by_key[evidence_key] = record

        if step_result.summary:
            self._append_item(
                {
                    "source_id": evidence_key,
                    "kind": "tool_evidence",
                    "title": step_result.title
                    or step_result.skill
                    or step_result.capability_name
                    or step_result.step_id,
                    "snippet": step_result.summary[:240],
                    "tool": step_result.skill or step_result.capability_name,
                    "step_id": step_result.step_id,
                    "status": step_result.status,
                }
            )
        for source in step_result.sources:
            if isinstance(source, dict):
                self._append_item(source)

    def _append_item(self, item: Dict[str, Any]) -> None:
        identity = (
            str(item.get("source_id") or ""),
            str(item.get("title") or ""),
            str(item.get("snippet") or ""),
        )
        if identity in self._seen_sources:
            return
        self._seen_sources.add(identity)
        self._items.append(dict(item))

    def snapshot(self) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        return dict(self._by_key), list(self._items)
