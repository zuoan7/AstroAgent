from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SkillResult:
    skill_name: str
    success: bool
    data: Dict[str, Any]
    summary: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    latency_ms: Optional[float] = None
    logical_skill: Optional[str] = None
    operation: Optional[str] = None
    expected_mcp_tools: List[str] = field(default_factory=list)
    allowed_child_tools: List[str] = field(default_factory=list)
    forbidden_child_tools: List[str] = field(default_factory=list)

    def to_legacy_str(self) -> str:
        if not self.success and self.error_message:
            return f"[错误] {self.error_message}"
        return self.summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "success": self.success,
            "data": self.data,
            "summary": self.summary,
            "sources": self.sources,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "latency_ms": self.latency_ms,
            "logical_skill": self.logical_skill,
            "operation": self.operation,
            "expected_mcp_tools": list(self.expected_mcp_tools),
            "allowed_child_tools": list(self.allowed_child_tools),
            "forbidden_child_tools": list(self.forbidden_child_tools),
        }

    def to_tool_timeline_entry(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        return {
            "run_id": run_id or self.skill_name,
            "tool": self.skill_name,
            "input": {},
            "output_summary": self.summary[:240],
            "status": "success" if self.success else "error",
            "latency_ms": self.latency_ms,
            "logical_skill": self.logical_skill or self.skill_name,
            "operation": self.operation,
            "expected_mcp_tools": list(self.expected_mcp_tools),
        }

    def to_evidence_entry(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        return {
            "source_id": run_id or self.skill_name,
            "kind": "tool_output",
            "title": self.skill_name,
            "snippet": self.summary[:240],
            "tool": self.skill_name,
        }

    @classmethod
    def from_error(
        cls,
        skill_name: str,
        error_code: str,
        error_message: str,
        latency_ms: Optional[float] = None,
    ) -> SkillResult:
        return cls(
            skill_name=skill_name,
            success=False,
            data={},
            summary=f"[错误] {error_message}",
            error_code=error_code,
            error_message=error_message,
            latency_ms=latency_ms,
        )
