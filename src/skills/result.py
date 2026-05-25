"""技能调用结果模型，统一技能数据、摘要、来源、错误和工具审计字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SkillResult:
    """技能或原子工具调用的结构化结果。"""
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
        """把技能结果转换为 legacy LangChain Tool 字符串。"""
        if not self.success and self.error_message:
            return f"[错误] {self.error_message}"
        return self.summary

    def to_dict(self) -> Dict[str, Any]:
        """将当前模型转换为可序列化字典。"""
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
        """把技能结果转换为前端工具时间线条目。"""
        return {
            "run_id": run_id or self.skill_name,
            "tool": self.skill_name,
            "display_tool": self.skill_name,
            "input": {},
            "output_summary": self.summary[:240],
            "status": "success" if self.success else "error",
            "latency_ms": self.latency_ms,
            "logical_skill": self.logical_skill or self.skill_name,
            "operation": self.operation,
            "mcp_tools_used": [
                source.get("tool")
                for source in self.sources
                if isinstance(source, dict)
                and source.get("kind") in {"mcp_tool", "tool_output"}
            ],
            "expected_mcp_tools": list(self.expected_mcp_tools),
        }

    def to_evidence_entry(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """把技能结果转换为统一证据来源条目。"""
        return {
            "source_id": run_id or self.skill_name,
            "kind": "tool_output",
            "title": self.skill_name,
            "snippet": self.summary[:240],
            "tool": self.skill_name,
            "logical_skill": self.logical_skill or self.skill_name,
            "expected_mcp_tools": list(self.expected_mcp_tools),
        }

    @classmethod
    def from_error(
        cls,
        skill_name: str,
        error_code: str,
        error_message: str,
        latency_ms: Optional[float] = None,
    ) -> SkillResult:
        """根据错误信息构造失败的 SkillResult。"""
        return cls(
            skill_name=skill_name,
            success=False,
            data={},
            summary=f"[错误] {error_message}",
            error_code=error_code,
            error_message=error_message,
            latency_ms=latency_ms,
        )
