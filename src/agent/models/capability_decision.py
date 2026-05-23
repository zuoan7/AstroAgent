"""能力选择结果模型，描述执行器下一步应调用的高层技能、原子工具或无需能力。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


CapabilityKind = Literal["none", "skill", "tool"]


@dataclass(frozen=True)
class CapabilityDecision:
    """Agent-facing decision for the next executable capability.

    This is intentionally narrower than TaskProfile and wider than a concrete
    MCP tool call. It gives executors a stable handoff point between task
    profiling and skill/tool execution while the legacy matched_skills path is
    still supported.
    """

    kind: CapabilityKind
    name: str = ""
    confidence: float = 0.0
    reason: str = ""
    required_params: List[str] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    forbidden_tools: List[str] = field(default_factory=list)
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def none(cls, *, reason: str = "", confidence: float = 0.0) -> "CapabilityDecision":
        """构造无需额外技能或工具的能力选择结果。"""
        return cls(kind="none", reason=reason, confidence=confidence)

    @classmethod
    def for_skill(
        cls,
        name: str,
        *,
        confidence: float,
        reason: str,
        required_params: Optional[List[str]] = None,
        allowed_tools: Optional[List[str]] = None,
        forbidden_tools: Optional[List[str]] = None,
        operation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CapabilityDecision":
        """构造指向高层技能的能力选择结果。"""
        return cls(
            kind="skill",
            name=name,
            confidence=confidence,
            reason=reason,
            required_params=list(required_params or []),
            allowed_tools=list(allowed_tools or []),
            forbidden_tools=list(forbidden_tools or []),
            operation=operation,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def for_tool(
        cls,
        name: str,
        *,
        confidence: float,
        reason: str,
        required_params: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CapabilityDecision":
        """构造指向原子工具的能力选择结果。"""
        return cls(
            kind="tool",
            name=name,
            confidence=confidence,
            reason=reason,
            required_params=list(required_params or []),
            allowed_tools=[name],
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """将当前模型转换为可序列化字典。"""
        return {
            "kind": self.kind,
            "name": self.name,
            "confidence": self.confidence,
            "reason": self.reason,
            "required_params": list(self.required_params),
            "allowed_tools": list(self.allowed_tools),
            "forbidden_tools": list(self.forbidden_tools),
            "operation": self.operation,
            "metadata": dict(self.metadata),
        }
