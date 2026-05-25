"""Capability-level plan step model shared by planners and adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PlanStep:
    """兼容计划步骤，描述单个 planned 节点的能力、参数和失败策略。"""

    id: str
    kind: str
    title: str = ""
    description: str = ""
    skill: Optional[str] = None
    capability_kind: str = ""
    capability_name: str = ""
    operation: Optional[str] = None
    allowed_tools: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    purpose: str = ""
    success_criteria: str = ""
    fallback_strategy: str = ""
    evidence_key: str = ""
    depends_on: List[str] = field(default_factory=list)
    planner_source: str = ""
    required: bool = True
    parallel_group: Optional[str] = None
    retry_policy: int = 0
    timeout_ms: Optional[int] = None

    def __post_init__(self) -> None:
        """在 dataclass 初始化后校验和规范化字段。"""
        if self.skill and not self.capability_name and not self.capability_kind:
            self.capability_kind = self.capability_kind or "skill"
            self.capability_name = self.skill
        elif (
            self.skill
            and not self.capability_name
            and self.capability_kind == "skill"
        ):
            self.capability_name = self.skill

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        """从字典恢复当前模型对象。"""
        skill = data.get("skill")
        capability_kind = str(data.get("capability_kind", "") or "")
        capability_name = str(data.get("capability_name", "") or "")

        if skill and not capability_name:
            capability_kind = capability_kind or "skill"
            capability_name = str(skill)
        if capability_kind == "skill" and capability_name and skill is None:
            skill = capability_name

        return cls(
            id=str(data.get("id", "")),
            kind=str(data.get("kind", "tool")),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            skill=skill,
            capability_kind=capability_kind,
            capability_name=capability_name,
            operation=data.get("operation"),
            allowed_tools=list(data.get("allowed_tools", []) or []),
            params=dict(data.get("params", {}) or {}),
            purpose=str(data.get("purpose", "")),
            success_criteria=str(data.get("success_criteria", "")),
            fallback_strategy=str(data.get("fallback_strategy", "")),
            evidence_key=str(data.get("evidence_key", "")),
            depends_on=list(data.get("depends_on", []) or []),
            planner_source=str(data.get("planner_source", "")),
            required=bool(data.get("required", True)),
            parallel_group=data.get("parallel_group"),
            retry_policy=int(data.get("retry_policy", 0) or 0),
            timeout_ms=data.get("timeout_ms"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """将当前模型转换为可序列化字典。"""
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "description": self.description,
            "skill": self.skill,
            "capability_kind": self.capability_kind,
            "capability_name": self.capability_name,
            "operation": self.operation,
            "allowed_tools": list(self.allowed_tools),
            "params": dict(self.params),
            "purpose": self.purpose,
            "success_criteria": self.success_criteria,
            "fallback_strategy": self.fallback_strategy,
            "evidence_key": self.evidence_key,
            "depends_on": list(self.depends_on),
            "planner_source": self.planner_source,
            "required": self.required,
            "parallel_group": self.parallel_group,
            "retry_policy": self.retry_policy,
            "timeout_ms": self.timeout_ms,
        }
