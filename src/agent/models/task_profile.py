"""
TaskProfile — 中性任务画像，Phase 1 引入。

当前阶段：仅构造，不参与主执行路径。
收敛计划：Phase 3 (UnifiedExecutionEngine) 时替代 RouteDecision 在 TaskOrchestrator 中的使用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


LEGACY_ROUTE_MAP: Dict[str, str] = {
    "direct_task": "direct_task",
    "planned_task": "planned_task",
    "fallback_react": "fallback_react",
}


@dataclass
class TaskProfile:
    task_type: str
    complexity: str          # low | medium | high
    openness: str            # low | medium | high
    tool_need: str           # none | single | multi
    matched_skills: List[str] = field(default_factory=list)
    confidence: float = 0.0
    expected_output_schema: str = "generic_answer_v1"
    legacy_route: str = "fallback_react"

    def to_dict(self) -> Dict:
        return {
            "task_type": self.task_type,
            "complexity": self.complexity,
            "openness": self.openness,
            "tool_need": self.tool_need,
            "matched_skills": list(self.matched_skills),
            "confidence": self.confidence,
            "expected_output_schema": self.expected_output_schema,
            "legacy_route": self.legacy_route,
        }

    @classmethod
    def from_legacy_route(
        cls,
        route: str,
        task_type: str,
        confidence: float,
        matched_skills: Optional[List[str]] = None,
        expected_output_schema: str = "generic_answer_v1",
    ) -> "TaskProfile":
        """从旧 RouteDecision 字段推断 TaskProfile，保证双向可转换。"""
        skills = list(matched_skills or [])

        if route == "direct_task" and task_type == "smalltalk":
            complexity, openness, tool_need = "low", "low", "none"
        elif route == "direct_task" and task_type == "simple_qa":
            complexity, openness, tool_need = "low", "low", "none"
        elif route == "direct_task" and task_type == "single_tool_lookup":
            complexity, openness, tool_need = "low", "low", "single"
        elif route == "planned_task":
            skill_count = len(skills)
            complexity = "high" if skill_count >= 2 else "medium"
            openness = "low"
            tool_need = "multi" if skill_count >= 2 else "single"
        elif route == "fallback_react":
            complexity, openness, tool_need = "high", "high", "none"
        else:
            complexity, openness, tool_need = "medium", "medium", "single"

        return cls(
            task_type=task_type,
            complexity=complexity,
            openness=openness,
            tool_need=tool_need,
            matched_skills=skills,
            confidence=confidence,
            expected_output_schema=expected_output_schema,
            legacy_route=LEGACY_ROUTE_MAP.get(route, route),
        )
