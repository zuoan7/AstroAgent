from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PlanStep:
    id: str
    kind: str
    title: str = ""
    description: str = ""
    skill: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    required: bool = True
    parallel_group: Optional[str] = None
    retry_policy: int = 0
    timeout_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "description": self.description,
            "skill": self.skill,
            "params": dict(self.params),
            "required": self.required,
            "parallel_group": self.parallel_group,
            "retry_policy": self.retry_policy,
            "timeout_ms": self.timeout_ms,
        }


@dataclass
class ExecutionPlan:
    task_type: str
    output_schema: str
    steps: List[PlanStep] = field(default_factory=list)
    planner_type: str = "template"
    rationale: str = ""
    planner_version: str = "planner_v2"
    schema_version: str = "schema_v2"
    budget_policy_version: str = "budget_v1"

    def tool_steps(self) -> List[PlanStep]:
        return [s for s in self.steps if s.kind == "tool"]

    def required_steps(self) -> List[PlanStep]:
        return [s for s in self.steps if s.required]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "output_schema": self.output_schema,
            "planner_type": self.planner_type,
            "rationale": self.rationale,
            "planner_version": self.planner_version,
            "schema_version": self.schema_version,
            "budget_policy_version": self.budget_policy_version,
            "steps": [step.to_dict() for step in self.steps],
        }

    def to_frontend_steps(self) -> List[Dict[str, Any]]:
        frontend_steps: List[Dict[str, Any]] = []
        for step in self.steps:
            frontend_steps.append(
                {
                    "id": step.id,
                    "title": step.title or step.skill or step.id,
                    "description": step.description
                    or (f"执行技能 {step.skill}" if step.skill else step.kind),
                    "status": "pending",
                    "kind": step.kind,
                    "skill": step.skill,
                    "required": step.required,
                    "parallel_group": step.parallel_group,
                }
            )
        return frontend_steps
