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
    """Compatibility plan model.

    planned 主路径已优先使用 WorkflowGraph；ExecutionPlan 继续保留给旧序列化、
    旧接口输入以及 TaskOrchestrator / StreamingService 兼容回退链路。
    """
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

    @classmethod
    def from_workflow_graph(
        cls,
        graph: Any,
        *,
        task_type: Optional[str] = None,
    ) -> "ExecutionPlan":
        """[Compatibility adapter] 从 WorkflowGraph 恢复 ExecutionPlan。

        用于 graph-first planned 路径下，向旧序列化/观测/Outcome 结构提供兼容 Plan。
        """
        steps: List[PlanStep] = []
        for node in graph.topological_order():
            steps.append(
                PlanStep(
                    id=node.id,
                    kind=node.kind,
                    title=node.title,
                    skill=node.skill,
                    params=dict(node.inputs or {}),
                    required=not node.optional,
                    timeout_ms=node.timeout_ms,
                )
            )

        metadata = getattr(graph, "metadata", {}) or {}
        resolved_task_type = (
            task_type
            or metadata.get("task_type")
            or "observation_recommendation"
        )

        return cls(
            task_type=resolved_task_type,
            output_schema=getattr(graph, "output_schema", "generic_answer_v1"),
            steps=steps,
            planner_type=metadata.get("planner_type", "graph_native"),
            rationale=metadata.get("rationale", ""),
            planner_version=metadata.get("planner_version", "planner_v2"),
            schema_version=metadata.get("schema_version", "schema_v2"),
            budget_policy_version=metadata.get(
                "budget_policy_version", "budget_v1"
            ),
        )
