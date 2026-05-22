from __future__ import annotations

from typing import Any, Optional

from src.agent.models.execution_plan import PlanStep
from src.capabilities.registry import CapabilityRegistry, get_default_capability_registry


class CapabilityPlanAdapter:
    """Build PlanStep objects from capability metadata.

    `PlanStep.skill` is kept as a legacy mirror for skill capabilities. New code
    should read `capability_kind` and `capability_name` first.
    """

    def __init__(self, registry: Optional[CapabilityRegistry] = None) -> None:
        self._registry = registry or get_default_capability_registry()

    def make_skill_step(
        self,
        *,
        skill_name: str,
        id: str,
        title: str,
        description: str = "",
        params: Optional[dict[str, Any]] = None,
        operation: Optional[str] = None,
        purpose: str = "",
        success_criteria: str = "",
        fallback_strategy: str = "",
        evidence_key: str = "",
        depends_on: Optional[list[str]] = None,
        planner_source: str = "",
        required: bool = True,
        parallel_group: Optional[str] = None,
        retry_policy: int = 0,
        timeout_ms: Optional[int] = None,
    ) -> PlanStep:
        spec = self._registry.get_skill(skill_name)
        return PlanStep(
            id=id,
            kind="tool",
            title=title,
            description=description,
            skill=skill_name,
            capability_kind="skill",
            capability_name=skill_name,
            operation=operation,
            allowed_tools=list(spec.allowed_tools),
            params=dict(params or {}),
            purpose=purpose,
            success_criteria=success_criteria,
            fallback_strategy=fallback_strategy,
            evidence_key=evidence_key,
            depends_on=list(depends_on or []),
            planner_source=planner_source,
            required=required,
            parallel_group=parallel_group,
            retry_policy=retry_policy,
            timeout_ms=timeout_ms,
        )

    def make_tool_step(
        self,
        *,
        tool_name: str,
        id: str,
        title: str,
        description: str = "",
        params: Optional[dict[str, Any]] = None,
        purpose: str = "",
        success_criteria: str = "",
        fallback_strategy: str = "",
        evidence_key: str = "",
        depends_on: Optional[list[str]] = None,
        planner_source: str = "",
        required: bool = True,
        parallel_group: Optional[str] = None,
        retry_policy: int = 0,
        timeout_ms: Optional[int] = None,
    ) -> PlanStep:
        self._registry.get_tool(tool_name)
        return PlanStep(
            id=id,
            kind="tool",
            title=title,
            description=description,
            skill=None,
            capability_kind="tool",
            capability_name=tool_name,
            allowed_tools=[tool_name],
            params=dict(params or {}),
            purpose=purpose,
            success_criteria=success_criteria,
            fallback_strategy=fallback_strategy,
            evidence_key=evidence_key,
            depends_on=list(depends_on or []),
            planner_source=planner_source,
            required=required,
            parallel_group=parallel_group,
            retry_policy=retry_policy,
            timeout_ms=timeout_ms,
        )
