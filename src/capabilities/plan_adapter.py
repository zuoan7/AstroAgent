"""能力计划适配器，把技能或原子工具能力转换为 planned 路径 PlanStep。"""

from __future__ import annotations

from typing import Any, Optional

from src.capabilities.plan import PlanStep
from src.skills.registry import SkillRegistry, get_default_skill_registry
from src.tools.registry import ToolRegistry, get_default_tool_registry


class CapabilityPlanAdapter:
    """根据能力元数据构建 PlanStep。

    PlanStep.skill 只保留为技能能力的 legacy 镜像；新代码应优先读取
    capability_kind 和 capability_name。
    """

    def __init__(
        self,
        skill_registry: Optional[SkillRegistry] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        """初始化技能和工具注册表依赖。"""
        self._skill_registry = skill_registry or get_default_skill_registry()
        self._tool_registry = tool_registry or get_default_tool_registry()

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
        """根据高层技能能力创建 planned 路径步骤。"""
        definition = self._skill_registry.get(skill_name)
        return PlanStep(
            id=id,
            kind="tool",
            title=title,
            description=description,
            skill=skill_name,
            capability_kind="skill",
            capability_name=skill_name,
            operation=operation,
            allowed_tools=list(definition.allowed_tools),
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
        """根据原子工具能力创建 planned 路径步骤。"""
        self._tool_registry.get_tool(tool_name)
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
