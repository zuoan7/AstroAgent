"""能力选择器，把 TaskProfile 和执行决策转换为下一步可执行能力。"""

from __future__ import annotations

from typing import Any, Optional

from src.capabilities.decision import CapabilityDecision
from src.capabilities.param_builder import CapabilityParamBuilder
from src.skills.registry import SkillRegistry, get_default_skill_registry
from src.tools.registry import ToolRegistry, get_default_tool_registry
from src.tools.selector import ToolSelector


class CapabilitySelector:
    """在 TaskProfile 和可执行能力之间做兼容选择。"""

    def __init__(
        self,
        skill_registry: Optional[SkillRegistry] = None,
        tool_registry: Optional[ToolRegistry] = None,
        tool_selector: Optional[ToolSelector] = None,
    ) -> None:
        """初始化技能/工具注册表和原子工具规则选择器。"""
        self._skill_registry = skill_registry or get_default_skill_registry()
        self._tool_registry = tool_registry or get_default_tool_registry()
        self._tool_selector = tool_selector or ToolSelector()

    def select(
        self,
        *,
        profile: Any,
        execution_decision: Optional[Any] = None,
        query: str = "",
    ) -> CapabilityDecision:
        """根据任务画像、执行模式和查询文本选择技能、工具或无能力。"""
        hints = list(getattr(profile, "capability_hints", []) or [])

        for hint in hints:
            if self._skill_registry.has_skill(hint):
                definition = self._skill_registry.get(hint)
                return CapabilityDecision.for_skill(
                    definition.name,
                    confidence=float(getattr(profile, "confidence", 0.0) or 0.0),
                    reason="matched_task_profile_capability_hint",
                    required_params=list(definition.required_params),
                    allowed_tools=list(definition.allowed_tools),
                    forbidden_tools=[],
                    metadata={
                        "route_type": "handler",
                        "task_type": getattr(profile, "task_type", ""),
                        "query": query,
                    },
                )
            if self._tool_registry.has_tool(hint):
                definition = self._tool_registry.get_tool(hint)
                params = CapabilityParamBuilder.build_atomic_tool_params(
                    definition.name,
                    query,
                )
                return CapabilityDecision.for_tool(
                    definition.name,
                    confidence=float(getattr(profile, "confidence", 0.0) or 0.0),
                    reason="matched_task_profile_tool_hint",
                    required_params=_input_field_names(definition),
                    metadata={"query": query, "params": params},
                )

        if getattr(profile, "tool_need", "none") == "none":
            return CapabilityDecision.none(
                reason="task_profile_declares_no_tool_need",
                confidence=float(getattr(profile, "confidence", 0.0) or 0.0),
            )

        mode = getattr(execution_decision, "mode", "") if execution_decision else ""
        if mode == "react":
            return CapabilityDecision.none(
                reason="react_mode_deferred_dynamic_selection",
                confidence=float(getattr(profile, "confidence", 0.0) or 0.0),
            )

        tool_decision = self._tool_selector.select(query, profile=profile)
        if tool_decision is not None and self._tool_registry.has_tool(
            tool_decision.tool_name
        ):
            definition = self._tool_registry.get_tool(tool_decision.tool_name)
            return CapabilityDecision.for_tool(
                definition.name,
                confidence=tool_decision.confidence,
                reason=tool_decision.reason,
                required_params=_input_field_names(definition),
                metadata={
                    "query": query,
                    "params": dict(tool_decision.params),
                    "tool_selection": tool_decision.to_dict(),
                },
            )

        return CapabilityDecision.none(
            reason="no_registered_capability_hint",
            confidence=float(getattr(profile, "confidence", 0.0) or 0.0),
        )


def _input_field_names(definition: Any) -> list[str]:
    return list(definition.input_model.model_fields.keys())
