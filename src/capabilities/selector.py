"""能力选择器，把 TaskProfile 和执行决策转换为下一步可执行能力。
"""

from __future__ import annotations

from typing import Any, Optional

from src.agent.models.capability_decision import CapabilityDecision
from src.capabilities.param_builder import CapabilityParamBuilder
from src.capabilities.registry import (
    CapabilityRegistry,
    get_default_capability_registry,
)
from src.tools.selector import ToolSelector


class CapabilitySelector:
    """在 TaskProfile 和可执行能力之间做兼容选择。"""

    def __init__(
        self,
        registry: Optional[CapabilityRegistry] = None,
        tool_selector: Optional[ToolSelector] = None,
    ) -> None:
        """初始化能力注册表和原子工具规则选择器。"""
        self._registry = registry or get_default_capability_registry()
        self._tool_selector = tool_selector or ToolSelector()

    @property
    def registry(self) -> CapabilityRegistry:
        """返回当前能力注册表。"""
        return self._registry

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
            if self._registry.has_skill(hint):
                spec = self._registry.get_skill(hint)
                return CapabilityDecision.for_skill(
                    spec.name,
                    confidence=float(getattr(profile, "confidence", 0.0) or 0.0),
                    reason="matched_task_profile_capability_hint",
                    required_params=spec.required_params,
                    allowed_tools=spec.allowed_tools,
                    forbidden_tools=spec.forbidden_tools,
                    metadata={
                        "route_type": spec.route_type,
                        "task_type": getattr(profile, "task_type", ""),
                        "query": query,
                    },
                )
            if self._registry.has_tool(hint):
                spec = self._registry.get_tool(hint)
                params = CapabilityParamBuilder.build_atomic_tool_params(
                    spec.name,
                    query,
                )
                return CapabilityDecision.for_tool(
                    spec.name,
                    confidence=float(getattr(profile, "confidence", 0.0) or 0.0),
                    reason="matched_task_profile_tool_hint",
                    required_params=spec.required_params,
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
        if tool_decision is not None and self._registry.has_tool(
            tool_decision.tool_name
        ):
            spec = self._registry.get_tool(tool_decision.tool_name)
            return CapabilityDecision.for_tool(
                spec.name,
                confidence=tool_decision.confidence,
                reason=tool_decision.reason,
                required_params=spec.required_params,
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
