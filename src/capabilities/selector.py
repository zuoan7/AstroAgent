from __future__ import annotations

from typing import Any, Optional

from src.agent.models.capability_decision import CapabilityDecision
from src.capabilities.registry import CapabilityRegistry, get_default_capability_registry


class CapabilitySelector:
    """Compatibility selector between TaskProfile and executable capabilities."""

    def __init__(self, registry: Optional[CapabilityRegistry] = None) -> None:
        self._registry = registry or get_default_capability_registry()

    @property
    def registry(self) -> CapabilityRegistry:
        return self._registry

    def select(
        self,
        *,
        profile: Any,
        execution_decision: Optional[Any] = None,
        query: str = "",
    ) -> CapabilityDecision:
        hints = list(getattr(profile, "capability_hints", []) or [])
        if not hints:
            hints = list(getattr(profile, "matched_skills", []) or [])

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
                return CapabilityDecision.for_tool(
                    spec.name,
                    confidence=float(getattr(profile, "confidence", 0.0) or 0.0),
                    reason="matched_task_profile_tool_hint",
                    required_params=spec.required_params,
                    metadata={"query": query},
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

        return CapabilityDecision.none(
            reason="no_registered_capability_hint",
            confidence=float(getattr(profile, "confidence", 0.0) or 0.0),
        )
