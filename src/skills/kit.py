"""Capability-facing runtime for high-level skills only."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import ValidationError

from src.skills.context import SkillContext
from src.skills.definition import SkillDefinition
from src.skills.errors import SkillError, validation_error_result
from src.skills.policies.skill_policy import SkillPolicy
from src.skills.registry import SkillRegistry, get_default_skill_registry
from src.skills.result import SkillResult
from src.tools.kit import ToolKit


class SkillKit:
    """Invoke registered high-level skills without falling back to atomic tools."""

    def __init__(
        self,
        *,
        tool_kit: ToolKit,
        registry: Optional[SkillRegistry] = None,
    ) -> None:
        self._tool_kit = tool_kit
        self._registry = registry or get_default_skill_registry()

    def list(self) -> list[SkillDefinition]:
        """Return all registered high-level skill definitions."""
        return self._registry.list()

    def get(self, name: str) -> SkillDefinition:
        """Return one high-level skill definition or raise KeyError."""
        return self._registry.get(name)

    def invoke(self, name: str, payload: dict[str, Any] | None = None) -> SkillResult:
        """Validate and invoke one high-level skill.

        Unknown names deliberately raise KeyError. Atomic tool names must be invoked
        through ToolKit/CapabilityKit.call_tool().
        """
        definition = self.get(name)
        try:
            typed_payload = definition.input_model.model_validate(payload or {})
        except ValidationError as exc:
            return validation_error_result(
                skill_name=definition.name,
                error=exc,
            )

        policy = SkillPolicy.from_definition(definition)
        tool_kit = self._tool_kit.with_policy(**policy.to_tool_policy_kwargs())
        ctx = SkillContext(tool_kit=tool_kit, skill_name=definition.name)
        try:
            result = definition.handler(ctx, typed_payload)
        except SkillError as exc:
            return exc.to_result()
        if result.logical_skill is None:
            result.logical_skill = definition.name
        return result
