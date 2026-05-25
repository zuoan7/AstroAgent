"""Skill-level policy resolution for ToolKit derivation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.skills.definition import SkillDefinition


@dataclass(frozen=True)
class SkillPolicy:
    """Resolved skill-level execution policy."""

    skill_name: str
    allowed_tools: tuple[str, ...] = field(default_factory=tuple)
    forbidden_tools: tuple[str, ...] = field(default_factory=tuple)
    required_params: tuple[str, ...] = field(default_factory=tuple)
    required_scopes: tuple[str, ...] = field(default_factory=tuple)
    enforce_allowed_tools: bool = True

    @classmethod
    def from_definition(cls, definition: SkillDefinition) -> "SkillPolicy":
        """Resolve policy fields from a SkillDefinition."""
        return cls(
            skill_name=definition.name,
            allowed_tools=tuple(definition.allowed_tools),
            forbidden_tools=(),
            required_params=tuple(definition.required_params),
            required_scopes=tuple(definition.required_scopes),
            enforce_allowed_tools=True,
        )

    def to_tool_policy_kwargs(
        self,
        *,
        include_required_params: bool = False,
    ) -> dict[str, Any]:
        """Return kwargs for ToolKit.with_policy().

        Skill required_params describe skill input payloads. They are not
        automatically atomic-tool params, so SkillKit keeps them out of
        ToolGuardContext by default.
        """
        kwargs: dict[str, Any] = {
            "logical_skill": self.skill_name,
            "allowed_tools": list(self.allowed_tools),
            "forbidden_tools": list(self.forbidden_tools),
            "enforce_allowed_tools": self.enforce_allowed_tools,
        }
        if include_required_params:
            kwargs["required_params"] = list(self.required_params)
        return kwargs


__all__ = ["SkillPolicy"]
