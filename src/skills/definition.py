"""Skill definition contracts for high-level orchestration skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Type

from pydantic import BaseModel

from src.agent.models.skill_result import SkillResult


class SkillHandler(Protocol):
    """Callable contract implemented by high-level skill handlers."""

    def __call__(self, ctx: Any, payload: BaseModel) -> SkillResult:
        """Execute one skill with a typed payload."""
        ...


@dataclass(frozen=True)
class SkillDefinition:
    """Static contract for one high-level skill."""

    name: str
    display_name: str
    summary: str
    description: str
    input_model: Type[BaseModel]
    handler: SkillHandler
    output_model: Any = SkillResult
    allowed_tools: tuple[str, ...] = field(default_factory=tuple)
    operations: tuple[Any, ...] = field(default_factory=tuple)
    required_params: tuple[str, ...] = field(default_factory=tuple)
    required_scopes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def param_names(self) -> list[str]:
        """Return input field names for legacy callers."""
        return list(self.input_model.model_fields.keys())
