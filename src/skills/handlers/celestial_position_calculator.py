"""Celestial position calculator skill handler."""

from __future__ import annotations

from src.skills.result import SkillResult
from src.skills.context import SkillContext
from src.skills.inputs import CelestialPositionCalculatorInput
from src.skills.services.celestial_position import calculate_celestial_position


def celestial_position_calculator_handler(
    ctx: SkillContext,
    payload: CelestialPositionCalculatorInput,
) -> SkillResult:
    """Calculate sky position, rise/set time, current sky or coordinate transform."""
    return calculate_celestial_position(ctx, payload)
