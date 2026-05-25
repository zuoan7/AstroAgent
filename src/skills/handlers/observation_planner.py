"""Observation planner skill handler."""

from __future__ import annotations

from src.skills.result import SkillResult
from src.skills.context import SkillContext
from src.skills.inputs import ObservationPlannerInput
from src.skills.services.observation_plan import build_observation_plan


def observation_planner_handler(
    ctx: SkillContext,
    payload: ObservationPlannerInput,
) -> SkillResult:
    """Generate a night-sky observing plan for a date and location."""
    return build_observation_plan(ctx, payload)
