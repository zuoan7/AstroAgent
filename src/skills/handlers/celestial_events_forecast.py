"""Celestial events forecast skill handler."""

from __future__ import annotations

from src.skills.result import SkillResult
from src.skills.context import SkillContext
from src.skills.inputs import CelestialEventsForecastInput
from src.skills.services.celestial_events import forecast_celestial_events


def celestial_events_forecast_handler(
    ctx: SkillContext,
    payload: CelestialEventsForecastInput,
) -> SkillResult:
    """Query weekly or monthly celestial events and summarize them."""
    return forecast_celestial_events(ctx, payload)
