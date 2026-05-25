"""Near-earth object tracker skill handler."""

from __future__ import annotations

from src.skills.result import SkillResult
from src.skills.context import SkillContext
from src.skills.inputs import NeoTrackerInput
from src.skills.services.near_earth_objects import track_near_earth_objects


def neo_tracker_handler(ctx: SkillContext, payload: NeoTrackerInput) -> SkillResult:
    """Track near-earth object close approaches."""
    return track_near_earth_objects(ctx, payload)
