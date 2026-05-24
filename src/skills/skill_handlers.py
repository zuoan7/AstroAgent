"""Compatibility facade for split skill handlers.

New code should use SkillRegistry definitions from ``src.skills.registry``.
The classes here preserve the legacy ``Handler()(mcp, **params)`` test/API shape.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from src.agent.models.skill_result import SkillResult
from src.skills.context import SkillContext
from src.skills.handlers.astrophotography_calculator import (
    astrophotography_calculator_handler,
)
from src.skills.handlers.celestial_events_forecast import (
    celestial_events_forecast_handler,
)
from src.skills.handlers.celestial_position_calculator import (
    celestial_position_calculator_handler,
)
from src.skills.handlers.deep_sky_observing_guide import (
    deep_sky_observing_guide_handler,
)
from src.skills.handlers.neo_tracker import neo_tracker_handler
from src.skills.handlers.observation_planner import observation_planner_handler
from src.skills.handlers.weather_lookup import weather_lookup_handler
from src.skills.inputs import (
    AstrophotographyCalculatorInput,
    CelestialEventsForecastInput,
    CelestialPositionCalculatorInput,
    DeepSkyObservingGuideInput,
    NeoTrackerInput,
    ObservationPlannerInput,
    WeatherLookupInput,
)
from src.tools.results import ToolResult


class _LegacyToolKit:
    """Small adapter that returns ToolResult without output schema validation."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def with_policy(self, **_: Any) -> "_LegacyToolKit":
        """Legacy direct handler tests do not exercise guard policy."""
        return self

    def invoke(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Call one legacy backend tool and wrap its raw value."""
        raw = self._backend.call_tool(tool_name, **kwargs)
        return ToolResult.from_raw(tool_name, raw, output_model=None)

    def invoke_parallel(self, calls: list[dict]) -> list[ToolResult]:
        """Call legacy backend batch API when available."""
        if hasattr(self._backend, "call_tools_parallel"):
            raw_results = self._backend.call_tools_parallel(calls)
        else:
            raw_results = [
                self._backend.call_tool(call["tool_name"], **call.get("kwargs", {}))
                for call in calls
            ]
        return [
            ToolResult.from_raw(call["tool_name"], raw, output_model=None)
            for call, raw in zip(calls, raw_results)
        ]


class _LegacyHandlerAdapter:
    skill_name: str
    input_model: Type[Any]
    handler: Any

    def __call__(self, mcp: Any, **params: Any) -> SkillResult:
        """Adapt legacy ``mcp, **params`` invocation to the new handler contract."""
        payload = self.input_model.model_validate(params or {})
        ctx = SkillContext(tool_kit=_LegacyToolKit(mcp), skill_name=self.skill_name)
        return self.handler(ctx, payload)


class WeatherLookupHandler(_LegacyHandlerAdapter):
    skill_name = "weather-lookup"
    input_model = WeatherLookupInput
    handler = staticmethod(weather_lookup_handler)


class ObservationPlannerHandler(_LegacyHandlerAdapter):
    skill_name = "observation-planner"
    input_model = ObservationPlannerInput
    handler = staticmethod(observation_planner_handler)


class CelestialEventsForecastHandler(_LegacyHandlerAdapter):
    skill_name = "celestial-events-forecast"
    input_model = CelestialEventsForecastInput
    handler = staticmethod(celestial_events_forecast_handler)


class DeepSkyObservingGuideHandler(_LegacyHandlerAdapter):
    skill_name = "deep-sky-observing-guide"
    input_model = DeepSkyObservingGuideInput
    handler = staticmethod(deep_sky_observing_guide_handler)


class NeoTrackerHandler(_LegacyHandlerAdapter):
    skill_name = "neo-tracker"
    input_model = NeoTrackerInput
    handler = staticmethod(neo_tracker_handler)


class AstrophotographyCalculatorHandler(_LegacyHandlerAdapter):
    skill_name = "astrophotography-calculator"
    input_model = AstrophotographyCalculatorInput
    handler = staticmethod(astrophotography_calculator_handler)


class CelestialPositionCalculatorHandler(_LegacyHandlerAdapter):
    skill_name = "celestial-position-calculator"
    input_model = CelestialPositionCalculatorInput
    handler = staticmethod(celestial_position_calculator_handler)


SKILL_HANDLERS: Dict[str, type] = {
    "weather-lookup": WeatherLookupHandler,
    "observation-planner": ObservationPlannerHandler,
    "celestial-events-forecast": CelestialEventsForecastHandler,
    "deep-sky-observing-guide": DeepSkyObservingGuideHandler,
    "neo-tracker": NeoTrackerHandler,
    "astrophotography-calculator": AstrophotographyCalculatorHandler,
    "celestial-position-calculator": CelestialPositionCalculatorHandler,
}
