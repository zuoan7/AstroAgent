"""Pydantic input models for high-level skills."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SkillInput(BaseModel):
    """Base model for skill inputs."""

    model_config = ConfigDict(extra="ignore")


class WeatherLookupInput(SkillInput):
    city: Optional[str] = None
    location: Optional[str] = None
    extensions: str = "all"

    @model_validator(mode="after")
    def merge_city_location(self) -> "WeatherLookupInput":
        """Treat location as a city alias for the weather skill."""
        if not self.city and self.location:
            self.city = self.location
        return self


class ObservationPlannerInput(SkillInput):
    date: Optional[Any] = None
    location: Optional[Any] = None
    duration: Optional[str] = None


class CelestialEventsForecastInput(SkillInput):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    event_type: Optional[str] = None
    operation: Optional[str] = None


class DeepSkyObservingGuideInput(SkillInput):
    target: str = ""
    observer_location: Optional[str] = None
    date: Optional[str] = None
    equipment: Optional[str] = None


class NeoTrackerInput(SkillInput):
    time_range: Optional[str] = None
    min_size: Optional[float] = None
    max_distance: Optional[float] = None
    observable_only: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_bool(cls, data: Any) -> Any:
        """Keep legacy bool parsing for string inputs."""
        if not isinstance(data, dict):
            return data
        value = data.get("observable_only")
        if isinstance(value, str):
            data = dict(data)
            data["observable_only"] = value.lower() in {"true", "1", "yes", "是"}
        return data


class AstrophotographyCalculatorInput(SkillInput):
    target: str = Field(default="")
    camera: str = Field(default="")
    telescope: Optional[str] = None
    mount: Optional[str] = None
    location: Optional[str] = None
    date: Optional[str] = None
    iso: Optional[str] = None
    aperture: Optional[str] = None


class CelestialPositionCalculatorInput(SkillInput):
    target: str = ""
    datetime: Optional[str] = None
    location: Optional[str] = None
    output_format: Optional[str] = None
    operation: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    epoch: Optional[str] = None
    target_system: Optional[str] = None
