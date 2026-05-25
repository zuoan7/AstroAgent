"""Astronomy calculation and catalog tool schemas."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class PlanetPositionInput(BaseModel):
    planet_name: str
    observation_time: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class AltAzInput(PlanetPositionInput):
    latitude: float
    longitude: float


class CoordinateTransformationInput(BaseModel):
    ra: float
    dec: float
    epoch: str = "J2000"
    target_system: str = "fk5"


class RiseSetTimesInput(BaseModel):
    body_name: str
    latitude: float
    longitude: float
    date: Optional[str] = None


class CurrentSkyObjectsInput(BaseModel):
    latitude: float
    longitude: float
    date: Optional[str] = None


class AstrophysicalObjectInfoInput(BaseModel):
    object_name: str


class GalaxyDataInput(BaseModel):
    galaxy_name: str


class PlanetPositionData(BaseModel):
    model_config = ConfigDict(extra="allow")

    ra_hours: Optional[float] = None
    ra_degrees: Optional[float] = None
    dec: Optional[float] = None
    distance_au: Optional[float] = None
    altitude: Optional[float] = None
    azimuth: Optional[float] = None


class AltAzData(BaseModel):
    model_config = ConfigDict(extra="allow")

    planet: str
    altitude: float
    azimuth: float
    distance_au: float


class CoordinateTransformationData(BaseModel):
    model_config = ConfigDict(extra="allow")

    ra_hours: float
    ra_degrees: float
    dec: float


class RiseSetTimesData(BaseModel):
    model_config = ConfigDict(extra="allow")

    rise_time: Any = None
    set_time: Any = None
