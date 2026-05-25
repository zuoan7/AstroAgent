"""Weather tool schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class WeatherInput(BaseModel):
    city: Optional[str] = None
    extensions: str = "base"
