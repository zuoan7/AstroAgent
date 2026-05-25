"""Celestial event tool schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class TonightBestInput(BaseModel):
    pass


class WeeklyEventsInput(BaseModel):
    start_date: Optional[str] = None


class MonthlyEventsInput(BaseModel):
    year: Optional[int] = None
    month: Optional[int] = None
