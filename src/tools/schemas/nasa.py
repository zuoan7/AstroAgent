"""NASA tool schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class NASAApodInput(BaseModel):
    date: Optional[str] = None
    hd: bool = False


class NeoDataInput(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    limit: int = 10
