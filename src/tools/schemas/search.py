"""Search tool schemas."""

from __future__ import annotations

from pydantic import BaseModel


class WebSearchInput(BaseModel):
    query: str
    max_results: int = 5
