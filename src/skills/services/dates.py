"""Date helpers shared by skill handlers."""

from __future__ import annotations

import re


def is_date_like(text: str) -> bool:
    """Return whether text resembles a date or common relative date phrase."""
    value = text.strip()
    if not value:
        return False
    if value in (
        "今天",
        "明天",
        "今晚",
        "明晚",
        "本周末",
        "这个周末",
        "周末",
        "今日",
        "次日",
        "today",
        "tomorrow",
    ):
        return True
    return bool(re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", value))
