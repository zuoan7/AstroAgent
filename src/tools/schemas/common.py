"""Shared tool schema primitives."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import RootModel


class JsonObjectData(RootModel[Dict[str, Any]]):
    pass
