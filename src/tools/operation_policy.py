from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.skills import registry


@dataclass(frozen=True)
class OperationToolPolicy:
    logical_skill: str
    operation: str
    allowed_tools: List[str] = field(default_factory=list)
    forbidden_tools: List[str] = field(default_factory=list)
    required_params: List[str] = field(default_factory=list)


class OperationPolicyResolver:
    """Resolve operation-scoped MCP tool policy for handler skills."""

    _POSITION_OPERATIONS = {
        "altaz",
        "rise_set",
        "planet_position",
        "current_sky",
        "coordinate_transformation",
    }
    _EVENT_OPERATIONS = {"weekly", "monthly"}

    def resolve(
        self,
        logical_skill: str,
        params: Dict[str, Any],
    ) -> Optional[OperationToolPolicy]:
        operation = self._resolve_operation(logical_skill, params)
        if not operation:
            return None

        spec = registry.get_operation_spec(logical_skill, operation)
        return OperationToolPolicy(
            logical_skill=logical_skill,
            operation=operation,
            allowed_tools=list(spec.allowed_child_tools),
            forbidden_tools=list(spec.forbidden_child_tools),
            required_params=list(spec.required_params),
        )

    def _resolve_operation(
        self,
        logical_skill: str,
        params: Dict[str, Any],
    ) -> Optional[str]:
        if logical_skill == "celestial-position-calculator":
            return self._resolve_position_operation(params)
        if logical_skill == "celestial-events-forecast":
            return self._resolve_event_operation(params)
        return None

    def _resolve_position_operation(self, params: Dict[str, Any]) -> str:
        requested = str(params.get("operation") or "").strip().lower()
        if requested in self._POSITION_OPERATIONS:
            return requested

        output_format = str(params.get("output_format") or "radec").strip().lower()
        if output_format in {"rise_set", "rise-set", "riseset"}:
            return "rise_set"
        if output_format == "altaz":
            return "altaz"
        return "planet_position"

    def _resolve_event_operation(self, params: Dict[str, Any]) -> str:
        requested = str(params.get("operation") or "").strip().lower()
        if requested in self._EVENT_OPERATIONS:
            return requested

        start_date = self._parse_iso_date(params.get("start_date"))
        end_date = self._parse_iso_date(params.get("end_date"))
        if start_date is not None and end_date is not None:
            return "weekly" if (end_date - start_date).days <= 7 else "monthly"
        return "weekly"

    @staticmethod
    def _parse_iso_date(value: Any) -> Optional[datetime]:
        if not value:
            return None
        text = str(value).strip()
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            try:
                return datetime.strptime(text[:10], "%Y-%m-%d")
            except ValueError:
                return None
