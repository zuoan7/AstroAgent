"""Skill-layer error contracts and conversion helpers."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Optional

from src.skills.result import SkillResult


class SkillErrorCode(str, Enum):
    """Stable skill-layer error codes."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    POLICY_ERROR = "POLICY_ERROR"
    HANDLER_ERROR = "HANDLER_ERROR"


class SkillError(Exception):
    """Structured exception for skill-layer failures."""

    def __init__(
        self,
        skill_name: str,
        code: SkillErrorCode | str,
        message: str,
        *,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.skill_name = skill_name
        self.code = code.value if isinstance(code, SkillErrorCode) else str(code)
        self.message = message
        self.details = dict(details or {})

    def to_result(self, *, latency_ms: Optional[float] = None) -> SkillResult:
        """Convert this error into a failed SkillResult."""
        return skill_error_result(
            skill_name=self.skill_name,
            code=self.code,
            message=self.message,
            latency_ms=latency_ms,
        )


def skill_error_result(
    *,
    skill_name: str,
    code: SkillErrorCode | str,
    message: str,
    latency_ms: Optional[float] = None,
) -> SkillResult:
    """Build a failed SkillResult for a skill-layer error."""
    error_code = code.value if isinstance(code, SkillErrorCode) else str(code)
    return SkillResult.from_error(
        skill_name=skill_name,
        error_code=error_code,
        error_message=message,
        latency_ms=latency_ms,
    )


def validation_error_result(
    *,
    skill_name: str,
    error: Exception,
    latency_ms: Optional[float] = None,
) -> SkillResult:
    """Build a failed SkillResult from input validation errors."""
    return skill_error_result(
        skill_name=skill_name,
        code=SkillErrorCode.VALIDATION_ERROR,
        message=str(error),
        latency_ms=latency_ms,
    )


__all__ = [
    "SkillError",
    "SkillErrorCode",
    "skill_error_result",
    "validation_error_result",
]
