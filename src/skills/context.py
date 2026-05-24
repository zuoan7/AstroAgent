"""Runtime context passed to high-level skill handlers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional
from uuid import uuid4

from src.core.config import settings as app_settings
from src.core.logger import logger as app_logger
from src.tools.runtime import ToolKit


@dataclass(frozen=True)
class SkillContext:
    """Execution context for one skill invocation."""

    tool_kit: ToolKit
    skill_name: str
    request_id: str = ""
    settings: Any = field(default_factory=lambda: app_settings)
    logger: Any = field(default_factory=lambda: app_logger)
    metrics: Any = None

    def __post_init__(self) -> None:
        """Populate request_id when callers do not provide one."""
        if not self.request_id:
            object.__setattr__(self, "request_id", uuid4().hex)

    def with_tool_policy(
        self,
        *,
        operation: Optional[str] = None,
        allowed_tools: Optional[list[str]] = None,
        forbidden_tools: Optional[list[str]] = None,
        required_params: Optional[list[str]] = None,
        enforce_allowed_tools: Optional[bool] = True,
    ) -> "SkillContext":
        """Return a context with a refined ToolKit policy."""
        return replace(
            self,
            tool_kit=self.tool_kit.with_policy(
                logical_skill=self.skill_name,
                operation=operation,
                allowed_tools=allowed_tools,
                forbidden_tools=forbidden_tools,
                required_params=required_params,
                enforce_allowed_tools=enforce_allowed_tools,
            ),
        )
