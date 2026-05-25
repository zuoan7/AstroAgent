"""Unified capability facade for high-level skills and atomic tools."""

from __future__ import annotations

from typing import Any, Optional

from src.skills.result import SkillResult
from src.skills.definition import SkillDefinition
from src.skills.kit import SkillKit
from src.tools.definition import ToolDefinition
from src.tools.results import ToolResult
from src.tools.kit import ToolKit
from src.transport.mcp.client import MCPClient


class CapabilityKit:
    """Top-level capability runtime with explicit skill/tool call boundaries."""

    def __init__(
        self,
        *,
        mcp_client: Optional[Any] = None,
        tool_kit: Optional[ToolKit] = None,
        skill_kit: Optional[SkillKit] = None,
        rag_retriever: Optional[Any] = None,
    ) -> None:
        self.mcp_client = (
            mcp_client or getattr(tool_kit, "_backend", None) or MCPClient()
        )
        self.tool_kit = tool_kit or ToolKit(self.mcp_client)
        self.skill_kit = skill_kit or SkillKit(tool_kit=self.tool_kit)
        self.rag_retriever = rag_retriever

    def list_skills(self) -> list[SkillDefinition]:
        """Return high-level skill definitions."""
        return self.skill_kit.list()

    def list_tools(self) -> list[ToolDefinition]:
        """Return atomic tool definitions."""
        return self.tool_kit.list()

    def call_skill(self, name: str, **payload: Any) -> SkillResult:
        """Invoke a high-level skill by name."""
        return self.skill_kit.invoke(name, payload)

    def call_tool(self, name: str, **payload: Any) -> ToolResult:
        """Invoke an atomic tool by name."""
        return self.tool_kit.invoke(name, **payload)

    def to_langchain_tools(self, expose_tools: Optional[list[str]] = None) -> list[Any]:
        """Build LangChain tools for ReAct fallback."""
        from src.agent.adapters.langchain_adapter import to_langchain_tools

        return to_langchain_tools(self, expose_tools=expose_tools)

    def prewarm(self) -> bool:
        """Prewarm the underlying MCP runtime."""
        return self.tool_kit.prewarm()

    def get_runtime_metrics_snapshot(self) -> dict[str, float]:
        """Return underlying MCP runtime metrics."""
        return self.tool_kit.get_runtime_metrics_snapshot()

    def shutdown(self) -> None:
        """Shutdown underlying runtime resources."""
        self.tool_kit.shutdown()
