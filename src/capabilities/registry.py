from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional

from src.skills import registry as skill_registry
from src.tools.catalog import ToolCatalog, get_default_tool_catalog


CapabilitySpecKind = Literal["skill", "tool"]


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    kind: CapabilitySpecKind
    summary: str
    description: str = ""
    required_params: List[str] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    forbidden_tools: List[str] = field(default_factory=list)
    operations: List[str] = field(default_factory=list)
    route_type: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)


_HANDLER_ALLOWED_TOOLS: Dict[str, List[str]] = {
    "observation-planner": [
        "get_weather",
        "get_weekly_events",
        "get_tonight_best",
    ],
    "deep-sky-observing-guide": [
        "get_astrophysical_object_info",
        "get_galaxy_data",
    ],
    "neo-tracker": ["get_neo_data"],
    "astrophotography-calculator": [],
}


_REQUIRED_PARAMS: Dict[str, List[str]] = {
    "weather-lookup": [],
    "observation-planner": ["location"],
    "celestial-events-forecast": [],
    "deep-sky-observing-guide": ["target"],
    "neo-tracker": [],
    "astrophotography-calculator": ["target", "camera"],
    "celestial-position-calculator": ["target"],
}


class CapabilityRegistry:
    """Read model that joins high-level skills with MCP atomic tools."""

    def __init__(self, tool_catalog: Optional[ToolCatalog] = None) -> None:
        self._tool_catalog = tool_catalog or get_default_tool_catalog()
        self._skill_specs = self._build_skill_specs()
        self._tool_specs = self._build_tool_specs()

    @property
    def tool_catalog(self) -> ToolCatalog:
        return self._tool_catalog

    def list_skills(self) -> List[CapabilitySpec]:
        return list(self._skill_specs.values())

    def list_tools(self) -> List[CapabilitySpec]:
        return list(self._tool_specs.values())

    def list_all(self) -> List[CapabilitySpec]:
        return [*self.list_skills(), *self.list_tools()]

    def has_skill(self, name: str) -> bool:
        return name in self._skill_specs

    def has_tool(self, name: str) -> bool:
        return name in self._tool_specs

    def get_skill(self, name: str) -> CapabilitySpec:
        try:
            return self._skill_specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown skill capability: {name}") from exc

    def get_tool(self, name: str) -> CapabilitySpec:
        try:
            return self._tool_specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool capability: {name}") from exc

    def get(self, name: str, *, kind: Optional[CapabilitySpecKind] = None) -> CapabilitySpec:
        if kind == "skill":
            return self.get_skill(name)
        if kind == "tool":
            return self.get_tool(name)
        if self.has_skill(name):
            return self.get_skill(name)
        return self.get_tool(name)

    def _build_skill_specs(self) -> Dict[str, CapabilitySpec]:
        specs: Dict[str, CapabilitySpec] = {}
        operation_specs = skill_registry.get_operation_specs()
        operations_by_skill: Dict[str, List[str]] = {}
        tools_by_skill: Dict[str, List[str]] = {}

        for op in operation_specs:
            operations_by_skill.setdefault(op.logical_skill, []).append(op.operation)
            tools_by_skill.setdefault(op.logical_skill, [])
            if op.atomic_tool_name not in tools_by_skill[op.logical_skill]:
                tools_by_skill[op.logical_skill].append(op.atomic_tool_name)

        for skill in skill_registry.get_skill_specs():
            if skill.route_type == "simple" and skill.mcp_tool_name:
                allowed_tools = [skill.mcp_tool_name]
            else:
                allowed_tools = [
                    *tools_by_skill.get(skill.skill_name, []),
                    *_HANDLER_ALLOWED_TOOLS.get(skill.skill_name, []),
                ]
                allowed_tools = list(dict.fromkeys(allowed_tools))

            specs[skill.skill_name] = CapabilitySpec(
                name=skill.skill_name,
                kind="skill",
                summary=skill.summary,
                description=skill.description,
                required_params=list(_REQUIRED_PARAMS.get(skill.skill_name, [])),
                allowed_tools=allowed_tools,
                forbidden_tools=[],
                operations=list(operations_by_skill.get(skill.skill_name, [])),
                route_type=skill.route_type,
                metadata={
                    "langchain_tool_name": skill.langchain_tool_name,
                    "mcp_tool_name": skill.mcp_tool_name,
                    "param_names": list(skill.param_names),
                    "defaults": dict(skill.defaults),
                },
            )

        return specs

    def _build_tool_specs(self) -> Dict[str, CapabilitySpec]:
        return {
            tool.name: CapabilitySpec(
                name=tool.name,
                kind="tool",
                summary=tool.summary,
                required_params=list(tool.param_names),
                allowed_tools=[tool.name],
            )
            for tool in self._tool_catalog.list_specs()
        }


def get_default_capability_registry() -> CapabilityRegistry:
    return CapabilityRegistry()


def capability_names(specs: Iterable[CapabilitySpec]) -> List[str]:
    return [spec.name for spec in specs]
