"""能力注册表，合并高层技能和底层原子 MCP 工具的可执行能力视图。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional

from src.skills import registry as skill_registry
from src.tools.catalog import ToolCatalog, get_default_tool_catalog


CapabilitySpecKind = Literal["skill", "tool"]


@dataclass(frozen=True)
class CapabilitySpec:
    """Agent 可执行能力的统一描述，可表示高层技能或原子工具。"""

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
    """连接高层技能和 MCP 原子工具的只读能力注册表。"""

    def __init__(self, tool_catalog: Optional[ToolCatalog] = None) -> None:
        """初始化工具目录，并构建技能能力和工具能力索引。"""
        self._tool_catalog = tool_catalog or get_default_tool_catalog()
        self._skill_specs = self._build_skill_specs()
        self._tool_specs = self._build_tool_specs()

    @property
    def tool_catalog(self) -> ToolCatalog:
        """返回底层原子工具目录。"""
        return self._tool_catalog

    def list_skills(self) -> List[CapabilitySpec]:
        """列出所有高层技能能力。"""
        return list(self._skill_specs.values())

    def list_tools(self) -> List[CapabilitySpec]:
        """列出所有原子工具能力。"""
        return list(self._tool_specs.values())

    def list_all(self) -> List[CapabilitySpec]:
        """列出所有高层技能和原子工具能力。"""
        return [*self.list_skills(), *self.list_tools()]

    def has_skill(self, name: str) -> bool:
        """判断指定高层技能能力是否存在。"""
        return name in self._skill_specs

    def has_tool(self, name: str) -> bool:
        """判断指定原子工具能力是否存在。"""
        return name in self._tool_specs

    def get_skill(self, name: str) -> CapabilitySpec:
        """读取指定高层技能能力描述。"""
        try:
            return self._skill_specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown skill capability: {name}") from exc

    def get_tool(self, name: str) -> CapabilitySpec:
        """读取指定原子工具能力描述。"""
        try:
            return self._tool_specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool capability: {name}") from exc

    def get(self, name: str, *, kind: Optional[CapabilitySpecKind] = None) -> CapabilitySpec:
        """按名称和可选类型读取能力描述。"""
        if kind == "skill":
            return self.get_skill(name)
        if kind == "tool":
            return self.get_tool(name)
        if self.has_skill(name):
            return self.get_skill(name)
        return self.get_tool(name)

    def _build_skill_specs(self) -> Dict[str, CapabilitySpec]:
        """从技能注册表和 operation 策略构建高层技能能力描述。"""
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
        """从原子工具目录构建工具能力描述。"""
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
    """构造默认能力注册表。"""
    return CapabilityRegistry()


def capability_names(specs: Iterable[CapabilitySpec]) -> List[str]:
    """从能力描述列表中提取能力名称。"""
    return [spec.name for spec in specs]
