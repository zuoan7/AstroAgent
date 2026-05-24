"""Legacy atomic tool catalog facade backed by ToolRegistry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from src.tools.registry import ToolRegistry, get_default_tool_registry


@dataclass(frozen=True)
class AtomicToolSpec:
    """Legacy static description for one atomic MCP tool."""

    name: str
    summary: str = ""
    param_names: List[str] = field(default_factory=list)


def _spec_from_definition(name: str, registry: ToolRegistry) -> AtomicToolSpec:
    definition = registry.get_tool(name)
    return AtomicToolSpec(
        name=definition.name,
        summary=definition.summary,
        param_names=list(definition.param_names),
    )


class ToolCatalog:
    """Legacy catalog API backed by ToolRegistry."""

    def __init__(self, specs: Optional[Iterable[AtomicToolSpec]] = None) -> None:
        if specs is None:
            self._registry = get_default_tool_registry()
            self._specs: Dict[str, AtomicToolSpec] = {
                definition.name: AtomicToolSpec(
                    name=definition.name,
                    summary=definition.summary,
                    param_names=list(definition.param_names),
                )
                for definition in self._registry.list_definitions()
            }
        else:
            self._registry = None
            self._specs = {spec.name: spec for spec in specs}

    def list_specs(self) -> List[AtomicToolSpec]:
        """Return all atomic tool specs."""
        return list(self._specs.values())

    def list_names(self) -> List[str]:
        """Return all atomic tool names."""
        return list(self._specs.keys())

    def has_tool(self, name: str) -> bool:
        """Return whether an atomic tool exists."""
        return name in self._specs

    def get_tool(self, name: str) -> AtomicToolSpec:
        """Return one atomic tool spec or raise KeyError."""
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown MCP atomic tool: {name}") from exc


def get_default_tool_catalog() -> ToolCatalog:
    """Construct the legacy default atomic tool catalog."""
    return ToolCatalog()
