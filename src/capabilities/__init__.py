from src.capabilities.registry import (
    CapabilityRegistry,
    CapabilitySpec,
    get_default_capability_registry,
)
from src.capabilities.param_builder import CapabilityParamBuilder

__all__ = [
    "CapabilityRegistry",
    "CapabilityParamBuilder",
    "CapabilityPlanAdapter",
    "CapabilitySelector",
    "CapabilitySpec",
    "get_default_capability_registry",
]


def __getattr__(name: str):
    if name == "CapabilityPlanAdapter":
        from src.capabilities.plan_adapter import CapabilityPlanAdapter

        return CapabilityPlanAdapter
    if name == "CapabilitySelector":
        from src.capabilities.selector import CapabilitySelector

        return CapabilitySelector
    raise AttributeError(f"module 'src.capabilities' has no attribute {name!r}")
