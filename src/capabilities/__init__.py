"""能力层公共导出，提供能力注册表、参数构建、计划适配和能力选择器。
"""

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
    """按需延迟导入计划适配器和能力选择器，避免循环依赖。"""
    if name == "CapabilityPlanAdapter":
        from src.capabilities.plan_adapter import CapabilityPlanAdapter

        return CapabilityPlanAdapter
    if name == "CapabilitySelector":
        from src.capabilities.selector import CapabilitySelector

        return CapabilitySelector
    raise AttributeError(f"module 'src.capabilities' has no attribute {name!r}")
