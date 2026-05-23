"""能力参数构建器，把高层技能或原子工具能力转换为可执行调用参数。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.tools.selector import AtomicToolParamAdapter


class CapabilityParamBuilder:
    """为高层技能或原子工具能力构建参数。

    可调用接口用于兼容旧的 param_builder(skill, query) 调用点；
    新代码优先使用 build_for_capability()。
    """

    def __init__(
        self,
        skill_param_builder: Any,
        *,
        chat_history: str = "",
        user_profile: str = "",
    ) -> None:
        """初始化技能参数构建器和上下文文本。"""
        self._skill_param_builder = skill_param_builder
        self._chat_history = chat_history
        self._user_profile = user_profile

    def __call__(self, skill_name: str, query: str) -> Dict[str, Any]:
        """兼容旧接口，按高层技能名构建参数。"""
        return self.build_for_capability(
            "skill",
            skill_name,
            query,
        )

    def build_for_capability(
        self,
        capability_kind: str,
        capability_name: str,
        query: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """根据能力类型和名称构建技能或原子工具参数。"""
        if capability_kind == "skill":
            try:
                return self._skill_param_builder.build(
                    capability_name,
                    query,
                    chat_history=self._chat_history,
                    user_profile=self._user_profile,
                )
            except KeyError:
                return self.build_atomic_tool_params(capability_name, query)
        if capability_kind == "tool":
            explicit_params = None
            if isinstance(metadata, dict) and isinstance(metadata.get("params"), dict):
                explicit_params = metadata["params"]
            return self.build_atomic_tool_params(
                capability_name,
                query,
                explicit_params=explicit_params,
            )
        return {}

    def build_for_decision(self, capability: Any, query: str) -> Dict[str, Any]:
        """根据 CapabilityDecision 构建对应调用参数。"""
        return self.build_for_capability(
            str(getattr(capability, "kind", "") or ""),
            str(getattr(capability, "name", "") or ""),
            query,
            metadata=getattr(capability, "metadata", {}) or {},
        )

    @staticmethod
    def build_atomic_tool_params(
        tool_name: str,
        query: str,
        *,
        explicit_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """为原子工具构建参数，并优先使用显式参数。"""
        return AtomicToolParamAdapter.build(
            tool_name,
            query,
            explicit_params=explicit_params,
        )

    @staticmethod
    def _build_atomic_tool_params(tool_name: str, query: str) -> Dict[str, Any]:
        """兼容旧调用点的原子工具参数构建方法。"""
        return CapabilityParamBuilder.build_atomic_tool_params(tool_name, query)
