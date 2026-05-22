from __future__ import annotations

from typing import Any, Dict, Optional

from src.tools.selector import AtomicToolParamAdapter


class CapabilityParamBuilder:
    """Build params for skill or atomic-tool capabilities.

    The callable interface keeps compatibility with old `param_builder(skill,
    query)` call sites. New code should prefer `build_for_capability()`.
    """

    def __init__(
        self,
        skill_param_builder: Any,
        *,
        chat_history: str = "",
        user_profile: str = "",
    ) -> None:
        self._skill_param_builder = skill_param_builder
        self._chat_history = chat_history
        self._user_profile = user_profile

    def __call__(self, skill_name: str, query: str) -> Dict[str, Any]:
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
        return AtomicToolParamAdapter.build(
            tool_name,
            query,
            explicit_params=explicit_params,
        )

    @staticmethod
    def _build_atomic_tool_params(tool_name: str, query: str) -> Dict[str, Any]:
        return CapabilityParamBuilder.build_atomic_tool_params(tool_name, query)
