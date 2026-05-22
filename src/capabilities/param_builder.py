from __future__ import annotations

from typing import Any, Dict


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
    ) -> Dict[str, Any]:
        if capability_kind == "skill":
            return self._skill_param_builder.build(
                capability_name,
                query,
                chat_history=self._chat_history,
                user_profile=self._user_profile,
            )
        if capability_kind == "tool":
            return self._build_atomic_tool_params(capability_name, query)
        return {}

    @staticmethod
    def _build_atomic_tool_params(tool_name: str, query: str) -> Dict[str, Any]:
        if tool_name == "web_search":
            return {"query": query, "max_results": 5}
        if tool_name == "get_weather":
            return {"city": query, "extensions": "all"}
        if tool_name == "get_nasa_apod":
            return {"date": None, "hd": False}
        return {}
