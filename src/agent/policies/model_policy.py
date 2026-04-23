from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from src.core.config import settings


@dataclass(frozen=True)
class ModelRoleSelection:
    provider: str
    model_name: str
    tier: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "tier": self.tier,
        }


class ModelPolicy:
    def __init__(self) -> None:
        self.version = str(getattr(settings, "MODEL_POLICY_VERSION", "model_policy_v1"))

    def select(self, role: str) -> ModelRoleSelection:
        role = (role or "main").strip().lower()
        main = ModelRoleSelection(
            provider=str(getattr(settings, "DEFAULT_LLM_PROVIDER", "dashscope")),
            model_name=str(getattr(settings, "MODEL_NAME", "qwen-max")),
            tier="main",
        )
        small = ModelRoleSelection(
            provider=str(getattr(settings, "SMALL_MODEL_PROVIDER", main.provider)),
            model_name=str(getattr(settings, "SMALL_MODEL_NAME", "qwen-plus")),
            tier="small",
        )

        if role in {"router", "param_parser", "summary"}:
            return small
        if role == "synthesizer" and str(
            getattr(settings, "SYNTHESIS_MODEL_TIER", "main")
        ).strip().lower() == "small":
            return small
        return main
