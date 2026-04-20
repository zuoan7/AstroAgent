from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.core.config import settings
from src.core.errors import AgentError, ErrorCode


@dataclass(frozen=True)
class ModelProviderSpec:
    provider: str
    display_name: str
    api_key_attr: str
    base_url_attr: str
    default_base_url: str
    default_model: str
    model_options: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class ResolvedModelConfig:
    provider: str
    display_name: str
    model_name: str
    api_key: str
    base_url: str
    configured: bool

    @property
    def label(self) -> str:
        return f"{self.display_name} / {self.model_name}"


MODEL_PROVIDER_SPECS: Dict[str, ModelProviderSpec] = {
    "dashscope": ModelProviderSpec(
        provider="dashscope",
        display_name="DashScope",
        api_key_attr="DASHSCOPE_API_KEY",
        base_url_attr="OPENAI_COMPATIBLE_BASE_URL",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen3.6-plus",
        model_options=("qwen3.6-plus", "qwen-plus", "qwen-max"),
        description="阿里云 DashScope OpenAI 兼容接口",
    ),
    "glm": ModelProviderSpec(
        provider="glm",
        display_name="GLM",
        api_key_attr="GLM_API_KEY",
        base_url_attr="GLM_API_BASE_URL",
        default_base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-5.1",
        model_options=("glm-5.1", "glm-5", "glm-4.7", "glm-4.5-air"),
        description="智谱 GLM OpenAI 兼容接口",
    ),
    "minimax": ModelProviderSpec(
        provider="minimax",
        display_name="MiniMax",
        api_key_attr="MINIMAX_API_KEY",
        base_url_attr="MINIMAX_API_BASE_URL",
        default_base_url="https://api.minimaxi.com/v1",
        default_model="MiniMax-M2.5",
        model_options=("MiniMax-M2.5", "MiniMax-M2.5-highspeed", "MiniMax-M2.1"),
        description="MiniMax OpenAI 兼容接口",
    ),
}


def get_default_provider() -> str:
    provider = (getattr(settings, "DEFAULT_LLM_PROVIDER", "") or "dashscope").strip().lower()
    return provider if provider in MODEL_PROVIDER_SPECS else "dashscope"


def get_provider_spec(provider: Optional[str] = None) -> ModelProviderSpec:
    normalized = (provider or get_default_provider()).strip().lower()
    spec = MODEL_PROVIDER_SPECS.get(normalized)
    if spec is None:
        raise AgentError(
            code=ErrorCode.VALIDATION_ERROR,
            message=f"不支持的模型提供商: {provider}",
            details={"provider": provider, "supported_providers": list(MODEL_PROVIDER_SPECS)},
        )
    return spec


def resolve_model_config(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    *,
    require_api_key: bool = True,
) -> ResolvedModelConfig:
    spec = get_provider_spec(provider)
    api_key = getattr(settings, spec.api_key_attr, None)
    base_url = (getattr(settings, spec.base_url_attr, None) or spec.default_base_url).strip()
    resolved_model = (model_name or "").strip() or _default_model_for_provider(spec.provider)

    if require_api_key and not api_key:
        raise AgentError(
            code=ErrorCode.LLM_ERROR,
            message=f"{spec.display_name} 未配置 API Key，请设置环境变量 {spec.api_key_attr}",
            details={
                "provider": spec.provider,
                "api_key_env": spec.api_key_attr,
                "base_url_env": spec.base_url_attr,
            },
        )

    return ResolvedModelConfig(
        provider=spec.provider,
        display_name=spec.display_name,
        model_name=resolved_model,
        api_key=api_key or "",
        base_url=base_url,
        configured=bool(api_key),
    )


def list_model_providers() -> List[Dict[str, object]]:
    providers: List[Dict[str, object]] = []
    for provider_id, spec in MODEL_PROVIDER_SPECS.items():
        configured = bool(getattr(settings, spec.api_key_attr, None))
        providers.append(
            {
                "provider": provider_id,
                "display_name": spec.display_name,
                "description": spec.description,
                "configured": configured,
                "api_key_env": spec.api_key_attr,
                "base_url_env": spec.base_url_attr,
                "base_url": getattr(settings, spec.base_url_attr, None) or spec.default_base_url,
                "default_model": _default_model_for_provider(provider_id),
                "model_options": list(spec.model_options),
            }
        )
    return providers


def model_selection_payload(provider: Optional[str] = None, model_name: Optional[str] = None) -> Dict[str, str]:
    resolved = resolve_model_config(provider, model_name, require_api_key=False)
    return {
        "model_provider": resolved.provider,
        "model_name": resolved.model_name,
        "model_label": resolved.label,
    }


def _default_model_for_provider(provider: str) -> str:
    if provider == "dashscope":
        return getattr(settings, "MODEL_NAME", None) or MODEL_PROVIDER_SPECS[provider].default_model
    return MODEL_PROVIDER_SPECS[provider].default_model
