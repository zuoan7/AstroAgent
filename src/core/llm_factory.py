from __future__ import annotations

from src.core.config import settings
from src.core.model_catalog import resolve_model_config


DEFAULT_DASHSCOPE_COMPAT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def build_chat_model(
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.1,
    request_timeout: float | None = None,
    streaming: bool = False,
):
    from langchain_openai import ChatOpenAI

    resolved = resolve_model_config(provider, model)

    return ChatOpenAI(
        model=resolved.model_name,
        api_key=resolved.api_key,
        base_url=resolved.base_url or DEFAULT_DASHSCOPE_COMPAT_BASE_URL,
        temperature=temperature,
        timeout=request_timeout,
        streaming=streaming,
        max_retries=2,
    )
