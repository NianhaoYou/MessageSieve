from __future__ import annotations

from dataclasses import replace

from .base import LLMConfig
from .openai_compatible import OpenAICompatibleLLM


_PROVIDER_DEFAULTS = {
    "openai_compatible": (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "http://127.0.0.1:8000/v1",
    ),
}


def build_llm(config: LLMConfig) -> OpenAICompatibleLLM:
    provider = config.provider.lower()

    try:
        api_key_env, base_url_env, base_url = _PROVIDER_DEFAULTS[provider]
    except KeyError:
        raise NotImplementedError(f"Unsupported provider: {config.provider}") from None
    normalized = replace(
        config,
        provider=provider,
        api_key_env=config.api_key_env or api_key_env,
        base_url_env=config.base_url_env or base_url_env,
        extra_headers=dict(config.extra_headers),
        extra_body=dict(config.extra_body),
    )
    return OpenAICompatibleLLM(normalized, default_base_url=base_url, default_api_key="EMPTY")
