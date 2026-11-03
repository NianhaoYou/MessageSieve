from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, TypedDict

from ..core.message import TokenUsage


class ChatMessage(TypedDict):
    role: str
    content: str


@dataclass(slots=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    proxy: str | None = None
    proxy_env: str | None = None
    temperature: float = 0.2
    max_tokens: int | None = None
    timeout: float = 180.0
    max_retries: int = 5
    disable_thinking: bool = True
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_body: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LLMConfig":
        return cls(
            provider=str(data["provider"]),
            model=str(data["model"]),
            api_key=data.get("api_key"),
            api_key_env=data.get("api_key_env"),
            base_url=data.get("base_url"),
            base_url_env=data.get("base_url_env"),
            proxy=data.get("proxy"),
            proxy_env=data.get("proxy_env"),
            temperature=float(data.get("temperature", 0.2)),
            max_tokens=int(data["max_tokens"]) if data.get("max_tokens") is not None else None,
            timeout=float(data.get("timeout", os.getenv("MASBENCH_DEFAULT_TIMEOUT", "180"))),
            max_retries=int(data.get("max_retries", 5)),
            disable_thinking=_coerce_bool(data.get("disable_thinking", True)),
            extra_headers=dict(data.get("extra_headers") or {}),
            extra_body=dict(data.get("extra_body") or {}),
        )

    def resolved_api_key(self, *, default: str | None = None) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            value = os.getenv(self.api_key_env)
            if value:
                return value
        return default

    def resolved_base_url(self, *, default: str | None = None) -> str | None:
        if self.base_url:
            return self.base_url.rstrip("/")
        if self.base_url_env:
            value = os.getenv(self.base_url_env)
            if value:
                return value.rstrip("/")
        return default.rstrip("/") if default else default

    def resolved_proxy(self) -> str | None:
        if self.proxy:
            return self.proxy.strip()
        if self.proxy_env:
            value = os.getenv(self.proxy_env)
            if value:
                return value.strip()
        return None


@dataclass(frozen=True, slots=True)
class LLMResult:
    text: str
    usage: TokenUsage = TokenUsage()
    raw_response: Any = None


class BaseLLM(ABC):
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, messages: Sequence[ChatMessage], **kwargs: Any) -> LLMResult:
        raise NotImplementedError


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)
