from __future__ import annotations

import time
from typing import Any, Dict, Sequence

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from .base import BaseLLM, ChatMessage, LLMResult
from ..core.message import TokenUsage


def _normalize_model_name(model_name: str) -> str:
    return model_name.strip()


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


def _is_prompt_too_long_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = (
        "context length",
        "maximum context length",
        "input_tokens",
        "reduce the length of the input prompt",
    )
    return any(marker in text for marker in markers)


def _is_content_block_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = (
        "data_inspection_failed",
        "inappropriate content",
        "content_policy",
        "content filter",
        "safety system",
        "violates our content",
    )
    return any(marker in text for marker in markers)


def _rate_limit_retry_after_seconds(exc: BaseException, *, fallback: float) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    retry_after = None
    if headers is not None and hasattr(headers, "get"):
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(float(retry_after), fallback)
        except (TypeError, ValueError):
            pass
    return fallback


def _status_error_retry_after_seconds(exc: BaseException, *, fallback: float) -> float:
    delay = _rate_limit_retry_after_seconds(exc, fallback=fallback)
    response = getattr(exc, "response", None)
    if response is None:
        return delay
    try:
        payload = response.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        retry_after = payload.get("retry_after")
        if retry_after is not None:
            try:
                return max(float(retry_after), delay)
            except (TypeError, ValueError):
                pass
    return delay


def _is_retryable_status_error(exc: APIStatusError) -> bool:
    status_code = int(getattr(exc, "status_code", 0) or 0)
    if status_code >= 500:
        return True
    response = getattr(exc, "response", None)
    if response is None:
        return False
    try:
        payload = response.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        if bool(payload.get("retryable")):
            return True
        error_code = str(payload.get("error_code") or "").strip().lower()
        error_name = str(payload.get("error_name") or "").strip().lower()
        error_category = str(payload.get("error_category") or "").strip().lower()
        if error_code in {"502", "503", "504"}:
            return True
        if error_name in {"origin_bad_gateway", "bad_gateway", "service_unavailable", "gateway_timeout"}:
            return True
        if error_category in {"origin", "overloaded", "transient"}:
            return True
    return False


class OpenAICompatibleLLM(BaseLLM):
    def __init__(
        self,
        config,
        *,
        default_base_url: str | None = None,
        default_api_key: str | None = None,
    ) -> None:
        super().__init__(config)
        api_key = config.resolved_api_key(default=default_api_key)
        if not api_key:
            raise ValueError(f"Provider '{config.provider}' requires an API key.")
        base_url = config.resolved_base_url(default=default_base_url)
        self._api_key = api_key
        self._base_url = base_url
        self.request_timeout_seconds = max(1.0, min(float(config.timeout), 120.0))
        self.client = self._build_client()

    def _build_client(self) -> OpenAI:
        proxy = self.config.resolved_proxy()
        http_client = httpx.Client(
            proxy=proxy or None,
            timeout=self.request_timeout_seconds,
            trust_env=False,
        )
        return OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self.request_timeout_seconds,
            max_retries=0,
            default_headers=self.config.extra_headers or None,
            http_client=http_client,
        )

    def _reset_client(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
        self.client = self._build_client()

    def generate(self, messages: Sequence[ChatMessage], **kwargs: Any) -> LLMResult:
        extra_body = self._build_extra_body()
        extra_body.update(dict(kwargs.pop("extra_body", {}) or {}))
        max_tokens = kwargs.pop("max_tokens", self.config.max_tokens)
        request_kwargs: Dict[str, Any] = {
            "model": _normalize_model_name(self.config.model),
            "messages": list(messages),
            "temperature": kwargs.pop("temperature", self.config.temperature),
            "extra_body": extra_body or None,
            **kwargs,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max(int(max_tokens), 1)
        request_kwargs["timeout"] = self.request_timeout_seconds
        base_retry_delays = (5.0, 10.0, 20.0, 30.0, 45.0, 60.0)
        retry_delays = base_retry_delays[: max(0, int(self.config.max_retries))]
        for attempt in range(len(retry_delays) + 1):
            try:
                response = self.client.chat.completions.create(**request_kwargs)
                if not getattr(response, "choices", None):
                    if attempt >= len(retry_delays):
                        raise RuntimeError(
                            f"Provider returned no choices for provider={self.config.provider} "
                            f"model={self.config.model}."
                        )
                    self._reset_client()
                    delay = retry_delays[attempt]
                    if delay > 0:
                        time.sleep(delay)
                    continue
                break
            except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                if attempt >= len(retry_delays):
                    raise
                if isinstance(exc, (APITimeoutError, APIConnectionError)):
                    self._reset_client()
                delay = 0.0
                if isinstance(exc, RateLimitError):
                    delay = _rate_limit_retry_after_seconds(exc, fallback=retry_delays[attempt])
                else:
                    delay = retry_delays[attempt]
                if delay > 0:
                    time.sleep(delay)
            except BadRequestError as exc:
                if _is_prompt_too_long_error(exc):
                    raise RuntimeError(
                        f"Prompt exceeds model context length for model={self.config.model}. "
                        "Reduce the prompt size or use a model with a larger context window."
                    ) from exc
                if _is_content_block_error(exc):
                    return LLMResult(
                        text="",
                        usage=TokenUsage(),
                        raw_response={
                            "blocked": True,
                            "error": str(exc),
                            "provider": self.config.provider,
                            "model": self.config.model,
                        },
                    )
                raise
            except APIStatusError as exc:
                if not _is_retryable_status_error(exc) or attempt >= len(retry_delays):
                    raise
                delay = _status_error_retry_after_seconds(exc, fallback=retry_delays[attempt])
                if delay > 0:
                    time.sleep(delay)
        message = response.choices[0].message
        usage = getattr(response, "usage", None)
        result_usage = TokenUsage(
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )
        return LLMResult(
            text=_extract_text(message.content),
            usage=result_usage,
            raw_response=response,
        )

    def _build_extra_body(self) -> Dict[str, Any]:
        extra_body = dict(self.config.extra_body)
        if self.config.disable_thinking:
            template = dict(extra_body.get("chat_template_kwargs") or {})
            template.setdefault("enable_thinking", False)
            extra_body["chat_template_kwargs"] = template
        return extra_body
