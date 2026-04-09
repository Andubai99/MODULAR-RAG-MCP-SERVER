"""OpenAI-compatible LLM implementation."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence
from urllib import error, request

from libs.llm.base_llm import BaseLLM, ChatMessage

_DEFAULT_TIMEOUT_SECONDS = 30


def _normalize_messages(messages: Sequence[ChatMessage], provider: str) -> list[dict[str, str]]:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError(
            f"provider={provider}: invalid message input type, expected sequence of dict."
        )
    if not messages:
        raise ValueError(f"provider={provider}: messages cannot be empty.")

    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(
                f"provider={provider}: message[{index}] must be a mapping with role/content."
            )
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"provider={provider}: message[{index}].role must be a non-empty string.")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"provider={provider}: message[{index}].content must be a non-empty string."
            )
        normalized.append({"role": role, "content": content})
    return normalized


def _extract_response_text(payload: Any, provider: str) -> str:
    try:
        choices = payload["choices"]
        first_choice = choices[0]
        message = first_choice["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"provider={provider}: invalid response payload shape (error_type={type(exc).__name__})."
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"provider={provider}: response content is empty or invalid.")
    return content


def _post_json(
    endpoint: str, headers: dict[str, str], payload: dict[str, Any], provider: str
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=_DEFAULT_TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        raise RuntimeError(
            f"provider={provider}: HTTP request failed (error_type=HTTPError, status={exc.code})."
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(
            f"provider={provider}: HTTP request failed (error_type=URLError, reason={exc.reason})."
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"provider={provider}: HTTP request failed (error_type=TimeoutError)."
        ) from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"provider={provider}: invalid JSON response (error_type=JSONDecodeError)."
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"provider={provider}: invalid JSON response root type.")
    return parsed


def _require_llm_str(settings: object, field_name: str, provider: str) -> str:
    llm_settings = getattr(settings, "llm", None)
    value = getattr(llm_settings, field_name, "")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"provider={provider}: settings.llm.{field_name} must be a non-empty string.")
    return value.strip()


def _read_optional_llm_str(settings: object, field_name: str, default: str) -> str:
    llm_settings = getattr(settings, "llm", None)
    value = getattr(llm_settings, field_name, "")
    if not isinstance(value, str) or not value.strip():
        return default
    return value.strip()


class OpenAILLM(BaseLLM):
    """LLM client for OpenAI chat-completions endpoint."""

    provider_name = "openai"

    def __init__(self, settings: object) -> None:
        self._model = _require_llm_str(settings, "model", self.provider_name)
        self._api_key = _require_llm_str(settings, "api_key", self.provider_name)
        base_url = _read_optional_llm_str(
            settings, "base_url", "https://api.openai.com/v1"
        ).rstrip("/")
        self._endpoint = f"{base_url}/chat/completions"
        llm_settings = getattr(settings, "llm", None)
        self._temperature = float(getattr(llm_settings, "temperature", 0.0))
        self._max_tokens = int(getattr(llm_settings, "max_tokens", 1024))

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        normalized_messages = _normalize_messages(messages, self.provider_name)
        payload = {
            "model": self._model,
            "messages": normalized_messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        data = _post_json(self._endpoint, headers, payload, self.provider_name)
        return _extract_response_text(data, self.provider_name)
