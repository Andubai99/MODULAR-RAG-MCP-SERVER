"""DeepSeek OpenAI-compatible LLM implementation."""

from __future__ import annotations

from typing import Sequence

from libs.llm.base_llm import BaseLLM, ChatMessage
from libs.llm.openai_llm import (
    _extract_response_text,
    _normalize_messages,
    _post_json,
    _read_optional_llm_str,
    _require_llm_str,
)


class DeepSeekLLM(BaseLLM):
    """LLM client for DeepSeek chat-completions endpoint."""

    provider_name = "deepseek"

    def __init__(self, settings: object) -> None:
        self._model = _require_llm_str(settings, "model", self.provider_name)
        self._api_key = _require_llm_str(settings, "api_key", self.provider_name)
        base_url = _read_optional_llm_str(
            settings, "base_url", "https://api.deepseek.com/v1"
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
