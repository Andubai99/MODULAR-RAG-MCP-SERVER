"""Azure OpenAI-compatible LLM implementation."""

from __future__ import annotations

from typing import Any
from typing import Sequence

from libs.llm.base_llm import BaseLLM, ChatMessage
from libs.llm.openai_llm import (
    _extract_response_text,
    _normalize_messages,
    _post_json,
    _read_optional_llm_str,
    _require_llm_str,
)


class AzureLLM(BaseLLM):
    """LLM client for Azure OpenAI chat-completions endpoint."""

    provider_name = "azure"

    def __init__(self, settings: object) -> None:
        self._api_key = _require_llm_str(settings, "api_key", self.provider_name)
        self._deployment = _require_llm_str(
            settings, "deployment_name", self.provider_name
        )
        self._api_version = _require_llm_str(settings, "api_version", self.provider_name)
        self._endpoint_root = _require_llm_str(
            settings, "azure_endpoint", self.provider_name
        ).rstrip("/")
        llm_settings = getattr(settings, "llm", None)
        self._temperature = float(getattr(llm_settings, "temperature", 0.0))
        self._max_tokens = int(getattr(llm_settings, "max_tokens", 1024))
        self._model = _read_optional_llm_str(settings, "model", "")

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        normalized_messages = _normalize_messages(messages, self.provider_name)
        payload: dict[str, Any] = {
            "messages": normalized_messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if self._model:
            payload["model"] = self._model
        endpoint = (
            f"{self._endpoint_root}/openai/deployments/{self._deployment}/chat/completions"
            f"?api-version={self._api_version}"
        )
        headers = {"Content-Type": "application/json", "api-key": self._api_key}
        data = _post_json(endpoint, headers, payload, self.provider_name)
        return _extract_response_text(data, self.provider_name)
