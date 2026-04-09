"""Ollama LLM implementation over local HTTP endpoint."""

from __future__ import annotations

from typing import Any, Sequence

from libs.llm.base_llm import BaseLLM, ChatMessage
from libs.llm.openai_llm import (
    _normalize_messages,
    _post_json,
    _read_optional_llm_str,
    _require_llm_str,
)


class OllamaLLM(BaseLLM): 
    # OllamaLLM类继承自BaseLLM，提供了一个实现了Ollama聊天接口的LLM客户端。
    # 它在初始化时从settings对象中读取必要的配置参数，如模型名称、可选的基础URL、温度和最大令牌数，并在chat方法中处理输入消息，构建请求负载，
    # 发送HTTP POST请求到Ollama的聊天端点，并从响应中提取生成的文本内容返回。
    """LLM client for Ollama `/api/chat` endpoint."""

    provider_name = "ollama"

    def __init__(self, settings: object) -> None:
        self._model = _require_llm_str(settings, "model", self.provider_name)
        base_url = _read_optional_llm_str(settings, "base_url", "http://localhost:11434")
        self._endpoint = f"{base_url.rstrip('/')}/api/chat"
        llm_settings = getattr(settings, "llm", None)
        self._temperature = float(getattr(llm_settings, "temperature", 0.0))
        self._max_tokens = int(getattr(llm_settings, "max_tokens", 1024))

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        normalized_messages = _normalize_messages(messages, self.provider_name)
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": normalized_messages,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        headers = {"Content-Type": "application/json"}
        data = _post_json(self._endpoint, headers, payload, self.provider_name)
        return self._extract_response_text(data)

    def _extract_response_text(self, payload: dict[str, Any]) -> str:
        try:
            message = payload["message"]
            content = message["content"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "provider=ollama: invalid response payload shape "
                f"(error_type={type(exc).__name__})."
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError("provider=ollama: response content is empty or invalid.")
        return content
