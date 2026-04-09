"""OpenAI-compatible LLM implementation."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence
from urllib import error, request

from libs.llm.base_llm import BaseLLM, ChatMessage

_DEFAULT_TIMEOUT_SECONDS = 30


def _normalize_messages(messages: Sequence[ChatMessage], provider: str) -> list[dict[str, str]]:  
    # 检查messages是否是一个非空的Sequence，并且每个元素都是一个包含role和content的Mapping，最后返回一个规范化的list[dict[str, str]]格式的消息列表。
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
    # 从OpenAI的响应payload中提取生成的文本内容，并进行必要的错误检查和验证，确保返回一个有效的字符串结果。
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
    # 发送HTTP POST请求到指定的endpoint，携带给定的headers和payload，并处理可能出现的HTTP错误、URL错误、超时错误以及JSON解析错误，最终返回解析后的JSON响应数据。
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
    # 从settings对象中读取指定field_name的值，并验证它是否是一个非空字符串，如果验证失败则抛出ValueError异常，确保返回一个有效的字符串结果。
    llm_settings = getattr(settings, "llm", None)
    value = getattr(llm_settings, field_name, "")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"provider={provider}: settings.llm.{field_name} must be a non-empty string.")
    return value.strip()


def _read_optional_llm_str(settings: object, field_name: str, default: str) -> str: 
    # 从settings对象中读取指定field_name的值，如果该值不存在或不是一个非空字符串，则返回提供的default值，确保最终返回一个有效的字符串结果。
    llm_settings = getattr(settings, "llm", None)
    value = getattr(llm_settings, field_name, "")
    if not isinstance(value, str) or not value.strip():
        return default
    return value.strip()


class OpenAILLM(BaseLLM): 
    # OpenAILLM类继承自BaseLLM，提供了一个实现了OpenAI聊天完成接口的LLM客户端。
    # 它在初始化时从settings对象中读取必要的配置参数，如模型名称、API密钥、可选的基础URL、温度和最大令牌数，并在chat方法中处理输入消息，
    # 构建请求负载，发送HTTP POST请求到OpenAI的聊天完成端点，并从响应中提取生成的文本内容返回。 
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
