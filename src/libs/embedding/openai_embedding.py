"""OpenAI embedding provider implementation."""

from __future__ import annotations

import json
from typing import Any
from urllib import error, request

from libs.embedding.base_embedding import BaseEmbedding

_DEFAULT_TIMEOUT_SECONDS = 30


def _require_embedding_str(settings: object, field_name: str, provider: str) -> str:
    # 从settings对象中读取指定field_name的值，并验证它是否是一个非空字符串，如果验证失败则抛出ValueError异常，确保返回一个有效的字符串结果。
    embedding_settings = getattr(settings, "embedding", None)
    value = getattr(embedding_settings, field_name, "")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"provider={provider}: settings.embedding.{field_name} must be a non-empty string."
        )
    return value.strip()


def _read_optional_embedding_str(settings: object, field_name: str, default: str) -> str:
    # 从settings对象中读取指定field_name的值，如果该值不存在或不是一个非空字符串，则返回提供的default值，确保最终返回一个有效的字符串结果。
    embedding_settings = getattr(settings, "embedding", None)
    value = getattr(embedding_settings, field_name, "")
    if not isinstance(value, str) or not value.strip():
        return default
    return value.strip()


def _normalize_texts(texts: list[str], provider: str) -> list[str]:
# 验证输入的texts是否是一个非空的字符串列表，并返回一个规范化后的文本列表，如果验证失败则抛出ValueError异常。
    if not isinstance(texts, list):
        raise ValueError(f"provider={provider}: texts must be a list of strings.")
    if not texts:
        raise ValueError(f"provider={provider}: texts cannot be empty.")
    normalized: list[str] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"provider={provider}: texts[{index}] must be a non-empty string."
            )
        normalized.append(text)
    return normalized


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


def _extract_vectors(payload: Any, provider: str) -> list[list[float]]:
# 从响应payload中提取嵌入向量，验证其结构和类型是否符合预期，如果验证失败则抛出ValueError异常，最终返回一个二维列表，其中每个子列表表示一个文本的嵌入向量。
    try:
        data = payload["data"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"provider={provider}: invalid response payload shape (error_type={type(exc).__name__})."
        ) from exc
    if not isinstance(data, list):
        raise ValueError(f"provider={provider}: response data must be a list.")
    vectors: list[list[float]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"provider={provider}: response data[{index}] must be an object."
            )
        embedding = item.get("embedding")
        if not isinstance(embedding, list) or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in embedding
        ):
            raise ValueError(
                f"provider={provider}: response data[{index}].embedding must be a numeric list."
            )
        vectors.append([float(v) for v in embedding])
    return vectors


class OpenAIEmbedding(BaseEmbedding):
# OpenAIEmbedding类实现了BaseEmbedding接口，提供了一个用于与OpenAI嵌入API交互的客户端实现。
    """Embedding client for OpenAI embeddings endpoint."""

    provider_name = "openai"

    def __init__(self, settings: object) -> None:
        self._model = _require_embedding_str(settings, "model", self.provider_name)
        self._api_key = _require_embedding_str(settings, "api_key", self.provider_name)
        base_url = _read_optional_embedding_str(
            settings, "base_url", "https://api.openai.com/v1"
        ).rstrip("/")
        self._endpoint = f"{base_url}/embeddings"

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        del trace
        normalized_texts = _normalize_texts(texts, self.provider_name)
        payload = {"model": self._model, "input": normalized_texts}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        response = _post_json(self._endpoint, headers, payload, self.provider_name)
        return _extract_vectors(response, self.provider_name)
