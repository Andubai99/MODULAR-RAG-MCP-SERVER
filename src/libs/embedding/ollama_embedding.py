"""Ollama embedding provider implementation."""

from __future__ import annotations

from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.openai_embedding import (
    _normalize_texts,
    _post_json,
    _read_optional_embedding_str,
    _require_embedding_str,
)


class OllamaEmbedding(BaseEmbedding):
    """Embedding client for Ollama `/api/embed` endpoint."""

    provider_name = "ollama"

    def __init__(self, settings: object) -> None:
        self._model = _require_embedding_str(settings, "model", self.provider_name)
        base_url = _read_optional_embedding_str(
            settings, "base_url", "http://localhost:11434"
        ).rstrip("/")
        self._endpoint = f"{base_url}/api/embed"

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        del trace
        normalized_texts = _normalize_texts(texts, self.provider_name)
        payload = {"model": self._model, "input": normalized_texts}
        headers = {"Content-Type": "application/json"}
        data = _post_json(self._endpoint, headers, payload, self.provider_name)
        return self._extract_vectors(data)

    def _extract_vectors(self, payload: dict[str, object]) -> list[list[float]]:
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list):
            raise ValueError(
                "provider=ollama: invalid response payload shape (error_type=KeyError)."
            )

        vectors: list[list[float]] = []
        for index, vector in enumerate(embeddings):
            if not isinstance(vector, list) or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in vector
            ):
                raise ValueError(
                    f"provider=ollama: response embeddings[{index}] must be a numeric list."
                )
            vectors.append([float(v) for v in vector])
        return vectors
