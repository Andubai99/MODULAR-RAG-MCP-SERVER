"""Azure OpenAI embedding provider implementation."""

from __future__ import annotations

from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.openai_embedding import (
    _extract_vectors,
    _normalize_texts,
    _post_json,
    _require_embedding_str,
)


class AzureEmbedding(BaseEmbedding):
    """Embedding client for Azure OpenAI embeddings endpoint."""

    provider_name = "azure"

    def __init__(self, settings: object) -> None:
        self._api_key = _require_embedding_str(settings, "api_key", self.provider_name)
        self._deployment = _require_embedding_str(
            settings, "deployment_name", self.provider_name
        )
        self._api_version = _require_embedding_str(
            settings, "api_version", self.provider_name
        )
        self._endpoint_root = _require_embedding_str(
            settings, "azure_endpoint", self.provider_name
        ).rstrip("/")

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        del trace
        normalized_texts = _normalize_texts(texts, self.provider_name)
        endpoint = (
            f"{self._endpoint_root}/openai/deployments/{self._deployment}/embeddings"
            f"?api-version={self._api_version}"
        )
        headers = {"Content-Type": "application/json", "api-key": self._api_key}
        payload = {"input": normalized_texts}
        response = _post_json(endpoint, headers, payload, self.provider_name)
        return _extract_vectors(response, self.provider_name)
