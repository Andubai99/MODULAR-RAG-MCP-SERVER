"""Factory for creating embedding backends by provider name."""

from __future__ import annotations

from typing import Callable

from libs.embedding.base_embedding import BaseEmbedding

FactoryBuilder = Callable[[object], BaseEmbedding]


class EmbeddingFactory:
    """Provider registry and constructor for embedding implementations."""

    _REGISTRY: dict[str, FactoryBuilder] = {}

    @classmethod
    def register(cls, provider: str, builder: FactoryBuilder) -> None:
        normalized = provider.strip().lower()
        if not normalized:
            raise ValueError("Embedding provider name cannot be empty.")
        cls._REGISTRY[normalized] = builder

    @classmethod
    def create(cls, settings: object) -> BaseEmbedding:
        provider = cls._extract_provider(settings)
        builder = cls._REGISTRY.get(provider)
        if builder is None:
            supported = ", ".join(sorted(cls._REGISTRY)) or "<none>"
            raise ValueError(
                f"Unknown embedding provider: '{provider}'. "
                f"Supported providers: {supported}."
            )
        return builder(settings)

    @staticmethod
    def _extract_provider(settings: object) -> str:
        embedding_settings = getattr(settings, "embedding", None)
        provider = getattr(embedding_settings, "provider", "")
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("settings.embedding.provider must be a non-empty string.")
        return provider.strip().lower()
