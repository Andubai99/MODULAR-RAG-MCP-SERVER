"""Factory for creating vector store backends by provider name."""

from __future__ import annotations

from typing import Callable

from libs.vector_store.base_vector_store import BaseVectorStore

FactoryBuilder = Callable[[object], BaseVectorStore]


class VectorStoreFactory:
    """Provider registry and constructor for vector store implementations."""

    _REGISTRY: dict[str, FactoryBuilder] = {}

    @classmethod
    def register(cls, provider: str, builder: FactoryBuilder) -> None:
        normalized = provider.strip().lower()
        if not normalized:
            raise ValueError("Vector store provider name cannot be empty.")
        cls._REGISTRY[normalized] = builder

    @classmethod
    def create(cls, settings: object) -> BaseVectorStore:
        provider = cls._extract_provider(settings)
        builder = cls._REGISTRY.get(provider)
        if builder is None:
            supported = ", ".join(sorted(cls._REGISTRY)) or "<none>"
            raise ValueError(
                f"Unknown vector store provider: '{provider}'. "
                f"Supported providers: {supported}."
            )
        return builder(settings)

    @staticmethod
    def _extract_provider(settings: object) -> str:
        vector_store_settings = getattr(settings, "vector_store", None)
        provider = getattr(vector_store_settings, "provider", "")
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError(
                "settings.vector_store.provider must be a non-empty string."
            )
        return provider.strip().lower()
