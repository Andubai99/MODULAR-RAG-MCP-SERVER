"""Factory for creating vector store backends by provider name."""

from __future__ import annotations

from typing import Callable

from libs.vector_store.base_vector_store import BaseVectorStore

FactoryBuilder = Callable[[object], BaseVectorStore]


class VectorStoreFactory:
    """Provider registry and constructor for vector store implementations."""

    _REGISTRY: dict[str, FactoryBuilder] = {}
    _BUILTIN_PROVIDERS = {"chroma"}

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
            builder = cls._builtin_builder(provider)
        if builder is None:
            supported_providers = sorted(set(cls._REGISTRY) | cls._BUILTIN_PROVIDERS)
            supported = ", ".join(supported_providers) or "<none>"
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

    @staticmethod
    def _builtin_builder(provider: str) -> FactoryBuilder | None:
        if provider == "chroma":
            from libs.vector_store.chroma_store import ChromaStore

            return ChromaStore
        return None
