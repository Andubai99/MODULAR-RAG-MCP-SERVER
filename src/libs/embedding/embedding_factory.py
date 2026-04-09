"""Factory for creating embedding backends by provider name."""

from __future__ import annotations

from typing import Callable

from libs.embedding.base_embedding import BaseEmbedding

FactoryBuilder = Callable[[object], BaseEmbedding]


class EmbeddingFactory:
    # EmbeddingFactory类提供了一个工厂方法create，用于根据给定的settings对象创建和返回一个BaseEmbedding实例。
    # 它维护了一个注册表来存储不同embedding provider的构造函数，并且支持内置的OpenAI和Azure提供者。
    """Provider registry and constructor for embedding implementations."""

    _REGISTRY: dict[str, FactoryBuilder] = {}
    _BUILTIN_PROVIDERS = {"openai", "azure", "ollama"}

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
            builder = cls._builtin_builder(provider)
        if builder is None:
            supported_providers = sorted(set(cls._REGISTRY) | cls._BUILTIN_PROVIDERS)
            supported = ", ".join(supported_providers) or "<none>"
            raise ValueError(
                f"Unknown embedding provider: '{provider}'. "
                f"Supported providers: {supported}."
            )
        return builder(settings)

    @staticmethod 
    def _extract_provider(settings: object) -> str:
    # 从settings对象中提取embedding provider的名称，确保它是一个非空字符串，并返回规范化后的provider名称。
        embedding_settings = getattr(settings, "embedding", None)
        provider = getattr(embedding_settings, "provider", "")
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("settings.embedding.provider must be a non-empty string.")
        return provider.strip().lower()

    @staticmethod
    def _builtin_builder(provider: str) -> FactoryBuilder | None:
    # 根据提供的provider名称返回一个内置的FactoryBuilder，
    # 如果provider是"openai"或"azure"，则分别返回OpenAIEmbedding或AzureEmbedding的构造函数，否则返回None。
        if provider == "openai":
            from libs.embedding.openai_embedding import OpenAIEmbedding

            return OpenAIEmbedding
        if provider == "azure":
            from libs.embedding.azure_embedding import AzureEmbedding

            return AzureEmbedding
        if provider == "ollama":
            from libs.embedding.ollama_embedding import OllamaEmbedding

            return OllamaEmbedding
        return None
