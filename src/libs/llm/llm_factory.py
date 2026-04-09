"""Factory for creating LLM backends by provider name."""

from __future__ import annotations

from typing import Callable

from libs.llm.base_llm import BaseLLM

FactoryBuilder = Callable[[object], BaseLLM]


class LLMFactory:
    """Provider registry and constructor for LLM implementations."""

    _REGISTRY: dict[str, FactoryBuilder] = {}
    _BUILTIN_PROVIDERS = {"openai", "azure", "deepseek", "ollama"}

    @classmethod
    def register(cls, provider: str, builder: FactoryBuilder) -> None:
        normalized = provider.strip().lower()
        if not normalized:
            raise ValueError("LLM provider name cannot be empty.")
        cls._REGISTRY[normalized] = builder

    @classmethod
    def create(cls, settings: object) -> BaseLLM:
        provider = cls._extract_provider(settings)
        builder = cls._REGISTRY.get(provider)
        if builder is None:
            builder = cls._builtin_builder(provider)
        if builder is None:
            supported_providers = sorted(set(cls._REGISTRY) | cls._BUILTIN_PROVIDERS)
            supported = ", ".join(supported_providers) or "<none>"
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. Supported providers: {supported}."
            )
        return builder(settings)

    @staticmethod
    def _extract_provider(settings: object) -> str:
        llm_settings = getattr(settings, "llm", None)
        provider = getattr(llm_settings, "provider", "")
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("settings.llm.provider must be a non-empty string.")
        return provider.strip().lower()

    @staticmethod
    def _builtin_builder(provider: str) -> FactoryBuilder | None:
        if provider == "openai":
            from libs.llm.openai_llm import OpenAILLM

            return OpenAILLM
        if provider == "azure":
            from libs.llm.azure_llm import AzureLLM

            return AzureLLM
        if provider == "deepseek":
            from libs.llm.deepseek_llm import DeepSeekLLM

            return DeepSeekLLM
        if provider == "ollama":
            from libs.llm.ollama_llm import OllamaLLM

            return OllamaLLM
        return None
