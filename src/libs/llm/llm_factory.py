"""Factory for creating LLM backends by provider name."""

from __future__ import annotations

from typing import Callable

from libs.llm.base_llm import BaseLLM
from libs.llm.base_vision_llm import BaseVisionLLM

FactoryBuilder = Callable[[object], BaseLLM]
VisionFactoryBuilder = Callable[[object], BaseVisionLLM]


class LLMFactory:
    """Provider registry and constructor for LLM implementations."""

    _REGISTRY: dict[str, FactoryBuilder] = {}
    _VISION_REGISTRY: dict[str, VisionFactoryBuilder] = {}
    _BUILTIN_PROVIDERS = {"openai", "azure", "deepseek", "ollama"}
    _BUILTIN_VISION_PROVIDERS: set[str] = set()

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

    @classmethod
    def register_vision(cls, provider: str, builder: VisionFactoryBuilder) -> None:
        normalized = provider.strip().lower()
        if not normalized:
            raise ValueError("Vision LLM provider name cannot be empty.")
        cls._VISION_REGISTRY[normalized] = builder

    @classmethod
    def create_vision_llm(cls, settings: object) -> BaseVisionLLM:
        provider = cls._extract_vision_provider(settings)
        builder = cls._VISION_REGISTRY.get(provider)
        if builder is None:
            builder = cls._builtin_vision_builder(provider)
        if builder is None:
            supported_providers = sorted(
                set(cls._VISION_REGISTRY) | cls._BUILTIN_VISION_PROVIDERS
            )
            supported = ", ".join(supported_providers) or "<none>"
            raise ValueError(
                f"Unknown Vision LLM provider: '{provider}'. Supported providers: {supported}."
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
    def _extract_vision_provider(settings: object) -> str:
        vision_settings = getattr(settings, "vision_llm", None)
        provider = getattr(vision_settings, "provider", "")
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("settings.vision_llm.provider must be a non-empty string.")
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

    @staticmethod
    def _builtin_vision_builder(provider: str) -> VisionFactoryBuilder | None:
        del provider
        return None
