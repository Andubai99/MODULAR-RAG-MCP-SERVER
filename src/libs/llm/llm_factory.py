"""Factory for creating LLM backends by provider name."""

from __future__ import annotations

from typing import Callable

from libs.llm.base_llm import BaseLLM

FactoryBuilder = Callable[[object], BaseLLM]


class LLMFactory:
    """Provider registry and constructor for LLM implementations."""

    _REGISTRY: dict[str, FactoryBuilder] = {}

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
            supported = ", ".join(sorted(cls._REGISTRY)) or "<none>"
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
