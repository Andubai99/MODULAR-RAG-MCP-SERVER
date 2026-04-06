"""Factory for creating reranker backends by provider name."""

from __future__ import annotations

from typing import Callable

from libs.reranker.base_reranker import BaseReranker

FactoryBuilder = Callable[[object], BaseReranker]


class NoneReranker(BaseReranker):
    """Fallback reranker that preserves original candidate order."""

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, object]],
        trace: object | None = None,
    ) -> list[dict[str, object]]:
        del query, trace
        return list(candidates)


class RerankerFactory:
    """Provider registry and constructor for reranker implementations."""

    _REGISTRY: dict[str, FactoryBuilder] = {}

    @classmethod
    def register(cls, provider: str, builder: FactoryBuilder) -> None:
        normalized = provider.strip().lower()
        if not normalized:
            raise ValueError("Reranker provider name cannot be empty.")
        cls._REGISTRY[normalized] = builder

    @classmethod
    def create(cls, settings: object) -> BaseReranker:
        provider = cls._extract_provider(settings)

        if provider == "none":
            builder = cls._REGISTRY.get("none")
            if builder is not None:
                return builder(settings)
            return NoneReranker()

        builder = cls._REGISTRY.get(provider)
        if builder is None:
            supported = sorted(set(cls._REGISTRY) | {"none"})
            supported_text = ", ".join(supported) or "<none>"
            raise ValueError(
                f"Unknown reranker provider: '{provider}'. "
                f"Supported providers: {supported_text}."
            )
        return builder(settings)

    @staticmethod
    def _extract_provider(settings: object) -> str:
        rerank_settings = getattr(settings, "rerank", None)
        provider = getattr(rerank_settings, "provider", "")
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("settings.rerank.provider must be a non-empty string.")
        return provider.strip().lower()
