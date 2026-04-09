"""Factory for creating splitter backends by strategy name."""

from __future__ import annotations

from typing import Callable

from libs.splitter.base_splitter import BaseSplitter

FactoryBuilder = Callable[[object], BaseSplitter]


class SplitterFactory:
    """Strategy registry and constructor for splitter implementations."""

    _REGISTRY: dict[str, FactoryBuilder] = {}
    _BUILTIN_STRATEGIES = {"recursive"}

    @classmethod
    def register(cls, strategy: str, builder: FactoryBuilder) -> None:
        normalized = strategy.strip().lower()
        if not normalized:
            raise ValueError("Splitter strategy name cannot be empty.")
        cls._REGISTRY[normalized] = builder

    @classmethod
    def create(cls, settings: object) -> BaseSplitter:
        strategy = cls._extract_strategy(settings)
        builder = cls._REGISTRY.get(strategy)
        if builder is None:
            builder = cls._builtin_builder(strategy)
        if builder is None:
            supported_strategies = sorted(set(cls._REGISTRY) | cls._BUILTIN_STRATEGIES)
            supported = ", ".join(supported_strategies) or "<none>"
            raise ValueError(
                f"Unknown splitter strategy: '{strategy}'. "
                f"Supported strategies: {supported}."
            )
        return builder(settings)

    @staticmethod
    def _extract_strategy(settings: object) -> str:
        ingestion_settings = getattr(settings, "ingestion", None)
        strategy = getattr(ingestion_settings, "splitter", "")
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError("settings.ingestion.splitter must be a non-empty string.")
        return strategy.strip().lower()

    @staticmethod
    def _builtin_builder(strategy: str) -> FactoryBuilder | None:
        if strategy == "recursive":
            from libs.splitter.recursive_splitter import RecursiveSplitter

            return RecursiveSplitter
        return None
