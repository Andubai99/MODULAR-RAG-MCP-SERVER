"""Factory for creating evaluator backends by provider name."""

from __future__ import annotations

from typing import Callable

from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator

FactoryBuilder = Callable[[object], BaseEvaluator]


class EvaluatorFactory:
    """Provider registry and constructor for evaluator implementations."""

    _REGISTRY: dict[str, FactoryBuilder] = {}

    @classmethod
    def register(cls, provider: str, builder: FactoryBuilder) -> None:
        normalized = provider.strip().lower()
        if not normalized:
            raise ValueError("Evaluator provider name cannot be empty.")
        cls._REGISTRY[normalized] = builder

    @classmethod
    def create(cls, settings: object) -> BaseEvaluator:
        provider = cls._extract_provider(settings)

        if provider == "custom":
            builder = cls._REGISTRY.get("custom")
            if builder is not None:
                return builder(settings)
            metrics = cls._extract_metrics(settings)
            return CustomEvaluator(metrics=metrics)

        builder = cls._REGISTRY.get(provider)
        if builder is None:
            supported = sorted(set(cls._REGISTRY) | {"custom"})
            supported_text = ", ".join(supported) or "<none>"
            raise ValueError(
                f"Unknown evaluator provider: '{provider}'. "
                f"Supported providers: {supported_text}."
            )
        return builder(settings)

    @staticmethod
    def _extract_provider(settings: object) -> str:
        evaluation_settings = getattr(settings, "evaluation", None)
        provider = getattr(evaluation_settings, "provider", "")
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("settings.evaluation.provider must be a non-empty string.")
        return provider.strip().lower()

    @staticmethod
    def _extract_metrics(settings: object) -> list[str]:
        evaluation_settings = getattr(settings, "evaluation", None)
        metrics = getattr(evaluation_settings, "metrics", None)
        if metrics is None:
            return ["hit_rate", "mrr"]
        if not isinstance(metrics, list) or not all(isinstance(item, str) for item in metrics):
            raise ValueError("settings.evaluation.metrics must be a list of strings.")
        return metrics
