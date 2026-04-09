"""Base contract for evaluator providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    """Abstract evaluator interface."""

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: object | None = None,
    ) -> dict[str, float]:
        """Evaluate retrieval quality metrics for one query."""
        raise NotImplementedError
