"""Base contract for reranker providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseReranker(ABC):
    """Abstract reranker interface."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, object]],
        trace: object | None = None,
    ) -> list[dict[str, object]]:
        """Rerank retrieved candidates for one query."""
        raise NotImplementedError
