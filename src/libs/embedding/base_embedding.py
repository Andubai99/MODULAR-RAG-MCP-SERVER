"""Base contract for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """Abstract embedding client interface."""

    @abstractmethod
    def embed(
        self, texts: list[str], trace: object | None = None
    ) -> list[list[float]]:
        """Convert input texts into embedding vectors."""
        raise NotImplementedError
