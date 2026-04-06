"""Base contract for vector store providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, NotRequired, TypedDict


class VectorRecord(TypedDict):
    """Canonical vector record written into a vector store."""

    id: str
    vector: list[float]
    metadata: dict[str, object]
    content: NotRequired[str]


class QueryResult(TypedDict):
    """Canonical retrieval result returned by a vector store query."""

    id: str
    score: float
    metadata: dict[str, object]
    content: NotRequired[str]


class BaseVectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def upsert(
        self, records: list[VectorRecord], trace: object | None = None
    ) -> None:
        """Insert or update vector records."""
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, object] | None = None,
        trace: object | None = None,
    ) -> list[QueryResult]:
        """Retrieve top-k records by vector similarity."""
        raise NotImplementedError
