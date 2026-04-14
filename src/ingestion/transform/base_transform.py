"""Base contract for ingestion transform modules."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import Chunk
from core.trace.trace_context import TraceContext


class BaseTransform(ABC):
    """Abstract transform interface for chunk-level processing."""

    @abstractmethod
    def transform(
        self, chunks: list[Chunk], trace: TraceContext | None = None
    ) -> list[Chunk]:
        """Transform chunks and return new chunk list."""
        raise NotImplementedError
