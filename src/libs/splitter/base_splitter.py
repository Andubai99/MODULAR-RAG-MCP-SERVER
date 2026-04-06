"""Base contract for text splitter providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSplitter(ABC):
    """Abstract splitter interface."""

    @abstractmethod
    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        """Split one text into multiple chunks."""
        raise NotImplementedError
