"""Base contract for document loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import Document


class BaseLoader(ABC):
    """Abstract loader interface."""

    @abstractmethod
    def load(self, path: str) -> Document:
        """Load one source file into a canonical Document."""
        raise NotImplementedError
