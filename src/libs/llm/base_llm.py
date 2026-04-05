"""Base contract for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

ChatMessage = Mapping[str, str]


class BaseLLM(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    def chat(self, messages: Sequence[ChatMessage]) -> str:
        """Generate a response from chat messages."""
        raise NotImplementedError
