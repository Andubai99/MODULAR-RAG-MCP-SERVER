"""Base contract for Vision LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NotRequired, TypedDict


class ChatResponse(TypedDict):
    """Canonical response payload for vision-chat calls."""

    text: str
    metadata: NotRequired[dict[str, object]]


class BaseVisionLLM(ABC):
    """Abstract vision-LLM client interface."""

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: object | None = None,
    ) -> ChatResponse:
        """Generate a response from text and image input."""
        raise NotImplementedError
