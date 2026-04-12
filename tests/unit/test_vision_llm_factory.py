"""Unit tests for vision LLM factory routing."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libs.llm.base_vision_llm import BaseVisionLLM, ChatResponse
from libs.llm.llm_factory import LLMFactory


class _FakeAzureVisionLLM(BaseVisionLLM):
    def chat_with_image(
        self, text: str, image_path: str | bytes, trace: object | None = None
    ) -> ChatResponse:
        del trace
        image_type = "bytes" if isinstance(image_path, bytes) else "path"
        return {"text": f"azure:{text}:{image_type}"}


class _FakeOpenAIVisionLLM(BaseVisionLLM):
    def chat_with_image(
        self, text: str, image_path: str | bytes, trace: object | None = None
    ) -> ChatResponse:
        del image_path, trace
        return {"text": f"openai:{text}"}


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    old_registry = dict(LLMFactory._REGISTRY)
    old_vision_registry = dict(LLMFactory._VISION_REGISTRY)
    LLMFactory._REGISTRY.clear()
    LLMFactory._VISION_REGISTRY.clear()
    yield
    LLMFactory._REGISTRY = old_registry
    LLMFactory._VISION_REGISTRY = old_vision_registry


def _make_settings(provider: str) -> SimpleNamespace:
    return SimpleNamespace(vision_llm=SimpleNamespace(provider=provider))


def test_vision_llm_factory_routes_by_provider() -> None:
    LLMFactory.register_vision("azure", lambda _settings: _FakeAzureVisionLLM())
    LLMFactory.register_vision("openai", lambda _settings: _FakeOpenAIVisionLLM())

    azure_llm = LLMFactory.create_vision_llm(_make_settings("azure"))
    openai_llm = LLMFactory.create_vision_llm(_make_settings("openai"))

    assert isinstance(azure_llm, _FakeAzureVisionLLM)
    assert isinstance(openai_llm, _FakeOpenAIVisionLLM)
    assert azure_llm.chat_with_image("describe", b"img")["text"] == "azure:describe:bytes"
    assert openai_llm.chat_with_image("describe", "a.png")["text"] == "openai:describe"


def test_vision_llm_factory_unknown_provider() -> None:
    LLMFactory.register_vision("azure", lambda _settings: _FakeAzureVisionLLM())

    with pytest.raises(ValueError, match="Unknown Vision LLM provider"):
        LLMFactory.create_vision_llm(_make_settings("unknown"))


def test_vision_llm_factory_requires_provider_field() -> None:
    settings = SimpleNamespace(vision_llm=SimpleNamespace(provider=""))

    with pytest.raises(ValueError, match="settings\\.vision_llm\\.provider"):
        LLMFactory.create_vision_llm(settings)
