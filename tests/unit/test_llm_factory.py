"""Unit tests for LLM factory routing."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory


class _FakeOpenAILLM(BaseLLM):
    def chat(self, messages: list[dict[str, str]]) -> str:
        return f"openai:{len(messages)}"


class _FakeOllamaLLM(BaseLLM):
    def chat(self, messages: list[dict[str, str]]) -> str:
        return f"ollama:{len(messages)}"


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    old_registry = dict(LLMFactory._REGISTRY)
    LLMFactory._REGISTRY.clear()
    yield
    LLMFactory._REGISTRY = old_registry


def _make_settings(provider: str) -> SimpleNamespace:
    return SimpleNamespace(llm=SimpleNamespace(provider=provider))


def test_llm_factory_routes_by_provider() -> None:
    LLMFactory.register("openai", lambda _settings: _FakeOpenAILLM())
    LLMFactory.register("ollama", lambda _settings: _FakeOllamaLLM())

    openai_llm = LLMFactory.create(_make_settings("openai"))
    ollama_llm = LLMFactory.create(_make_settings("ollama"))

    assert isinstance(openai_llm, _FakeOpenAILLM)
    assert isinstance(ollama_llm, _FakeOllamaLLM)
    assert openai_llm.chat([{"role": "user", "content": "hi"}]) == "openai:1"
    assert ollama_llm.chat([{"role": "user", "content": "hi"}]) == "ollama:1"


def test_llm_factory_unknown_provider() -> None:
    LLMFactory.register("openai", lambda _settings: _FakeOpenAILLM())

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        LLMFactory.create(_make_settings("unknown"))


def test_llm_factory_requires_provider_field() -> None:
    settings = SimpleNamespace(llm=SimpleNamespace(provider=""))

    with pytest.raises(ValueError, match="settings\\.llm\\.provider"):
        LLMFactory.create(settings)
