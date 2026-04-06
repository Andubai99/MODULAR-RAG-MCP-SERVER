"""Unit tests for embedding factory routing."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class _FakeOpenAIEmbedding(BaseEmbedding):
    def embed(
        self, texts: list[str], trace: object | None = None
    ) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


class _FakeOllamaEmbedding(BaseEmbedding):
    def embed(
        self, texts: list[str], trace: object | None = None
    ) -> list[list[float]]:
        return [[float(len(text)), 2.0] for text in texts]


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    old_registry = dict(EmbeddingFactory._REGISTRY)
    EmbeddingFactory._REGISTRY.clear()
    yield
    EmbeddingFactory._REGISTRY = old_registry


def _make_settings(provider: str) -> SimpleNamespace:
    return SimpleNamespace(embedding=SimpleNamespace(provider=provider))


def test_embedding_factory_routes_by_provider() -> None:
    EmbeddingFactory.register("openai", lambda _settings: _FakeOpenAIEmbedding())
    EmbeddingFactory.register("ollama", lambda _settings: _FakeOllamaEmbedding())

    openai_embedding = EmbeddingFactory.create(_make_settings("openai"))
    ollama_embedding = EmbeddingFactory.create(_make_settings("ollama"))

    assert isinstance(openai_embedding, _FakeOpenAIEmbedding)
    assert isinstance(ollama_embedding, _FakeOllamaEmbedding)
    assert openai_embedding.embed(["abc"]) == [[3.0, 1.0]]
    assert ollama_embedding.embed(["abc"]) == [[3.0, 2.0]]


def test_embedding_factory_unknown_provider() -> None:
    EmbeddingFactory.register("openai", lambda _settings: _FakeOpenAIEmbedding())

    with pytest.raises(ValueError, match="Unknown embedding provider"):
        EmbeddingFactory.create(_make_settings("unknown"))


def test_embedding_factory_requires_provider_field() -> None:
    settings = SimpleNamespace(embedding=SimpleNamespace(provider=""))

    with pytest.raises(ValueError, match="settings\\.embedding\\.provider"):
        EmbeddingFactory.create(settings)
