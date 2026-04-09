"""Unit tests for Ollama embedding provider."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib import error

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.openai_embedding import request as openai_request_module


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    old_registry = dict(EmbeddingFactory._REGISTRY)
    EmbeddingFactory._REGISTRY.clear()
    yield
    EmbeddingFactory._REGISTRY = old_registry


def _make_settings(base_url: str = "", api_key: str = "secret-should-not-leak") -> SimpleNamespace:
    return SimpleNamespace(
        embedding=SimpleNamespace(
            provider="ollama",
            model="nomic-embed-text",
            dimensions=768,
            azure_endpoint="",
            deployment_name="",
            api_version="",
            api_key=api_key,
            base_url=base_url,
        )
    )


def test_embedding_factory_routes_ollama_provider() -> None:
    embedding = EmbeddingFactory.create(_make_settings())
    assert embedding.__class__.__name__ == "OllamaEmbedding"


def test_ollama_embedding_embed_with_mock_http(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_urlopen(req: object, timeout: int = 0) -> _FakeHTTPResponse:
        del timeout
        captured["url"] = req.full_url
        captured["headers"] = {key.lower(): value for key, value in req.header_items()}
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    monkeypatch.setattr(openai_request_module, "urlopen", _fake_urlopen)
    embedding = EmbeddingFactory.create(_make_settings("http://localhost:11434"))
    vectors = embedding.embed(["hello", "world"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert str(captured["url"]).endswith("/api/embed")
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert "content-type" in headers
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "nomic-embed-text"
    assert body["input"] == ["hello", "world"]


def test_ollama_embedding_connection_failure_message_does_not_leak_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_url_error(req: object, timeout: int = 0) -> _FakeHTTPResponse:
        del req, timeout
        raise error.URLError("connection refused")

    monkeypatch.setattr(openai_request_module, "urlopen", _raise_url_error)
    embedding = EmbeddingFactory.create(_make_settings())

    with pytest.raises(
        RuntimeError, match=r"provider=ollama: HTTP request failed \(error_type=URLError,"
    ) as exc_info:
        embedding.embed(["hello"])

    assert "secret-should-not-leak" not in str(exc_info.value)


def test_ollama_embedding_timeout_message_does_not_leak_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_timeout(req: object, timeout: int = 0) -> _FakeHTTPResponse:
        del req, timeout
        raise TimeoutError("timed out")

    monkeypatch.setattr(openai_request_module, "urlopen", _raise_timeout)
    embedding = EmbeddingFactory.create(_make_settings())

    with pytest.raises(
        RuntimeError, match=r"provider=ollama: HTTP request failed \(error_type=TimeoutError\)"
    ) as exc_info:
        embedding.embed(["hello"])

    assert "secret-should-not-leak" not in str(exc_info.value)
