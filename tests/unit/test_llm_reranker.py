"""Unit tests for LLM reranker provider."""

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
from libs.reranker.llm_reranker import LLMReranker
from libs.reranker.reranker_factory import RerankerFactory


class _FakeLLM(BaseLLM):
    def __init__(self, response_text: str, raise_error: Exception | None = None) -> None:
        self._response_text = response_text
        self._raise_error = raise_error
        self.last_messages: list[dict[str, str]] | None = None

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.last_messages = list(messages)
        if self._raise_error is not None:
            raise self._raise_error
        return self._response_text


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    old_registry = dict(RerankerFactory._REGISTRY)
    RerankerFactory._REGISTRY.clear()
    yield
    RerankerFactory._REGISTRY = old_registry


def _make_settings(provider: str = "llm") -> SimpleNamespace:
    return SimpleNamespace(
        llm=SimpleNamespace(
            provider="ollama",
            model="qwen2.5:7b",
            deployment_name="",
            azure_endpoint="",
            api_version="",
            api_key="",
            base_url="http://localhost:11434",
            temperature=0.0,
            max_tokens=128,
        ),
        rerank=SimpleNamespace(provider=provider, model="", top_k=5, enabled=True),
    )


def test_reranker_factory_routes_llm_provider() -> None:
    reranker = RerankerFactory.create(_make_settings("llm"))
    assert isinstance(reranker, LLMReranker)


def test_llm_reranker_returns_structured_rank_result() -> None:
    fake_llm = _FakeLLM('{"ranked_ids": ["c3", "c1"]}')
    reranker = LLMReranker(
        _make_settings(),
        llm_client=fake_llm,
        prompt_template="RERANK_TEMPLATE_FOR_TEST",
    )
    candidates = [
        {"id": "c1", "text": "first"},
        {"id": "c2", "text": "second"},
        {"id": "c3", "text": "third"},
    ]

    result = reranker.rerank("what is best", candidates)

    assert [item["id"] for item in result] == ["c3", "c1", "c2"]
    assert reranker.last_fallback_reason is None
    assert fake_llm.last_messages is not None
    assert "RERANK_TEMPLATE_FOR_TEST" in fake_llm.last_messages[0]["content"]


def test_llm_reranker_invalid_schema_raises_readable_error() -> None:
    reranker = LLMReranker(
        _make_settings(),
        llm_client=_FakeLLM('{"ids": ["a"]}'),
        prompt_template="PROMPT",
    )
    candidates = [{"id": "a", "text": "candidate"}]

    with pytest.raises(ValueError, match="ranked_ids"):
        reranker.rerank("query", candidates)


def test_llm_reranker_unknown_ranked_id_raises_readable_error() -> None:
    reranker = LLMReranker(
        _make_settings(),
        llm_client=_FakeLLM('{"ranked_ids": ["unknown"]}'),
        prompt_template="PROMPT",
    )
    candidates = [{"id": "a", "text": "candidate"}]

    with pytest.raises(ValueError, match="unknown candidate id"):
        reranker.rerank("query", candidates)


def test_llm_reranker_runtime_error_returns_fallback_signal() -> None:
    reranker = LLMReranker(
        _make_settings(),
        llm_client=_FakeLLM("", raise_error=RuntimeError("timeout")),
        prompt_template="PROMPT",
    )
    candidates = [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}]

    result = reranker.rerank("query", candidates)

    assert [item["id"] for item in result] == ["a", "b"]
    assert all(item["rerank_fallback"] is True for item in result)
    assert "llm_error:RuntimeError" == reranker.last_fallback_reason
