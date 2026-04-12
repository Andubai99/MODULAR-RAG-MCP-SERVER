"""Unit tests for cross-encoder reranker provider."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from libs.reranker.reranker_factory import RerankerFactory


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    old_registry = dict(RerankerFactory._REGISTRY)
    RerankerFactory._REGISTRY.clear()
    yield
    RerankerFactory._REGISTRY = old_registry


def _make_settings(provider: str = "cross_encoder", top_k: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        rerank=SimpleNamespace(provider=provider, top_k=top_k, model="bge-reranker", enabled=True)
    )


def test_reranker_factory_routes_cross_encoder_provider() -> None:
    reranker = RerankerFactory.create(_make_settings())
    assert isinstance(reranker, CrossEncoderReranker)


def test_cross_encoder_reranker_sorts_top_m_with_mock_scorer() -> None:
    def _mock_scorer(query: str, candidates: list[dict[str, str]]) -> list[float]:
        del query
        assert [c["id"] for c in candidates] == ["a", "b", "c"]
        return [0.2, 0.9, 0.5]

    reranker = CrossEncoderReranker(_make_settings(top_k=3), scorer=_mock_scorer)
    candidates = [
        {"id": "a", "text": "doc-a"},
        {"id": "b", "text": "doc-b"},
        {"id": "c", "text": "doc-c"},
    ]

    result = reranker.rerank("query", candidates)

    assert [item["id"] for item in result] == ["b", "c", "a"]
    assert reranker.last_fallback_reason is None


def test_cross_encoder_reranker_only_reorders_head_top_m() -> None:
    def _mock_scorer(query: str, candidates: list[dict[str, str]]) -> list[float]:
        del query
        assert [c["id"] for c in candidates] == ["a", "b"]
        return [0.1, 0.9]

    reranker = CrossEncoderReranker(_make_settings(top_k=2), scorer=_mock_scorer)
    candidates = [
        {"id": "a", "text": "doc-a"},
        {"id": "b", "text": "doc-b"},
        {"id": "c", "text": "doc-c"},
    ]

    result = reranker.rerank("query", candidates)

    assert [item["id"] for item in result] == ["b", "a", "c"]


def test_cross_encoder_reranker_invalid_scorer_output_raises_readable_error() -> None:
    def _bad_scorer(query: str, candidates: list[dict[str, str]]) -> list[float]:
        del query, candidates
        return [0.5]

    reranker = CrossEncoderReranker(_make_settings(top_k=2), scorer=_bad_scorer)
    candidates = [{"id": "a", "text": "doc-a"}, {"id": "b", "text": "doc-b"}]

    result = reranker.rerank("query", candidates)

    assert [item["id"] for item in result] == ["a", "b"]
    assert all(item["rerank_fallback"] is True for item in result)
    assert reranker.last_fallback_reason == "cross_encoder_error:ValueError"


def test_cross_encoder_reranker_timeout_returns_fallback_signal() -> None:
    def _timeout_scorer(query: str, candidates: list[dict[str, str]]) -> list[float]:
        del query, candidates
        raise TimeoutError("timeout")

    reranker = CrossEncoderReranker(_make_settings(), scorer=_timeout_scorer)
    candidates = [{"id": "a", "text": "doc-a"}, {"id": "b", "text": "doc-b"}]

    result = reranker.rerank("query", candidates)

    assert [item["id"] for item in result] == ["a", "b"]
    assert all(item["rerank_fallback"] is True for item in result)
    assert reranker.last_fallback_reason == "cross_encoder_error:TimeoutError"
