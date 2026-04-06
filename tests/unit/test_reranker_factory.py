"""Unit tests for reranker factory routing and fallback behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libs.reranker.base_reranker import BaseReranker
from libs.reranker.reranker_factory import NoneReranker, RerankerFactory


class _FakeCrossEncoderReranker(BaseReranker):
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, object]],
        trace: object | None = None,
    ) -> list[dict[str, object]]:
        del query, trace
        return list(reversed(candidates))


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    old_registry = dict(RerankerFactory._REGISTRY)
    RerankerFactory._REGISTRY.clear()
    yield
    RerankerFactory._REGISTRY = old_registry


def _make_settings(provider: str) -> SimpleNamespace:
    return SimpleNamespace(rerank=SimpleNamespace(provider=provider))


def test_reranker_factory_none_preserves_original_order() -> None:
    reranker = RerankerFactory.create(_make_settings("none"))
    candidates = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    result = reranker.rerank("query", candidates)

    assert isinstance(reranker, NoneReranker)
    assert [item["id"] for item in result] == ["a", "b", "c"]


def test_reranker_factory_routes_registered_provider() -> None:
    RerankerFactory.register(
        "cross_encoder", lambda _settings: _FakeCrossEncoderReranker()
    )

    reranker = RerankerFactory.create(_make_settings("cross_encoder"))
    result = reranker.rerank("query", [{"id": "a"}, {"id": "b"}])

    assert isinstance(reranker, _FakeCrossEncoderReranker)
    assert [item["id"] for item in result] == ["b", "a"]


def test_reranker_factory_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown reranker provider"):
        RerankerFactory.create(_make_settings("unknown"))


def test_reranker_factory_requires_provider_field() -> None:
    settings = SimpleNamespace(rerank=SimpleNamespace(provider=""))

    with pytest.raises(ValueError, match="settings\\.rerank\\.provider"):
        RerankerFactory.create(settings)
