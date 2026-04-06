"""Unit tests for splitter factory routing."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory


class _FakeRecursiveSplitter(BaseSplitter):
    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        return [f"recursive:{text}"]


class _FakeSemanticSplitter(BaseSplitter):
    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        return [f"semantic:{text}"]


class _FakeFixedSplitter(BaseSplitter):
    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        return [f"fixed:{text}"]


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    old_registry = dict(SplitterFactory._REGISTRY)
    SplitterFactory._REGISTRY.clear()
    yield
    SplitterFactory._REGISTRY = old_registry


def _make_settings(strategy: str) -> SimpleNamespace:
    return SimpleNamespace(ingestion=SimpleNamespace(splitter=strategy))


def test_splitter_factory_routes_by_strategy() -> None:
    SplitterFactory.register("recursive", lambda _settings: _FakeRecursiveSplitter())
    SplitterFactory.register("semantic", lambda _settings: _FakeSemanticSplitter())
    SplitterFactory.register("fixed", lambda _settings: _FakeFixedSplitter())

    recursive_splitter = SplitterFactory.create(_make_settings("recursive"))
    semantic_splitter = SplitterFactory.create(_make_settings("semantic"))
    fixed_splitter = SplitterFactory.create(_make_settings("fixed"))

    assert isinstance(recursive_splitter, _FakeRecursiveSplitter)
    assert isinstance(semantic_splitter, _FakeSemanticSplitter)
    assert isinstance(fixed_splitter, _FakeFixedSplitter)
    assert recursive_splitter.split_text("abc") == ["recursive:abc"]
    assert semantic_splitter.split_text("abc") == ["semantic:abc"]
    assert fixed_splitter.split_text("abc") == ["fixed:abc"]


def test_splitter_factory_unknown_strategy() -> None:
    SplitterFactory.register("recursive", lambda _settings: _FakeRecursiveSplitter())

    with pytest.raises(ValueError, match="Unknown splitter strategy"):
        SplitterFactory.create(_make_settings("unknown"))


def test_splitter_factory_requires_strategy_field() -> None:
    settings = SimpleNamespace(ingestion=SimpleNamespace(splitter=""))

    with pytest.raises(ValueError, match="settings\\.ingestion\\.splitter"):
        SplitterFactory.create(settings)
