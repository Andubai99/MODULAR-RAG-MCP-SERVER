"""Unit tests for recursive splitter default implementation."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libs.splitter.recursive_splitter import RecursiveSplitter
from libs.splitter.splitter_factory import SplitterFactory


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    old_registry = dict(SplitterFactory._REGISTRY)
    SplitterFactory._REGISTRY.clear()
    yield
    SplitterFactory._REGISTRY = old_registry


def _make_settings(
    strategy: str = "recursive", chunk_size: int = 120, chunk_overlap: int = 20
) -> SimpleNamespace:
    return SimpleNamespace(
        ingestion=SimpleNamespace(
            splitter=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    )


def test_splitter_factory_routes_recursive_strategy() -> None:
    splitter = SplitterFactory.create(_make_settings(strategy="recursive"))
    assert isinstance(splitter, RecursiveSplitter)


def test_recursive_splitter_preserves_code_fence_integrity() -> None:
    text = (
        "# Title\n\n"
        "This is a long paragraph used to force splitting into multiple chunks. "
        "It should preserve markdown structures and keep code fences intact.\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "After code block there is additional context that should still be split "
        "without breaking the fenced block."
    )
    splitter = RecursiveSplitter(_make_settings(chunk_size=140, chunk_overlap=15))
    chunks = splitter.split_text(text)

    assert len(chunks) >= 2
    # no chunk should contain an unmatched single code fence marker
    assert all(chunk.count("```") in (0, 2) for chunk in chunks)
    code_chunks = [chunk for chunk in chunks if "def add(a, b)" in chunk]
    assert len(code_chunks) == 1
    assert code_chunks[0].count("```") == 2


def test_recursive_splitter_honors_configured_chunk_size() -> None:
    text = " ".join([f"token{i}" for i in range(120)])
    splitter_small = RecursiveSplitter(_make_settings(chunk_size=60, chunk_overlap=10))
    splitter_large = RecursiveSplitter(_make_settings(chunk_size=240, chunk_overlap=20))

    small_chunks = splitter_small.split_text(text)
    large_chunks = splitter_large.split_text(text)

    assert len(small_chunks) > len(large_chunks)
    assert all(len(chunk) <= 60 for chunk in small_chunks)
