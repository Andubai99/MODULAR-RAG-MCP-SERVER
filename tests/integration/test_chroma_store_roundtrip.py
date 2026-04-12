"""Integration tests for ChromaStore upsert/query roundtrip."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libs.vector_store.vector_store_factory import VectorStoreFactory


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    old_registry = dict(VectorStoreFactory._REGISTRY)
    VectorStoreFactory._REGISTRY.clear()
    yield
    VectorStoreFactory._REGISTRY = old_registry


def _make_settings(persist_directory: Path) -> SimpleNamespace:
    return SimpleNamespace(
        vector_store=SimpleNamespace(
            provider="chroma",
            persist_directory=str(persist_directory),
            collection_name="test_collection",
        )
    )


def test_chroma_store_upsert_query_roundtrip(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path / "chroma")
    vector_store = VectorStoreFactory.create(settings)

    vector_store.upsert(
        [
            {
                "id": "chunk-1",
                "vector": [1.0, 0.0, 0.0],
                "metadata": {"source": "doc-a", "lang": "zh"},
                "content": "alpha",
            },
            {
                "id": "chunk-2",
                "vector": [0.0, 1.0, 0.0],
                "metadata": {"source": "doc-b", "lang": "en"},
                "content": "beta",
            },
            {
                "id": "chunk-3",
                "vector": [0.0, 0.0, 1.0],
                "metadata": {"source": "doc-c", "lang": "zh"},
                "content": "gamma",
            },
        ]
    )

    results = vector_store.query(vector=[0.9, 0.1, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "chunk-1"
    assert results[0]["score"] >= results[1]["score"]

    filtered = vector_store.query(vector=[0.0, 0.0, 1.0], top_k=5, filters={"lang": "zh"})
    assert len(filtered) == 2
    assert all(item["metadata"]["lang"] == "zh" for item in filtered)


def test_chroma_store_persistence_across_instances(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path / "chroma")
    first_instance = VectorStoreFactory.create(settings)
    first_instance.upsert(
        [
            {
                "id": "chunk-persist",
                "vector": [0.2, 0.8],
                "metadata": {"source": "doc-persist"},
                "content": "persist-me",
            }
        ]
    )

    second_instance = VectorStoreFactory.create(settings)
    results = second_instance.query(vector=[0.1, 0.9], top_k=1)

    assert len(results) == 1
    assert results[0]["id"] == "chunk-persist"
    assert results[0]["metadata"]["source"] == "doc-persist"
