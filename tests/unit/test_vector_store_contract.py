"""Contract tests for vector store interface and factory routing."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libs.vector_store.base_vector_store import BaseVectorStore, QueryResult, VectorRecord
from libs.vector_store.vector_store_factory import VectorStoreFactory


class _FakeVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    def upsert(self, records: list[VectorRecord], trace: object | None = None) -> None:
        existing = {record["id"]: record for record in self._records}
        for record in records:
            existing[record["id"]] = record
        self._records = list(existing.values())

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, object] | None = None,
        trace: object | None = None,
    ) -> list[QueryResult]:
        del vector, trace
        candidates = self._records
        if filters:
            def _match(record: VectorRecord) -> bool:
                metadata = record.get("metadata", {})
                return all(metadata.get(key) == value for key, value in filters.items())

            candidates = [record for record in candidates if _match(record)]
        selected = candidates[:top_k]
        return [
            {
                "id": record["id"],
                "score": 1.0,
                "metadata": record.get("metadata", {}),
                "content": record.get("content", ""),
            }
            for record in selected
        ]


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    old_registry = dict(VectorStoreFactory._REGISTRY)
    VectorStoreFactory._REGISTRY.clear()
    yield
    VectorStoreFactory._REGISTRY = old_registry


def _make_settings(provider: str) -> SimpleNamespace:
    return SimpleNamespace(vector_store=SimpleNamespace(provider=provider))


def test_vector_store_factory_routes_provider() -> None:
    VectorStoreFactory.register("chroma", lambda _settings: _FakeVectorStore())

    vector_store = VectorStoreFactory.create(_make_settings("chroma"))

    assert isinstance(vector_store, _FakeVectorStore)


def test_vector_store_contract_upsert_and_query_shape() -> None:
    vector_store = _FakeVectorStore()
    vector_store.upsert(
        [
            {
                "id": "chunk-1",
                "vector": [0.1, 0.2],
                "metadata": {"source": "doc-a"},
                "content": "content-a",
            },
            {
                "id": "chunk-2",
                "vector": [0.3, 0.4],
                "metadata": {"source": "doc-b"},
                "content": "content-b",
            },
        ]
    )

    results = vector_store.query(vector=[0.0, 0.0], top_k=2, filters={"source": "doc-a"})

    assert isinstance(results, list)
    assert len(results) == 1
    first = results[0]
    assert isinstance(first["id"], str)
    assert isinstance(first["score"], float)
    assert isinstance(first["metadata"], dict)
    assert first["metadata"]["source"] == "doc-a"


def test_vector_store_factory_unknown_provider() -> None:
    VectorStoreFactory.register("chroma", lambda _settings: _FakeVectorStore())

    with pytest.raises(ValueError, match="Unknown vector store provider"):
        VectorStoreFactory.create(_make_settings("unknown"))


def test_vector_store_factory_requires_provider_field() -> None:
    settings = SimpleNamespace(vector_store=SimpleNamespace(provider=""))

    with pytest.raises(ValueError, match="settings\\.vector_store\\.provider"):
        VectorStoreFactory.create(settings)
