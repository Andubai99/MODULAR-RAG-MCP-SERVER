"""Unit tests for VectorUpserter deterministic ID and idempotent upsert behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.types import ChunkRecord
from ingestion.storage.vector_upserter import VectorUpserter
from libs.vector_store.base_vector_store import BaseVectorStore, QueryResult, VectorRecord


class _FakeVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.calls: list[list[VectorRecord]] = []
        self.records_by_id: dict[str, VectorRecord] = {}

    def upsert(self, records: list[VectorRecord], trace: object | None = None) -> None:
        del trace
        snapshot: list[VectorRecord] = []
        for record in records:
            normalized: VectorRecord = {
                "id": record["id"],
                "vector": list(record["vector"]),
                "metadata": dict(record.get("metadata", {})),
            }
            if "content" in record:
                normalized["content"] = str(record.get("content", ""))
            snapshot.append(normalized)
            self.records_by_id[normalized["id"]] = normalized
        self.calls.append(snapshot)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, object] | None = None,
        trace: object | None = None,
    ) -> list[QueryResult]:
        del vector, top_k, filters, trace
        return []


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        vector_store=SimpleNamespace(
            provider="chroma",
            persist_directory="data/db/chroma",
            collection_name="test_collection",
        )
    )


def _record(chunk_index: int, text: str) -> ChunkRecord:
    return ChunkRecord(
        id=f"legacy-{chunk_index}",
        text=text,
        metadata={
            "source_path": "tests/fixtures/sample_documents/sample.pdf",
            "chunk_index": chunk_index,
        },
        dense_vector=[float(chunk_index), 1.0, 2.0],
    )


def test_same_chunk_upsert_twice_returns_same_id_and_deduplicates() -> None:
    vector_store = _FakeVectorStore()
    upserter = VectorUpserter(_settings(), vector_store=vector_store)
    record = _record(0, "same content")

    first_ids = upserter.upsert([record])
    second_ids = upserter.upsert([record])

    assert first_ids == second_ids
    assert len(vector_store.records_by_id) == 1
    assert [item["id"] for item in vector_store.calls[0]] == first_ids
    assert [item["id"] for item in vector_store.calls[1]] == second_ids


def test_content_change_produces_new_id() -> None:
    vector_store = _FakeVectorStore()
    upserter = VectorUpserter(_settings(), vector_store=vector_store)

    first_id = upserter.upsert([_record(0, "alpha content")])[0]
    second_id = upserter.upsert([_record(0, "alpha content changed")])[0]

    assert first_id != second_id
    assert set(vector_store.records_by_id.keys()) == {first_id, second_id}


def test_batch_upsert_preserves_input_order() -> None:
    vector_store = _FakeVectorStore()
    upserter = VectorUpserter(_settings(), vector_store=vector_store)
    batch = [_record(2, "third"), _record(0, "first"), _record(1, "second")]

    generated_ids = upserter.upsert(batch)
    latest_call = vector_store.calls[-1]

    assert [item["id"] for item in latest_call] == generated_ids
    assert [item.get("content") for item in latest_call] == ["third", "first", "second"]


def test_upsert_requires_dense_vector() -> None:
    vector_store = _FakeVectorStore()
    upserter = VectorUpserter(_settings(), vector_store=vector_store)
    record = ChunkRecord(
        id="legacy-0",
        text="text without vector",
        metadata={"source_path": "tests/fixtures/sample_documents/sample.pdf", "chunk_index": 0},
    )

    with pytest.raises(ValueError, match="dense_vector is required"):
        upserter.upsert([record])
