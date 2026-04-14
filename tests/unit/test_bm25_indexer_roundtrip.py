"""Roundtrip tests for BM25Indexer."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.types import ChunkRecord
from ingestion.storage.bm25_indexer import BM25Indexer


def _make_settings(collection_name: str = "demo") -> SimpleNamespace:
    return SimpleNamespace(
        vector_store=SimpleNamespace(
            persist_directory="data/db/chroma",
            collection_name=collection_name,
        )
    )


def _make_persist_dir(prefix: str) -> Path:
    path = ROOT_DIR / "cache" / "bm25_tests" / f"{prefix}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _record(
    chunk_id: str,
    source_path: str,
    sparse: dict[str, float],
    token_count: int | None = None,
) -> ChunkRecord:
    metadata = {"source_path": source_path}
    if token_count is not None:
        metadata["sparse_token_count"] = token_count
    return ChunkRecord(
        id=chunk_id,
        text=f"text for {chunk_id}",
        metadata=metadata,
        sparse_vector=sparse,
    )


def test_roundtrip_build_save_load_query_stable_top_ids() -> None:
    persist_dir = _make_persist_dir("roundtrip")
    records = [
        _record("c1", "a.pdf", {"rag": 1.0, "retrieval": 0.8}, token_count=10),
        _record("c2", "b.pdf", {"retrieval": 1.0, "bm25": 0.9}, token_count=11),
        _record("c3", "c.pdf", {"rag": 0.7, "index": 0.5}, token_count=9),
    ]

    indexer = BM25Indexer(_make_settings("demo"), persist_dir=persist_dir)
    indexer.rebuild(records, collection="demo")
    first = indexer.query("rag retrieval", top_k=3, collection="demo")

    reloaded = BM25Indexer(_make_settings("demo"), persist_dir=persist_dir)
    second = reloaded.query("rag retrieval", top_k=3, collection="demo")

    assert [item["chunk_id"] for item in first] == [item["chunk_id"] for item in second]
    assert [item["chunk_id"] for item in second] == ["c1", "c2", "c3"]


def test_idf_formula_matches_spec() -> None:
    persist_dir = _make_persist_dir("idf")
    records = [
        _record("c1", "a.pdf", {"rag": 1.0}),
        _record("c2", "b.pdf", {"rag": 0.9, "bm25": 1.0}),
        _record("c3", "c.pdf", {"bm25": 0.7}),
    ]
    indexer = BM25Indexer(_make_settings("demo"), persist_dir=persist_dir)
    indexer.rebuild(records, collection="demo")

    payload = indexer.get_index("demo")
    rag_idf = payload["inverted_index"]["rag"]["idf"]
    bm25_idf = payload["inverted_index"]["bm25"]["idf"]

    expected = math.log((3 - 2 + 0.5) / (2 + 0.5))
    assert rag_idf == pytest.approx(expected)
    assert bm25_idf == pytest.approx(expected)


def test_incremental_upsert_updates_index_without_rebuild_loss() -> None:
    persist_dir = _make_persist_dir("incremental")
    indexer = BM25Indexer(_make_settings("demo"), persist_dir=persist_dir)
    base = [
        _record("c1", "a.pdf", {"alpha": 1.0}),
        _record("c2", "b.pdf", {"beta": 1.0}),
    ]
    indexer.rebuild(base, collection="demo")

    indexer.upsert([_record("c3", "c.pdf", {"gamma": 1.0})], collection="demo")

    gamma_results = indexer.query("gamma", top_k=3, collection="demo")
    alpha_results = indexer.query("alpha", top_k=3, collection="demo")
    assert [item["chunk_id"] for item in gamma_results] == ["c3"]
    assert [item["chunk_id"] for item in alpha_results] == ["c1"]


def test_remove_document_deletes_matching_source_path() -> None:
    persist_dir = _make_persist_dir("remove_doc")
    indexer = BM25Indexer(_make_settings("demo"), persist_dir=persist_dir)
    indexer.rebuild(
        [
            _record("c1", "same.pdf", {"alpha": 1.0}),
            _record("c2", "same.pdf", {"beta": 1.0}),
            _record("c3", "other.pdf", {"alpha": 0.5}),
        ],
        collection="demo",
    )

    indexer.remove_document("same.pdf", collection="demo")
    results = indexer.query("alpha beta", top_k=5, collection="demo")

    assert [item["chunk_id"] for item in results] == ["c3"]


def test_query_validates_top_k() -> None:
    indexer = BM25Indexer(_make_settings("demo"), persist_dir=_make_persist_dir("query_k"))

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        indexer.query("alpha", top_k=0, collection="demo")
