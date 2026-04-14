"""Unit tests for SparseEncoder."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from ingestion.embedding.sparse_encoder import SparseEncoder


def _chunk(text: str, idx: int) -> Chunk:
    return Chunk(
        id=f"chunk-{idx}",
        text=text,
        metadata={"source_path": "tests/fixtures/sample_documents/sample.pdf"},
        start_offset=0,
        end_offset=max(1, len(text)),
        source_ref="doc-1",
    )


def test_encode_returns_sparse_vectors_for_each_chunk() -> None:
    encoder = SparseEncoder()
    chunks = [
        _chunk("Hybrid search uses BM25 and vector retrieval.", 0),
        _chunk("BM25 retrieval improves exact keyword recall.", 1),
    ]

    records = encoder.encode(chunks)

    assert len(records) == 2
    assert all(isinstance(record, ChunkRecord) for record in records)
    assert all(record.sparse_vector is not None for record in records)
    assert all(isinstance(value, float) for value in records[0].sparse_vector.values())


def test_encode_is_case_insensitive_and_filters_stopwords() -> None:
    encoder = SparseEncoder()
    chunk = _chunk("The THE retrieval Retrieval and BM25", 0)

    record = encoder.encode([chunk])[0]

    assert "the" not in record.sparse_vector
    assert "and" not in record.sparse_vector
    assert "retrieval" in record.sparse_vector
    assert "bm25" in record.sparse_vector


def test_encode_empty_tokens_returns_empty_sparse_vector() -> None:
    encoder = SparseEncoder()
    chunk = _chunk("... !!! ???", 0)

    record = encoder.encode([chunk])[0]

    assert record.sparse_vector == {}
    assert record.metadata["sparse_token_count"] == 0
    assert record.metadata["sparse_unique_terms"] == 0


def test_encode_weights_are_tf_normalized() -> None:
    encoder = SparseEncoder(stopwords=set())
    chunk = _chunk("alpha alpha alpha beta beta gamma", 0)

    record = encoder.encode([chunk])[0]

    assert record.sparse_vector["alpha"] == 1.0
    assert record.sparse_vector["beta"] == pytest.approx(2.0 / 3.0)
    assert record.sparse_vector["gamma"] == pytest.approx(1.0 / 3.0)


def test_encode_validates_input_types() -> None:
    encoder = SparseEncoder()

    with pytest.raises(ValueError, match="expects chunks as a list"):
        encoder.encode("not-list")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must be Chunk"):
        encoder.encode([_chunk("ok", 0), "bad-item"])  # type: ignore[list-item]


def test_encode_empty_list_returns_empty() -> None:
    encoder = SparseEncoder()

    assert encoder.encode([]) == []


def test_encode_records_trace_stage() -> None:
    encoder = SparseEncoder()
    trace = TraceContext(trace_type="ingestion")

    _ = encoder.encode([_chunk("sparse encoder trace sample", 0)], trace=trace)

    assert len(trace.stages) == 1
    stage = trace.stages[0]
    assert stage["stage"] == "sparse_encode"
    assert stage["method"] == "bm25"
    assert stage["details"]["chunk_count"] == 1
