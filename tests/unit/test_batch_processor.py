"""Unit tests for BatchProcessor."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from ingestion.embedding.batch_processor import BatchProcessor


class _FakeDenseEncoder:
    def __init__(self, drop_last: bool = False) -> None:
        self.drop_last = drop_last
        self.calls: list[list[str]] = []

    def encode(
        self, chunks: list[Chunk], trace: TraceContext | None = None
    ) -> list[ChunkRecord]:
        del trace
        self.calls.append([chunk.id for chunk in chunks])
        records = [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=dict(chunk.metadata),
                dense_vector=[float(index), 1.0],
            )
            for index, chunk in enumerate(chunks)
        ]
        if self.drop_last and records:
            return records[:-1]
        return records


class _FakeSparseEncoder:
    def __init__(self, drop_last: bool = False) -> None:
        self.drop_last = drop_last
        self.calls: list[list[str]] = []

    def encode(
        self, chunks: list[Chunk], trace: TraceContext | None = None
    ) -> list[ChunkRecord]:
        del trace
        self.calls.append([chunk.id for chunk in chunks])
        records = [
            ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata={**dict(chunk.metadata), "sparse_unique_terms": 3},
                sparse_vector={f"token_{chunk.id}": 1.0},
            )
            for chunk in chunks
        ]
        if self.drop_last and records:
            return records[:-1]
        return records


def _make_settings(batch_size: int = 2) -> SimpleNamespace:
    return SimpleNamespace(ingestion=SimpleNamespace(batch_size=batch_size))


def _chunk(idx: int) -> Chunk:
    text = f"chunk text {idx}"
    return Chunk(
        id=f"chunk-{idx}",
        text=text,
        metadata={"source_path": "tests/fixtures/sample_documents/sample.pdf"},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-1",
    )


def test_batch_size_two_splits_five_chunks_into_three_batches_and_keeps_order() -> None:
    dense = _FakeDenseEncoder()
    sparse = _FakeSparseEncoder()
    processor = BatchProcessor(_make_settings(batch_size=2), dense_encoder=dense, sparse_encoder=sparse)
    chunks = [_chunk(i) for i in range(5)]

    records = processor.process(chunks)

    assert [record.id for record in records] == [chunk.id for chunk in chunks]
    assert dense.calls == [["chunk-0", "chunk-1"], ["chunk-2", "chunk-3"], ["chunk-4"]]
    assert sparse.calls == [["chunk-0", "chunk-1"], ["chunk-2", "chunk-3"], ["chunk-4"]]


def test_process_merges_dense_and_sparse_vectors() -> None:
    processor = BatchProcessor(
        _make_settings(batch_size=3),
        dense_encoder=_FakeDenseEncoder(),
        sparse_encoder=_FakeSparseEncoder(),
    )

    records = processor.process([_chunk(0), _chunk(1)])

    assert records[0].dense_vector is not None
    assert records[0].sparse_vector is not None
    assert "sparse_unique_terms" in records[0].metadata


def test_process_empty_returns_empty_without_encoder_calls() -> None:
    dense = _FakeDenseEncoder()
    sparse = _FakeSparseEncoder()
    processor = BatchProcessor(_make_settings(batch_size=2), dense_encoder=dense, sparse_encoder=sparse)

    assert processor.process([]) == []
    assert dense.calls == []
    assert sparse.calls == []


def test_process_validates_input_types() -> None:
    processor = BatchProcessor(
        _make_settings(batch_size=2),
        dense_encoder=_FakeDenseEncoder(),
        sparse_encoder=_FakeSparseEncoder(),
    )

    with pytest.raises(ValueError, match="expects chunks as a list"):
        processor.process("not-list")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must be Chunk"):
        processor.process([_chunk(0), "bad-item"])  # type: ignore[list-item]


def test_process_raises_when_dense_count_mismatch() -> None:
    processor = BatchProcessor(
        _make_settings(batch_size=2),
        dense_encoder=_FakeDenseEncoder(drop_last=True),
        sparse_encoder=_FakeSparseEncoder(),
    )

    with pytest.raises(ValueError, match="dense record count mismatch"):
        processor.process([_chunk(0), _chunk(1)])


def test_process_raises_when_sparse_count_mismatch() -> None:
    processor = BatchProcessor(
        _make_settings(batch_size=2),
        dense_encoder=_FakeDenseEncoder(),
        sparse_encoder=_FakeSparseEncoder(drop_last=True),
    )

    with pytest.raises(ValueError, match="sparse record count mismatch"):
        processor.process([_chunk(0), _chunk(1)])


def test_trace_records_each_batch() -> None:
    processor = BatchProcessor(
        _make_settings(batch_size=2),
        dense_encoder=_FakeDenseEncoder(),
        sparse_encoder=_FakeSparseEncoder(),
    )
    trace = TraceContext(trace_type="ingestion")

    _ = processor.process([_chunk(0), _chunk(1), _chunk(2)], trace=trace)

    assert len(trace.stages) == 2
    batch_stages = [stage for stage in trace.stages if stage["stage"] == "batch_process"]
    assert len(batch_stages) == 2
    assert batch_stages[0]["details"]["batch_index"] == 1
    assert batch_stages[0]["details"]["total_batches"] == 2


def test_requires_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size as a positive integer"):
        BatchProcessor(
            _make_settings(batch_size=0),
            dense_encoder=_FakeDenseEncoder(),
            sparse_encoder=_FakeSparseEncoder(),
        )
