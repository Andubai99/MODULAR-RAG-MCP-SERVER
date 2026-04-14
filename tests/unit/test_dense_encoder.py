"""Unit tests for DenseEncoder."""

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
from ingestion.embedding.dense_encoder import DenseEncoder
from libs.embedding.base_embedding import BaseEmbedding


class _FakeEmbedding(BaseEmbedding):
    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self._vectors = vectors
        self.calls: list[list[str]] = []

    def embed(
        self, texts: list[str], trace: object | None = None
    ) -> list[list[float]]:
        del trace
        self.calls.append(list(texts))
        if self._vectors is not None:
            return self._vectors
        return [[float(len(text)), 1.0, 2.0] for text in texts]


def _make_settings() -> SimpleNamespace:
    return SimpleNamespace(
        embedding=SimpleNamespace(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
            azure_endpoint="",
            deployment_name="",
            api_version="",
            api_key="dummy",
            base_url="https://api.openai.com/v1",
        )
    )


def _chunk(text: str, idx: int) -> Chunk:
    return Chunk(
        id=f"chunk-{idx}",
        text=text,
        metadata={
            "source_path": "tests/fixtures/sample_documents/sample.pdf",
            "chunk_index": idx,
        },
        start_offset=0,
        end_offset=max(1, len(text)),
        source_ref="doc-1",
    )


def test_encode_returns_chunk_records_with_dense_vectors() -> None:
    embedding = _FakeEmbedding()
    encoder = DenseEncoder(_make_settings(), embedding=embedding)
    chunks = [_chunk("alpha", 0), _chunk("beta", 1)]

    records = encoder.encode(chunks)

    assert len(records) == 2
    assert all(isinstance(record, ChunkRecord) for record in records)
    assert records[0].id == "chunk-0"
    assert records[1].id == "chunk-1"
    assert records[0].dense_vector is not None
    assert len(records[0].dense_vector) == len(records[1].dense_vector)


def test_encode_batches_texts_in_single_embedding_call() -> None:
    embedding = _FakeEmbedding()
    encoder = DenseEncoder(_make_settings(), embedding=embedding)
    chunks = [_chunk("first text", 0), _chunk("second text", 1), _chunk("third", 2)]

    _ = encoder.encode(chunks)

    assert embedding.calls == [["first text", "second text", "third"]]


def test_encode_empty_input_returns_empty_list() -> None:
    embedding = _FakeEmbedding()
    encoder = DenseEncoder(_make_settings(), embedding=embedding)

    assert encoder.encode([]) == []
    assert embedding.calls == []


def test_encode_raises_on_vector_count_mismatch() -> None:
    embedding = _FakeEmbedding(vectors=[[0.1, 0.2]])
    encoder = DenseEncoder(_make_settings(), embedding=embedding)

    with pytest.raises(ValueError, match="vector count mismatch"):
        encoder.encode([_chunk("a", 0), _chunk("b", 1)])


def test_encode_raises_on_dimension_mismatch() -> None:
    embedding = _FakeEmbedding(vectors=[[0.1, 0.2], [0.3, 0.4, 0.5]])
    encoder = DenseEncoder(_make_settings(), embedding=embedding)

    with pytest.raises(ValueError, match="dimension mismatch"):
        encoder.encode([_chunk("a", 0), _chunk("b", 1)])


def test_encode_validates_input_types() -> None:
    encoder = DenseEncoder(_make_settings(), embedding=_FakeEmbedding())

    with pytest.raises(ValueError, match="expects chunks as a list"):
        encoder.encode("not-list")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must be Chunk"):
        encoder.encode([_chunk("ok", 0), "bad-item"])  # type: ignore[list-item]


def test_encode_records_trace_stage() -> None:
    encoder = DenseEncoder(_make_settings(), embedding=_FakeEmbedding())
    trace = TraceContext(trace_type="ingestion")

    _ = encoder.encode([_chunk("trace text", 0)], trace=trace)

    assert len(trace.stages) == 1
    stage = trace.stages[0]
    assert stage["stage"] == "dense_encode"
    assert stage["method"] == "openai"
    assert stage["details"]["chunk_count"] == 1
    assert stage["details"]["dimension"] == 3
