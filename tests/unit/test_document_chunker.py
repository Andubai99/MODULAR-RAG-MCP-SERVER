"""Unit tests for DocumentChunker adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.types import Document
from ingestion.chunking.document_chunker import DocumentChunker
from libs.splitter.base_splitter import BaseSplitter


class _FakeSplitter(BaseSplitter):
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        del text, trace
        return list(self._chunks)


def _make_settings(chunk_size: int, chunk_overlap: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        ingestion=SimpleNamespace(
            splitter="recursive",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    )


def _make_document(text: str, images: list[dict[str, object]] | None = None) -> Document:
    metadata: dict[str, object] = {
        "source_path": "tests/fixtures/sample_documents/sample.pdf",
        "doc_type": "pdf",
        "title": "Sample",
    }
    if images is not None:
        metadata["images"] = images
    return Document(id="doc_abc123", text=text, metadata=metadata)


def test_config_driven_splitter_changes_chunk_granularity() -> None:
    text = ("A" * 220) + "\n\n" + ("B" * 220) + "\n\n" + ("C" * 220)
    document = _make_document(text)

    chunker_small = DocumentChunker(_make_settings(chunk_size=80, chunk_overlap=10))
    chunker_large = DocumentChunker(_make_settings(chunk_size=220, chunk_overlap=20))

    small_chunks = chunker_small.split_document(document)
    large_chunks = chunker_large.split_document(document)

    assert len(small_chunks) > len(large_chunks)
    assert all(len(chunk.text) <= 240 for chunk in large_chunks)


def test_chunk_ids_are_unique_and_deterministic() -> None:
    text = "alpha beta gamma\n\ndelta epsilon zeta\n\neta theta iota"
    document = _make_document(text)
    chunker = DocumentChunker(_make_settings(chunk_size=30, chunk_overlap=5))

    first = chunker.split_document(document)
    second = chunker.split_document(document)
    first_ids = [chunk.id for chunk in first]
    second_ids = [chunk.id for chunk in second]

    assert len(set(first_ids)) == len(first_ids)
    assert first_ids == second_ids


def test_chunk_metadata_inherits_document_metadata_and_index() -> None:
    document = _make_document("section one\n\nsection two")
    chunker = DocumentChunker(
        _make_settings(chunk_size=100),
        splitter=_FakeSplitter(["section one", "section two"]),
    )
    chunks = chunker.split_document(document)

    assert len(chunks) == 2
    assert chunks[0].metadata["source_path"] == document.metadata["source_path"]
    assert chunks[0].metadata["doc_type"] == "pdf"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[1].metadata["chunk_index"] == 1


def test_image_references_distributed_per_chunk() -> None:
    images = [
        {
            "id": "img-1",
            "path": "data/images/demo/img-1.png",
            "text_offset": 10,
            "text_length": 14,
        },
        {
            "id": "img-2",
            "path": "data/images/demo/img-2.png",
            "text_offset": 40,
            "text_length": 14,
        },
    ]
    document = _make_document("irrelevant", images=images)
    chunker = DocumentChunker(
        _make_settings(chunk_size=200),
        splitter=_FakeSplitter(
            [
                "chunk one [IMAGE: img-1]",
                "chunk two without image",
                "chunk three [IMAGE: img-2] and [IMAGE: img-1]",
            ]
        ),
    )
    chunks = chunker.split_document(document)

    first = chunks[0]
    second = chunks[1]
    third = chunks[2]

    assert first.metadata["image_refs"] == ["img-1"]
    assert [img["id"] for img in first.metadata["images"]] == ["img-1"]
    assert second.metadata["image_refs"] == []
    assert "images" not in second.metadata
    assert third.metadata["image_refs"] == ["img-2", "img-1"]
    assert sorted(img["id"] for img in third.metadata["images"]) == ["img-1", "img-2"]


def test_source_ref_and_chunk_contract_fields() -> None:
    document = _make_document("alpha\n\nbeta")
    chunker = DocumentChunker(
        _make_settings(chunk_size=100), splitter=_FakeSplitter(["alpha", "beta"])
    )
    chunks = chunker.split_document(document)

    assert all(chunk.source_ref == document.id for chunk in chunks)
    payload = chunks[0].to_dict()
    assert set(payload.keys()) == {
        "id",
        "text",
        "metadata",
        "start_offset",
        "end_offset",
        "source_ref",
    }


def test_invalid_document_input_raises() -> None:
    chunker = DocumentChunker(_make_settings(chunk_size=100), splitter=_FakeSplitter(["x"]))
    with pytest.raises(ValueError, match="expects a Document instance"):
        chunker.split_document("not-a-document")  # type: ignore[arg-type]
