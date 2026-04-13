"""Unit tests for core shared data contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.types import IMAGE_PLACEHOLDER_TEMPLATE, Chunk, ChunkRecord, Document


def _base_metadata() -> dict[str, object]:
    return {"source_path": "tests/fixtures/sample_documents/sample.txt", "doc_type": "text"}


def test_document_roundtrip_serialization_stable() -> None:
    metadata = _base_metadata()
    metadata["images"] = [
        {
            "id": "doc_hash_1_0",
            "path": "data/images/demo/doc_hash_1_0.png",
            "page": 1,
            "text_offset": 12,
            "text_length": 21,
            "position": {"x": 10, "y": 20, "width": 128, "height": 64},
        }
    ]
    text = f"before {IMAGE_PLACEHOLDER_TEMPLATE.format(image_id='doc_hash_1_0')} after"
    document = Document(id="doc-1", text=text, metadata=metadata)

    payload_dict = document.to_dict()
    payload_json = document.to_json()
    restored = Document.from_json(payload_json)

    assert list(payload_dict.keys()) == ["id", "text", "metadata"]
    assert payload_dict["metadata"]["source_path"] == metadata["source_path"]
    assert json.loads(payload_json)["id"] == "doc-1"
    assert restored.to_dict() == payload_dict


def test_chunk_and_chunkrecord_roundtrip_serialization() -> None:
    chunk = Chunk(
        id="chunk-1",
        text="chunk text",
        metadata=_base_metadata(),
        start_offset=0,
        end_offset=10,
        source_ref="doc-1",
    )
    chunk_record = ChunkRecord(
        id="chunk-1",
        text="chunk text",
        metadata=_base_metadata(),
        dense_vector=[0.1, 0.2, 0.3],
        sparse_vector={"chunk": 1.5},
    )

    assert Chunk.from_json(chunk.to_json()).to_dict() == chunk.to_dict()
    assert ChunkRecord.from_json(chunk_record.to_json()).to_dict() == chunk_record.to_dict()


@pytest.mark.parametrize(
    ("factory", "match_text"),
    [
        (lambda: Document(id="doc", text="text", metadata={}), "source_path"),
        (
            lambda: Chunk(id="c", text="t", metadata={}, start_offset=0, end_offset=1),
            "source_path",
        ),
        (lambda: ChunkRecord(id="c", text="t", metadata={}), "source_path"),
    ],
)
def test_metadata_requires_source_path(factory: object, match_text: str) -> None:
    with pytest.raises(ValueError, match=match_text):
        factory()  # type: ignore[misc]


def test_metadata_images_schema_validation() -> None:
    metadata = _base_metadata()
    metadata["images"] = [
        {
            "id": "doc_hash_1_0",
            "path": "data/images/demo/doc_hash_1_0.png",
            "text_offset": 0,
            "text_length": 18,
            "position": {"x": 1, "y": 2},
        }
    ]
    document = Document(id="doc-1", text="hello", metadata=metadata)
    images = document.metadata["images"]

    assert isinstance(images, list)
    assert images[0]["id"] == "doc_hash_1_0"
    assert images[0]["path"] == "data/images/demo/doc_hash_1_0.png"
    assert images[0]["text_offset"] == 0
    assert images[0]["text_length"] == 18


def test_metadata_images_schema_rejects_invalid_shape() -> None:
    metadata = _base_metadata()
    metadata["images"] = [{"id": "img-1", "path": "a.png", "text_offset": -1, "text_length": 3}]

    with pytest.raises(ValueError, match="text_offset"):
        Document(id="doc-1", text="hello", metadata=metadata)


def test_chunk_offsets_are_validated() -> None:
    with pytest.raises(ValueError, match="end_offset must be >="):
        Chunk(
            id="chunk-1",
            text="chunk text",
            metadata=_base_metadata(),
            start_offset=5,
            end_offset=2,
        )


def test_metadata_extension_fields_are_preserved() -> None:
    metadata = _base_metadata()
    metadata["tenant_id"] = "acme"
    metadata["custom_flag"] = True

    document = Document(id="doc-1", text="text", metadata=metadata)
    payload = document.to_dict()

    assert payload["metadata"]["tenant_id"] == "acme"
    assert payload["metadata"]["custom_flag"] is True
