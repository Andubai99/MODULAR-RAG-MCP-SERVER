"""Core shared data contracts for ingestion, retrieval, and MCP layers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

IMAGE_PLACEHOLDER_TEMPLATE = "[IMAGE: {image_id}]"


def _is_non_empty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_source_path(metadata: Mapping[str, object], owner: str) -> None:
    source_path = metadata.get("source_path")
    if not _is_non_empty_str(source_path):
        raise ValueError(f"{owner}.metadata.source_path must be a non-empty string.")


def _validate_images(images: object, owner: str) -> list[dict[str, object]]:
    if not isinstance(images, list):
        raise ValueError(f"{owner}.metadata.images must be a list.")

    normalized: list[dict[str, object]] = []
    for index, image in enumerate(images):
        if not isinstance(image, Mapping):
            raise ValueError(f"{owner}.metadata.images[{index}] must be an object.")

        image_id = image.get("id")
        path = image.get("path")
        text_offset = image.get("text_offset")
        text_length = image.get("text_length")

        if not _is_non_empty_str(image_id):
            raise ValueError(
                f"{owner}.metadata.images[{index}].id must be a non-empty string."
            )
        if not _is_non_empty_str(path):
            raise ValueError(
                f"{owner}.metadata.images[{index}].path must be a non-empty string."
            )
        if isinstance(text_offset, bool) or not isinstance(text_offset, int) or text_offset < 0:
            raise ValueError(
                f"{owner}.metadata.images[{index}].text_offset must be a non-negative integer."
            )
        if isinstance(text_length, bool) or not isinstance(text_length, int) or text_length < 0:
            raise ValueError(
                f"{owner}.metadata.images[{index}].text_length must be a non-negative integer."
            )

        page = image.get("page")
        if page is not None and (
            isinstance(page, bool) or not isinstance(page, int) or page < 0
        ):
            raise ValueError(
                f"{owner}.metadata.images[{index}].page must be a non-negative integer when provided."
            )

        position = image.get("position")
        if position is not None and not isinstance(position, Mapping):
            raise ValueError(
                f"{owner}.metadata.images[{index}].position must be an object when provided."
            )

        normalized_image = {
            "id": str(image_id).strip(),
            "path": str(path).strip(),
            "text_offset": int(text_offset),
            "text_length": int(text_length),
        }
        if page is not None:
            normalized_image["page"] = int(page)
        if position is not None:
            normalized_image["position"] = dict(position)
        normalized.append(normalized_image)
    return normalized


def _normalize_metadata(metadata: object, owner: str) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{owner}.metadata must be an object.")

    normalized = dict(metadata)
    _validate_source_path(normalized, owner)

    if "images" in normalized:
        normalized["images"] = _validate_images(normalized["images"], owner)
    return normalized


def _ensure_non_empty_string(value: object, field_path: str) -> str:
    if not _is_non_empty_str(value):
        raise ValueError(f"{field_path} must be a non-empty string.")
    return str(value).strip()


def _normalize_dense_vector(value: object) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("ChunkRecord.dense_vector must be a numeric list when provided.")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        raise ValueError("ChunkRecord.dense_vector must be a numeric list when provided.")
    return [float(v) for v in value]


def _normalize_sparse_vector(value: object) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(
            "ChunkRecord.sparse_vector must be an object of token->weight when provided."
        )
    normalized: dict[str, float] = {}
    for key, weight in value.items():
        if not _is_non_empty_str(key):
            raise ValueError("ChunkRecord.sparse_vector keys must be non-empty strings.")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("ChunkRecord.sparse_vector weights must be numeric.")
        normalized[str(key).strip()] = float(weight)
    return normalized


@dataclass(slots=True)
class Document:
    id: str
    text: str
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        self.id = _ensure_non_empty_string(self.id, "Document.id")
        self.text = _ensure_non_empty_string(self.text, "Document.text")
        self.metadata = _normalize_metadata(self.metadata, "Document")

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "text": self.text, "metadata": dict(self.metadata)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Document:
        return cls(id=data.get("id", ""), text=data.get("text", ""), metadata=data.get("metadata", {}))

    @classmethod
    def from_json(cls, payload: str) -> Document:
        data = json.loads(payload)
        if not isinstance(data, Mapping):
            raise ValueError("Document JSON payload must be an object.")
        return cls.from_dict(data)


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, object]
    start_offset: int
    end_offset: int
    source_ref: str | None = None

    def __post_init__(self) -> None:
        self.id = _ensure_non_empty_string(self.id, "Chunk.id")
        self.text = _ensure_non_empty_string(self.text, "Chunk.text")
        self.metadata = _normalize_metadata(self.metadata, "Chunk")

        if isinstance(self.start_offset, bool) or not isinstance(self.start_offset, int):
            raise ValueError("Chunk.start_offset must be an integer.")
        if isinstance(self.end_offset, bool) or not isinstance(self.end_offset, int):
            raise ValueError("Chunk.end_offset must be an integer.")
        if self.start_offset < 0:
            raise ValueError("Chunk.start_offset must be >= 0.")
        if self.end_offset < self.start_offset:
            raise ValueError("Chunk.end_offset must be >= start_offset.")

        if self.source_ref is not None:
            self.source_ref = _ensure_non_empty_string(self.source_ref, "Chunk.source_ref")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "text": self.text,
            "metadata": dict(self.metadata),
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
        }
        if self.source_ref is not None:
            payload["source_ref"] = self.source_ref
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Chunk:
        return cls(
            id=data.get("id", ""),
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
            start_offset=int(data.get("start_offset", -1)),
            end_offset=int(data.get("end_offset", -1)),
            source_ref=data.get("source_ref") if isinstance(data.get("source_ref"), str) else None,
        )

    @classmethod
    def from_json(cls, payload: str) -> Chunk:
        data = json.loads(payload)
        if not isinstance(data, Mapping):
            raise ValueError("Chunk JSON payload must be an object.")
        return cls.from_dict(data)


@dataclass(slots=True)
class ChunkRecord:
    id: str
    text: str
    metadata: dict[str, object]
    dense_vector: list[float] | None = None
    sparse_vector: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self.id = _ensure_non_empty_string(self.id, "ChunkRecord.id")
        self.text = _ensure_non_empty_string(self.text, "ChunkRecord.text")
        self.metadata = _normalize_metadata(self.metadata, "ChunkRecord")
        self.dense_vector = _normalize_dense_vector(self.dense_vector)
        self.sparse_vector = _normalize_sparse_vector(self.sparse_vector)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "text": self.text,
            "metadata": dict(self.metadata),
        }
        if self.dense_vector is not None:
            payload["dense_vector"] = list(self.dense_vector)
        if self.sparse_vector is not None:
            payload["sparse_vector"] = dict(self.sparse_vector)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ChunkRecord:
        return cls(
            id=data.get("id", ""),
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
            dense_vector=data.get("dense_vector") if isinstance(data.get("dense_vector"), list) else None,
            sparse_vector=data.get("sparse_vector") if isinstance(data.get("sparse_vector"), Mapping) else None,
        )

    @classmethod
    def from_json(cls, payload: str) -> ChunkRecord:
        data = json.loads(payload)
        if not isinstance(data, Mapping):
            raise ValueError("ChunkRecord JSON payload must be an object.")
        return cls.from_dict(data)
