"""Document-to-Chunk adapter on top of libs.splitter."""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Mapping

from core.types import Chunk, Document
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory

_IMAGE_PLACEHOLDER_RE = re.compile(r"\[IMAGE:\s*([^\]]+)\]")


class DocumentChunker:
    """Convert Document objects into Chunk objects with business metadata."""

    def __init__(self, settings: object, splitter: BaseSplitter | None = None) -> None:
        self._settings = settings
        self._splitter = splitter or SplitterFactory.create(settings)

    def split_document(self, document: Document) -> list[Chunk]:
        if not isinstance(document, Document):
            raise ValueError("DocumentChunker.split_document expects a Document instance.")

        chunk_texts = self._splitter.split_text(document.text)
        chunks: list[Chunk] = []
        search_cursor = 0

        for index, chunk_text in enumerate(chunk_texts):
            normalized_text = chunk_text.strip()
            if not normalized_text:
                continue

            start_offset, end_offset, search_cursor = self._locate_offsets(
                document.text, normalized_text, search_cursor
            )
            chunk_id = self._generate_chunk_id(document.id, index, normalized_text)
            metadata = self._inherit_metadata(document, index, normalized_text)
            chunk = Chunk(
                id=chunk_id,
                text=normalized_text,
                metadata=metadata,
                start_offset=start_offset,
                end_offset=end_offset,
                source_ref=document.id,
            )
            chunks.append(chunk)
        return chunks

    @staticmethod
    def _generate_chunk_id(doc_id: str, index: int, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{digest}"

    @staticmethod
    def _locate_offsets(
        full_text: str, chunk_text: str, cursor: int
    ) -> tuple[int, int, int]:
        start = full_text.find(chunk_text, cursor)
        if start < 0:
            start = full_text.find(chunk_text)
        if start < 0:
            start = cursor
        end = start + len(chunk_text)
        new_cursor = max(cursor, end)
        return start, end, new_cursor

    def _inherit_metadata(
        self, document: Document, chunk_index: int, chunk_text: str
    ) -> dict[str, object]:
        metadata = copy.deepcopy(document.metadata)
        metadata["chunk_index"] = chunk_index

        document_images_raw = document.metadata.get("images", [])
        document_images: list[dict[str, object]] = []
        if isinstance(document_images_raw, list):
            for item in document_images_raw:
                if isinstance(item, Mapping):
                    document_images.append(dict(item))

        refs = self._extract_image_refs(chunk_text)
        metadata["image_refs"] = refs

        if refs:
            ref_set = set(refs)
            metadata["images"] = [
                image
                for image in document_images
                if isinstance(image.get("id"), str) and image["id"] in ref_set
            ]
        else:
            metadata.pop("images", None)
        return metadata

    @staticmethod
    def _extract_image_refs(chunk_text: str) -> list[str]:
        refs: list[str] = []
        for match in _IMAGE_PLACEHOLDER_RE.findall(chunk_text):
            image_id = match.strip()
            if image_id and image_id not in refs:
                refs.append(image_id)
        return refs
