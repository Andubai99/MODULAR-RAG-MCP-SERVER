"""Ingestion pipeline orchestration for MVP flow."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord, Document
from ingestion.chunking.document_chunker import DocumentChunker
from ingestion.embedding.batch_processor import BatchProcessor
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from ingestion.storage.vector_upserter import VectorUpserter
from ingestion.transform.chunk_refiner import ChunkRefiner
from ingestion.transform.image_captioner import ImageCaptioner
from ingestion.transform.metadata_enricher import MetadataEnricher
from libs.loader.base_loader import BaseLoader
from libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker
from libs.loader.pdf_loader import PdfLoader


class PipelineStageError(RuntimeError):
    """Raised when a pipeline stage fails."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(f"IngestionPipeline failed at stage '{stage}': {message}")


@dataclass(slots=True)
class PipelineResult:
    """Result snapshot returned by one pipeline run."""

    status: str
    source_path: str
    collection: str
    file_hash: str
    chunk_count: int
    vector_count: int
    image_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "source_path": self.source_path,
            "collection": self.collection,
            "file_hash": self.file_hash,
            "chunk_count": self.chunk_count,
            "vector_count": self.vector_count,
            "image_count": self.image_count,
        }


class _SupportsIntegrity(Protocol):
    def compute_sha256(self, path: str) -> str: ...
    def should_skip(self, file_hash: str) -> bool: ...
    def mark_success(self, file_hash: str, file_path: str, **kwargs: object) -> None: ...
    def mark_failed(self, file_hash: str, error_msg: str) -> None: ...


class IngestionPipeline:
    """Run ingestion stages in sequence and persist outputs."""

    def __init__(
        self,
        settings: object,
        integrity_checker: _SupportsIntegrity | None = None,
        loader: BaseLoader | None = None,
        chunker: DocumentChunker | None = None,
        transforms: list[object] | None = None,
        batch_processor: BatchProcessor | None = None,
        vector_upserter: VectorUpserter | None = None,
        bm25_indexer: BM25Indexer | None = None,
        image_storage: ImageStorage | None = None,
    ) -> None:
        self._settings = settings
        self._integrity = integrity_checker or self._default_integrity_checker()
        self._loader = loader or PdfLoader()
        self._chunker = chunker or DocumentChunker(settings)
        self._transforms = transforms or [
            ChunkRefiner(settings),
            MetadataEnricher(settings),
            ImageCaptioner(settings),
        ]
        self._batch_processor = batch_processor or BatchProcessor(settings)
        self._vector_upserter = vector_upserter or VectorUpserter(settings)
        self._bm25_indexer = bm25_indexer or BM25Indexer(settings)
        self._image_storage = image_storage or ImageStorage()
        self._default_collection = self._resolve_default_collection(settings)

    def run(
        self,
        source_path: str,
        collection: str | None = None,
        force: bool = False,
        trace: TraceContext | None = None,
    ) -> PipelineResult:
        normalized_source = self._normalize_source_path(source_path)
        normalized_collection = self._normalize_collection(collection)

        file_hash = ""
        chunks: list[Chunk] = []
        records: list[ChunkRecord] = []
        image_count = 0

        try:
            file_hash = self._run_integrity_stage(
                normalized_source, force=force, trace=trace
            )
            if file_hash.startswith("skip:"):
                digest = file_hash.replace("skip:", "", 1)
                return PipelineResult(
                    status="skipped",
                    source_path=normalized_source,
                    collection=normalized_collection,
                    file_hash=digest,
                    chunk_count=0,
                    vector_count=0,
                    image_count=0,
                )

            document = self._run_load_stage(normalized_source, trace=trace)
            chunks = self._run_split_stage(document, trace=trace)
            chunks = self._run_transform_stage(chunks, trace=trace)
            records = self._run_encode_stage(chunks, trace=trace)
            image_count = self._run_store_stage(
                records=records,
                document=document,
                collection=normalized_collection,
                trace=trace,
            )

            self._integrity.mark_success(
                file_hash=file_hash,
                file_path=normalized_source,
                chunk_count=len(chunks),
            )
            return PipelineResult(
                status="processed",
                source_path=normalized_source,
                collection=normalized_collection,
                file_hash=file_hash,
                chunk_count=len(chunks),
                vector_count=len(records),
                image_count=image_count,
            )
        except PipelineStageError as exc:
            if file_hash and not file_hash.startswith("skip:"):
                self._integrity.mark_failed(file_hash=file_hash, error_msg=str(exc))
            raise
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            wrapped = PipelineStageError("unknown", message)
            if file_hash and not file_hash.startswith("skip:"):
                self._integrity.mark_failed(file_hash=file_hash, error_msg=str(wrapped))
            raise wrapped from exc

    def _run_integrity_stage(
        self, source_path: str, force: bool, trace: TraceContext | None = None
    ) -> str:
        stage_start = trace.stage_timer() if trace is not None else None
        try:
            file_hash = self._integrity.compute_sha256(source_path)
            skipped = False
            if not force and self._integrity.should_skip(file_hash):
                skipped = True
                file_hash = f"skip:{file_hash}"
        except Exception as exc:
            raise PipelineStageError("integrity", f"{type(exc).__name__}: {exc}") from exc

        if trace is not None:
            elapsed_ms = (
                trace.stage_elapsed_ms(stage_start) if stage_start is not None else None
            )
            trace.record_stage(
                "integrity",
                method="sha256",
                details={"source_path": source_path, "skipped": skipped},
                elapsed_ms=elapsed_ms,
            )
        return file_hash

    def _run_load_stage(
        self, source_path: str, trace: TraceContext | None = None
    ) -> Document:
        stage_start = trace.stage_timer() if trace is not None else None
        try:
            document = self._loader.load(source_path)
        except Exception as exc:
            raise PipelineStageError("load", f"{type(exc).__name__}: {exc}") from exc
        if not isinstance(document, Document):
            raise PipelineStageError("load", "loader must return a Document instance.")
        if trace is not None:
            elapsed_ms = (
                trace.stage_elapsed_ms(stage_start) if stage_start is not None else None
            )
            trace.record_stage(
                "load",
                method=type(self._loader).__name__,
                details={"document_id": document.id},
                elapsed_ms=elapsed_ms,
            )
        return document

    def _run_split_stage(
        self, document: Document, trace: TraceContext | None = None
    ) -> list[Chunk]:
        stage_start = trace.stage_timer() if trace is not None else None
        try:
            chunks = self._chunker.split_document(document)
        except Exception as exc:
            raise PipelineStageError("split", f"{type(exc).__name__}: {exc}") from exc
        if trace is not None:
            elapsed_ms = (
                trace.stage_elapsed_ms(stage_start) if stage_start is not None else None
            )
            trace.record_stage(
                "split",
                method=type(self._chunker).__name__,
                details={"chunk_count": len(chunks)},
                elapsed_ms=elapsed_ms,
            )
        return chunks

    def _run_transform_stage(
        self, chunks: list[Chunk], trace: TraceContext | None = None
    ) -> list[Chunk]:
        stage_start = trace.stage_timer() if trace is not None else None
        transformed = chunks
        try:
            for transform in self._transforms:
                transform_name = type(transform).__name__
                if not hasattr(transform, "transform"):
                    raise ValueError(f"Transform '{transform_name}' has no transform() method.")
                transformed = transform.transform(transformed, trace=trace)
        except Exception as exc:
            raise PipelineStageError("transform", f"{type(exc).__name__}: {exc}") from exc
        if trace is not None:
            elapsed_ms = (
                trace.stage_elapsed_ms(stage_start) if stage_start is not None else None
            )
            trace.record_stage(
                "transform",
                method="sequential",
                details={"chunk_count": len(transformed), "steps": len(self._transforms)},
                elapsed_ms=elapsed_ms,
            )
        return transformed

    def _run_encode_stage(
        self, chunks: list[Chunk], trace: TraceContext | None = None
    ) -> list[ChunkRecord]:
        stage_start = trace.stage_timer() if trace is not None else None
        try:
            records = self._batch_processor.process(chunks, trace=trace)
        except Exception as exc:
            raise PipelineStageError("encode", f"{type(exc).__name__}: {exc}") from exc
        if trace is not None:
            elapsed_ms = (
                trace.stage_elapsed_ms(stage_start) if stage_start is not None else None
            )
            trace.record_stage(
                "encode",
                method=type(self._batch_processor).__name__,
                details={"record_count": len(records)},
                elapsed_ms=elapsed_ms,
            )
        return records

    def _run_store_stage(
        self,
        records: list[ChunkRecord],
        document: Document,
        collection: str,
        trace: TraceContext | None = None,
    ) -> int:
        stage_start = trace.stage_timer() if trace is not None else None
        try:
            self._vector_upserter.upsert(records, trace=trace)
            self._bm25_indexer.upsert(records, collection=collection)
            image_count = self._persist_images(document=document, collection=collection)
        except Exception as exc:
            raise PipelineStageError("store", f"{type(exc).__name__}: {exc}") from exc
        if trace is not None:
            elapsed_ms = (
                trace.stage_elapsed_ms(stage_start) if stage_start is not None else None
            )
            trace.record_stage(
                "store",
                method="vector+bm25+image",
                details={
                    "record_count": len(records),
                    "image_count": image_count,
                    "collection": collection,
                },
                elapsed_ms=elapsed_ms,
            )
        return image_count

    def _persist_images(self, document: Document, collection: str) -> int:
        raw_images = document.metadata.get("images")
        if not isinstance(raw_images, list):
            return 0

        persisted = 0
        doc_hash = self._extract_doc_hash(document)
        for image in raw_images:
            if not isinstance(image, dict):
                continue
            image_id = image.get("id")
            image_path = image.get("path")
            page_num = image.get("page")
            if not isinstance(image_id, str) or not image_id.strip():
                continue
            if not isinstance(image_path, str) or not image_path.strip():
                continue
            source_file = Path(image_path.strip())
            if not source_file.exists() or not source_file.is_file():
                continue
            payload = source_file.read_bytes()
            extension = source_file.suffix or ".png"
            normalized_page = (
                page_num if isinstance(page_num, int) and not isinstance(page_num, bool) else None
            )
            self._image_storage.save_image(
                image_id=image_id.strip(),
                image_bytes=payload,
                collection=collection,
                doc_hash=doc_hash,
                page_num=normalized_page,
                extension=extension,
            )
            persisted += 1
        return persisted

    @staticmethod
    def _extract_doc_hash(document: Document) -> str:
        if document.id.startswith("doc_") and len(document.id) > 4:
            return document.id[4:]
        digest = hashlib.sha256(document.id.encode("utf-8")).hexdigest()
        return digest

    def _default_integrity_checker(self) -> FileIntegrityChecker:
        return SQLiteIntegrityChecker()

    @staticmethod
    def _normalize_source_path(source_path: str) -> str:
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError("source_path must be a non-empty string.")
        normalized = Path(source_path.strip())
        if not normalized.exists() or not normalized.is_file():
            raise ValueError(f"source_path not found: {normalized.as_posix()}")
        return normalized.as_posix()

    def _normalize_collection(self, collection: str | None) -> str:
        if collection is None:
            return self._default_collection
        if not isinstance(collection, str) or not collection.strip():
            raise ValueError("collection must be a non-empty string when provided.")
        return collection.strip()

    @staticmethod
    def _resolve_default_collection(settings: object) -> str:
        vector_store_settings = getattr(settings, "vector_store", None)
        candidate = getattr(vector_store_settings, "collection_name", "")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        return "default"
