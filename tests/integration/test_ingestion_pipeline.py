"""Integration tests for ingestion pipeline orchestration."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.trace.trace_context import TraceContext
from ingestion.pipeline import IngestionPipeline, PipelineStageError
from ingestion.storage.image_storage import ImageStorage
from libs.embedding.base_embedding import BaseEmbedding
from libs.loader.base_loader import BaseLoader
from libs.loader.pdf_loader import PdfLoader
from ingestion.embedding.batch_processor import BatchProcessor
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder


class _MemoryIntegrity:
    def __init__(self) -> None:
        self._status: dict[str, str] = {}

    def compute_sha256(self, path: str) -> str:
        payload = Path(path).read_bytes()
        return hashlib.sha256(payload).hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        return self._status.get(file_hash) == "success"

    def mark_success(self, file_hash: str, file_path: str, **kwargs: object) -> None:
        del file_path, kwargs
        self._status[file_hash] = "success"

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        del error_msg
        self._status[file_hash] = "failed"


class _FakeEmbedding(BaseEmbedding):
    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        del trace
        return [[float(index + 1), float(len(text)), 1.0] for index, text in enumerate(texts)]


class _BrokenLoader(BaseLoader):
    def load(self, path: str):  # type: ignore[override]
        raise RuntimeError(f"cannot parse: {path}")


def _workspace(prefix: str) -> Path:
    path = ROOT_DIR / "cache" / "ingestion_pipeline_tests" / f"{prefix}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_settings(workspace: Path) -> SimpleNamespace:
    return SimpleNamespace(
        ingestion=SimpleNamespace(
            splitter="recursive",
            chunk_size=120,
            chunk_overlap=20,
            batch_size=2,
            chunk_refiner=SimpleNamespace(use_llm=False),
            metadata_enricher=SimpleNamespace(use_llm=False),
        ),
        vision_llm=SimpleNamespace(enabled=False),
        embedding=SimpleNamespace(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=3,
            api_key="dummy",
            base_url="https://api.openai.com/v1",
            azure_endpoint="",
            deployment_name="",
            api_version="",
        ),
        vector_store=SimpleNamespace(
            provider="chroma",
            persist_directory=(workspace / "data" / "db" / "chroma").as_posix(),
            collection_name="demo",
        ),
        llm=SimpleNamespace(
            provider="openai_compatible",
            model="dummy",
            deployment_name="",
            azure_endpoint="",
            api_version="",
            api_key="dummy",
            base_url="https://api.openai.com/v1",
            temperature=0.0,
            max_tokens=128,
        ),
    )


def _create_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4 fixture")


def test_pipeline_runs_end_to_end_and_persists_outputs() -> None:
    workspace = _workspace("success")
    settings = _make_settings(workspace)
    complex_pdf = workspace / "fixtures" / "complex_technical_doc.pdf"
    _create_pdf(complex_pdf)

    raw_images = workspace / "raw_images"
    raw_images.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for index in range(3):
        path = raw_images / f"raw_{index}.png"
        path.write_bytes(f"raw-image-{index}".encode("utf-8"))
        image_paths.append(path)

    image_ids = [f"dochash_1_{index}" for index in range(3)]

    def _parser(_: Path) -> dict[str, object]:
        text = (
            "# Chapter 1\nRAG architecture overview.\n\n"
            f"[IMAGE: {image_ids[0]}]\n\n"
            "## Chapter 2\nDense retrieval details.\n\n"
            f"[IMAGE: {image_ids[1]}]\n\n"
            "## Chapter 3\nSparse retrieval and fusion.\n\n"
            f"[IMAGE: {image_ids[2]}]\n"
        )
        images = []
        for index, image_id in enumerate(image_ids):
            placeholder = f"[IMAGE: {image_id}]"
            images.append(
                {
                    "id": image_id,
                    "path": image_paths[index].as_posix(),
                    "text_offset": text.find(placeholder),
                    "text_length": len(placeholder),
                    "page": index + 1,
                }
            )
        return {"text": text, "images": images}

    loader = PdfLoader(parser=_parser)
    dense_encoder = DenseEncoder(settings, embedding=_FakeEmbedding())
    batch_processor = BatchProcessor(
        settings,
        dense_encoder=dense_encoder,
        sparse_encoder=SparseEncoder(),
    )
    image_storage = ImageStorage(
        db_path=(workspace / "data" / "db" / "image_index.db").as_posix(),
        image_root=(workspace / "data" / "images").as_posix(),
    )
    pipeline = IngestionPipeline(
        settings=settings,
        integrity_checker=_MemoryIntegrity(),
        loader=loader,
        batch_processor=batch_processor,
        image_storage=image_storage,
    )

    trace = TraceContext(trace_type="ingestion")
    result = pipeline.run(complex_pdf.as_posix(), collection="demo", trace=trace)

    assert result.status == "processed"
    assert result.chunk_count > 0
    assert result.vector_count == result.chunk_count
    assert result.image_count == 3

    vector_file = workspace / "data" / "db" / "chroma" / "demo.json"
    bm25_file = workspace / "data" / "db" / "bm25" / "demo.pkl"
    persisted_images = sorted((workspace / "data" / "images" / "demo").glob("*.png"))
    assert vector_file.exists()
    assert bm25_file.exists()
    assert len(persisted_images) == 3
    assert len(image_storage.list_images(collection="demo")) == 3

    stage_names = [stage["stage"] for stage in trace.stages]
    for expected in ["integrity", "load", "split", "transform", "encode", "store"]:
        assert expected in stage_names


def test_pipeline_surfaces_clear_stage_error_when_loader_fails() -> None:
    workspace = _workspace("loader_fail")
    settings = _make_settings(workspace)
    simple_pdf = workspace / "fixtures" / "simple.pdf"
    _create_pdf(simple_pdf)
    integrity = _MemoryIntegrity()

    pipeline = IngestionPipeline(
        settings=settings,
        integrity_checker=integrity,
        loader=_BrokenLoader(),
    )

    with pytest.raises(PipelineStageError, match="stage 'load'"):
        pipeline.run(simple_pdf.as_posix(), collection="demo")
