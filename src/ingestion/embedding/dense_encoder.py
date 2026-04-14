"""Dense embedding encoder for chunk batches."""

from __future__ import annotations

from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class DenseEncoder:
    """Batch-encode chunk texts into dense vectors via embedding providers."""

    def __init__(self, settings: object, embedding: BaseEmbedding | None = None) -> None:
        self._settings = settings
        self._embedding = embedding or EmbeddingFactory.create(settings)
        embedding_settings = getattr(settings, "embedding", None)
        provider = getattr(embedding_settings, "provider", "embedding")
        self._provider = provider if isinstance(provider, str) and provider.strip() else "embedding"

    def encode(
        self, chunks: list[Chunk], trace: TraceContext | None = None
    ) -> list[ChunkRecord]:
        if not isinstance(chunks, list):
            raise ValueError("DenseEncoder.encode expects chunks as a list.")
        if not chunks:
            return []

        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise ValueError(f"DenseEncoder.encode chunks[{index}] must be Chunk.")

        texts = [chunk.text for chunk in chunks]
        stage_start = trace.stage_timer() if trace is not None else None
        vectors = self._embedding.embed(texts, trace=trace)

        if not isinstance(vectors, list):
            raise ValueError("DenseEncoder.encode embedding output must be a list.")
        if len(vectors) != len(chunks):
            raise ValueError(
                "DenseEncoder.encode vector count mismatch: "
                f"expected {len(chunks)}, got {len(vectors)}."
            )

        expected_dim: int | None = None
        records: list[ChunkRecord] = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            if not isinstance(vector, list) or not vector:
                raise ValueError(
                    f"DenseEncoder.encode vectors[{index}] must be a non-empty list."
                )
            if expected_dim is None:
                expected_dim = len(vector)
            elif len(vector) != expected_dim:
                raise ValueError(
                    "DenseEncoder.encode vector dimension mismatch: "
                    f"expected {expected_dim}, got {len(vector)} at index {index}."
                )

            records.append(
                ChunkRecord(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=dict(chunk.metadata),
                    dense_vector=vector,
                )
            )

        if trace is not None:
            elapsed_ms = (
                trace.stage_elapsed_ms(stage_start) if stage_start is not None else None
            )
            trace.record_stage(
                "dense_encode",
                method=self._provider.strip().lower(),
                details={
                    "chunk_count": len(chunks),
                    "dimension": expected_dim or 0,
                },
                elapsed_ms=elapsed_ms,
            )

        return records
