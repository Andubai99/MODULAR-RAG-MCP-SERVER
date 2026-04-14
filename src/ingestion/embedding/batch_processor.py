"""Batch orchestration for dense and sparse encoding."""

from __future__ import annotations

from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder


class BatchProcessor:
    """Split chunks into batches and run dense/sparse encoders in order."""

    def __init__(
        self,
        settings: object,
        dense_encoder: DenseEncoder | None = None,
        sparse_encoder: SparseEncoder | None = None,
    ) -> None:
        ingestion_settings = getattr(settings, "ingestion", None)
        batch_size = getattr(ingestion_settings, "batch_size", 100)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("BatchProcessor requires ingestion.batch_size as a positive integer.")
        self._batch_size = batch_size

        self._dense_encoder = dense_encoder or DenseEncoder(settings)
        self._sparse_encoder = sparse_encoder or SparseEncoder()

    def process(
        self, chunks: list[Chunk], trace: TraceContext | None = None
    ) -> list[ChunkRecord]:
        if not isinstance(chunks, list):
            raise ValueError("BatchProcessor.process expects chunks as a list.")
        if not chunks:
            return []

        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise ValueError(f"BatchProcessor.process chunks[{index}] must be Chunk.")

        output: list[ChunkRecord] = []
        batches = list(self._build_batches(chunks))
        total_batches = len(batches)

        for batch_index, batch in enumerate(batches, start=1):
            stage_start = trace.stage_timer() if trace is not None else None
            dense_records = self._dense_encoder.encode(batch, trace=trace)
            sparse_records = self._sparse_encoder.encode(batch, trace=trace)
            merged = self._merge_records(batch, dense_records, sparse_records)
            output.extend(merged)

            if trace is not None:
                elapsed_ms = (
                    trace.stage_elapsed_ms(stage_start)
                    if stage_start is not None
                    else None
                )
                trace.record_stage(
                    "batch_process",
                    method="dense+sparse",
                    details={
                        "batch_index": batch_index,
                        "total_batches": total_batches,
                        "batch_size": len(batch),
                    },
                    elapsed_ms=elapsed_ms,
                )
        return output

    def _build_batches(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        return [
            chunks[index : index + self._batch_size]
            for index in range(0, len(chunks), self._batch_size)
        ]

    @staticmethod
    def _merge_records(
        batch: list[Chunk],
        dense_records: list[ChunkRecord],
        sparse_records: list[ChunkRecord],
    ) -> list[ChunkRecord]:
        if len(dense_records) != len(batch):
            raise ValueError(
                "BatchProcessor.process dense record count mismatch: "
                f"expected {len(batch)}, got {len(dense_records)}."
            )
        if len(sparse_records) != len(batch):
            raise ValueError(
                "BatchProcessor.process sparse record count mismatch: "
                f"expected {len(batch)}, got {len(sparse_records)}."
            )

        merged: list[ChunkRecord] = []
        for index, (chunk, dense_record, sparse_record) in enumerate(
            zip(batch, dense_records, sparse_records, strict=True)
        ):
            if dense_record.id != chunk.id or sparse_record.id != chunk.id:
                raise ValueError(
                    "BatchProcessor.process record id mismatch at index "
                    f"{index}: expected {chunk.id}, got dense={dense_record.id}, "
                    f"sparse={sparse_record.id}."
                )

            metadata = dict(dense_record.metadata)
            metadata.update(sparse_record.metadata)
            merged.append(
                ChunkRecord(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=metadata,
                    dense_vector=dense_record.dense_vector,
                    sparse_vector=sparse_record.sparse_vector,
                )
            )
        return merged
