"""Dense-vector to vector-store upsert adapter with deterministic chunk IDs."""

from __future__ import annotations

import hashlib

from core.trace.trace_context import TraceContext
from core.types import ChunkRecord
from libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord
from libs.vector_store.vector_store_factory import VectorStoreFactory


class VectorUpserter:
    """Persist dense vectors into the configured vector store idempotently."""

    def __init__(
        self, settings: object, vector_store: BaseVectorStore | None = None
    ) -> None:
        self._vector_store = vector_store or VectorStoreFactory.create(settings)
        vector_store_settings = getattr(settings, "vector_store", None)
        provider = getattr(vector_store_settings, "provider", "vector_store")
        if isinstance(provider, str) and provider.strip():
            self._provider = provider.strip().lower()
        else:
            self._provider = "vector_store"

    def upsert(
        self, records: list[ChunkRecord], trace: TraceContext | None = None
    ) -> list[str]:
        if not isinstance(records, list):
            raise ValueError("VectorUpserter.upsert expects records as a list.")
        if not records:
            return []

        stage_start = trace.stage_timer() if trace is not None else None

        vector_records: list[VectorRecord] = []
        stable_ids: list[str] = []
        for index, record in enumerate(records):
            if not isinstance(record, ChunkRecord):
                raise ValueError(
                    f"VectorUpserter.upsert records[{index}] must be ChunkRecord."
                )
            if record.dense_vector is None:
                raise ValueError(
                    f"VectorUpserter.upsert records[{index}].dense_vector is required."
                )

            chunk_id = self._generate_chunk_id(record)
            stable_ids.append(chunk_id)
            vector_records.append(
                {
                    "id": chunk_id,
                    "vector": list(record.dense_vector),
                    "metadata": dict(record.metadata),
                    "content": record.text,
                }
            )

        self._vector_store.upsert(vector_records, trace=trace)

        if trace is not None:
            elapsed_ms = (
                trace.stage_elapsed_ms(stage_start) if stage_start is not None else None
            )
            trace.record_stage(
                "vector_upsert",
                method=self._provider,
                details={"record_count": len(vector_records)},
                elapsed_ms=elapsed_ms,
            )
        return stable_ids

    @staticmethod
    def _generate_chunk_id(record: ChunkRecord) -> str:
        source_path = record.metadata.get("source_path")
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError(
                "VectorUpserter requires metadata.source_path as a non-empty string."
            )

        chunk_index = record.metadata.get("chunk_index")
        if (
            isinstance(chunk_index, bool)
            or not isinstance(chunk_index, int)
            or chunk_index < 0
        ):
            raise ValueError(
                "VectorUpserter requires metadata.chunk_index as a non-negative integer."
            )

        content_hash = hashlib.sha256(record.text.encode("utf-8")).hexdigest()[:8]
        identity = f"{source_path.strip()}::{chunk_index}::{content_hash}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()
