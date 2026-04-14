"""Sparse encoder for BM25-style term weight preparation."""

from __future__ import annotations

import re
from collections import Counter

from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")
_DEFAULT_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "into",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "not",
    "you",
    "your",
    "but",
    "can",
    "will",
}


class SparseEncoder:
    """Encode chunk text into sparse term-weight vectors."""

    def __init__(self, stopwords: set[str] | None = None) -> None:
        self._stopwords = set(stopwords or _DEFAULT_STOPWORDS)

    def encode(
        self, chunks: list[Chunk], trace: TraceContext | None = None
    ) -> list[ChunkRecord]:
        if not isinstance(chunks, list):
            raise ValueError("SparseEncoder.encode expects chunks as a list.")
        if not chunks:
            return []

        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise ValueError(f"SparseEncoder.encode chunks[{index}] must be Chunk.")

        stage_start = trace.stage_timer() if trace is not None else None
        records: list[ChunkRecord] = []
        unique_total = 0
        token_total = 0

        for chunk in chunks:
            terms = self._tokenize(chunk.text)
            freqs = Counter(terms)
            weights = self._to_term_weights(freqs)

            metadata = dict(chunk.metadata)
            metadata["sparse_token_count"] = int(sum(freqs.values()))
            metadata["sparse_unique_terms"] = len(freqs)

            records.append(
                ChunkRecord(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=metadata,
                    sparse_vector=weights,
                )
            )
            unique_total += len(freqs)
            token_total += int(sum(freqs.values()))

        if trace is not None:
            elapsed_ms = (
                trace.stage_elapsed_ms(stage_start) if stage_start is not None else None
            )
            trace.record_stage(
                "sparse_encode",
                method="bm25",
                details={
                    "chunk_count": len(chunks),
                    "token_count": token_total,
                    "unique_term_count": unique_total,
                },
                elapsed_ms=elapsed_ms,
            )
        return records

    def _tokenize(self, text: str) -> list[str]:
        tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
        return [token for token in tokens if token not in self._stopwords]

    @staticmethod
    def _to_term_weights(freqs: Counter[str]) -> dict[str, float]:
        if not freqs:
            return {}
        max_tf = max(freqs.values())
        if max_tf <= 0:
            return {}
        return {term: float(count) / float(max_tf) for term, count in freqs.items()}
