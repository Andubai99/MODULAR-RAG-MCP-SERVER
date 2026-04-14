"""BM25 inverted index builder and persistence layer."""

from __future__ import annotations

import math
import pickle
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.types import ChunkRecord

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")
_INDEX_VERSION = 1


class BM25Indexer:
    """Build, persist, load, and query BM25-style sparse indexes."""

    def __init__(
        self,
        settings: object | None = None,
        persist_dir: str | Path | None = None,
    ) -> None:
        if persist_dir is None:
            persist_dir = self._resolve_default_persist_dir(settings)
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._default_collection = self._resolve_default_collection(settings)
        self._cache: dict[str, dict[str, Any]] = {}

    def rebuild(self, records: list[ChunkRecord], collection: str | None = None) -> None:
        """Replace an index collection with records and persist."""
        normalized_collection = self._normalize_collection(collection)
        documents = self._records_to_documents(records)
        payload = self._build_payload(documents)
        self._cache[normalized_collection] = payload
        self._save_payload(normalized_collection, payload)

    def upsert(self, records: list[ChunkRecord], collection: str | None = None) -> None:
        """Incrementally insert or update records into an index collection."""
        normalized_collection = self._normalize_collection(collection)
        payload = self._load_payload(normalized_collection)
        documents = payload.get("documents", {})
        if not isinstance(documents, dict):
            documents = {}
        updates = self._records_to_documents(records)
        documents.update(updates)
        rebuilt = self._build_payload(documents)
        self._cache[normalized_collection] = rebuilt
        self._save_payload(normalized_collection, rebuilt)

    def remove_document(self, source_path: str, collection: str | None = None) -> None:
        """Remove all chunks that belong to a source document path."""
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError("BM25Indexer.remove_document source_path must be non-empty string.")

        normalized_collection = self._normalize_collection(collection)
        payload = self._load_payload(normalized_collection)
        documents = payload.get("documents", {})
        if not isinstance(documents, dict):
            return

        target = source_path.strip()
        filtered: dict[str, dict[str, Any]] = {}
        for chunk_id, doc in documents.items():
            if not isinstance(doc, Mapping):
                continue
            if doc.get("source_path") == target:
                continue
            filtered[str(chunk_id)] = dict(doc)

        rebuilt = self._build_payload(filtered)
        self._cache[normalized_collection] = rebuilt
        self._save_payload(normalized_collection, rebuilt)

    def query(
        self,
        query_terms: str | Iterable[str],
        top_k: int = 10,
        collection: str | None = None,
    ) -> list[dict[str, object]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("BM25Indexer.query top_k must be a positive integer.")

        normalized_collection = self._normalize_collection(collection)
        payload = self._load_payload(normalized_collection)
        inverted = payload.get("inverted_index", {})
        documents = payload.get("documents", {})
        if not isinstance(inverted, Mapping) or not isinstance(documents, Mapping):
            return []

        tokens = self._normalize_query_terms(query_terms)
        if not tokens:
            return []

        scores: dict[str, float] = {}
        for token in tokens:
            term_entry = inverted.get(token)
            if not isinstance(term_entry, Mapping):
                continue
            idf = float(term_entry.get("idf", 0.0))
            postings = term_entry.get("postings", [])
            if not isinstance(postings, list):
                continue
            for posting in postings:
                if not isinstance(posting, Mapping):
                    continue
                chunk_id = posting.get("chunk_id")
                tf = posting.get("tf")
                if not isinstance(chunk_id, str):
                    continue
                if isinstance(tf, bool) or not isinstance(tf, (int, float)):
                    continue
                effective_idf = idf if idf > 0 else 1.0
                score = effective_idf * float(tf)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + score

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        results: list[dict[str, object]] = []
        for chunk_id, score in ranked:
            doc = documents.get(chunk_id, {})
            source_path = doc.get("source_path") if isinstance(doc, Mapping) else None
            results.append(
                {
                    "chunk_id": chunk_id,
                    "score": float(score),
                    "source_path": source_path if isinstance(source_path, str) else "",
                }
            )
        return results

    def get_index(self, collection: str | None = None) -> dict[str, Any]:
        """Return loaded index payload for inspection/testing."""
        normalized_collection = self._normalize_collection(collection)
        payload = self._load_payload(normalized_collection)
        return dict(payload)

    @staticmethod
    def _resolve_default_persist_dir(settings: object | None) -> str:
        if settings is None:
            return "data/db/bm25"
        vector_store = getattr(settings, "vector_store", None)
        candidate = getattr(vector_store, "persist_directory", "")
        if isinstance(candidate, str) and candidate.strip():
            base = Path(candidate.strip())
            return (base.parent / "bm25").as_posix()
        return "data/db/bm25"

    @staticmethod
    def _resolve_default_collection(settings: object | None) -> str:
        if settings is None:
            return "default"
        vector_store = getattr(settings, "vector_store", None)
        candidate = getattr(vector_store, "collection_name", "")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        return "default"

    def _normalize_collection(self, collection: str | None) -> str:
        if collection is None:
            return self._default_collection
        if not isinstance(collection, str) or not collection.strip():
            raise ValueError("BM25Indexer collection must be a non-empty string.")
        return collection.strip()

    def _collection_path(self, collection: str) -> Path:
        return self._persist_dir / f"{collection}.pkl"

    def _load_payload(self, collection: str) -> dict[str, Any]:
        cached = self._cache.get(collection)
        if cached is not None:
            return cached

        path = self._collection_path(collection)
        if not path.exists():
            payload = self._build_payload({})
            self._cache[collection] = payload
            return payload

        raw = path.read_bytes()
        loaded = pickle.loads(raw)
        if not isinstance(loaded, dict):
            raise ValueError("BM25Indexer index payload must be dict.")
        version = loaded.get("version")
        if version != _INDEX_VERSION:
            raise ValueError(
                f"BM25Indexer unsupported index version: {version} (expected {_INDEX_VERSION})."
            )
        self._cache[collection] = loaded
        return loaded

    def _save_payload(self, collection: str, payload: dict[str, Any]) -> None:
        path = self._collection_path(collection)
        temp_path = path.with_suffix(".pkl.tmp")
        content = pickle.dumps(payload)
        temp_path.write_bytes(content)
        try:
            temp_path.replace(path)
        except PermissionError:
            # Fallback for restrictive Windows environments where atomic replace is blocked.
            path.write_bytes(content)

    def _records_to_documents(self, records: list[ChunkRecord]) -> dict[str, dict[str, Any]]:
        if not isinstance(records, list):
            raise ValueError("BM25Indexer expects records as a list.")

        documents: dict[str, dict[str, Any]] = {}
        for index, record in enumerate(records):
            if not isinstance(record, ChunkRecord):
                raise ValueError(f"BM25Indexer records[{index}] must be ChunkRecord.")

            sparse = record.sparse_vector or {}
            if not isinstance(sparse, Mapping):
                raise ValueError(f"BM25Indexer records[{index}].sparse_vector must be mapping.")

            terms: dict[str, float] = {}
            for term, value in sparse.items():
                if not isinstance(term, str) or not term.strip():
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                terms[term.strip().lower()] = float(value)

            metadata = dict(record.metadata)
            source_path = metadata.get("source_path", "")
            source_value = source_path if isinstance(source_path, str) else ""
            doc_length_raw = metadata.get("sparse_token_count")
            if isinstance(doc_length_raw, bool) or not isinstance(doc_length_raw, int):
                doc_length = len(terms)
            else:
                doc_length = max(0, int(doc_length_raw))

            documents[record.id] = {
                "source_path": source_value,
                "doc_length": doc_length,
                "terms": terms,
            }
        return documents

    def _build_payload(self, documents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        normalized_docs: dict[str, dict[str, Any]] = {}
        inverted: dict[str, dict[str, Any]] = {}
        doc_count = len(documents)

        for chunk_id, doc in documents.items():
            if not isinstance(chunk_id, str) or not chunk_id:
                continue
            if not isinstance(doc, Mapping):
                continue
            terms_raw = doc.get("terms", {})
            if not isinstance(terms_raw, Mapping):
                terms_raw = {}
            terms: dict[str, float] = {}
            for term, tf in terms_raw.items():
                if not isinstance(term, str) or not term.strip():
                    continue
                if isinstance(tf, bool) or not isinstance(tf, (int, float)):
                    continue
                normalized_term = term.strip().lower()
                tf_value = float(tf)
                terms[normalized_term] = tf_value
                term_entry = inverted.setdefault(normalized_term, {"idf": 0.0, "postings": []})
                postings = term_entry["postings"]
                if isinstance(postings, list):
                    postings.append(
                        {
                            "chunk_id": chunk_id,
                            "tf": tf_value,
                            "doc_length": int(doc.get("doc_length", len(terms))),
                        }
                    )

            source_path = doc.get("source_path")
            normalized_docs[chunk_id] = {
                "source_path": source_path if isinstance(source_path, str) else "",
                "doc_length": int(doc.get("doc_length", len(terms))),
                "terms": terms,
            }

        for term, entry in inverted.items():
            postings = entry.get("postings", [])
            if not isinstance(postings, list):
                continue
            df = len(postings)
            entry["idf"] = self._compute_idf(doc_count, df)
            postings.sort(key=lambda item: item["chunk_id"])

        return {
            "version": _INDEX_VERSION,
            "doc_count": doc_count,
            "documents": normalized_docs,
            "inverted_index": inverted,
        }

    @staticmethod
    def _compute_idf(doc_count: int, doc_freq: int) -> float:
        if doc_count <= 0 or doc_freq <= 0:
            return 0.0
        return math.log((doc_count - doc_freq + 0.5) / (doc_freq + 0.5))

    @staticmethod
    def _normalize_query_terms(query_terms: str | Iterable[str]) -> list[str]:
        if isinstance(query_terms, str):
            raw_terms = [token.lower() for token in _TOKEN_RE.findall(query_terms)]
        else:
            raw_terms = []
            for item in query_terms:
                if isinstance(item, str):
                    raw_terms.extend([token.lower() for token in _TOKEN_RE.findall(item)])
        seen: set[str] = set()
        normalized: list[str] = []
        for term in raw_terms:
            if not term or term in seen:
                continue
            seen.add(term)
            normalized.append(term)
        return normalized
