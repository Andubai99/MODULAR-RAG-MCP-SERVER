"""Image file persistence with SQLite-backed image_id index."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class ImageStorage:
    """Persist image bytes and maintain image_id -> file path mapping."""

    def __init__(
        self,
        db_path: str = "data/db/image_index.db",
        image_root: str = "data/images",
    ) -> None:
        self._db_path = Path(db_path)
        self._image_root = Path(image_root)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._image_root.mkdir(parents=True, exist_ok=True)
        self._journal_mode = self._detect_journal_mode()
        self._init_db()

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def image_root(self) -> Path:
        return self._image_root

    def save_image(
        self,
        image_id: str,
        image_bytes: bytes | bytearray | memoryview,
        collection: str,
        doc_hash: str | None = None,
        page_num: int | None = None,
        extension: str = ".png",
    ) -> str:
        normalized_id = self._normalize_required_text(image_id, "image_id")
        normalized_collection = self._normalize_required_text(collection, "collection")
        normalized_bytes = self._normalize_image_bytes(image_bytes)
        normalized_extension = self._normalize_extension(extension)
        normalized_doc_hash = self._normalize_optional_text(doc_hash, "doc_hash")
        normalized_page = self._normalize_page_num(page_num)

        collection_dir = self._image_root / normalized_collection
        collection_dir.mkdir(parents=True, exist_ok=True)
        file_path = collection_dir / f"{normalized_id}{normalized_extension}"
        temp_path = file_path.parent / f"{file_path.name}.tmp"
        temp_path.write_bytes(normalized_bytes)
        try:
            temp_path.replace(file_path)
        except PermissionError:
            file_path.write_bytes(normalized_bytes)

        self._upsert_mapping(
            image_id=normalized_id,
            file_path=file_path.as_posix(),
            collection=normalized_collection,
            doc_hash=normalized_doc_hash,
            page_num=normalized_page,
        )
        return file_path.as_posix()

    def get_image_path(self, image_id: str) -> str | None:
        normalized_id = self._normalize_required_text(image_id, "image_id")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT file_path FROM image_index WHERE image_id = ?",
                (normalized_id,),
            ).fetchone()
        if row is None:
            return None
        value = row["file_path"]
        return str(value) if isinstance(value, str) else None

    def list_images(
        self, collection: str | None = None, doc_hash: str | None = None
    ) -> list[dict[str, Any]]:
        normalized_collection = self._normalize_optional_text(collection, "collection")
        normalized_doc_hash = self._normalize_optional_text(doc_hash, "doc_hash")

        clauses: list[str] = []
        params: list[object] = []
        if normalized_collection is not None:
            clauses.append("collection = ?")
            params.append(normalized_collection)
        if normalized_doc_hash is not None:
            clauses.append("doc_hash = ?")
            params.append(normalized_doc_hash)

        query = (
            "SELECT image_id, file_path, collection, doc_hash, page_num, created_at "
            "FROM image_index"
        )
        if clauses:
            query = f"{query} WHERE {' AND '.join(clauses)}"
        query = f"{query} ORDER BY created_at DESC, image_id"

        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def _upsert_mapping(
        self,
        image_id: str,
        file_path: str,
        collection: str,
        doc_hash: str | None,
        page_num: int | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO image_index (
                    image_id, file_path, collection, doc_hash, page_num, created_at
                ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(image_id) DO UPDATE SET
                    file_path=excluded.file_path,
                    collection=excluded.collection,
                    doc_hash=excluded.doc_hash,
                    page_num=excluded.page_num,
                    created_at=CURRENT_TIMESTAMP
                """,
                (image_id, file_path, collection, doc_hash, page_num),
            )

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS image_index (
                    image_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    collection TEXT,
                    doc_hash TEXT,
                    page_num INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_collection ON image_index(collection)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_hash ON image_index(doc_hash)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._db_path.as_posix(),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        if self._journal_mode == "wal":
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        else:
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _detect_journal_mode(self) -> str:
        probe_path = self._db_path.with_suffix(f"{self._db_path.suffix}.wal_probe")
        try:
            probe_conn = sqlite3.connect(
                probe_path.as_posix(),
                timeout=5.0,
                isolation_level=None,
                check_same_thread=False,
            )
            try:
                row = probe_conn.execute("PRAGMA journal_mode=WAL").fetchone()
            finally:
                probe_conn.close()
            if row and isinstance(row[0], str) and row[0].strip().lower() == "wal":
                return "wal"
        except sqlite3.OperationalError:
            return "off"
        finally:
            for candidate in (
                probe_path,
                Path(f"{probe_path.as_posix()}-wal"),
                Path(f"{probe_path.as_posix()}-shm"),
            ):
                try:
                    if candidate.exists():
                        candidate.unlink()
                except OSError:
                    pass
        return "off"

    @staticmethod
    def _normalize_required_text(value: str, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")
        return value.strip()

    @staticmethod
    def _normalize_optional_text(value: str | None, field_name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string when provided.")
        return value.strip()

    @staticmethod
    def _normalize_image_bytes(
        value: bytes | bytearray | memoryview,
    ) -> bytes:
        if isinstance(value, bytes):
            payload = value
        elif isinstance(value, bytearray):
            payload = bytes(value)
        elif isinstance(value, memoryview):
            payload = value.tobytes()
        else:
            raise ValueError(
                "image_bytes must be bytes-like (bytes, bytearray, or memoryview)."
            )
        if not payload:
            raise ValueError("image_bytes cannot be empty.")
        return payload

    @staticmethod
    def _normalize_page_num(value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("page_num must be a non-negative integer when provided.")
        return value

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        if not isinstance(extension, str) or not extension.strip():
            raise ValueError("extension must be a non-empty string.")
        normalized = extension.strip()
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        return normalized
