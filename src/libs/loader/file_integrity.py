"""File integrity checker with SQLite-backed ingestion history."""

from __future__ import annotations

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class FileIntegrityChecker(ABC):
    """Abstract integrity checker contract."""

    @abstractmethod
    def compute_sha256(self, path: str) -> str:
        """Return SHA256 hex digest for a file path."""
        raise NotImplementedError

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """Return whether ingestion can skip this file hash."""
        raise NotImplementedError

    @abstractmethod
    def mark_success(self, file_hash: str, file_path: str, **kwargs: object) -> None:
        """Mark ingestion success for a file hash."""
        raise NotImplementedError

    @abstractmethod
    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        """Mark ingestion failure for a file hash."""
        raise NotImplementedError


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """SQLite implementation for ingestion history tracking."""

    def __init__(self, db_path: str = "data/db/ingestion_history.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def compute_sha256(self, path: str) -> str:
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            raise ValueError(f"File not found: {file_path.as_posix()}")

        hasher = hashlib.sha256()
        with file_path.open("rb") as reader:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        normalized_hash = self._normalize_hash(file_hash)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ?",
                (normalized_hash,),
            ).fetchone()
        return bool(row and row[0] == "success")

    def mark_success(self, file_hash: str, file_path: str, **kwargs: object) -> None:
        normalized_hash = self._normalize_hash(file_hash)
        normalized_path = self._normalize_path(file_path)

        file_size = kwargs.get("file_size")
        if file_size is None:
            file_size = self._safe_file_size(normalized_path)
        if file_size is not None and (isinstance(file_size, bool) or not isinstance(file_size, int)):
            raise ValueError("file_size must be an integer when provided.")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_history (
                    file_hash, file_path, status, error_msg, file_size, updated_at
                ) VALUES (?, ?, 'success', NULL, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path=excluded.file_path,
                    status='success',
                    error_msg=NULL,
                    file_size=excluded.file_size,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (normalized_hash, normalized_path, file_size),
            )

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        normalized_hash = self._normalize_hash(file_hash)
        if not isinstance(error_msg, str) or not error_msg.strip():
            raise ValueError("error_msg must be a non-empty string.")
        normalized_error = error_msg.strip()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_history (
                    file_hash, file_path, status, error_msg, file_size, updated_at
                ) VALUES (?, '', 'failed', ?, NULL, CURRENT_TIMESTAMP)
                ON CONFLICT(file_hash) DO UPDATE SET
                    status='failed',
                    error_msg=excluded.error_msg,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (normalized_hash, normalized_error),
            )

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('success', 'failed')),
                    error_msg TEXT,
                    file_size INTEGER,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ingestion_status ON ingestion_history(status)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._db_path.as_posix(),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @staticmethod
    def _normalize_hash(file_hash: str) -> str:
        if not isinstance(file_hash, str) or not file_hash.strip():
            raise ValueError("file_hash must be a non-empty string.")
        normalized = file_hash.strip().lower()
        if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
            raise ValueError("file_hash must be a SHA256 hex string.")
        return normalized

    @staticmethod
    def _normalize_path(file_path: str) -> str:
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string.")
        return file_path.strip()

    @staticmethod
    def _safe_file_size(file_path: str) -> int | None:
        try:
            return Path(file_path).stat().st_size
        except OSError:
            return None

    def count_records(self) -> int:
        """Testing helper: return row count."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM ingestion_history").fetchone()
        return int(row["cnt"]) if row is not None else 0

    def fetch_status(self, file_hash: str) -> dict[str, Any] | None:
        """Testing/helper API for inspecting status by hash."""
        normalized_hash = self._normalize_hash(file_hash)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT file_hash, file_path, status, error_msg, file_size, updated_at
                FROM ingestion_history
                WHERE file_hash = ?
                """,
                (normalized_hash,),
            ).fetchone()
        if row is None:
            return None
        return dict(row)
