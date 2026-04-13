"""Unit tests for file integrity checker and SQLite history backend."""

from __future__ import annotations

import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libs.loader.file_integrity import SQLiteIntegrityChecker


def test_compute_sha256_is_stable(tmp_path: Path) -> None:
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("same-content", encoding="utf-8")
    checker = SQLiteIntegrityChecker(db_path=str(tmp_path / "db" / "ingestion_history.db"))

    first = checker.compute_sha256(str(sample_file))
    second = checker.compute_sha256(str(sample_file))

    assert first == second
    assert len(first) == 64


def test_mark_success_then_should_skip_returns_true(tmp_path: Path) -> None:
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("hello", encoding="utf-8")
    checker = SQLiteIntegrityChecker(db_path=str(tmp_path / "db" / "ingestion_history.db"))
    file_hash = checker.compute_sha256(str(sample_file))

    assert checker.should_skip(file_hash) is False
    checker.mark_success(file_hash=file_hash, file_path=str(sample_file))
    assert checker.should_skip(file_hash) is True


def test_default_db_created_under_data_db() -> None:
    checker = SQLiteIntegrityChecker()
    db_path = checker.db_path

    assert db_path.as_posix().endswith("data/db/ingestion_history.db")
    assert db_path.exists()
    assert db_path.is_file()


def test_mark_failed_does_not_enable_skip(tmp_path: Path) -> None:
    checker = SQLiteIntegrityChecker(db_path=str(tmp_path / "db" / "ingestion_history.db"))
    file_hash = "a" * 64

    checker.mark_failed(file_hash=file_hash, error_msg="parse failed")

    assert checker.should_skip(file_hash) is False
    status = checker.fetch_status(file_hash)
    assert status is not None
    assert status["status"] == "failed"
    assert status["error_msg"] == "parse failed"


def test_sqlite_wal_mode_enabled_and_supports_concurrent_writes(tmp_path: Path) -> None:
    db_path = tmp_path / "db" / "ingestion_history.db"
    checker = SQLiteIntegrityChecker(db_path=str(db_path))

    with sqlite3.connect(str(db_path)) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"

    file_hashes = [f"{i:064x}" for i in range(1, 16)]

    def _writer(file_hash: str) -> None:
        checker.mark_success(file_hash=file_hash, file_path=f"/tmp/{file_hash}.txt")

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(_writer, file_hashes))

    assert checker.count_records() >= len(file_hashes)
    for file_hash in file_hashes:
        assert checker.should_skip(file_hash) is True


def test_compute_sha256_missing_file_raises(tmp_path: Path) -> None:
    checker = SQLiteIntegrityChecker(db_path=str(tmp_path / "db" / "ingestion_history.db"))
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(ValueError, match="File not found"):
        checker.compute_sha256(str(missing_file))


def test_normalizes_hash_case_and_validates_shape(tmp_path: Path) -> None:
    checker = SQLiteIntegrityChecker(db_path=str(tmp_path / "db" / "ingestion_history.db"))
    upper_hash = "A" * 64

    checker.mark_success(file_hash=upper_hash, file_path="a.txt")
    assert checker.should_skip(upper_hash.lower()) is True

    with pytest.raises(ValueError, match="SHA256"):
        checker.should_skip("abc")
