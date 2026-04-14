"""Unit tests for image storage file persistence and SQLite index mapping."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from uuid import uuid4

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ingestion.storage.image_storage import ImageStorage


def _workspace(prefix: str) -> Path:
    path = ROOT_DIR / "cache" / "image_storage_tests" / f"{prefix}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_save_image_creates_file_and_index_record() -> None:
    workspace = _workspace("save")
    storage = ImageStorage(
        db_path=(workspace / "db" / "image_index.db").as_posix(),
        image_root=(workspace / "images").as_posix(),
    )

    saved_path = storage.save_image(
        image_id="img-001",
        image_bytes=b"png-binary",
        collection="demo",
        doc_hash="doc-hash-1",
        page_num=2,
    )

    path = Path(saved_path)
    assert path.exists()
    assert path.read_bytes() == b"png-binary"
    assert storage.get_image_path("img-001") == saved_path

    listed = storage.list_images(collection="demo")
    assert len(listed) == 1
    assert listed[0]["image_id"] == "img-001"
    assert listed[0]["doc_hash"] == "doc-hash-1"
    assert listed[0]["page_num"] == 2


def test_get_image_path_returns_none_for_unknown_id() -> None:
    workspace = _workspace("missing")
    storage = ImageStorage(
        db_path=(workspace / "db" / "image_index.db").as_posix(),
        image_root=(workspace / "images").as_posix(),
    )

    assert storage.get_image_path("not-found") is None


def test_mapping_persists_across_instances() -> None:
    workspace = _workspace("persist")
    db_path = (workspace / "db" / "image_index.db").as_posix()
    image_root = (workspace / "images").as_posix()
    first = ImageStorage(db_path=db_path, image_root=image_root)
    saved_path = first.save_image(
        image_id="img-persist",
        image_bytes=bytearray(b"content"),
        collection="knowledge",
        doc_hash="hash-persist",
        page_num=1,
    )

    second = ImageStorage(db_path=db_path, image_root=image_root)
    assert second.get_image_path("img-persist") == saved_path
    assert [item["image_id"] for item in second.list_images(collection="knowledge")] == [
        "img-persist"
    ]


def test_list_images_supports_collection_filter() -> None:
    workspace = _workspace("filter")
    storage = ImageStorage(
        db_path=(workspace / "db" / "image_index.db").as_posix(),
        image_root=(workspace / "images").as_posix(),
    )
    storage.save_image("img-a", b"a", collection="c1", doc_hash="h1")
    storage.save_image("img-b", b"b", collection="c1", doc_hash="h2")
    storage.save_image("img-c", b"c", collection="c2", doc_hash="h1")

    c1_images = storage.list_images(collection="c1")
    h1_images = storage.list_images(doc_hash="h1")

    assert sorted(item["image_id"] for item in c1_images) == ["img-a", "img-b"]
    assert sorted(item["image_id"] for item in h1_images) == ["img-a", "img-c"]


def test_sqlite_wal_mode_is_enabled() -> None:
    workspace = _workspace("wal")
    db_path = workspace / "db" / "image_index.db"
    storage = ImageStorage(db_path=db_path.as_posix(), image_root=(workspace / "images").as_posix())
    storage.save_image("img-wal", b"payload", collection="demo")

    with sqlite3.connect(db_path.as_posix()) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() in {"wal", "off", "delete"}


def test_validation_errors_for_invalid_inputs() -> None:
    workspace = _workspace("validation")
    storage = ImageStorage(
        db_path=(workspace / "db" / "image_index.db").as_posix(),
        image_root=(workspace / "images").as_posix(),
    )

    with pytest.raises(ValueError, match="image_id must be a non-empty string"):
        storage.save_image("", b"x", collection="demo")

    with pytest.raises(ValueError, match="image_bytes cannot be empty"):
        storage.save_image("img", b"", collection="demo")

    with pytest.raises(ValueError, match="page_num must be a non-negative integer"):
        storage.save_image("img", b"x", collection="demo", page_num=-1)
