"""E2E-style tests for ingest CLI entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import scripts.ingest as ingest_script


class _FakePipeline:
    def __init__(self) -> None:
        self._settings: object | None = None
        self._seen: set[tuple[str, str]] = set()

    def bind(self, settings: object) -> _FakePipeline:
        self._settings = settings
        return self

    def run(
        self,
        source_path: str,
        collection: str | None = None,
        force: bool = False,
        trace: object | None = None,
    ) -> SimpleNamespace:
        del trace
        if self._settings is None:
            raise RuntimeError("settings not bound")

        normalized_collection = (
            collection
            if isinstance(collection, str) and collection.strip()
            else getattr(getattr(self._settings, "vector_store", None), "collection_name", "default")
        )
        key = (source_path, str(normalized_collection))
        status = "processed"
        if key in self._seen and not force:
            status = "skipped"
        else:
            self._seen.add(key)

        persist_directory = Path(
            getattr(getattr(self._settings, "vector_store", None), "persist_directory")
        )
        chroma_file = persist_directory / f"{normalized_collection}.json"
        bm25_file = persist_directory.parent / "bm25" / f"{normalized_collection}.pkl"
        chroma_file.parent.mkdir(parents=True, exist_ok=True)
        bm25_file.parent.mkdir(parents=True, exist_ok=True)
        if status == "processed":
            chroma_file.write_text("[]", encoding="utf-8")
            bm25_file.write_bytes(b"bm25")

        return SimpleNamespace(
            status=status,
            source_path=source_path,
            collection=normalized_collection,
            file_hash="f" * 64,
            chunk_count=2 if status == "processed" else 0,
            vector_count=2 if status == "processed" else 0,
            image_count=0,
        )


def _workspace(prefix: str) -> Path:
    path = ROOT_DIR / "cache" / "e2e_ingest_tests" / f"{prefix}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_settings(path: Path, persist_directory: Path, collection_name: str = "demo") -> None:
    payload = {
        "vector_store": {
            "provider": "chroma",
            "persist_directory": persist_directory.as_posix(),
            "collection_name": collection_name,
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_ingest_cli_generates_artifacts_and_supports_skip_and_force(
    monkeypatch, capsys
) -> None:
    workspace = _workspace("ingest_cli")
    pdf_path = workspace / "docs" / "sample.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4 test fixture")

    settings_path = workspace / "config" / "settings.yaml"
    persist_directory = workspace / "data" / "db" / "chroma"
    _write_settings(settings_path, persist_directory)

    fake_pipeline = _FakePipeline()
    monkeypatch.setattr(
        ingest_script, "_build_pipeline", lambda settings: fake_pipeline.bind(settings)
    )

    first_exit = ingest_script.main(
        ["--path", pdf_path.as_posix(), "--settings", settings_path.as_posix()]
    )
    assert first_exit == 0
    assert (workspace / "data" / "db" / "chroma" / "demo.json").exists()
    assert (workspace / "data" / "db" / "bm25" / "demo.pkl").exists()
    first_output = capsys.readouterr().out
    assert "[PROCESSED]" in first_output
    assert "processed=1" in first_output

    second_exit = ingest_script.main(
        ["--path", pdf_path.as_posix(), "--settings", settings_path.as_posix()]
    )
    assert second_exit == 0
    second_output = capsys.readouterr().out
    assert "[SKIPPED]" in second_output
    assert "skipped=1" in second_output

    force_exit = ingest_script.main(
        ["--path", pdf_path.as_posix(), "--settings", settings_path.as_posix(), "--force"]
    )
    assert force_exit == 0
    force_output = capsys.readouterr().out
    assert "[PROCESSED]" in force_output
    assert "processed=1" in force_output


def test_ingest_cli_returns_error_on_missing_input(capsys) -> None:
    workspace = _workspace("missing_input")
    settings_path = workspace / "config" / "settings.yaml"
    _write_settings(settings_path, workspace / "data" / "db" / "chroma")

    exit_code = ingest_script.main(
        [
            "--path",
            (workspace / "docs" / "not_exists.pdf").as_posix(),
            "--settings",
            settings_path.as_posix(),
        ]
    )

    assert exit_code == 2
    error_output = capsys.readouterr().err
    assert "[ERROR]" in error_output
