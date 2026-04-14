"""Offline ingestion CLI entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ingestion.pipeline import IngestionPipeline


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def load_runtime_settings(path: str | Path) -> object:
    config_path = Path(path)
    if not config_path.exists() or not config_path.is_file():
        raise ValueError(f"settings file not found: {config_path.as_posix()}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("settings root must be an object.")
    return _to_namespace(raw)


def collect_pdf_files(path: str | Path) -> list[Path]:
    input_path = Path(path)
    if not input_path.exists():
        raise ValueError(f"input path not found: {input_path.as_posix()}")
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError("input file must be a .pdf document.")
        return [input_path]

    files = sorted(
        file_path for file_path in input_path.rglob("*.pdf") if file_path.is_file()
    )
    if not files:
        raise ValueError(f"no PDF files found under: {input_path.as_posix()}")
    return files


def _build_pipeline(settings: object) -> IngestionPipeline:
    return IngestionPipeline(settings)


def run_ingest(
    path: str,
    collection: str | None,
    force: bool,
    settings_path: str,
) -> int:
    settings = load_runtime_settings(settings_path)
    pipeline = _build_pipeline(settings)
    files = collect_pdf_files(path)

    default_collection = getattr(getattr(settings, "vector_store", None), "collection_name", "")
    target_collection = collection.strip() if isinstance(collection, str) and collection.strip() else ""
    if not target_collection:
        target_collection = (
            default_collection.strip()
            if isinstance(default_collection, str) and default_collection.strip()
            else "default"
        )

    processed = 0
    skipped = 0
    failed = 0

    for file_path in files:
        try:
            result = pipeline.run(
                file_path.as_posix(),
                collection=target_collection,
                force=force,
                trace=None,
            )
        except Exception as exc:
            failed += 1
            print(
                f"[FAILED] {file_path.as_posix()} :: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue

        status = str(getattr(result, "status", "processed")).strip().lower()
        if status == "skipped":
            skipped += 1
        else:
            processed += 1

        chunk_count = int(getattr(result, "chunk_count", 0))
        vector_count = int(getattr(result, "vector_count", 0))
        image_count = int(getattr(result, "image_count", 0))
        print(
            f"[{status.upper()}] {file_path.as_posix()} "
            f"chunks={chunk_count} vectors={vector_count} images={image_count}"
        )

    print(
        "SUMMARY "
        f"collection={target_collection} total={len(files)} "
        f"processed={processed} skipped={skipped} failed={failed}"
    )
    return 1 if failed else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline ingestion pipeline.")
    parser.add_argument("--path", required=True, help="PDF file path or directory path.")
    parser.add_argument("--collection", default="", help="Target collection name.")
    parser.add_argument("--force", action="store_true", help="Force re-ingest even if unchanged.")
    parser.add_argument(
        "--settings",
        default="config/settings.yaml",
        help="Path to settings YAML file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_ingest(
            path=args.path,
            collection=args.collection,
            force=bool(args.force),
            settings_path=args.settings,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
