"""Contract tests for BaseLoader and PdfLoader."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.types import IMAGE_PLACEHOLDER_TEMPLATE, Document
from libs.loader.base_loader import BaseLoader
from libs.loader.pdf_loader import PdfLoader


class _FakeLoader(BaseLoader):
    def load(self, path: str) -> Document:
        return Document(
            id="fake-doc",
            text=f"loaded:{path}",
            metadata={"source_path": path, "doc_type": "fake"},
        )


def test_base_loader_contract_shape() -> None:
    loader = _FakeLoader()
    document = loader.load("tests/fixtures/sample_documents/sample.txt")

    assert isinstance(document, Document)
    assert document.metadata["source_path"] == "tests/fixtures/sample_documents/sample.txt"


def test_pdf_loader_returns_document_with_required_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "simple.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")

    def _parser(_: Path) -> str:
        return "This is extracted PDF text."

    loader = PdfLoader(parser=_parser)
    document = loader.load(str(pdf_path))

    assert isinstance(document, Document)
    assert document.metadata["source_path"] == pdf_path.as_posix()
    assert document.metadata["doc_type"] == "pdf"
    assert document.text == "This is extracted PDF text."
    assert isinstance(document.metadata.get("images"), list)


def test_pdf_loader_supports_images_metadata_schema(tmp_path: Path) -> None:
    pdf_path = tmp_path / "with_images.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 with image")
    image_id = "dochash_1_0"
    placeholder = IMAGE_PLACEHOLDER_TEMPLATE.format(image_id=image_id)
    text = f"Before image {placeholder} after image."

    def _parser(_: Path) -> dict[str, object]:
        return {
            "text": text,
            "images": [
                {
                    "id": image_id,
                    "path": f"data/images/demo/{image_id}.png",
                    "page": 1,
                    "text_offset": text.find(placeholder),
                    "text_length": len(placeholder),
                    "position": {"x": 10, "y": 20},
                }
            ],
        }

    loader = PdfLoader(parser=_parser)
    document = loader.load(str(pdf_path))

    images = document.metadata["images"]
    assert isinstance(images, list)
    assert images[0]["id"] == image_id
    assert images[0]["path"] == f"data/images/demo/{image_id}.png"
    assert images[0]["text_length"] == len(placeholder)


def test_pdf_loader_rejects_missing_file(tmp_path: Path) -> None:
    loader = PdfLoader(parser=lambda _: "ignored")
    missing = tmp_path / "not_exists.pdf"

    with pytest.raises(ValueError, match="file not found"):
        loader.load(str(missing))


def test_pdf_loader_rejects_non_pdf_extension(tmp_path: Path) -> None:
    text_path = tmp_path / "input.txt"
    text_path.write_text("abc", encoding="utf-8")
    loader = PdfLoader(parser=lambda _: "text")

    with pytest.raises(ValueError, match="only \\.pdf"):
        loader.load(str(text_path))


def test_pdf_loader_raises_on_empty_extracted_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "simple.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    loader = PdfLoader(parser=lambda _: "   ")

    with pytest.raises(ValueError, match="parsed text must be a non-empty string"):
        loader.load(str(pdf_path))


def test_pdf_loader_parser_output_shape_validation(tmp_path: Path) -> None:
    pdf_path = tmp_path / "simple.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    loader = PdfLoader(parser=lambda _: 123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="parser output"):
        loader.load(str(pdf_path))
