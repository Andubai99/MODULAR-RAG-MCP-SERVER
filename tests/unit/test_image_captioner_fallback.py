"""Unit tests for ImageCaptioner transform and fallback behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.image_captioner import ImageCaptioner
from libs.llm.base_vision_llm import BaseVisionLLM, ChatResponse


class _FakeVisionLLM(BaseVisionLLM):
    def __init__(
        self, response_text: str = "A line chart showing steady growth.", raise_error: Exception | None = None
    ) -> None:
        self._response_text = response_text
        self._raise_error = raise_error
        self.calls: list[tuple[str, str]] = []

    def chat_with_image(
        self, text: str, image_path: str | bytes, trace: object | None = None
    ) -> ChatResponse:
        del trace
        if isinstance(image_path, bytes):
            path_value = "<bytes>"
        else:
            path_value = image_path
        self.calls.append((text, path_value))
        if self._raise_error is not None:
            raise self._raise_error
        return {"text": self._response_text}


def _make_settings(vision_enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        vision_llm=SimpleNamespace(
            enabled=vision_enabled,
            provider="azure",
            model="gpt-4o",
            azure_endpoint="https://example.openai.azure.com/",
            deployment_name="vision",
            api_version="2024-02-15-preview",
            api_key="dummy",
            max_image_size=2048,
        )
    )


def _chunk_with_image(text: str = "Section text with image.", image_id: str = "img-1") -> Chunk:
    return Chunk(
        id="chunk-1",
        text=text,
        metadata={
            "source_path": "tests/fixtures/sample_documents/sample.pdf",
            "image_refs": [image_id],
            "images": [
                {
                    "id": image_id,
                    "path": f"data/images/demo/{image_id}.png",
                    "text_offset": 0,
                    "text_length": 12,
                }
            ],
        },
        start_offset=0,
        end_offset=max(1, len(text)),
        source_ref="doc-1",
    )


def test_enabled_mode_generates_caption_and_appends_to_text() -> None:
    fake_llm = _FakeVisionLLM(response_text="A chart rising from Q1 to Q4.")
    captioner = ImageCaptioner(_make_settings(vision_enabled=True), vision_llm=fake_llm)

    result = captioner.transform([_chunk_with_image()])[0]

    assert result.metadata["image_captioned_by"] == "llm"
    assert result.metadata["image_captions"] == [
        {"id": "img-1", "caption": "A chart rising from Q1 to Q4."}
    ]
    assert "A chart rising from Q1 to Q4." in result.text
    assert not result.metadata.get("has_unprocessed_images", False)
    assert len(fake_llm.calls) == 1


def test_disabled_mode_keeps_image_refs_and_marks_unprocessed() -> None:
    captioner = ImageCaptioner(_make_settings(vision_enabled=False))

    result = captioner.transform([_chunk_with_image()])[0]

    assert result.metadata["image_captioned_by"] == "rule"
    assert result.metadata["has_unprocessed_images"] is True
    assert result.metadata["image_caption_fallback_reason"] == "vision_llm_disabled_or_unavailable"
    assert "image_captions" not in result.metadata
    assert result.metadata["image_refs"] == ["img-1"]


def test_llm_failure_falls_back_without_raising() -> None:
    fake_llm = _FakeVisionLLM(raise_error=RuntimeError("vision service unavailable"))
    captioner = ImageCaptioner(_make_settings(vision_enabled=True), vision_llm=fake_llm)

    result = captioner.transform([_chunk_with_image()])[0]

    assert result.metadata["image_captioned_by"] == "rule"
    assert result.metadata["has_unprocessed_images"] is True
    assert result.metadata["image_caption_fallback_reason"] == "vision_llm_failed_or_partial"
    assert "image_captions" not in result.metadata


def test_missing_image_path_marks_unprocessed() -> None:
    fake_llm = _FakeVisionLLM()
    captioner = ImageCaptioner(_make_settings(vision_enabled=True), vision_llm=fake_llm)
    chunk = Chunk(
        id="chunk-2",
        text="Text with unresolved image ref.",
        metadata={
            "source_path": "tests/fixtures/sample_documents/sample.pdf",
            "image_refs": ["img-x"],
            "images": [],
        },
        start_offset=0,
        end_offset=20,
        source_ref="doc-2",
    )

    result = captioner.transform([chunk])[0]

    assert result.metadata["image_captioned_by"] == "rule"
    assert result.metadata["has_unprocessed_images"] is True
    assert result.metadata["image_caption_fallback_reason"] == "vision_llm_failed_or_partial"


def test_no_image_refs_skips_without_mutation() -> None:
    fake_llm = _FakeVisionLLM()
    captioner = ImageCaptioner(_make_settings(vision_enabled=True), vision_llm=fake_llm)
    chunk = Chunk(
        id="chunk-3",
        text="No images in this chunk.",
        metadata={"source_path": "tests/fixtures/sample_documents/sample.pdf"},
        start_offset=0,
        end_offset=22,
        source_ref="doc-3",
    )

    result = captioner.transform([chunk])[0]

    assert "image_captioned_by" not in result.metadata
    assert "image_captions" not in result.metadata
    assert not fake_llm.calls


def test_prompt_loading_appends_placeholders_when_missing() -> None:
    prompt_path = ROOT_DIR / "cache" / "caption_prompt_test.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("Describe image.", encoding="utf-8")
    captioner = ImageCaptioner(
        _make_settings(vision_enabled=False), prompt_path=prompt_path
    )

    assert "{text}" in captioner._prompt_template
    assert "{image_id}" in captioner._prompt_template


def test_transform_validates_input_type() -> None:
    captioner = ImageCaptioner(_make_settings(vision_enabled=False))

    with pytest.raises(ValueError, match="expects chunks as a list"):
        captioner.transform("not-list")  # type: ignore[arg-type]


def test_transform_validates_chunk_item_type() -> None:
    captioner = ImageCaptioner(_make_settings(vision_enabled=False))

    with pytest.raises(ValueError, match="must be Chunk"):
        captioner.transform([_chunk_with_image(), "bad-item"])  # type: ignore[list-item]


def test_trace_records_each_chunk_stage() -> None:
    captioner = ImageCaptioner(_make_settings(vision_enabled=False))
    trace = TraceContext(trace_type="ingestion")
    chunks = [_chunk_with_image(), _chunk_with_image(image_id="img-2")]

    _ = captioner.transform(chunks, trace=trace)

    assert len(trace.stages) == 2
    assert all(stage["stage"] == "image_caption" for stage in trace.stages)
