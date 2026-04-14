"""Unit tests for ChunkRefiner transform."""

from __future__ import annotations

import json
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
from ingestion.transform.chunk_refiner import ChunkRefiner
from libs.llm.base_llm import BaseLLM


class _FakeLLM(BaseLLM):
    def __init__(self, response: str = "llm refined text", raise_error: Exception | None = None) -> None:
        self._response = response
        self._raise_error = raise_error
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if self._raise_error is not None:
            raise self._raise_error
        return self._response


def _make_settings(use_llm: bool) -> SimpleNamespace:
    return SimpleNamespace(
        llm=SimpleNamespace(
            provider="openai",
            model="gpt-4o-mini",
            deployment_name="",
            azure_endpoint="",
            api_version="",
            api_key="dummy",
            base_url="https://api.openai.com/v1",
            temperature=0.0,
            max_tokens=128,
        ),
        ingestion=SimpleNamespace(
            chunk_refiner=SimpleNamespace(use_llm=use_llm),
        ),
    )


def _chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(
        id=f"chunk-{idx}",
        text=text,
        metadata={"source_path": "tests/fixtures/sample_documents/sample.pdf"},
        start_offset=0,
        end_offset=max(1, len(text)),
        source_ref="doc-1",
    )


def test_rule_based_cleanup_from_fixture_cases() -> None:
    fixture = ROOT_DIR / "tests" / "fixtures" / "noisy_chunks.json"
    noisy_cases = json.loads(fixture.read_text(encoding="utf-8"))
    refiner = ChunkRefiner(_make_settings(use_llm=False))

    cleaned_typical = refiner._rule_based_refine(noisy_cases["typical_noise_scenario"])
    cleaned_header = refiner._rule_based_refine(noisy_cases["page_header_footer"])
    cleaned_whitespace = refiner._rule_based_refine(noisy_cases["excessive_whitespace"])
    cleaned_markers = refiner._rule_based_refine(noisy_cases["format_markers"])

    assert "Page 1" not in cleaned_typical
    assert "Page 9" not in cleaned_header
    assert "\n\n\n" not in cleaned_whitespace
    assert "<!--" not in cleaned_markers


def test_rule_based_cleanup_preserves_code_blocks() -> None:
    text = "Intro\n\n```python\nx = 1\nprint(x)\n```\n\nPage 1"
    refiner = ChunkRefiner(_make_settings(use_llm=False))
    cleaned = refiner._rule_based_refine(text)

    assert "```python\nx = 1\nprint(x)\n```" in cleaned
    assert "Page 1" not in cleaned


def test_transform_rule_mode_marks_metadata() -> None:
    refiner = ChunkRefiner(_make_settings(use_llm=False))
    chunks = [_chunk("Page 1\n\ntext", 0), _chunk("clean text", 1)]

    transformed = refiner.transform(chunks)

    assert len(transformed) == 2
    assert all(c.metadata["refined_by"] == "rule" for c in transformed)
    assert "Page 1" not in transformed[0].text


def test_transform_llm_mode_uses_llm_response() -> None:
    fake_llm = _FakeLLM(response="LLM polished output")
    refiner = ChunkRefiner(_make_settings(use_llm=True), llm=fake_llm)
    chunk = _chunk("noisy content", 0)

    transformed = refiner.transform([chunk])

    assert transformed[0].text == "LLM polished output"
    assert transformed[0].metadata["refined_by"] == "llm"
    assert fake_llm.calls


def test_transform_llm_failure_fallbacks_to_rule_result() -> None:
    fake_llm = _FakeLLM(raise_error=RuntimeError("llm down"))
    refiner = ChunkRefiner(_make_settings(use_llm=True), llm=fake_llm)
    chunk = _chunk("Page 7\n\ndata text", 0)

    transformed = refiner.transform([chunk])

    assert transformed[0].metadata["refined_by"] == "rule"
    assert transformed[0].metadata["refine_fallback_reason"] == "llm_unavailable_or_failed"
    assert "Page 7" not in transformed[0].text


def test_transform_chunk_error_isolated_without_breaking_others(monkeypatch: pytest.MonkeyPatch) -> None:
    refiner = ChunkRefiner(_make_settings(use_llm=False))

    original_method = refiner._rule_based_refine

    def _raise_for_one(text: str) -> str:
        if "bad" in text:
            raise ValueError("bad chunk")
        return original_method(text)

    monkeypatch.setattr(refiner, "_rule_based_refine", _raise_for_one)
    chunks = [_chunk("good text", 0), _chunk("bad text", 1), _chunk("Page 1\nok", 2)]

    transformed = refiner.transform(chunks)

    assert transformed[0].text == "good text"
    assert transformed[1].text == "bad text"
    assert transformed[1].metadata["refine_fallback_reason"] == "chunk_error:ValueError"
    assert transformed[2].text == "ok"


def test_prompt_loading_appends_placeholder_when_missing(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Refine this chunk please.", encoding="utf-8")

    refiner = ChunkRefiner(_make_settings(use_llm=False), prompt_path=prompt_path)

    assert "{text}" in refiner._prompt_template


def test_prompt_loading_uses_default_when_missing_file(tmp_path: Path) -> None:
    prompt_path = tmp_path / "missing_prompt.txt"
    refiner = ChunkRefiner(_make_settings(use_llm=False), prompt_path=prompt_path)

    assert "{text}" in refiner._prompt_template
    assert "Do not invent information" in refiner._prompt_template


def test_trace_records_each_chunk_stage() -> None:
    refiner = ChunkRefiner(_make_settings(use_llm=False))
    trace = TraceContext(trace_type="ingestion")
    chunks = [_chunk("a", 0), _chunk("b", 1), _chunk("c", 2)]

    _ = refiner.transform(chunks, trace=trace)

    assert len(trace.stages) == 3
    assert all(stage["stage"] == "chunk_refine" for stage in trace.stages)


def test_transform_validates_input_type() -> None:
    refiner = ChunkRefiner(_make_settings(use_llm=False))

    with pytest.raises(ValueError, match="expects chunks as a list"):
        refiner.transform("not-list")  # type: ignore[arg-type]


def test_transform_validates_chunk_item_type() -> None:
    refiner = ChunkRefiner(_make_settings(use_llm=False))

    with pytest.raises(ValueError, match="must be Chunk"):
        refiner.transform([_chunk("ok"), "bad-item"])  # type: ignore[list-item]
