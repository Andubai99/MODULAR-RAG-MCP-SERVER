"""Contract tests for MetadataEnricher."""

from __future__ import annotations

import os
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
from ingestion.transform.metadata_enricher import MetadataEnricher
from libs.llm.base_llm import BaseLLM


class _FakeLLM(BaseLLM):
    def __init__(
        self, response: str = "", raise_error: Exception | None = None
    ) -> None:
        self._response = response
        self._raise_error = raise_error
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if self._raise_error is not None:
            raise self._raise_error
        return self._response


def _make_settings(
    use_llm: bool,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str = "dummy",
    base_url: str = "https://api.openai.com/v1",
) -> SimpleNamespace:
    return SimpleNamespace(
        llm=SimpleNamespace(
            provider=provider,
            model=model,
            deployment_name="",
            azure_endpoint="",
            api_version="",
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            max_tokens=256,
        ),
        ingestion=SimpleNamespace(
            metadata_enricher=SimpleNamespace(use_llm=use_llm),
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


def test_rule_mode_populates_required_metadata_fields() -> None:
    enricher = MetadataEnricher(_make_settings(use_llm=False))
    text = "Hybrid retrieval combines dense embeddings and sparse BM25 scoring."

    result = enricher.transform([_chunk(text)])[0]

    assert result.metadata["metadata_enriched_by"] == "rule"
    assert isinstance(result.metadata.get("title"), str) and result.metadata["title"].strip()
    assert isinstance(result.metadata.get("summary"), str) and result.metadata["summary"].strip()
    assert isinstance(result.metadata.get("tags"), list)
    assert result.metadata["tags"]
    assert all(isinstance(tag, str) and tag.strip() for tag in result.metadata["tags"])


def test_rule_mode_preserves_existing_metadata() -> None:
    chunk = Chunk(
        id="chunk-1",
        text="Document explains vector indexing internals.",
        metadata={
            "source_path": "docs/a.pdf",
            "section": "architecture",
            "chunk_index": 3,
        },
        start_offset=0,
        end_offset=48,
        source_ref="doc-arch",
    )
    enricher = MetadataEnricher(_make_settings(use_llm=False))

    result = enricher.transform([chunk])[0]

    assert result.metadata["source_path"] == "docs/a.pdf"
    assert result.metadata["section"] == "architecture"
    assert result.metadata["chunk_index"] == 3
    assert result.metadata["metadata_enriched_by"] == "rule"


def test_llm_mode_prefers_llm_json_metadata() -> None:
    fake_llm = _FakeLLM(
        response=(
            '{"title":"Hybrid Search Workflow",'
            '"summary":"Explains dense and sparse retrieval with fusion.",'
            '"tags":["retrieval","hybrid","rrf"]}'
        )
    )
    enricher = MetadataEnricher(_make_settings(use_llm=True), llm=fake_llm)

    result = enricher.transform(
        [_chunk("Dense + sparse retrieval pipelines are fused with RRF.")]
    )[0]

    assert result.metadata["metadata_enriched_by"] == "llm"
    assert result.metadata["title"] == "Hybrid Search Workflow"
    assert result.metadata["summary"] == "Explains dense and sparse retrieval with fusion."
    assert result.metadata["tags"] == ["retrieval", "hybrid", "rrf"]
    assert fake_llm.calls


def test_llm_mode_accepts_fenced_json_response() -> None:
    fake_llm = _FakeLLM(
        response=(
            "```json\n"
            '{"title":"Chunk Title","summary":"Chunk Summary","tags":["a","b","c"]}\n'
            "```"
        )
    )
    enricher = MetadataEnricher(_make_settings(use_llm=True), llm=fake_llm)

    result = enricher.transform([_chunk("Some content text here.")])[0]

    assert result.metadata["metadata_enriched_by"] == "llm"
    assert result.metadata["title"] == "Chunk Title"
    assert result.metadata["summary"] == "Chunk Summary"
    assert result.metadata["tags"] == ["a", "b", "c"]


def test_llm_failure_falls_back_to_rule_metadata() -> None:
    fake_llm = _FakeLLM(raise_error=RuntimeError("llm temporarily unavailable"))
    enricher = MetadataEnricher(_make_settings(use_llm=True), llm=fake_llm)

    result = enricher.transform([_chunk("Page title and content to summarize.")])[0]

    assert result.metadata["metadata_enriched_by"] == "rule"
    assert result.metadata["metadata_enrich_fallback_reason"] == "llm_unavailable_or_failed"
    assert isinstance(result.metadata["title"], str) and result.metadata["title"].strip()
    assert isinstance(result.metadata["summary"], str) and result.metadata["summary"].strip()
    assert isinstance(result.metadata["tags"], list) and result.metadata["tags"]


def test_invalid_llm_response_falls_back_to_rule_metadata() -> None:
    fake_llm = _FakeLLM(response="not-a-json-response")
    enricher = MetadataEnricher(_make_settings(use_llm=True), llm=fake_llm)

    result = enricher.transform([_chunk("Content for fallback validation.")])[0]

    assert result.metadata["metadata_enriched_by"] == "rule"
    assert result.metadata["metadata_enrich_fallback_reason"] == "llm_unavailable_or_failed"


def test_transform_validates_input_types() -> None:
    enricher = MetadataEnricher(_make_settings(use_llm=False))

    with pytest.raises(ValueError, match="expects chunks as a list"):
        enricher.transform("not-list")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must be Chunk"):
        enricher.transform([_chunk("ok"), "bad-item"])  # type: ignore[list-item]


def test_trace_records_stage_for_each_chunk() -> None:
    enricher = MetadataEnricher(_make_settings(use_llm=False))
    trace = TraceContext(trace_type="ingestion")
    chunks = [_chunk("alpha", 0), _chunk("beta", 1)]

    _ = enricher.transform(chunks, trace=trace)

    assert len(trace.stages) == 2
    assert all(stage["stage"] == "metadata_enrich" for stage in trace.stages)


def test_metadata_enricher_real_llm_optional() -> None:
    if os.getenv("RUN_REAL_LLM_TESTS", "0") != "1":
        pytest.skip("Set RUN_REAL_LLM_TESTS=1 to run real LLM metadata enrichment test.")

    settings = _make_settings(
        use_llm=True,
        provider=os.getenv("REFINER_LLM_PROVIDER", "openai").strip().lower(),
        model=os.getenv("REFINER_LLM_MODEL", "moonshot-v1-8k"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("REFINER_LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    enricher = MetadataEnricher(settings)
    chunk = _chunk(
        "Page 1\n\nHybrid retrieval combines vector search with BM25 for better recall.\n\nPage 1 of 10"
    )

    result = enricher.transform([chunk])[0]

    assert result.metadata["metadata_enriched_by"] == "llm"
    assert isinstance(result.metadata.get("title"), str) and result.metadata["title"].strip()
    assert isinstance(result.metadata.get("summary"), str) and result.metadata["summary"].strip()
    assert isinstance(result.metadata.get("tags"), list) and result.metadata["tags"]
