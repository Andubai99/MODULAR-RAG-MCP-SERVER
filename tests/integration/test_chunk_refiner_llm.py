"""Integration tests for ChunkRefiner with optional real LLM calls."""

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

from core.types import Chunk
from ingestion.transform.chunk_refiner import ChunkRefiner
from libs.llm.base_llm import BaseLLM

pytestmark = pytest.mark.integration


class _FailingLLM(BaseLLM):
    def chat(self, messages: list[dict[str, str]]) -> str:
        del messages
        raise RuntimeError("forced failure")


def _make_settings_for_real_llm() -> SimpleNamespace:
    provider = os.getenv("REFINER_LLM_PROVIDER", "openai").strip().lower()
    model = os.getenv("REFINER_LLM_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("REFINER_LLM_BASE_URL", "https://api.openai.com/v1")

    return SimpleNamespace(
        llm=SimpleNamespace(
            provider=provider,
            model=model,
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", ""),
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            max_tokens=256,
        ),
        ingestion=SimpleNamespace(chunk_refiner=SimpleNamespace(use_llm=True)),
    )


def _chunk(text: str) -> Chunk:
    return Chunk(
        id="chunk-real-1",
        text=text,
        metadata={"source_path": "tests/fixtures/sample_documents/sample.pdf"},
        start_offset=0,
        end_offset=max(1, len(text)),
        source_ref="doc-real",
    )


def test_chunk_refiner_real_llm_refinement_optional() -> None:
    if os.getenv("RUN_REAL_LLM_TESTS", "0") != "1":
        pytest.skip("Set RUN_REAL_LLM_TESTS=1 to run real LLM integration test.")

    settings = _make_settings_for_real_llm()
    refiner = ChunkRefiner(settings)
    noisy_text = "Page 1\n\nThis    paragraph has   noisy spacing.\n\nPage 1 of 10"

    result = refiner.transform([_chunk(noisy_text)])

    assert len(result) == 1
    assert result[0].metadata["refined_by"] == "llm"
    assert isinstance(result[0].text, str) and result[0].text.strip()


def test_chunk_refiner_fallback_when_llm_fails_in_integration_path() -> None:
    settings = _make_settings_for_real_llm()
    refiner = ChunkRefiner(settings, llm=_FailingLLM())
    noisy_text = "Page 2\n\nSome text.\n\nPage 2 of 9"

    result = refiner.transform([_chunk(noisy_text)])

    assert len(result) == 1
    assert result[0].metadata["refined_by"] == "rule"
    assert result[0].metadata["refine_fallback_reason"] == "llm_unavailable_or_failed"
