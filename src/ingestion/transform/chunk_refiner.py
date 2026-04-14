"""Chunk refinement transform with rule-based cleanup and optional LLM polish."""

from __future__ import annotations

import re
from pathlib import Path

from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory

_DEFAULT_PROMPT_PATH = Path("config/prompts/chunk_refinement.txt")
_DEFAULT_PROMPT_TEMPLATE = (
    "Refine the following chunk for semantic retrieval while preserving facts.\n"
    "Do not invent information.\n\n"
    "Text:\n{text}"
)
_CODE_BLOCK_RE = re.compile(r"(```[\s\S]*?```)")
_HTML_COMMENT_RE = re.compile(r"<!--[\s\S]*?-->")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_HEADER_FOOTER_PATTERNS = (
    re.compile(r"^page\s*\d+(\s*of\s*\d+)?$", re.IGNORECASE),
    re.compile(r"^第\s*\d+\s*页$"),
    re.compile(r"^[-_=]{3,}$"),
)


class ChunkRefiner(BaseTransform):
    """Refine noisy chunks via rules first, then optional LLM enhancement."""

    def __init__(
        self,
        settings: object,
        llm: BaseLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        ingestion_settings = getattr(settings, "ingestion", None)
        refiner_settings = getattr(ingestion_settings, "chunk_refiner", None)
        self._use_llm = bool(getattr(refiner_settings, "use_llm", False))
        self._prompt_template = self._load_prompt(prompt_path)

        if self._use_llm and llm is None:
            try:
                llm = LLMFactory.create(settings)
            except Exception:
                llm = None
        self._llm = llm

    def transform(
        self, chunks: list[Chunk], trace: TraceContext | None = None
    ) -> list[Chunk]:
        if not isinstance(chunks, list):
            raise ValueError("ChunkRefiner.transform expects chunks as a list.")

        results: list[Chunk] = []
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise ValueError(f"ChunkRefiner.transform chunks[{index}] must be Chunk.")

            stage_start = trace.stage_timer() if trace is not None else None
            method = "rule"
            fallback_reason: str | None = None

            try:
                rule_text = self._rule_based_refine(chunk.text)
                refined_text = rule_text

                llm_text = self._llm_refine(rule_text, trace)
                if llm_text is not None:
                    refined_text = llm_text
                    method = "llm"
                elif self._use_llm:
                    fallback_reason = "llm_unavailable_or_failed"

                new_metadata = dict(chunk.metadata)
                new_metadata["refined_by"] = method
                if fallback_reason:
                    new_metadata["refine_fallback_reason"] = fallback_reason

                refined_chunk = Chunk(
                    id=chunk.id,
                    text=refined_text if refined_text.strip() else chunk.text,
                    metadata=new_metadata,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    source_ref=chunk.source_ref,
                )
            except Exception as exc:
                fallback_reason = f"chunk_error:{type(exc).__name__}"
                new_metadata = dict(chunk.metadata)
                new_metadata["refined_by"] = "rule"
                new_metadata["refine_fallback_reason"] = fallback_reason
                refined_chunk = Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=new_metadata,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    source_ref=chunk.source_ref,
                )
                method = "rule"

            if trace is not None:
                elapsed_ms = (
                    trace.stage_elapsed_ms(stage_start) if stage_start is not None else None
                )
                trace.record_stage(
                    "chunk_refine",
                    method=method,
                    details={"chunk_id": chunk.id, "chunk_index": index},
                    elapsed_ms=elapsed_ms,
                )
            results.append(refined_chunk)
        return results

    def _rule_based_refine(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return text

        sections = _CODE_BLOCK_RE.split(text)
        cleaned_parts: list[str] = []
        for section in sections:
            if not section:
                continue
            if section.startswith("```") and section.endswith("```"):
                cleaned_parts.append(section)
                continue

            section = _HTML_COMMENT_RE.sub(" ", section)
            lines = section.splitlines()
            normalized_lines: list[str] = []
            blank_pending = False
            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    if normalized_lines:
                        blank_pending = True
                    continue
                if self._is_header_footer_line(line):
                    continue

                line = _MULTI_SPACE_RE.sub(" ", line)
                if blank_pending and normalized_lines:
                    normalized_lines.append("")
                    blank_pending = False
                normalized_lines.append(line)

            cleaned = "\n".join(normalized_lines).strip()
            cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
            if cleaned:
                cleaned_parts.append(cleaned)

        refined = "\n\n".join(part for part in cleaned_parts if part.strip())
        refined = _MULTI_NEWLINE_RE.sub("\n\n", refined).strip()
        return refined if refined else text.strip()

    def _llm_refine(self, text: str, trace: TraceContext | None = None) -> str | None:
        if not self._use_llm or self._llm is None:
            return None

        prompt = self._prompt_template.replace("{text}", text)
        try:
            response = self._llm.chat([{"role": "user", "content": prompt}])
        except Exception as exc:
            if trace is not None:
                trace.record_stage(
                    "chunk_refine_fallback",
                    method="llm",
                    details={"reason": type(exc).__name__},
                )
            return None

        if not isinstance(response, str) or not response.strip():
            return None
        return response.strip()

    def _load_prompt(self, prompt_path: str | Path | None = None) -> str:
        candidate = Path(prompt_path) if prompt_path is not None else _DEFAULT_PROMPT_PATH
        if not candidate.exists():
            return _DEFAULT_PROMPT_TEMPLATE
        try:
            content = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            return _DEFAULT_PROMPT_TEMPLATE
        if not content:
            return _DEFAULT_PROMPT_TEMPLATE
        if "{text}" not in content:
            content = f"{content}\n\n{{text}}"
        return content

    @staticmethod
    def _is_header_footer_line(line: str) -> bool:
        lowered = line.strip().lower()
        for pattern in _HEADER_FOOTER_PATTERNS:
            if pattern.match(lowered):
                return True
        return False
