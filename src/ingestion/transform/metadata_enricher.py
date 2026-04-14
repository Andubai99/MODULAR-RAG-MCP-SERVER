"""Metadata enrichment transform with rule-based defaults and optional LLM refinement."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory

_DEFAULT_PROMPT_PATH = Path("config/prompts/metadata_enrichment.txt")
_DEFAULT_PROMPT_TEMPLATE = (
# 这个默认提示模板提供了清晰的指导，帮助LLM生成符合要求的元数据，同时也提供了基于规则的提示供LLM参考和改进。
# 用户可以通过提供自定义提示文件来覆盖默认模板，以适应特定的领域或风格需求。
    "Generate retrieval metadata for the following chunk.\n"
    "Return JSON only with keys: title, summary, tags.\n"
    "- title: concise and factual\n"
    "- summary: 1-2 sentences, preserve facts\n"
    "- tags: 3-6 short topical tags\n"
    "Do not invent information.\n\n"
    "Text:\n{text}\n\n"
    "Rule Hints:\n"
    "title_hint: {title}\n"
    "summary_hint: {summary}\n"
    "tags_hint: {tags}"
)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)# 匹配被```json```或```包围的JSON对象，非贪婪模式捕获内容
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}")# 匹配长度至少为3的英文单词（以字母开头，后续可以包含字母、数字、下划线或连字符）或长度至少为2的中文词语。
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")# 匹配连续的空格或制表符，用于文本规范化，将多个空格或制表符替换为单个空格。
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "into",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "but",
    "not",
    "you",
    "your",
    "about",
    "into",
    "onto",
    "then",
    "than",
    "them",
    "they",
    "their",
    "will",
    "shall",
    "can",
    "could",
    "would",
    "should",
}


class MetadataEnricher(BaseTransform):
    """Add stable metadata fields (title/summary/tags) to each chunk."""

    def __init__(
        self,
        settings: object,
        llm: BaseLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        ingestion_settings = getattr(settings, "ingestion", None)
        enricher_settings = getattr(ingestion_settings, "metadata_enricher", None)
        self._use_llm = bool(getattr(enricher_settings, "use_llm", False))
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
        # 这个方法是MetadataEnricher的核心，负责对输入的文本块进行元数据丰富处理。它首先验证输入是否为Chunk列表，然后逐个处理每个Chunk。
        if not isinstance(chunks, list):
            raise ValueError("MetadataEnricher.transform expects chunks as a list.")

        results: list[Chunk] = []
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise ValueError(
                    f"MetadataEnricher.transform chunks[{index}] must be Chunk."
                )

            stage_start = trace.stage_timer() if trace is not None else None
            method = "rule"
            fallback_reason: str | None = None

            try:
                rule_metadata = self._rule_based_metadata(chunk.text)
                enriched_metadata = rule_metadata

                llm_metadata = self._llm_enrich(chunk.text, rule_metadata, trace)
                if llm_metadata is not None:
                    enriched_metadata = llm_metadata
                    method = "llm"
                elif self._use_llm:
                    fallback_reason = "llm_unavailable_or_failed"

                new_metadata = dict(chunk.metadata)
                new_metadata.update(enriched_metadata)
                new_metadata["metadata_enriched_by"] = method
                if fallback_reason:
                    new_metadata["metadata_enrich_fallback_reason"] = fallback_reason

                enriched_chunk = Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=new_metadata,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    source_ref=chunk.source_ref,
                )
            except Exception as exc:
                fallback_reason = f"chunk_error:{type(exc).__name__}"
                safe_rule_metadata = self._safe_rule_metadata(chunk.text)
                new_metadata = dict(chunk.metadata)
                new_metadata.update(safe_rule_metadata)
                new_metadata["metadata_enriched_by"] = "rule"
                new_metadata["metadata_enrich_fallback_reason"] = fallback_reason
                enriched_chunk = Chunk(
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
                    "metadata_enrich",
                    method=method,
                    details={"chunk_id": chunk.id, "chunk_index": index},
                    elapsed_ms=elapsed_ms,
                )
            results.append(enriched_chunk)
        return results

    def _rule_based_metadata(self, text: str) -> dict[str, object]:
    # 这个方法实现了基于规则的元数据提取逻辑。它首先对输入文本进行规范化处理，然后通过一系列启发式规则来构建标题、摘要和标签。
    # 这些规则包括提取第一行作为标题、使用标点符号分割句子来生成摘要，以及从文本中提取候选标签。最终返回一个包含生成的标题、摘要和标签的字典。
        normalized = self._normalize_text(text)
        title = self._build_title(normalized)
        summary = self._build_summary(normalized, title)
        tags = self._build_tags(normalized, title)
        return {"title": title, "summary": summary, "tags": tags}

    def _llm_enrich(
        self,
        text: str,
        rule_metadata: Mapping[str, object],
        trace: TraceContext | None = None,
    ) -> dict[str, object] | None:
        # 这个方法负责使用LLM对基于规则的元数据进行进一步丰富。它首先检查是否启用了LLM以及LLM实例是否可用。如果条件不满足，则直接返回None。
        # 如果LLM可用，它会构建一个提示，将原始文本和基于规则的元数据作为提示的一部分提供给LLM。
        # 然后，它调用LLM的chat方法获取响应，并尝试解析LLM返回的JSON格式的元数据。如果解析成功，返回合并后的元数据；如果失败，则记录相应的失败原因并返回None。
        if not self._use_llm or self._llm is None:
            return None

        prompt = (
            self._prompt_template.replace("{text}", text)
            .replace("{title}", str(rule_metadata.get("title", "")))
            .replace("{summary}", str(rule_metadata.get("summary", "")))
            .replace("{tags}", ", ".join(self._normalize_tags(rule_metadata.get("tags"))))
        )
        try:
            response = self._llm.chat([{"role": "user", "content": prompt}])
        except Exception as exc:
            if trace is not None:
                trace.record_stage(
                    "metadata_enrich_fallback",
                    method="llm",
                    details={"reason": type(exc).__name__},
                )
            return None

        parsed = self._parse_llm_metadata(response)
        if parsed is None:
            if trace is not None:
                trace.record_stage(
                    "metadata_enrich_fallback",
                    method="llm",
                    details={"reason": "invalid_llm_response"},
                )
            return None

        merged = dict(rule_metadata)
        merged.update(parsed)
        merged["tags"] = self._normalize_tags(merged.get("tags"))
        return merged

    def _parse_llm_metadata(self, response: object) -> dict[str, object] | None:
        # 这个方法尝试从LLM的响应中提取和解析元数据。它首先验证响应是否为非空字符串，然后使用正则表达式尝试从响应中提取JSON对象。
        # 如果成功提取并解析出一个字典对象，它会进一步规范化标题、摘要和标签，并返回一个包含这些字段的字典。
        # 如果任何步骤失败，都会返回None，表示无法从LLM响应中提取有效的元数据。
        if not isinstance(response, str) or not response.strip():
            return None

        parsed_obj = self._extract_json_object(response)
        if parsed_obj is None:
            return None

        normalized: dict[str, object] = {}
        title = parsed_obj.get("title")
        summary = parsed_obj.get("summary")
        tags = parsed_obj.get("tags")

        if isinstance(title, str) and title.strip():
            normalized["title"] = self._truncate(title.strip(), 90)
        if isinstance(summary, str) and summary.strip():
            normalized["summary"] = self._truncate(summary.strip(), 260)

        normalized_tags = self._normalize_tags(tags)
        if normalized_tags:
            normalized["tags"] = normalized_tags

        if not normalized:
            return None
        return normalized

    def _load_prompt(self, prompt_path: str | Path | None = None) -> str:
        # 这个方法负责加载用于LLM提示的模板。它首先确定要使用的提示文件路径，如果用户提供了自定义路径则使用它，否则使用默认路径。
        # 然后，它检查该路径是否存在，如果不存在则返回内置的默认提示模板。如果文件存在，它尝试读取文件内容并进行基本的验证（如检查是否包含{text}占位符）。
        # 如果读取或验证失败，也会返回默认提示模板。最终返回一个有效的提示模板字符串供LLM使用。
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

    def _safe_rule_metadata(self, text: str) -> dict[str, object]:
        try:
            return self._rule_based_metadata(text)
        except Exception:
            fallback_text = text.strip() or "No content"
            return {
                "title": self._truncate(fallback_text, 80),
                "summary": self._truncate(fallback_text, 260),
                "tags": ["chunk"],
            }

    @staticmethod
    def _normalize_text(text: str) -> str:
        # 这个方法对输入文本进行规范化处理，主要通过去除多余的空白字符和行来简化文本结构。它首先将文本按行分割，并去除每行的前后空白，同时过滤掉完全为空的行。
        # 然后，它将剩余的行重新连接成一个单一的字符串，并使用正则表达式将连续的空格或制表符替换为单个空格。
        # 最终返回一个更紧凑、规范化的文本字符串，供后续的元数据提取使用。
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        joined = "\n".join(lines).strip()
        joined = _MULTI_SPACE_RE.sub(" ", joined)
        return joined or text.strip()

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        # 这个方法用于将输入文本截断到指定的最大字符数。它首先去除文本两端的空白字符，然后检查文本长度是否超过最大限制。
        # 如果文本长度在限制范围内，则直接返回规范化后的文本；如果超过限制，则将文本截断到最大字符数减去3的位置，
        # 并在末尾添加省略号（...）以指示文本被截断。最终返回一个符合长度要求的文本字符串。
        trimmed = text.strip()
        if len(trimmed) <= max_chars:
            return trimmed
        return f"{trimmed[: max_chars - 3].rstrip()}..."

    def _build_title(self, text: str) -> str:
        # 这个方法负责从输入文本中构建一个简洁且具有事实性的标题。它首先检查文本是否为空，如果是，则返回一个默认的标题"Untitled Chunk"。
        # 如果文本不为空，它会提取文本的第一行，并使用正则表达式去除常见的标题前缀（如#、>、-、*、数字等）。如果处理后的第一行仍然为空，则使用整个文本作为标题的候选。
        # 如果候选标题超过90个字符，则进一步使用句子分割来尝试提取更短的标题。最后，它将标题截断到90个字符，并返回一个非空的标题字符串。
        if not text:
            return "Untitled Chunk"

        first_line = text.splitlines()[0].strip()
        first_line = re.sub(r"^[#>\-\*\d\.\)\s]+", "", first_line).strip()
        if not first_line:
            first_line = text.replace("\n", " ").strip()

        if len(first_line) > 90:
            sentence = self._split_sentences(first_line)
            if sentence:
                first_line = sentence[0]
        title = self._truncate(first_line, 90)
        return title if title else "Untitled Chunk"

    def _build_summary(self, text: str, title: str) -> str:
        # 这个方法负责从输入文本中构建一个1-2句的摘要，尽可能保留事实信息。它首先将文本转换为一个紧凑的单行字符串，并使用正则表达式将连续的空格替换为单个空格。
        # 然后，它使用句子分割方法将文本分割成句子，并尝试将前两句组合成摘要。如果文本中没有足够的句子，则使用整个文本作为摘要的候选。
        # 最后，它将摘要截断到260个字符，并返回一个非空的摘要字符串，如果摘要为空则返回标题作为摘要。
        compact = text.replace("\n", " ").strip()
        compact = _MULTI_SPACE_RE.sub(" ", compact)
        sentences = self._split_sentences(compact)
        if len(sentences) >= 2:
            summary = f"{sentences[0]} {sentences[1]}"
        elif sentences:
            summary = sentences[0]
        else:
            summary = compact
        summary = self._truncate(summary, 260)
        return summary if summary else title

    def _build_tags(self, text: str, title: str) -> list[str]:
        # 这个方法负责从输入文本中提取3-6个简短的主题标签。它首先尝试从文本中提取候选标签，如果没有找到合适的标签，则尝试从标题中提取标签。
        # 如果仍然没有找到标签，则返回一个默认标签列表["chunk"]。最终返回一个包含提取到的标签的列表，最多包含6个标签。
        # 标签提取过程中会进行规范化处理，如去除停用词、数字和重复项，以确保标签的质量和相关性。
        candidates = self._extract_candidate_tags(text)
        if not candidates:
            candidates = self._extract_candidate_tags(title)
        if not candidates:
            return ["chunk"]
        return candidates[:6]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        if not text:
            return []
        parts = re.split(r"(?<=[。！？.!?])\s+", text)
        return [part.strip() for part in parts if part.strip()]

    def _extract_candidate_tags(self, text: str) -> list[str]:
        tags: list[str] = []
        seen: set[str] = set()
        for token in _TOKEN_RE.findall(text):
            normalized = token.strip().lower()
            if not normalized or normalized in _STOPWORDS:
                continue
            if normalized.isdigit():
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            tags.append(normalized)
        return tags

    def _normalize_tags(self, value: object) -> list[str]:
        if isinstance(value, str):
            candidates = [part.strip() for part in value.split(",")]
        elif isinstance(value, list):
            candidates = [str(item).strip() for item in value]
        else:
            candidates = []

        normalized: list[str] = []
        seen: set[str] = set()
        for tag in candidates:
            if not tag:
                continue
            lowered = tag.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(lowered)
        return normalized[:6]

    @staticmethod
    def _extract_json_object(response: str) -> dict[str, Any] | None:
        candidates: list[str] = []
        for match in _FENCED_JSON_RE.findall(response):
            candidate = match.strip()
            if candidate:
                candidates.append(candidate)

        stripped = response.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidates.append(stripped)

        brace_start = stripped.find("{")
        brace_end = stripped.rfind("}")
        if 0 <= brace_start < brace_end:
            candidates.append(stripped[brace_start : brace_end + 1])

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None


