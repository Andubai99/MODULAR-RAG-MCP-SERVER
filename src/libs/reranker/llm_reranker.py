"""LLM reranker implementation with prompt loading and fallback signal."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory
from libs.reranker.base_reranker import BaseReranker

_DEFAULT_PROMPT_PATH = Path("config/prompts/rerank.txt")


class LLMReranker(BaseReranker):
    """Reranker that asks an LLM to output ranked candidate ids as JSON."""

    provider_name = "llm"

    def __init__(
        self,
        settings: object,
        llm_client: BaseLLM | None = None,
        prompt_template: str | None = None,
        prompt_path: str | Path = _DEFAULT_PROMPT_PATH,
    ) -> None:
        self.last_fallback_reason: str | None = None
        self._llm = llm_client or LLMFactory.create(self._build_llm_settings(settings))
        self._prompt_template = (
            prompt_template
            if prompt_template is not None
            else self._load_prompt_template(prompt_path)
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, object]],
        trace: object | None = None,
    ) -> list[dict[str, object]]:
        del trace
        normalized_query = self._normalize_query(query)
        normalized_candidates = self._normalize_candidates(candidates)
        candidate_ids = [item["id"] for item in normalized_candidates]
        prompt = self._build_prompt(normalized_query, normalized_candidates)

        try:
            response_text = self._llm.chat([{"role": "user", "content": prompt}])
        except Exception as exc:  # pragma: no cover - exact exception type is backend-specific.
            reason = f"llm_error:{type(exc).__name__}"
            self.last_fallback_reason = reason
            return self._mark_fallback(candidates, reason)

        ranked_ids = self._parse_ranked_ids(response_text, candidate_ids)
        self.last_fallback_reason = None
        return self._apply_ranked_ids(candidates, ranked_ids)

    def _build_prompt(
        self, query: str, candidates: list[dict[str, str]]
    ) -> str:
        payload = json.dumps(
            {"query": query, "candidates": candidates},
            ensure_ascii=False,
            indent=2,
        )
        return (
            f"{self._prompt_template.strip()}\n\n"
            "Input JSON:\n"
            f"{payload}\n\n"
            'Output JSON schema: {"ranked_ids": ["id1", "id2"]}'
        )

    @staticmethod
    def _normalize_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("provider=llm: query must be a non-empty string.")
        return query.strip()

    @staticmethod
    def _normalize_candidates(
        candidates: list[dict[str, object]],
    ) -> list[dict[str, str]]:
        if not isinstance(candidates, list):
            raise ValueError("provider=llm: candidates must be a list.")
        normalized: list[dict[str, str]] = []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ValueError(
                    f"provider=llm: candidates[{index}] must be an object."
                )
            candidate_id = candidate.get("id")
            if candidate_id is None:
                candidate_id = candidate.get("chunk_id")
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                raise ValueError(
                    f"provider=llm: candidates[{index}] must contain non-empty id/chunk_id."
                )
            text_value = candidate.get("text")
            if text_value is None:
                text_value = candidate.get("content", "")
            if not isinstance(text_value, str):
                text_value = str(text_value)
            normalized.append({"id": candidate_id.strip(), "text": text_value})
        return normalized

    @staticmethod
    def _extract_json_object(raw_text: str) -> dict[str, object]:
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError("provider=llm: empty rerank response from llm.")

        payload_text = raw_text.strip()
        try:
            parsed = json.loads(payload_text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", payload_text)
            if not match:
                raise ValueError("provider=llm: rerank response is not valid JSON.")
            parsed = json.loads(match.group(0))

        if not isinstance(parsed, dict):
            raise ValueError("provider=llm: rerank response root must be a JSON object.")
        return parsed

    def _parse_ranked_ids(
        self, response_text: str, candidate_ids: list[str]
    ) -> list[str]:
        parsed = self._extract_json_object(response_text)
        ranked_ids = parsed.get("ranked_ids")
        if not isinstance(ranked_ids, list) or not ranked_ids:
            raise ValueError(
                "provider=llm: rerank response must include non-empty ranked_ids list."
            )
        if not all(isinstance(item, str) and item.strip() for item in ranked_ids):
            raise ValueError("provider=llm: ranked_ids must contain non-empty strings.")

        normalized_ranked_ids = [item.strip() for item in ranked_ids]
        if len(set(normalized_ranked_ids)) != len(normalized_ranked_ids):
            raise ValueError("provider=llm: ranked_ids contains duplicate ids.")

        candidate_id_set = set(candidate_ids)
        unknown_ids = [item for item in normalized_ranked_ids if item not in candidate_id_set]
        if unknown_ids:
            unknown_text = ", ".join(unknown_ids)
            raise ValueError(
                f"provider=llm: ranked_ids contains unknown candidate id(s): {unknown_text}."
            )

        missing_ids = [item for item in candidate_ids if item not in set(normalized_ranked_ids)]
        return normalized_ranked_ids + missing_ids

    @staticmethod
    def _apply_ranked_ids(
        candidates: list[dict[str, object]], ranked_ids: list[str]
    ) -> list[dict[str, object]]:
        by_id: dict[str, dict[str, object]] = {}
        for candidate in candidates:
            candidate_id = candidate.get("id")
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                candidate_id = candidate.get("chunk_id")
            if isinstance(candidate_id, str) and candidate_id.strip():
                by_id[candidate_id.strip()] = dict(candidate)
        return [by_id[candidate_id] for candidate_id in ranked_ids]

    @staticmethod
    def _mark_fallback(
        candidates: list[dict[str, object]], reason: str
    ) -> list[dict[str, object]]:
        marked: list[dict[str, object]] = []
        for candidate in candidates:
            item = dict(candidate)
            item["rerank_fallback"] = True
            item["rerank_fallback_reason"] = reason
            marked.append(item)
        return marked

    @staticmethod
    def _load_prompt_template(prompt_path: str | Path) -> str:
        file_path = Path(prompt_path)
        if not file_path.exists():
            raise ValueError(
                f"provider=llm: rerank prompt file not found: {file_path.as_posix()}"
            )
        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(
                f"provider=llm: rerank prompt file is empty: {file_path.as_posix()}"
            )
        return content

    @staticmethod
    def _build_llm_settings(settings: object) -> object:
        llm_settings = getattr(settings, "llm", None)
        rerank_settings = getattr(settings, "rerank", None)

        provider = getattr(llm_settings, "provider", "")
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider=llm: settings.llm.provider must be a non-empty string.")

        base_model = getattr(llm_settings, "model", "")
        rerank_model = getattr(rerank_settings, "model", "")
        model = rerank_model if isinstance(rerank_model, str) and rerank_model.strip() else base_model
        if not isinstance(model, str) or not model.strip():
            raise ValueError("provider=llm: settings.rerank.model or settings.llm.model must be set.")

        llm_for_rerank = SimpleNamespace(
            provider=provider.strip(),
            model=model.strip(),
            deployment_name=getattr(llm_settings, "deployment_name", ""),
            azure_endpoint=getattr(llm_settings, "azure_endpoint", ""),
            api_version=getattr(llm_settings, "api_version", ""),
            api_key=getattr(llm_settings, "api_key", ""),
            base_url=getattr(llm_settings, "base_url", ""),
            temperature=getattr(llm_settings, "temperature", 0.0),
            max_tokens=getattr(llm_settings, "max_tokens", 1024),
        )
        return SimpleNamespace(llm=llm_for_rerank)
