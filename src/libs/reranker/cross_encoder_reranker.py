"""Cross-encoder reranker placeholder with deterministic scoring and fallback."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

from libs.reranker.base_reranker import BaseReranker

Scorer = Callable[[str, list[dict[str, str]]], list[float]]


class CrossEncoderReranker(BaseReranker):
    """Reranker that scores top-M candidates and returns sorted results."""

    provider_name = "cross_encoder"

    def __init__(self, settings: object, scorer: Scorer | None = None) -> None:
        self.last_fallback_reason: str | None = None
        self._top_m = self._read_top_m(settings)
        self._scorer = scorer or self._default_score

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, object]],
        trace: object | None = None,
    ) -> list[dict[str, object]]:
        del trace
        normalized_query = self._normalize_query(query)
        normalized_candidates = self._normalize_candidates(candidates)
        top_m = min(self._top_m, len(normalized_candidates))
        head = normalized_candidates[:top_m]
        tail = normalized_candidates[top_m:]

        try:
            scores = self._scorer(normalized_query, head)
            self._validate_scores(scores, len(head))
        except Exception as exc:  # pragma: no cover - backend scorer errors are external.
            reason = f"cross_encoder_error:{type(exc).__name__}"
            self.last_fallback_reason = reason
            return self._mark_fallback(candidates, reason)

        scored_rows = []
        for index, (candidate, score) in enumerate(zip(head, scores)):
            scored_rows.append((float(score), index, candidate))
        scored_rows.sort(key=lambda row: (-row[0], row[1]))

        ordered_ids = [row[2]["id"] for row in scored_rows] + [row["id"] for row in tail]
        self.last_fallback_reason = None
        return self._apply_order(candidates, ordered_ids)

    @staticmethod
    def _read_top_m(settings: object) -> int:
        rerank_settings = getattr(settings, "rerank", None)
        top_m = getattr(rerank_settings, "top_k", 5)
        if isinstance(top_m, bool) or not isinstance(top_m, int) or top_m <= 0:
            raise ValueError("provider=cross_encoder: settings.rerank.top_k must be a positive integer.")
        return top_m

    @staticmethod
    def _normalize_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("provider=cross_encoder: query must be a non-empty string.")
        return query.strip()

    @staticmethod
    def _normalize_candidates(
        candidates: list[dict[str, object]],
    ) -> list[dict[str, str]]:
        if not isinstance(candidates, list):
            raise ValueError("provider=cross_encoder: candidates must be a list.")

        normalized: list[dict[str, str]] = []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ValueError(
                    f"provider=cross_encoder: candidates[{index}] must be an object."
                )
            candidate_id = candidate.get("id")
            if candidate_id is None:
                candidate_id = candidate.get("chunk_id")
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                raise ValueError(
                    f"provider=cross_encoder: candidates[{index}] must contain non-empty id/chunk_id."
                )
            text = candidate.get("text")
            if text is None:
                text = candidate.get("content", "")
            if not isinstance(text, str):
                text = str(text)
            normalized.append({"id": candidate_id.strip(), "text": text})
        return normalized

    @staticmethod
    def _default_score(query: str, candidates: list[dict[str, str]]) -> list[float]:
        query_tokens = set(re.findall(r"\w+", query.lower()))
        if not query_tokens:
            return [0.0 for _ in candidates]

        scores: list[float] = []
        for candidate in candidates:
            text_tokens = set(re.findall(r"\w+", candidate["text"].lower()))
            overlap = len(query_tokens & text_tokens)
            scores.append(float(overlap))
        return scores

    @staticmethod
    def _validate_scores(scores: list[float], expected_size: int) -> None:
        if not isinstance(scores, list):
            raise ValueError("provider=cross_encoder: scorer must return a score list.")
        if len(scores) != expected_size:
            raise ValueError(
                "provider=cross_encoder: scorer returned invalid score length."
            )
        if not all(isinstance(score, (int, float)) and not isinstance(score, bool) for score in scores):
            raise ValueError("provider=cross_encoder: scorer returned non-numeric score.")

    @staticmethod
    def _apply_order(
        candidates: list[dict[str, object]], ordered_ids: list[str]
    ) -> list[dict[str, object]]:
        by_id: dict[str, dict[str, object]] = {}
        for candidate in candidates:
            candidate_id = candidate.get("id")
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                candidate_id = candidate.get("chunk_id")
            if isinstance(candidate_id, str) and candidate_id.strip():
                by_id[candidate_id.strip()] = dict(candidate)
        return [by_id[candidate_id] for candidate_id in ordered_ids]

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
