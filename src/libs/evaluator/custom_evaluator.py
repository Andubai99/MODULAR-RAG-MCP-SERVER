"""Lightweight custom evaluator metrics."""

from __future__ import annotations

from libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """Evaluate retrieval with lightweight custom metrics."""

    _SUPPORTED_METRICS = {"hit_rate", "mrr", "faithfulness"}

    def __init__(self, metrics: list[str] | None = None) -> None:
        configured = metrics or ["hit_rate", "mrr"]
        normalized = [metric.strip().lower() for metric in configured]
        invalid = [metric for metric in normalized if metric not in self._SUPPORTED_METRICS]
        if invalid:
            names = ", ".join(sorted(self._SUPPORTED_METRICS))
            invalid_names = ", ".join(invalid)
            raise ValueError(
                f"Unsupported custom evaluator metrics: {invalid_names}. "
                f"Supported metrics: {names}."
            )
        self._metrics = normalized

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: object | None = None,
    ) -> dict[str, float]:
        del query, trace
        relevant = set(golden_ids)
        scores: dict[str, float] = {}

        for metric in self._metrics:
            if metric == "hit_rate":
                scores[metric] = self._hit_rate(retrieved_ids, relevant)
            elif metric == "mrr":
                scores[metric] = self._mrr(retrieved_ids, relevant)
            elif metric == "faithfulness":
                # Placeholder until generation-level evaluation is integrated.
                scores[metric] = 0.0
        return scores

    @staticmethod
    def _hit_rate(retrieved_ids: list[str], relevant: set[str]) -> float:
        return 1.0 if any(item_id in relevant for item_id in retrieved_ids) else 0.0

    @staticmethod
    def _mrr(retrieved_ids: list[str], relevant: set[str]) -> float:
        for idx, item_id in enumerate(retrieved_ids, start=1):
            if item_id in relevant:
                return 1.0 / float(idx)
        return 0.0
