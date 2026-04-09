"""Unit tests for custom evaluator and evaluator factory."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory


class _FakeEvaluator(BaseEvaluator):
    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: object | None = None,
    ) -> dict[str, float]:
        del query, retrieved_ids, golden_ids, trace
        return {"mock": 1.0}


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    old_registry = dict(EvaluatorFactory._REGISTRY)
    EvaluatorFactory._REGISTRY.clear()
    yield
    EvaluatorFactory._REGISTRY = old_registry


def _make_settings(provider: str, metrics: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        evaluation=SimpleNamespace(provider=provider, metrics=metrics or ["hit_rate", "mrr"])
    )


def test_custom_evaluator_outputs_stable_metrics() -> None:
    evaluator = CustomEvaluator(metrics=["hit_rate", "mrr"])
    scores = evaluator.evaluate(
        query="what is x",
        retrieved_ids=["doc-3", "doc-7", "doc-2"],
        golden_ids=["doc-2", "doc-9"],
    )

    assert scores["hit_rate"] == 1.0
    assert math.isclose(scores["mrr"], 1.0 / 3.0)


def test_custom_evaluator_no_match_returns_zero_scores() -> None:
    evaluator = CustomEvaluator(metrics=["hit_rate", "mrr"])
    scores = evaluator.evaluate(
        query="what is x",
        retrieved_ids=["doc-3", "doc-7"],
        golden_ids=["doc-2", "doc-9"],
    )

    assert scores["hit_rate"] == 0.0
    assert scores["mrr"] == 0.0


def test_evaluator_factory_default_custom_provider() -> None:
    evaluator = EvaluatorFactory.create(
        _make_settings(provider="custom", metrics=["hit_rate", "mrr", "faithfulness"])
    )
    scores = evaluator.evaluate("q", ["a"], ["a"])

    assert isinstance(evaluator, CustomEvaluator)
    assert set(scores.keys()) == {"hit_rate", "mrr", "faithfulness"}


def test_evaluator_factory_routes_registered_provider() -> None:
    EvaluatorFactory.register("mock_eval", lambda _settings: _FakeEvaluator())
    evaluator = EvaluatorFactory.create(_make_settings(provider="mock_eval"))

    assert isinstance(evaluator, _FakeEvaluator)
    assert evaluator.evaluate("q", [], []) == {"mock": 1.0}


def test_evaluator_factory_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown evaluator provider"):
        EvaluatorFactory.create(_make_settings(provider="unknown"))


def test_evaluator_factory_requires_provider_field() -> None:
    settings = SimpleNamespace(evaluation=SimpleNamespace(provider="", metrics=["hit_rate"]))

    with pytest.raises(ValueError, match="settings\\.evaluation\\.provider"):
        EvaluatorFactory.create(settings)
