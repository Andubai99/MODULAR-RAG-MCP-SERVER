"""配置加载与校验。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class SettingsError(ValueError):
    """配置文件加载或校验失败。"""


@dataclass(slots=True)
class LLMSettings:
    provider: str
    model: str
    deployment_name: str
    azure_endpoint: str
    api_version: str
    api_key: str
    base_url: str
    temperature: float
    max_tokens: int


@dataclass(slots=True)
class EmbeddingSettings:
    provider: str
    model: str
    dimensions: int
    azure_endpoint: str
    deployment_name: str
    api_version: str
    api_key: str
    base_url: str


@dataclass(slots=True)
class VectorStoreSettings:
    provider: str
    persist_directory: str
    collection_name: str


@dataclass(slots=True)
class RetrievalSettings:
    dense_top_k: int
    sparse_top_k: int
    fusion_top_k: int
    rrf_k: int


@dataclass(slots=True)
class RerankSettings:
    enabled: bool
    provider: str
    model: str
    top_k: int


@dataclass(slots=True)
class EvaluationSettings:
    enabled: bool
    provider: str
    metrics: list[str]


@dataclass(slots=True)
class ObservabilitySettings:
    log_level: str
    trace_enabled: bool
    trace_file: str
    structured_logging: bool


@dataclass(slots=True)
class Settings:
    llm: LLMSettings
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings


def _require_dict(data: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise SettingsError(f"Missing required field: {path}.{key}")
    return value


def _require_str(data: dict[str, Any], key: str, path: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"Missing required field: {path}.{key}")
    return value


def _optional_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise SettingsError(f"Invalid type for field: {key}, expected string")
    return value


def _require_int(data: dict[str, Any], key: str, path: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SettingsError(f"Missing required field: {path}.{key}")
    return value


def _require_bool(data: dict[str, Any], key: str, path: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise SettingsError(f"Missing required field: {path}.{key}")
    return value


def _require_float(data: dict[str, Any], key: str, path: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SettingsError(f"Missing required field: {path}.{key}")
    return float(value)


def _require_str_list(data: dict[str, Any], key: str, path: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise SettingsError(f"Missing required field: {path}.{key}")
    return value


def validate_settings(settings: Settings) -> None:
    """执行最小值校验。"""
    if settings.embedding.dimensions <= 0:
        raise SettingsError("embedding.dimensions must be > 0")
    if settings.retrieval.dense_top_k <= 0:
        raise SettingsError("retrieval.dense_top_k must be > 0")
    if settings.retrieval.sparse_top_k <= 0:
        raise SettingsError("retrieval.sparse_top_k must be > 0")
    if settings.retrieval.fusion_top_k <= 0:
        raise SettingsError("retrieval.fusion_top_k must be > 0")
    if settings.rerank.top_k <= 0:
        raise SettingsError("rerank.top_k must be > 0")


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    """读取 YAML 并转换为 Settings。"""
    file_path = Path(path)
    if not file_path.exists():
        raise SettingsError(f"Settings file not found: {file_path}")

    try:
        raw_obj = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SettingsError(f"Invalid YAML: {exc}") from exc

    if not isinstance(raw_obj, dict):
        raise SettingsError("Invalid root object in settings.yaml")

    llm_raw = _require_dict(raw_obj, "llm", "root")
    embedding_raw = _require_dict(raw_obj, "embedding", "root")
    vector_store_raw = _require_dict(raw_obj, "vector_store", "root")
    retrieval_raw = _require_dict(raw_obj, "retrieval", "root")
    rerank_raw = _require_dict(raw_obj, "rerank", "root")
    evaluation_raw = _require_dict(raw_obj, "evaluation", "root")
    observability_raw = _require_dict(raw_obj, "observability", "root")

    settings = Settings(
        llm=LLMSettings(
            provider=_require_str(llm_raw, "provider", "llm"),
            model=_require_str(llm_raw, "model", "llm"),
            deployment_name=_optional_str(llm_raw, "deployment_name"),
            azure_endpoint=_optional_str(llm_raw, "azure_endpoint"),
            api_version=_optional_str(llm_raw, "api_version"),
            api_key=_optional_str(llm_raw, "api_key"),
            base_url=_optional_str(llm_raw, "base_url"),
            temperature=_require_float(llm_raw, "temperature", "llm"),
            max_tokens=_require_int(llm_raw, "max_tokens", "llm"),
        ),
        embedding=EmbeddingSettings(
            provider=_require_str(embedding_raw, "provider", "embedding"),
            model=_require_str(embedding_raw, "model", "embedding"),
            dimensions=_require_int(embedding_raw, "dimensions", "embedding"),
            azure_endpoint=_optional_str(embedding_raw, "azure_endpoint"),
            deployment_name=_optional_str(embedding_raw, "deployment_name"),
            api_version=_optional_str(embedding_raw, "api_version"),
            api_key=_optional_str(embedding_raw, "api_key"),
            base_url=_optional_str(embedding_raw, "base_url"),
        ),
        vector_store=VectorStoreSettings(
            provider=_require_str(vector_store_raw, "provider", "vector_store"),
            persist_directory=_require_str(
                vector_store_raw, "persist_directory", "vector_store"
            ),
            collection_name=_require_str(
                vector_store_raw, "collection_name", "vector_store"
            ),
        ),
        retrieval=RetrievalSettings(
            dense_top_k=_require_int(retrieval_raw, "dense_top_k", "retrieval"),
            sparse_top_k=_require_int(retrieval_raw, "sparse_top_k", "retrieval"),
            fusion_top_k=_require_int(retrieval_raw, "fusion_top_k", "retrieval"),
            rrf_k=_require_int(retrieval_raw, "rrf_k", "retrieval"),
        ),
        rerank=RerankSettings(
            enabled=_require_bool(rerank_raw, "enabled", "rerank"),
            provider=_require_str(rerank_raw, "provider", "rerank"),
            model=_optional_str(rerank_raw, "model"),
            top_k=_require_int(rerank_raw, "top_k", "rerank"),
        ),
        evaluation=EvaluationSettings(
            enabled=_require_bool(evaluation_raw, "enabled", "evaluation"),
            provider=_require_str(evaluation_raw, "provider", "evaluation"),
            metrics=_require_str_list(evaluation_raw, "metrics", "evaluation"),
        ),
        observability=ObservabilitySettings(
            log_level=_require_str(observability_raw, "log_level", "observability"),
            trace_enabled=_require_bool(
                observability_raw, "trace_enabled", "observability"
            ),
            trace_file=_require_str(observability_raw, "trace_file", "observability"),
            structured_logging=_require_bool(
                observability_raw, "structured_logging", "observability"
            ),
        ),
    )

    validate_settings(settings)
    return settings
