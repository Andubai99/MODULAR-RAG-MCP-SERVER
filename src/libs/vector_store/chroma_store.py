"""Default Chroma-like vector store implementation with local persistence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from libs.vector_store.base_vector_store import BaseVectorStore, QueryResult, VectorRecord


class ChromaStore(BaseVectorStore):
    """A lightweight local vector store with Chroma-compatible role in architecture."""

    provider_name = "chroma"

    def __init__(self, settings: object) -> None:
        # 验证并提取必要的配置项
        vector_store_settings = getattr(settings, "vector_store", None)
        persist_directory = getattr(vector_store_settings, "persist_directory", "")
        collection_name = getattr(vector_store_settings, "collection_name", "")

        if not isinstance(persist_directory, str) or not persist_directory.strip():
            raise ValueError(
                "provider=chroma: settings.vector_store.persist_directory must be a non-empty string."
            )
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError(
                "provider=chroma: settings.vector_store.collection_name must be a non-empty string."
            )

        self._persist_dir = Path(persist_directory)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name.strip()
        self._store_path = self._persist_dir / f"{self._collection_name}.json"
        self._records_by_id: dict[str, VectorRecord] = {}
        self._load()

    def upsert(self, records: list[VectorRecord], trace: object | None = None) -> None:
        # 这里的 upsert 实现是：对于每个输入记录，如果它的 id 已经存在于当前存储中，则覆盖原有记录；
        # 如果 id 不存在，则添加新记录。最后将更新后的整个记录集合保存到磁盘。
        del trace
        if not isinstance(records, list):
            raise ValueError("provider=chroma: records must be a list.")
        for record in records:
            normalized = self._normalize_record(record)
            self._records_by_id[normalized["id"]] = normalized
        self._save()

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: Mapping[str, object] | None = None,
        trace: object | None = None,
    ) -> list[QueryResult]:
        # 这里的 query 实现是：首先验证输入参数的类型和有效性，然后从当前存储中获取所有记录，并根据可选的 filters 进行筛选；
        # 接着计算每个候选记录与查询向量之间的余弦相似度得分，并按照得分从高到低排序，最后返回 top_k 个得分最高的记录作为查询结果。
        del trace
        query_vector = self._normalize_vector(vector, "vector")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("provider=chroma: top_k must be a positive integer.")

        candidates = list(self._records_by_id.values())
        if filters is not None:
            if not isinstance(filters, Mapping):
                raise ValueError("provider=chroma: filters must be a mapping when provided.")
            candidates = [record for record in candidates if self._matches_filters(record, filters)]

        scored: list[tuple[float, VectorRecord]] = []
        for record in candidates:
            score = self._cosine_similarity(query_vector, record["vector"])
            scored.append((score, record))

        scored.sort(key=lambda item: (item[0], item[1]["id"]), reverse=True)
        selected = scored[:top_k]

        results: list[QueryResult] = []
        for score, record in selected:
            result: QueryResult = {
                "id": record["id"],
                "score": float(score),
                "metadata": dict(record.get("metadata", {})),
            }
            if "content" in record:
                result["content"] = str(record.get("content", ""))
            results.append(result)
        return results

    def _load(self) -> None:
        if not self._store_path.exists():
            return
        payload = json.loads(self._store_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("provider=chroma: persisted store payload must be a list.")
        for item in payload:
            normalized = self._normalize_record(item)
            self._records_by_id[normalized["id"]] = normalized

    def _save(self) -> None:
        payload = list(self._records_by_id.values())
        tmp_path = self._store_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self._store_path)

    @staticmethod
    def _normalize_record(raw_record: Any) -> VectorRecord:
        # 这里的 _normalize_record 实现是：首先验证输入记录的类型和必要字段的存在性，然后对 record.id 进行去除前后空白的处理，
        # 对 record.vector 进行数值化处理，并确保 record.metadata 是一个字典对象；
        if not isinstance(raw_record, Mapping):
            raise ValueError("provider=chroma: each record must be an object.")

        record_id = raw_record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("provider=chroma: record.id must be a non-empty string.")

        vector = ChromaStore._normalize_vector(raw_record.get("vector"), "record.vector")

        metadata = raw_record.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("provider=chroma: record.metadata must be an object.")

        normalized: VectorRecord = {
            "id": record_id.strip(),
            "vector": vector,
            "metadata": dict(metadata),
        }
        if "content" in raw_record and raw_record.get("content") is not None:
            normalized["content"] = str(raw_record.get("content"))
        return normalized

    @staticmethod
    def _normalize_vector(raw_vector: Any, field_name: str) -> list[float]:
        # 这里的 _normalize_vector 实现是：首先验证输入向量的类型必须是一个非空列表，并且列表中的每个元素都必须是一个数字（整数或浮点数，但不能是布尔值）；
        # 然后将列表中的每个元素转换为浮点数，并返回一个新的浮点数列表作为规范化后的向量。
        if not isinstance(raw_vector, list) or not raw_vector:
            raise ValueError(f"provider=chroma: {field_name} must be a non-empty numeric list.")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw_vector):
            raise ValueError(f"provider=chroma: {field_name} must be a non-empty numeric list.")
        return [float(v) for v in raw_vector]

    @staticmethod
    def _matches_filters(record: VectorRecord, filters: Mapping[str, object]) -> bool:
        # 这里的 _matches_filters 实现是：对于每个输入记录和过滤条件，首先获取记录的 metadata 字段（如果不存在则默认为空字典），
        # 然后检查 filters 中的每个键值对是否都存在于记录的 metadata 中，并且值完全匹配（使用 == 运算符）。
        # 如果所有过滤条件都满足，则返回 True；如果有任何一个过滤条件不满足，则返回 False。
        metadata = record.get("metadata", {})
        return all(metadata.get(key) == value for key, value in filters.items())

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        # 这里的 _cosine_similarity 实现是：首先验证输入的两个向量的类型必须是列表，并且它们的长度必须相等；
        # 然后计算两个向量之间的余弦相似度得分，具体步骤包括计算点积、计算每个向量的范数，并使用这些值来计算最终的相似度得分。
        # 如果任一向量的范数为零，则相似度得分定义为 0.0。
        if len(left) != len(right):
            raise ValueError(
                "provider=chroma: query vector dimension mismatch with stored vector."
            )
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)
