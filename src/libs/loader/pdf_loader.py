"""PDF loader implementation with core contract normalization."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping

from core.types import Document, IMAGE_PLACEHOLDER_TEMPLATE
from libs.loader.base_loader import BaseLoader

ParserOutput = str | tuple[str, list[dict[str, object]]] | Mapping[str, object]
PdfParser = Callable[[Path], ParserOutput]


class PdfLoader(BaseLoader):
    """Load PDF files into canonical Document objects."""

    def __init__(self, parser: PdfParser | None = None) -> None:
        self._parser = parser or self._default_parse_pdf

    def load(self, path: str) -> Document:
        # 验证输入路径的有效性，确保文件存在且是 PDF 格式。然后调用解析器获取原始输出，
        # 并通过 _coerce_parser_output 方法将其转换为标准化的文本和图像列表。
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            raise ValueError(f"PdfLoader: file not found: {file_path.as_posix()}")
        if file_path.suffix.lower() != ".pdf":
            raise ValueError("PdfLoader: only .pdf files are supported.")

        parsed = self._coerce_parser_output(self._parser(file_path))
        text = self._normalize_text(parsed["text"])
        doc_hash = self._compute_sha256(file_path)
        images = self._normalize_images(parsed.get("images"), text, doc_hash)
        doc_id = f"doc_{doc_hash[:16]}"

        metadata: dict[str, object] = {
            "source_path": file_path.as_posix(),
            "doc_type": "pdf",
        }
        if images:
            metadata["images"] = images
        else:
            metadata["images"] = []
        return Document(id=doc_id, text=text, metadata=metadata)

    @staticmethod
    def _default_parse_pdf(path: Path) -> Mapping[str, object]:
        # 默认的 PDF 解析器实现使用 pypdf 库来提取文本内容。它会遍历 PDF 的每一页，提取文本并将非空文本块连接起来形成最终的文档文本。
        # 如果解析过程中发生任何错误（例如文件损坏或格式不兼容），它会捕获异常并抛出一个更具描述性的 RuntimeError。最终返回一个包含文本和空图像列表的字典。
        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "PdfLoader: default PDF parser requires 'pypdf'. "
                "Install pypdf or inject a custom parser."
            ) from exc

        try:
            reader = PdfReader(path.as_posix())
            page_texts = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    page_texts.append(text.strip())
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"PdfLoader: failed to parse PDF '{path.as_posix()}': {type(exc).__name__}"
            ) from exc

        joined = "\n\n".join(page_texts).strip()
        if not joined:
            raise ValueError(f"PdfLoader: parsed empty text from '{path.as_posix()}'.")
        return {"text": joined, "images": []}

    @staticmethod
    def _coerce_parser_output(output: ParserOutput) -> dict[str, object]:
        # _coerce_parser_output 方法的实现是：
        # 首先检查解析器输出的类型，如果是字符串，则将其视为文本并返回一个包含文本和空图像列表的字典；
        # 如果是一个包含两个元素的元组，则将第一个元素视为文本，第二个元素视为图像列表，并返回相应的字典；
        # 如果是一个映射对象，则从中提取 "text" 和 "images" 字段，提供默认值，并返回一个包含这些字段的字典；
        # 如果输出不符合上述任何一种格式，则抛出一个 ValueError，指示解析器输出的格式无效。
        if isinstance(output, str):
            return {"text": output, "images": []}
        if isinstance(output, tuple) and len(output) == 2:
            text, images = output
            return {"text": text, "images": images}
        if isinstance(output, Mapping):
            text = output.get("text", "")
            images = output.get("images", [])
            return {"text": text, "images": images}
        raise ValueError("PdfLoader: parser output must be str, (text, images), or mapping.")

    @staticmethod
    def _normalize_text(text: object) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("PdfLoader: parsed text must be a non-empty string.")
        return text.strip()

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        # _compute_sha256 方法的实现是：使用 hashlib 库创建一个 SHA-256 哈希对象，然后以二进制模式打开指定路径的文件，
        # 并逐块读取文件内容（每块大小为 1MB），将每块数据更新到哈希对象中。最后返回计算完成的哈希值的十六进制字符串表示。
        hasher = hashlib.sha256()
        with path.open("rb") as reader:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _normalize_images(
        # _normalize_images 方法的实现是：首先检查输入的 raw_images 是否为 None，如果是，则返回一个空列表。
        # 然后验证 raw_images 是否为一个列表，如果不是，则抛出一个 ValueError。
        raw_images: object, text: str, doc_hash: str
    ) -> list[dict[str, object]]:
        if raw_images is None:
            return []
        if not isinstance(raw_images, list):
            raise ValueError("PdfLoader: images must be a list when provided.")

        normalized: list[dict[str, object]] = []
        for index, item in enumerate(raw_images):
            if not isinstance(item, Mapping):
                raise ValueError(f"PdfLoader: images[{index}] must be an object.")

            image_id = item.get("id")
            if not isinstance(image_id, str) or not image_id.strip():
                image_id = f"{doc_hash}_{index}"
            image_id = image_id.strip()

            placeholder = IMAGE_PLACEHOLDER_TEMPLATE.format(image_id=image_id)
            text_offset = item.get("text_offset")
            if isinstance(text_offset, bool) or not isinstance(text_offset, int) or text_offset < 0:
                text_offset = text.find(placeholder)
                if text_offset < 0:
                    text_offset = 0

            text_length = item.get("text_length")
            if isinstance(text_length, bool) or not isinstance(text_length, int) or text_length < 0:
                text_length = len(placeholder)

            path = item.get("path")
            if not isinstance(path, str) or not path.strip():
                path = f"data/images/{doc_hash}/{image_id}.png"

            record: dict[str, object] = {
                "id": image_id,
                "path": path.strip(),
                "text_offset": int(text_offset),
                "text_length": int(text_length),
            }
            page = item.get("page")
            if isinstance(page, int) and not isinstance(page, bool) and page >= 0:
                record["page"] = page
            position = item.get("position")
            if isinstance(position, Mapping):
                record["position"] = dict(position)
            normalized.append(record)
        return normalized
