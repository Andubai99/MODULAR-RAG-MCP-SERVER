"""Default recursive markdown splitter implementation."""

from __future__ import annotations

from dataclasses import dataclass

from libs.splitter.base_splitter import BaseSplitter

"""
该文件实现一个递归文本切分器：
先识别并保护 Markdown 代码块，再按空行、换行、空格等分隔符逐层递归切分普通文本，
最后将小片段按块大小和重叠规则合并成适合后续处理的文本块。
也就是说这个切分对于普通文本来说，刚开始没有考虑重叠规则，先按照各种分隔符切成了非常碎的各种句子，最后再统一合并成了最终文本块 chunks, 并应用重叠规则。
如果 overlap + 新片段组成的新块仍然超长，且不含代码块，那么就对这个“新块整体”再硬切，而不是只保留 overlap。
例如current = "CD\nEFGH", overlap = "CD", 新片段 = "EFGH", candidate = "CD\nEFGH" 超长且不含代码块，那么就直接把 candidate 切成 "CD\nEFG" 和 "H" 两块，而不是保留 overlap "CD" 作为下一块的开头。
"""

@dataclass(slots=True)
class _ChunkConfig:
    chunk_size: int
    chunk_overlap: int


class RecursiveSplitter(BaseSplitter):
    """Recursive text splitter with markdown/code-block awareness."""

    _SEPARATORS = ("\n\n", "\n", " ", "")

    def __init__(self, settings: object) -> None:
        ingestion_settings = getattr(settings, "ingestion", None)
        chunk_size = int(getattr(ingestion_settings, "chunk_size", 1000))
        chunk_overlap = int(getattr(ingestion_settings, "chunk_overlap", 200))
        self._config = self._validate_config(chunk_size, chunk_overlap)

    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        del trace
        if not isinstance(text, str):
            raise ValueError("split_text expects `text` to be a string.")
        if not text.strip():
            return []

        pieces: list[str] = []
        for section, is_code_block in self._split_code_block_sections(text):
            if is_code_block:
                pieces.append(section)
                continue
            pieces.extend(self._split_recursively(section, 0))

        return self._merge_pieces(pieces)

    @staticmethod
    def _validate_config(chunk_size: int, chunk_overlap: int) -> _ChunkConfig:
        if chunk_size <= 0:
            raise ValueError("ingestion.chunk_size must be > 0.")
        if chunk_overlap < 0:
            raise ValueError("ingestion.chunk_overlap must be >= 0.")
        if chunk_overlap >= chunk_size:
            raise ValueError("ingestion.chunk_overlap must be < ingestion.chunk_size.")
        return _ChunkConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def _split_recursively(self, text: str, level: int) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self._config.chunk_size:
            return [text]

        separator = self._SEPARATORS[level]
        if separator == "":
            return [
                text[i : i + self._config.chunk_size]
                for i in range(0, len(text), self._config.chunk_size)
            ]

        segments = text.split(separator)
        if len(segments) == 1:
            return self._split_recursively(text, level + 1)

        pieces: list[str] = []
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            if separator != " ":
                segment = f"{segment}{separator}"
            pieces.extend(self._split_recursively(segment, level + 1))
        return pieces

    @staticmethod
    def _split_code_block_sections(text: str) -> list[tuple[str, bool]]:
        sections: list[tuple[str, bool]] = []
        lines = text.splitlines(keepends=True)
        in_code_block = False
        buffer: list[str] = []

        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("```"):
                if in_code_block:
                    buffer.append(line)
                    sections.append(("".join(buffer).strip(), True))
                    buffer = []
                    in_code_block = False
                else:
                    if buffer:
                        sections.append(("".join(buffer).strip(), False))
                        buffer = []
                    buffer.append(line)
                    in_code_block = True
                continue
            buffer.append(line)

        if buffer:
            sections.append(("".join(buffer).strip(), in_code_block))
        return [item for item in sections if item[0]]

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current = ""

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue

            if not current:
                current = piece
                continue

            candidate = f"{current}\n{piece}".strip()
            if len(candidate) <= self._config.chunk_size:
                current = candidate
                continue

            chunks.append(current)
            overlap = self._safe_tail_overlap(current)
            current = f"{overlap}\n{piece}".strip() if overlap else piece

            if len(current) > self._config.chunk_size:
                if "```" in current:
                    chunks.append(current)
                    current = ""
                else:
                    chunks.extend(
                        self._split_recursively(current, len(self._SEPARATORS) - 1)
                    )
                    current = ""

        if current:
            chunks.append(current)
        return chunks

    def _safe_tail_overlap(self, text: str) -> str:
        overlap = self._config.chunk_overlap
        if overlap <= 0 or len(text) <= overlap:
            return ""
        tail = text[-overlap:]
        if "```" in tail:
            return ""
        return tail
