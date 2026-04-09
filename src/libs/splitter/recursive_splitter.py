"""Default recursive markdown splitter implementation."""
"""该文件实现一个默认的递归文本切分器。
其主要作用是：按照设定的 chunk_size 和 chunk_overlap，对输入文本进行分块。
切分时会优先依据较自然的分隔符（如空行、换行、空格）逐层递归拆分，
并对 Markdown 代码块进行特殊保护，尽量避免把代码块随意切开。
最终会将切分后的片段重新合并成适合后续处理的文本块。
"""

from __future__ import annotations

from dataclasses import dataclass

from libs.splitter.base_splitter import BaseSplitter


@dataclass(slots=True)
class _ChunkConfig:
    # 用于保存切分配置，包括块大小和块间重叠长度。
    chunk_size: int
    chunk_overlap: int


class RecursiveSplitter(BaseSplitter):
    # RecursiveSplitter类实现了BaseSplitter接口，提供了一种递归的文本分割方法，
    # 能够根据指定的chunk_size和chunk_overlap将输入文本分割成适合处理的块，同时保持markdown代码块的完整性。
    """Recursive text splitter with markdown/code-block awareness."""

    _SEPARATORS = ("\n\n", "\n", " ", "")

    def __init__(self, settings: object) -> None:
        """初始化切分器。
        从 settings.ingestion 中读取 chunk_size 和 chunk_overlap，
        若未提供则使用默认值；随后调用 _validate_config 校验配置合法性，
        并将合法配置保存到 self._config 中供后续切分使用。
        """
        ingestion_settings = getattr(settings, "ingestion", None)
        chunk_size = int(getattr(ingestion_settings, "chunk_size", 1000))
        chunk_overlap = int(getattr(ingestion_settings, "chunk_overlap", 200))
        self._config = self._validate_config(chunk_size, chunk_overlap)

    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        """对输入文本执行切分。
        先检查输入是否为合法字符串；若文本为空则直接返回空列表。
        然后先用 _split_code_block_sections 将普通文本和代码块分段：
        代码块直接保留，普通文本再交给 _split_recursively 递归切分。
        最后调用 _merge_pieces 将所有片段按块大小和重叠规则重新合并。
        """
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
        """校验切分配置是否合法。
        要求 chunk_size 必须大于 0，chunk_overlap 不能小于 0，
        且 overlap 必须严格小于 chunk_size。
        若配置合法，则封装成 _ChunkConfig 对象返回。
        """
        if chunk_size <= 0:
            raise ValueError("ingestion.chunk_size must be > 0.")
        if chunk_overlap < 0:
            raise ValueError("ingestion.chunk_overlap must be >= 0.")
        if chunk_overlap >= chunk_size:
            raise ValueError("ingestion.chunk_overlap must be < ingestion.chunk_size.")
        return _ChunkConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def _split_recursively(self, text: str, level: int) -> list[str]:
        """按分隔符层级递归切分普通文本。
        会先去除首尾空白；若文本长度已经不超过 chunk_size，则直接作为一个块返回。
        否则根据当前 level 选择分隔符：
        先尝试按空行，再按换行，再按空格，最后按固定长度硬切分。
        如果当前分隔符无法切开文本，则递归进入下一层分隔策略。
        """
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
        """将文本拆分为“普通文本段”和“代码块段”。
        通过识别 Markdown 中以 ``` 开始和结束的围栏代码块，
        把整段代码块作为一个整体标记为 True，
        普通文本段标记为 False。
        这样后续切分时可以避免把代码块内部内容随意拆开。
        """
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
        """将前面递归切出的较小片段重新合并成最终文本块。
        合并时会尽量把相邻片段拼接到 current 中，只要总长度不超过 chunk_size。
        如果超过限制，则先把当前块写入结果，再根据 _safe_tail_overlap 提取安全尾部重叠，
        用于与下一个片段衔接。
        若合并后仍过长，则对普通文本继续切分；若包含代码块标记，则直接整体保留。
        """
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
                    chunks.extend(self._split_recursively(current, len(self._SEPARATORS) - 1))
                    current = ""

        if current:
            chunks.append(current)
        return chunks

    def _safe_tail_overlap(self, text: str) -> str:
        """提取一个安全的尾部重叠片段。
        根据配置的 chunk_overlap，从当前文本末尾截取一段内容作为下一块的上下文衔接。
        但如果 overlap 不合法，或者截取出的尾部中包含 ```，
        则返回空字符串，以避免把 Markdown 代码块边界破坏掉。
        """
        overlap = self._config.chunk_overlap
        if overlap <= 0 or len(text) <= overlap:
            return ""
        tail = text[-overlap:]
        if "```" in tail:
            return ""
        return tail
