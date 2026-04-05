"""Modular RAG MCP Server 入口。"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许从仓库根目录直接执行 `python main.py`
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.settings import SettingsError, load_settings
from observability.logger import get_logger


def main() -> int:
    """启动时加载配置，失败即快速退出。"""
    logger = get_logger(__name__)
    try:
        settings = load_settings()
    except SettingsError as exc:
        logger.error("配置加载失败: %s", exc)
        return 1

    logger.info(
        "配置加载成功: LLM=%s/%s, EMBED=%s/%s",
        settings.llm.provider,
        settings.llm.model,
        settings.embedding.provider,
        settings.embedding.model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
