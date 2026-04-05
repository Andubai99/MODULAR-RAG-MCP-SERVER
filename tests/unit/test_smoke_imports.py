"""Smoke tests for top-level package imports."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_import_top_level_packages() -> None:
    import core  # noqa: F401
    import ingestion  # noqa: F401
    import libs  # noqa: F401
    import mcp_server  # noqa: F401
    import observability  # noqa: F401
