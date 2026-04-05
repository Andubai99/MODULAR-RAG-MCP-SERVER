"""配置加载模块测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.settings import SettingsError, load_settings


def test_load_settings_success() -> None:
    settings = load_settings(ROOT_DIR / "config" / "settings.yaml")
    assert settings.llm.provider == "openai"
    assert settings.embedding.provider == "openai"
    assert settings.vector_store.provider == "chroma"


def test_load_settings_missing_required_field() -> None:
    broken_file = (
        ROOT_DIR / "tests" / "fixtures" / "settings_missing_embedding_provider.yaml"
    )
    with pytest.raises(SettingsError, match="embedding.provider"):
        load_settings(broken_file)
