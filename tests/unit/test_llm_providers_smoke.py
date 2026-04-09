"""Smoke tests for OpenAI-compatible LLM providers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib import error

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import request as openai_request_module


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    old_registry = dict(LLMFactory._REGISTRY)
    LLMFactory._REGISTRY.clear()
    yield
    LLMFactory._REGISTRY = old_registry


def _make_settings(provider: str) -> SimpleNamespace:
    if provider == "azure":
        return SimpleNamespace(
            llm=SimpleNamespace(
                provider="azure",
                model="gpt-4o",
                deployment_name="test-deployment",
                azure_endpoint="https://example-resource.openai.azure.com",
                api_version="2024-10-21",
                api_key="test-key",
                base_url="",
                temperature=0.0,
                max_tokens=128,
            )
        )
    return SimpleNamespace(
        llm=SimpleNamespace(
            provider=provider,
            model="gpt-4o-mini",
            deployment_name="",
            azure_endpoint="",
            api_version="",
            api_key="test-key",
            base_url="",
            temperature=0.0,
            max_tokens=128,
        )
    )


@pytest.mark.parametrize(
    ("provider", "class_name"),
    [
        ("openai", "OpenAILLM"),
        ("azure", "AzureLLM"),
        ("deepseek", "DeepSeekLLM"),
    ],
)
def test_llm_factory_routes_openai_compatible_providers(
    provider: str, class_name: str
) -> None:
    llm = LLMFactory.create(_make_settings(provider))
    assert llm.__class__.__name__ == class_name


@pytest.mark.parametrize(
    ("provider", "url_fragment", "header_name"),
    [
        ("openai", "/chat/completions", "authorization"),
        ("azure", "/openai/deployments/test-deployment/chat/completions", "api-key"),
        ("deepseek", "/chat/completions", "authorization"),
    ],
)
def test_openai_compatible_llm_chat_with_mock_http(
    monkeypatch: pytest.MonkeyPatch, provider: str, url_fragment: str, header_name: str
) -> None:
    captured: dict[str, object] = {}

    def _fake_urlopen(req: object, timeout: int = 0) -> _FakeHTTPResponse:
        del timeout
        request_obj = req
        captured["url"] = request_obj.full_url
        captured["headers"] = {
            key.lower(): value for key, value in request_obj.header_items()
        }
        return _FakeHTTPResponse(
            {"choices": [{"message": {"content": f"{provider}-ok"}}]}
        )

    monkeypatch.setattr(openai_request_module, "urlopen", _fake_urlopen)

    llm = LLMFactory.create(_make_settings(provider))
    reply = llm.chat([{"role": "user", "content": "hello"}])

    assert reply == f"{provider}-ok"
    assert url_fragment in str(captured["url"])
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert header_name in headers


def test_openai_llm_message_shape_validation() -> None:
    llm = LLMFactory.create(_make_settings("openai"))

    with pytest.raises(
        ValueError, match=r"provider=openai: message\[0\]\.content must be a non-empty string"
    ):
        llm.chat([{"role": "user", "content": ""}])


def test_deepseek_llm_error_contains_provider_and_error_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_url_error(req: object, timeout: int = 0) -> _FakeHTTPResponse:
        del req, timeout
        raise error.URLError("network down")

    monkeypatch.setattr(openai_request_module, "urlopen", _raise_url_error)
    llm = LLMFactory.create(_make_settings("deepseek"))

    with pytest.raises(
        RuntimeError, match=r"provider=deepseek: HTTP request failed \(error_type=URLError,"
    ):
        llm.chat([{"role": "user", "content": "hello"}])
