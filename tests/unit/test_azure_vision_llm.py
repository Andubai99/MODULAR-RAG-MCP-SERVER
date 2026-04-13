"""Unit tests for Azure Vision LLM provider."""

from __future__ import annotations

import base64
import io
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

from libs.llm import azure_vision_llm as azure_vision_module
from libs.llm.azure_vision_llm import AzureVisionLLM
from libs.llm.llm_factory import LLMFactory

_PNG_1X1_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7ZxQAAAABJRU5ErkJggg=="
)
_PNG_1X1_BYTES = base64.b64decode(_PNG_1X1_BASE64)


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
    old_vision_registry = dict(LLMFactory._VISION_REGISTRY)
    LLMFactory._REGISTRY.clear()
    LLMFactory._VISION_REGISTRY.clear()
    yield
    LLMFactory._REGISTRY = old_registry
    LLMFactory._VISION_REGISTRY = old_vision_registry


def _make_settings() -> SimpleNamespace:
    return SimpleNamespace(
        vision_llm=SimpleNamespace(
            provider="azure",
            model="gpt-4o",
            azure_endpoint="https://example-resource.openai.azure.com",
            deployment_name="vision-deploy",
            api_version="2024-10-21",
            api_key="secret-should-not-leak",
            max_image_size=2048,
            max_tokens=256,
        )
    )


def test_vision_factory_routes_azure_provider() -> None:
    vision_llm = LLMFactory.create_vision_llm(_make_settings())
    assert isinstance(vision_llm, AzureVisionLLM)


def test_azure_vision_llm_chat_with_image_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_path = tmp_path / "tiny.png"
    image_path.write_bytes(_PNG_1X1_BYTES)
    captured: dict[str, object] = {}

    def _fake_urlopen(req: object, timeout: int = 0) -> _FakeHTTPResponse:
        del timeout
        request_obj = req
        captured["url"] = request_obj.full_url
        captured["headers"] = {k.lower(): v for k, v in request_obj.header_items()}
        captured["body"] = json.loads(request_obj.data.decode("utf-8"))
        return _FakeHTTPResponse({"choices": [{"message": {"content": "vision-ok"}}]})

    monkeypatch.setattr(azure_vision_module.request, "urlopen", _fake_urlopen)

    vision_llm = LLMFactory.create_vision_llm(_make_settings())
    result = vision_llm.chat_with_image("describe this image", str(image_path))

    assert result["text"] == "vision-ok"
    assert "/openai/deployments/vision-deploy/chat/completions" in str(captured["url"])
    assert "api-version=2024-10-21" in str(captured["url"])
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert "api-key" in headers
    body = captured["body"]
    assert isinstance(body, dict)
    image_url = body["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")


def test_azure_vision_llm_chat_with_bytes_input(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_urlopen(req: object, timeout: int = 0) -> _FakeHTTPResponse:
        del timeout
        request_obj = req
        captured["body"] = json.loads(request_obj.data.decode("utf-8"))
        return _FakeHTTPResponse({"choices": [{"message": {"content": "bytes-ok"}}]})

    monkeypatch.setattr(azure_vision_module.request, "urlopen", _fake_urlopen)
    vision_llm = LLMFactory.create_vision_llm(_make_settings())
    result = vision_llm.chat_with_image("bytes image", _PNG_1X1_BYTES)

    assert result["text"] == "bytes-ok"
    body = captured["body"]
    assert isinstance(body, dict)
    image_url = body["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")


def test_azure_vision_llm_compression_path(monkeypatch: pytest.MonkeyPatch) -> None:
    compressed_bytes = b"compressed-image-bytes"
    expected_base64 = base64.b64encode(compressed_bytes).decode("ascii")
    captured: dict[str, object] = {}

    def _fake_resize(image_bytes: bytes, max_size: int) -> tuple[bytes, bool]:
        del image_bytes, max_size
        return compressed_bytes, True

    def _fake_urlopen(req: object, timeout: int = 0) -> _FakeHTTPResponse:
        del timeout
        request_obj = req
        captured["body"] = json.loads(request_obj.data.decode("utf-8"))
        return _FakeHTTPResponse({"choices": [{"message": {"content": "compress-ok"}}]})

    monkeypatch.setattr(azure_vision_module.request, "urlopen", _fake_urlopen)
    vision_llm = LLMFactory.create_vision_llm(_make_settings())
    monkeypatch.setattr(vision_llm, "_resize_image_if_needed", _fake_resize)

    result = vision_llm.chat_with_image("compress", _PNG_1X1_BYTES)

    assert result["text"] == "compress-ok"
    assert result["metadata"]["compressed"] is True
    body = captured["body"]
    assert isinstance(body, dict)
    image_url = body["messages"][0]["content"][1]["image_url"]["url"]
    assert expected_base64 in image_url


def test_azure_vision_llm_timeout_error_does_not_leak_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_timeout(req: object, timeout: int = 0) -> _FakeHTTPResponse:
        del req, timeout
        raise TimeoutError("timeout")

    monkeypatch.setattr(azure_vision_module.request, "urlopen", _raise_timeout)
    vision_llm = LLMFactory.create_vision_llm(_make_settings())

    with pytest.raises(
        RuntimeError,
        match=r"provider=azure_vision: HTTP request failed \(error_type=TimeoutError\)",
    ) as exc_info:
        vision_llm.chat_with_image("describe", _PNG_1X1_BYTES)

    assert "secret-should-not-leak" not in str(exc_info.value)


def test_azure_vision_llm_auth_failure_contains_azure_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_auth_error(req: object, timeout: int = 0) -> _FakeHTTPResponse:
        del timeout
        payload = b'{"error":{"code":"401","message":"Unauthorized"}}'
        raise error.HTTPError(
            url=req.full_url,
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(payload),
        )

    monkeypatch.setattr(azure_vision_module.request, "urlopen", _raise_auth_error)
    vision_llm = LLMFactory.create_vision_llm(_make_settings())

    with pytest.raises(
        RuntimeError,
        match=r"azure_error_code=401",
    ) as exc_info:
        vision_llm.chat_with_image("describe", _PNG_1X1_BYTES)

    assert "secret-should-not-leak" not in str(exc_info.value)
