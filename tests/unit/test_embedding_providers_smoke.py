"""Smoke tests for OpenAI and Azure embedding providers."""

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

from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.openai_embedding import request as openai_request_module


class _FakeHTTPResponse:
    # _FakeHTTPResponse类模拟了一个HTTP响应对象，提供了一个read方法来返回预定义的JSON响应数据，并实现了上下文管理器协议以支持with语句。
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


@pytest.fixture(autouse=True) # 在每个测试函数运行前自动执行，确保EmbeddingFactory的注册表在测试之间被重置，以避免测试之间的相互影响。
def _reset_registry() -> None:
    # 保存当前的注册表状态，清空注册表以确保测试环境的干净，然后在测试完成后恢复原始的注册表状态。
    old_registry = dict(EmbeddingFactory._REGISTRY)
    EmbeddingFactory._REGISTRY.clear()
    yield
    EmbeddingFactory._REGISTRY = old_registry


def _make_settings(provider: str) -> SimpleNamespace:
    # 根据提供的provider名称构造一个模拟的settings对象，包含了必要的配置参数，以便在测试中使用EmbeddingFactory创建相应的embedding实例。
    if provider == "azure":
        return SimpleNamespace(
            embedding=SimpleNamespace(
                provider="azure",
                model="text-embedding-3-small",
                dimensions=1536,
                azure_endpoint="https://example-resource.openai.azure.com",
                deployment_name="embedding-deploy",
                api_version="2024-10-21",
                api_key="test-key",
                base_url="",
            )
        )
    # 默认情况下返回一个OpenAI的settings对象，包含了必要的配置参数，如provider名称、模型名称、API密钥等，以便在测试中使用EmbeddingFactory创建OpenAIEmbedding实例。
    return SimpleNamespace(
        embedding=SimpleNamespace(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
            azure_endpoint="",
            deployment_name="",
            api_version="",
            api_key="test-key",
            base_url="",
        )
    )

# 测试函数test_embedding_factory_routes_builtin_providers使用参数化测试，
# 分别验证EmbeddingFactory是否正确地根据提供的provider名称创建了相应的embedding实例，并且实例的类名与预期的class_name匹配。
@pytest.mark.parametrize(
    ("provider", "class_name"),
    [
        ("openai", "OpenAIEmbedding"),
        ("azure", "AzureEmbedding"),
    ],
)
def test_embedding_factory_routes_builtin_providers(
    # 使用参数化测试，分别验证EmbeddingFactory是否正确地根据提供的provider名称创建了相应的embedding实例，并且实例的类名与预期的class_name匹配。
    provider: str, class_name: str
) -> None:
    embedding = EmbeddingFactory.create(_make_settings(provider))
    assert embedding.__class__.__name__ == class_name


@pytest.mark.parametrize(
    ("provider", "url_fragment", "header_name"),
    [
        ("openai", "/embeddings", "authorization"),
        ("azure", "/openai/deployments/embedding-deploy/embeddings", "api-key"),
    ],
)
def test_openai_and_azure_embedding_with_mock_http(
    # 使用参数化测试，分别验证OpenAI和Azure嵌入提供者在发送HTTP请求时是否包含了正确的URL片段和请求头。
    monkeypatch: pytest.MonkeyPatch, provider: str, url_fragment: str, header_name: str
) -> None:
    captured: dict[str, object] = {}

    def _fake_urlopen(req: object, timeout: int = 0) -> _FakeHTTPResponse:
        del timeout
        captured["url"] = req.full_url
        captured["headers"] = {key.lower(): value for key, value in req.header_items()}
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse(
            {
                "data": [
                    {"embedding": [0.1, 0.2]},
                    {"embedding": [0.3, 0.4]},
                ]
            }
        )

    monkeypatch.setattr(openai_request_module, "urlopen", _fake_urlopen)
    embedding = EmbeddingFactory.create(_make_settings(provider))
    vectors = embedding.embed(["hello", "world"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert url_fragment in str(captured["url"])
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert header_name in headers
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["input"] == ["hello", "world"]


def test_embedding_empty_input_has_clear_error() -> None:
    # 测试函数test_embedding_empty_input_has_clear_error验证当输入文本列表为空时，
    # embedding实例的embed方法是否抛出了一个包含特定错误消息的ValueError异常，以确保提供了清晰的错误信息。
    embedding = EmbeddingFactory.create(_make_settings("openai"))

    with pytest.raises(ValueError, match=r"provider=openai: texts cannot be empty"):
        embedding.embed([])


def test_azure_embedding_http_error_contains_provider_and_error_type(
    # 测试函数test_azure_embedding_http_error_contains_provider_and_error_type
    # 验证当Azure嵌入提供者在发送HTTP请求时发生URLError异常，
    # 是否抛出了一个包含特定错误消息的RuntimeError异常，并且错误消息中包含了提供者名称和错误类型，以确保提供了清晰的错误信息。
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_url_error(req: object, timeout: int = 0) -> _FakeHTTPResponse:
        del req, timeout
        raise error.URLError("network down")

    monkeypatch.setattr(openai_request_module, "urlopen", _raise_url_error)
    embedding = EmbeddingFactory.create(_make_settings("azure"))

    with pytest.raises(
        RuntimeError, match=r"provider=azure: HTTP request failed \(error_type=URLError,"
    ):
        embedding.embed(["hello"])
