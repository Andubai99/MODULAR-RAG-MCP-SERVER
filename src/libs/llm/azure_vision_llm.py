"""Azure Vision LLM implementation over chat-completions API."""

from __future__ import annotations

import base64
import json
import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib import error, request

from libs.llm.base_vision_llm import BaseVisionLLM, ChatResponse

_DEFAULT_TIMEOUT_SECONDS = 30
_DEFAULT_MAX_IMAGE_SIZE = 2048

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency.
    Image = None  # type: ignore[assignment]


def _require_vision_str(settings: object, field_name: str) -> str:
    vision_settings = getattr(settings, "vision_llm", None)
    value = getattr(vision_settings, field_name, "")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"provider=azure_vision: settings.vision_llm.{field_name} must be a non-empty string."
        )
    return value.strip()


class AzureVisionLLM(BaseVisionLLM):
    """Vision-capable Azure OpenAI client."""

    provider_name = "azure"

    def __init__(self, settings: object) -> None:
        self._api_key = _require_vision_str(settings, "api_key")
        self._deployment = _require_vision_str(settings, "deployment_name")
        self._api_version = _require_vision_str(settings, "api_version")
        self._endpoint_root = _require_vision_str(settings, "azure_endpoint").rstrip("/")
        self._model = _require_vision_str(settings, "model")
        vision_settings = getattr(settings, "vision_llm", None)
        self._max_tokens = int(getattr(vision_settings, "max_tokens", 1024))
        max_image_size = getattr(vision_settings, "max_image_size", _DEFAULT_MAX_IMAGE_SIZE)
        if isinstance(max_image_size, bool) or not isinstance(max_image_size, int) or max_image_size <= 0:
            raise ValueError(
                "provider=azure_vision: settings.vision_llm.max_image_size must be a positive integer."
            )
        self._max_image_size = max_image_size

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: object | None = None,
    ) -> ChatResponse:
        del trace
        prompt_text = self._normalize_text(text)
        image_bytes, source_hint = self._load_image_bytes(image_path)
        resized_bytes, compressed = self._resize_image_if_needed(
            image_bytes, self._max_image_size
        )
        image_data_url = self._build_data_url(resized_bytes, source_hint)
        endpoint = (
            f"{self._endpoint_root}/openai/deployments/{self._deployment}/chat/completions"
            f"?api-version={self._api_version}"
        )
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            "max_tokens": self._max_tokens,
        }
        headers = {"Content-Type": "application/json", "api-key": self._api_key}
        response = self._post_json(endpoint, headers, payload)
        response_text = self._extract_response_text(response)
        return {
            "text": response_text,
            "metadata": {
                "provider": self.provider_name,
                "compressed": compressed,
                "max_image_size": self._max_image_size,
            },
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("provider=azure_vision: text must be a non-empty string.")
        return text.strip()

    @staticmethod
    def _load_image_bytes(image_input: str | bytes) -> tuple[bytes, str]:
        if isinstance(image_input, bytes):
            if not image_input:
                raise ValueError("provider=azure_vision: image bytes cannot be empty.")
            return image_input, "bytes"
        if isinstance(image_input, str):
            path = Path(image_input)
            if not path.exists() or not path.is_file():
                raise ValueError(
                    f"provider=azure_vision: image path not found: {path.as_posix()}"
                )
            return path.read_bytes(), path.suffix.lower()
        raise ValueError("provider=azure_vision: image_path must be str path or bytes.")

    @staticmethod
    def _guess_mime_type(source_hint: str, image_bytes: bytes) -> str:
        if source_hint and source_hint != "bytes":
            guessed, _ = mimetypes.guess_type(f"file{source_hint}")
            if isinstance(guessed, str) and guessed.startswith("image/"):
                return guessed

        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if image_bytes.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:16]:
            return "image/webp"
        return "application/octet-stream"

    def _build_data_url(self, image_bytes: bytes, source_hint: str) -> str:
        mime_type = self._guess_mime_type(source_hint, image_bytes)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _resize_image_if_needed(
        image_bytes: bytes, max_image_size: int
    ) -> tuple[bytes, bool]:
        if Image is None:
            return image_bytes, False

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                width, height = image.size
                if max(width, height) <= max_image_size:
                    return image_bytes, False

                resized = image.copy()
                resized.thumbnail((max_image_size, max_image_size))
                output = BytesIO()
                output_format = image.format or "PNG"
                resized.save(output, format=output_format)
                return output.getvalue(), True
        except Exception:
            return image_bytes, False

    def _post_json(
        self, endpoint: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=_DEFAULT_TIMEOUT_SECONDS) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            azure_code, azure_message = self._extract_azure_error(exc)
            raise RuntimeError(
                "provider=azure_vision: HTTP request failed "
                f"(error_type=HTTPError, status={exc.code}, "
                f"azure_error_code={azure_code}, azure_message={azure_message})."
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"provider=azure_vision: HTTP request failed "
                f"(error_type=URLError, reason={exc.reason})."
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                "provider=azure_vision: HTTP request failed (error_type=TimeoutError)."
            ) from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "provider=azure_vision: invalid JSON response "
                "(error_type=JSONDecodeError)."
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError("provider=azure_vision: invalid JSON response root type.")
        return parsed

    @staticmethod
    def _extract_azure_error(exc: error.HTTPError) -> tuple[str, str]:
        default_code = "unknown"
        default_message = "unknown"
        try:
            body = exc.read().decode("utf-8")
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                error_obj = parsed.get("error", {})
                if isinstance(error_obj, dict):
                    azure_code = error_obj.get("code", default_code)
                    azure_message = error_obj.get("message", default_message)
                    return str(azure_code), str(azure_message)
        except Exception:
            pass
        return default_code, default_message

    @staticmethod
    def _extract_response_text(payload: dict[str, Any]) -> str:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "provider=azure_vision: invalid response payload shape "
                f"(error_type={type(exc).__name__})."
            ) from exc

        if isinstance(content, str) and content.strip():
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts)

        raise ValueError("provider=azure_vision: response content is empty or invalid.")
