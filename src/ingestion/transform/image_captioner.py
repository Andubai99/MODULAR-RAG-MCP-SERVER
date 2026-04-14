"""Image caption transform with optional Vision LLM and safe fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_vision_llm import BaseVisionLLM, ChatResponse
from libs.llm.llm_factory import LLMFactory

_DEFAULT_PROMPT_PATH = Path("config/prompts/image_captioning.txt")
_DEFAULT_PROMPT_TEMPLATE = (
    "Describe the image accurately for retrieval.\n"
    "Focus on entities, actions, numbers, and chart/table trends.\n\n"
    "Chunk Context:\n{text}\n\n"
    "Image ID: {image_id}"
)


class ImageCaptioner(BaseTransform):
    """Generate captions for chunk image refs and attach them back to metadata/text."""

    def __init__(
        self,
        settings: object,
        vision_llm: BaseVisionLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        vision_settings = getattr(settings, "vision_llm", None)
        self._enabled = bool(getattr(vision_settings, "enabled", False))
        self._prompt_template = self._load_prompt(prompt_path)

        if self._enabled and vision_llm is None:
            try:
                vision_llm = LLMFactory.create_vision_llm(settings)
            except Exception:
                vision_llm = None
        self._vision_llm = vision_llm

    def transform(
        self, chunks: list[Chunk], trace: TraceContext | None = None
    ) -> list[Chunk]:
        if not isinstance(chunks, list):
            raise ValueError("ImageCaptioner.transform expects chunks as a list.")

        results: list[Chunk] = []
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise ValueError(f"ImageCaptioner.transform chunks[{index}] must be Chunk.")

            stage_start = trace.stage_timer() if trace is not None else None
            method = "rule"
            fallback_reason: str | None = None

            new_metadata = dict(chunk.metadata)
            new_text = chunk.text
            image_refs = self._read_image_refs(new_metadata)

            if image_refs:
                if not self._enabled or self._vision_llm is None:
                    new_metadata["has_unprocessed_images"] = True
                    new_metadata["image_captioned_by"] = "rule"
                    new_metadata["image_caption_fallback_reason"] = (
                        "vision_llm_disabled_or_unavailable"
                    )
                    method = "rule"
                else:
                    captions, failed = self._generate_captions(
                        chunk.text, image_refs, new_metadata, trace
                    )
                    if captions and not failed:
                        new_metadata["image_captions"] = captions
                        new_metadata["image_captioned_by"] = "llm"
                        new_metadata.pop("has_unprocessed_images", None)
                        new_metadata.pop("image_caption_fallback_reason", None)
                        new_text = self._append_captions(chunk.text, captions)
                        method = "llm"
                    else:
                        new_metadata["has_unprocessed_images"] = True
                        new_metadata["image_captioned_by"] = "rule"
                        fallback_reason = "vision_llm_failed_or_partial"
                        new_metadata["image_caption_fallback_reason"] = fallback_reason
                        method = "rule"

            enriched_chunk = Chunk(
                id=chunk.id,
                text=new_text,
                metadata=new_metadata,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                source_ref=chunk.source_ref,
            )

            if trace is not None:
                elapsed_ms = (
                    trace.stage_elapsed_ms(stage_start) if stage_start is not None else None
                )
                trace.record_stage(
                    "image_caption",
                    method=method,
                    details={
                        "chunk_id": chunk.id,
                        "chunk_index": index,
                        "image_count": len(image_refs),
                    },
                    elapsed_ms=elapsed_ms,
                )

            results.append(enriched_chunk)
        return results

    def _generate_captions(
        self,
        chunk_text: str,
        image_refs: list[str],
        metadata: Mapping[str, object],
        trace: TraceContext | None = None,
    ) -> tuple[list[dict[str, str]], bool]:
        if self._vision_llm is None:
            return [], True

        failed = False
        captions: list[dict[str, str]] = []
        id_to_path = self._build_image_path_index(metadata)
        for image_id in image_refs:
            image_path = id_to_path.get(image_id)
            if not image_path:
                failed = True
                continue

            prompt = (
                self._prompt_template.replace("{text}", chunk_text)
                .replace("{image_id}", image_id)
            )
            try:
                response = self._vision_llm.chat_with_image(
                    prompt, image_path=image_path, trace=trace
                )
            except Exception:
                failed = True
                continue

            caption = self._extract_caption_text(response)
            if not caption:
                failed = True
                continue
            captions.append({"id": image_id, "caption": caption})
        return captions, failed

    @staticmethod
    def _build_image_path_index(metadata: Mapping[str, object]) -> dict[str, str]:
        image_index: dict[str, str] = {}
        raw_images = metadata.get("images")
        if not isinstance(raw_images, list):
            return image_index

        for item in raw_images:
            if not isinstance(item, Mapping):
                continue
            image_id = item.get("id")
            image_path = item.get("path")
            if isinstance(image_id, str) and image_id.strip() and isinstance(image_path, str):
                path_value = image_path.strip()
                if path_value:
                    image_index[image_id.strip()] = path_value
        return image_index

    @staticmethod
    def _extract_caption_text(response: ChatResponse | object) -> str | None:
        if not isinstance(response, Mapping):
            return None
        text = response.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        return text.strip()

    @staticmethod
    def _read_image_refs(metadata: Mapping[str, object]) -> list[str]:
        raw_refs = metadata.get("image_refs")
        if not isinstance(raw_refs, list):
            return []

        refs: list[str] = []
        for ref in raw_refs:
            if isinstance(ref, str):
                normalized = ref.strip()
                if normalized and normalized not in refs:
                    refs.append(normalized)
        return refs

    @staticmethod
    def _append_captions(text: str, captions: list[dict[str, str]]) -> str:
        caption_lines = [f"[IMAGE:{item['id']}] {item['caption']}" for item in captions]
        if not caption_lines:
            return text
        suffix = "\n".join(caption_lines)
        base = text.rstrip()
        if not base:
            return suffix
        return f"{base}\n\n{suffix}"

    def _load_prompt(self, prompt_path: str | Path | None = None) -> str:
        candidate = Path(prompt_path) if prompt_path is not None else _DEFAULT_PROMPT_PATH
        if not candidate.exists():
            return _DEFAULT_PROMPT_TEMPLATE
        try:
            content = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            return _DEFAULT_PROMPT_TEMPLATE
        if not content:
            return _DEFAULT_PROMPT_TEMPLATE
        if "{text}" not in content:
            content = f"{content}\n\n{{text}}"
        if "{image_id}" not in content:
            content = f"{content}\n\nImage ID: {{image_id}}"
        return content
