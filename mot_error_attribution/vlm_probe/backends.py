from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import FRAMES, MAX_PIXELS, MAX_TOKENS, SEED, TEMPERATURE


@dataclass
class GenerateResult:
    text: str
    n_frames_used: int | None
    elapsed_sec: float
    extra: dict[str, Any]


class Backend(Protocol):
    name: str
    model_id: str

    def generate(self, video_path: Path, prompt: str) -> GenerateResult: ...


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "--").replace(" ", "_")


def file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def video_content(
    video_path: Path, n_frames: int = FRAMES, max_pixels: int = MAX_PIXELS
) -> dict[str, Any]:
    return {
        "type": "video",
        "video": file_uri(video_path),
        "nframes": n_frames,
        "max_pixels": max_pixels,
    }


def media_content(
    path: Path, n_frames: int = FRAMES, max_pixels: int = MAX_PIXELS
) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return {
            "type": "image",
            "image": file_uri(path),
            "max_pixels": max_pixels,
        }
    return video_content(path, n_frames, max_pixels=max_pixels)


def _n_frames_from_tensor(video: Any) -> int | None:
    shape = getattr(video, "shape", None)
    if shape is None or len(shape) < 3:
        return None
    if len(shape) == 4:
        time_or_batch, channels = int(shape[0]), int(shape[1])
        if channels in {1, 3, 4}:
            return time_or_batch
        if time_or_batch in {1, 3, 4}:
            return int(shape[1])
        return time_or_batch
    if len(shape) >= 5:
        return int(shape[1] if int(shape[0]) <= 4 else shape[0])
    return None


def unpack_qwen_videos(
    videos: Any,
) -> tuple[list[Any] | None, list[Any] | None, int | None]:
    """Split qwen-vl-utils `(tensor, metadata)` pairs before the HF processor.

    ``process_vision_info(..., return_video_metadata=True)`` returns a list of
    ``(video_tensor, metadata_dict)``. Passing that list through unchanged
    makes ``group_videos_by_shape`` see a dict and crash.
    """
    if videos is None:
        return None, None, None
    # A single (tensor, metadata) pair is a 2-tuple, not two videos.
    if (
        isinstance(videos, (tuple, list))
        and len(videos) == 2
        and hasattr(videos[0], "shape")
        and isinstance(videos[1], dict)
    ):
        items = [videos]
    elif isinstance(videos, (list, tuple)):
        items = list(videos)
    else:
        items = [videos]
    tensors: list[Any] = []
    metadatas: list[Any] = []
    n_frames: int | None = None
    for item in items:
        tensor, meta = item, None
        if isinstance(item, (tuple, list)) and item:
            first = item[0]
            if hasattr(first, "shape"):
                tensor = first
                meta = item[1] if len(item) > 1 else None
            elif (
                isinstance(first, dict)
                and len(item) == 2
                and hasattr(item[1], "shape")
            ):
                meta, tensor = first, item[1]
        elif isinstance(item, dict):
            tensor = item.get("video", item.get("pixel_values"))
            meta = {
                key: value
                for key, value in item.items()
                if key not in {"video", "pixel_values"}
            }
        tensors.append(tensor)
        metadatas.append(meta)
        n_frames = _n_frames_from_tensor(tensor) or n_frames
    if all(meta is None for meta in metadatas):
        metadatas = None
    return tensors, metadatas, n_frames


class DryRunBackend:
    """Valid JSON, no model. Used for tests and wiring checks."""

    name = "dry-run"

    def __init__(self, model_id: str = "dry-run", *, max_pixels: int = MAX_PIXELS):
        self.model_id = model_id
        self.max_pixels = max_pixels

    def generate(self, video_path: Path, prompt: str) -> GenerateResult:
        started = time.perf_counter()
        stem = video_path.stem.lower()
        if "highlighted_box_present" in prompt or "Is there ALSO a box" in prompt:
            payload = {"highlighted_box_present": "inside" not in stem}
        elif "found_highlighted_box" in prompt or "FIRST frame" in prompt:
            payload = {
                "track_number": "unreadable",
                "position": "centre",
                "clothing": "dark",
                "found_highlighted_box": True,
            }
        elif "n_moments" in prompt or "distinct moments" in prompt:
            payload = {"n_moments": 8}
        elif "box_behaviour" in prompt:
            payload = {
                "person_description": "dry-run person in dark clothing walking left",
                "box_behaviour": "disappears",
                "disappears_at": "middle",
                "person_still_visible_after": "yes",
                "confidence": "low",
            }
        else:
            payload = {
                "error_kind": "miss",
                "cause": "cannot_determine",
                "evidence": "dry-run backend; no video was inspected",
                "occluder_visible_in_frame": False,
                "confidence": "low",
            }
        return GenerateResult(
            text=json.dumps(payload),
            n_frames_used=1 if video_path.suffix.lower() in {".jpg", ".jpeg", ".png"} else FRAMES,
            elapsed_sec=time.perf_counter() - started,
            extra={"video": str(video_path)},
        )


class TransformersBackend:
    name = "transformers"

    def __init__(
        self,
        model_id: str,
        *,
        n_frames: int = FRAMES,
        max_pixels: int = MAX_PIXELS,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
    ):
        self.model_id = model_id
        self.n_frames = n_frames
        self.max_pixels = max_pixels
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelCls
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelCls

        kwargs: dict[str, Any] = {"device_map": "auto"}
        try:
            kwargs["torch_dtype"] = torch.bfloat16
        except Exception:
            pass
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = ModelCls.from_pretrained(self.model_id, **kwargs)
        try:
            from transformers import set_seed

            set_seed(SEED)
        except Exception:
            pass

    def generate(self, video_path: Path, prompt: str) -> GenerateResult:
        self._load()
        started = time.perf_counter()
        messages = [
            {
                "role": "user",
                "content": [
                    media_content(video_path, self.n_frames, max_pixels=self.max_pixels),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        processor = self._processor
        model = self._model
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError(
                "The transformers backend needs qwen-vl-utils. "
                "Install with: pip install -e '.[vlm]'"
            ) from exc

        video_kwargs: dict[str, Any] = {}
        try:
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
        except TypeError:
            try:
                images, videos, video_kwargs = process_vision_info(
                    messages, return_video_kwargs=True
                )
            except TypeError:
                images, videos = process_vision_info(messages)
                video_kwargs = {}

        videos, video_metadata, decoded_frames = unpack_qwen_videos(videos)
        is_image = video_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        if is_image or not videos:
            videos = None
            video_metadata = None
            decoded_frames = 1 if is_image else decoded_frames
        n_frames_used = 1 if is_image else (decoded_frames or self.n_frames)
        proc_kwargs: dict[str, Any] = dict(video_kwargs or {})
        proc_kwargs.update(
            {
                "text": [text],
                "images": images,
                "videos": videos,
                "padding": True,
                "return_tensors": "pt",
                "do_resize": False,
            }
        )
        if video_metadata is not None:
            proc_kwargs["video_metadata"] = video_metadata
        if not is_image:
            proc_kwargs.setdefault("do_sample_frames", False)
        inputs = processor(**proc_kwargs)
        device = getattr(model, "device", None) or next(model.parameters()).device
        inputs = inputs.to(device)
        gen_kwargs: dict[str, Any] = {"max_new_tokens": self.max_tokens}
        if self.temperature <= 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
        generated = model.generate(**inputs, **gen_kwargs)
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated)
        ]
        decoded = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return GenerateResult(
            text=decoded[0] if decoded else "",
            n_frames_used=n_frames_used,
            elapsed_sec=time.perf_counter() - started,
            extra={},
        )


class VllmBackend:
    name = "vllm"

    def __init__(
        self,
        model_id: str,
        *,
        n_frames: int = FRAMES,
        max_pixels: int = MAX_PIXELS,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
    ):
        self.model_id = model_id
        self.n_frames = n_frames
        self.max_pixels = max_pixels
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._llm = None

    def _load(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM

        self._llm = LLM(
            model=self.model_id,
            limit_mm_per_prompt={"video": 1},
            seed=SEED,
        )

    def generate(self, video_path: Path, prompt: str) -> GenerateResult:
        self._load()
        from vllm import SamplingParams

        started = time.perf_counter()
        sampling = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=SEED,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    media_content(video_path, self.n_frames, max_pixels=self.max_pixels),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        outputs = self._llm.chat(messages, sampling_params=sampling)
        text = ""
        if outputs:
            text = outputs[0].outputs[0].text if outputs[0].outputs else ""
        return GenerateResult(
            text=text,
            n_frames_used=self.n_frames,
            elapsed_sec=time.perf_counter() - started,
            extra={},
        )


class GeminiBackend:
    name = "gemini"

    def __init__(
        self,
        model_id: str,
        *,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
    ):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _load(self) -> None:
        if self._client is not None:
            return
        from google import genai

        self._client = genai.Client()

    def generate(self, video_path: Path, prompt: str) -> GenerateResult:
        self._load()
        from google.genai import types

        started = time.perf_counter()
        suffix = video_path.suffix.lower()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(suffix, "video/mp4")
        part = types.Part.from_bytes(data=video_path.read_bytes(), mime_type=mime)
        response = self._client.models.generate_content(
            model=self.model_id,
            contents=[part, prompt],
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        text = getattr(response, "text", None) or ""
        return GenerateResult(
            text=text,
            n_frames_used=None,
            elapsed_sec=time.perf_counter() - started,
            extra={},
        )


def build_backend(
    kind: str,
    model_id: str,
    *,
    n_frames: int = FRAMES,
    max_pixels: int = MAX_PIXELS,
) -> Backend:
    kind = kind.lower().replace("_", "-")
    if kind in {"dry-run", "dryrun", "mock"}:
        return DryRunBackend(model_id=model_id, max_pixels=max_pixels)
    if kind in {"transformers", "hf", "qwen"}:
        return TransformersBackend(model_id, n_frames=n_frames, max_pixels=max_pixels)
    if kind in {"vllm"}:
        return VllmBackend(model_id, n_frames=n_frames, max_pixels=max_pixels)
    if kind in {"gemini", "google"}:
        return GeminiBackend(model_id)
    raise ValueError(f"Unknown backend: {kind}")
