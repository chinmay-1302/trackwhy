"""P5: pinned-track preview renders. Same clip windows as regime A."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from mot_error_attribution.models import TrkBox
from mot_error_attribution.probe_clips import FPS, encode_h264
from vlm_clip_style_tests.build_dataset import _sha256
from vlm_clip_style_tests.render_annotation_styles import (
    STYLE_TOP_LEFT,
    _prediction_boxes,
    render_style_frame,
)

from .derive import highlight_id_for_frame


PREVIEW_STYLE = STYLE_TOP_LEFT
PREVIEW_PER_CLASS = 3


def pinned_source_frames(
    annotation: Mapping[str, Any],
    dataset_root: Path,
    highlight_track_id: int | None,
    style: str = PREVIEW_STYLE,
) -> Iterable[np.ndarray]:
    import cv2

    sequence = str(annotation["sequence"])
    by_frame = _prediction_boxes(dict(annotation))
    tracker_ids_by_frame = {
        frame: [int(box.trk_id) for box in boxes] for frame, boxes in by_frame.items()
    }
    expected_width = int(annotation["video"]["width"])
    expected_height = int(annotation["video"]["height"])
    for raw_frame in annotation["clip_frames"]:
        frame = int(raw_frame)
        image_path = dataset_root / "train" / sequence / "img1" / f"{frame:06d}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"OpenCV could not read {image_path}")
        if image.shape[1] > expected_width or image.shape[0] > expected_height:
            raise ValueError(f"Source image exceeds encoded dimensions: {image_path}")
        if image.shape[1] != expected_width or image.shape[0] != expected_height:
            image = cv2.copyMakeBorder(
                image,
                0,
                expected_height - image.shape[0],
                0,
                expected_width - image.shape[1],
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        highlight_id = highlight_id_for_frame(
            highlight_track_id, frame, tracker_ids_by_frame
        )
        yield render_style_frame(
            image,
            by_frame.get(frame, []),
            highlight_id,
            style,
        )


def select_preview_clip_ids(
    records: Sequence[Mapping[str, Any]],
    flagged: Sequence[str],
    *,
    per_class: int = PREVIEW_PER_CLASS,
) -> list[str]:
    by_key: dict[str, list[str]] = defaultdict(list)
    for row in records:
        primary = str(row.get("primary_error") or "non_core")
        subtype = str(row.get("identity_change_subtype") or "")
        key = f"{primary}:{subtype}" if primary == "identity_change" and subtype else primary
        clip_id = str(row["source_clip_id"] if "source_clip_id" in row else row["clip_id"])
        clip_id = clip_id.split("-")[0] if clip_id[:1] == "C" else clip_id
        if clip_id not in by_key[key]:
            by_key[key].append(clip_id)
    selected: list[str] = []
    for key in sorted(by_key):
        selected.extend(by_key[key][:per_class])
    for clip_id in flagged:
        source = str(clip_id).split("-")[0]
        if source not in selected:
            selected.append(source)
    return selected


def render_preview_clip(
    *,
    annotation: Mapping[str, Any],
    dataset_root: Path,
    highlight_track_id: int | None,
    output_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = int(annotation["video"]["width"])
    height = int(annotation["video"]["height"])
    encode_h264(
        pinned_source_frames(annotation, dataset_root, highlight_track_id),
        output_path,
        width,
        height,
        FPS,
    )
    return {
        "clip_id": annotation["clip_id"],
        "highlight_track_id": highlight_track_id,
        "style": PREVIEW_STYLE,
        "video": str(output_path),
        "n_frames": len(annotation["clip_frames"]),
        "width": width,
        "height": height,
        "sha256": _sha256(output_path),
    }
