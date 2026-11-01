from __future__ import annotations

import json
import os
from pathlib import Path

from .config import PipelineConfig
from .matching import MatchingResult
from .models import BBox, ErrorEvent, GTBox, SequenceData


RENDER_LAYOUT = "tracker_gt_side_by_side_focus_visibility_v5"
CONTEXT_OPACITY = 0.10
TRACKER_COLOR = (255, 190, 20)
GT_COLOR = (70, 220, 90)
FOCAL_COLOR = (0, 215, 255)


def _find_box(sequence: SequenceData, frame: int, gt_id: int | None) -> GTBox | None:
    if gt_id is None:
        return None
    return next(
        (
            box
            for box in sequence.gt_by_frame.get(frame, [])
            if box.gt_id == gt_id
        ),
        None,
    )


def _target_label(prefix: str, object_id: int, gt_box: GTBox | None) -> str:
    """Format a target ID with the visibility of its corresponding GT observation."""
    visibility = "NA" if gt_box is None else f"{gt_box.visibility:.2f}"
    return f"{prefix}:{object_id} V:{visibility}"


def _draw_labeled_box(
    image,
    bbox: BBox,
    label: str,
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
) -> None:
    import cv2

    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    x1 = min(max(x1, 0), max(0, width - 1))
    x2 = min(max(x2, 0), max(0, width - 1))
    y1 = min(max(y1, 0), max(0, height - 1))
    y2 = min(max(y2, 0), max(0, height - 1))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    text_thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(
        label, font, scale, text_thickness
    )
    label_left = min(x1, max(0, width - text_width - 8))
    text_y = max(text_height + 2, y1 - 4)
    # A thin outline preserves legibility without using a filled label background.
    cv2.putText(
        image,
        label,
        (label_left, text_y),
        font,
        scale,
        (25, 25, 25),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        (label_left, text_y),
        font,
        scale,
        color,
        text_thickness,
        cv2.LINE_AA,
    )


def _draw_context_boxes(
    image,
    boxes: list[BBox],
    color: tuple[int, int, int],
) -> None:
    """Draw unlabeled boxes with one efficient 10% alpha composite."""
    import cv2

    if not boxes:
        return
    overlay = image.copy()
    height, width = image.shape[:2]
    for bbox in boxes:
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        x1 = min(max(x1, 0), max(0, width - 1))
        x2 = min(max(x2, 0), max(0, width - 1))
        y1 = min(max(y1, 0), max(0, height - 1))
        y2 = min(max(y2, 0), max(0, height - 1))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
    cv2.addWeighted(
        overlay,
        CONTEXT_OPACITY,
        image,
        1.0 - CONTEXT_OPACITY,
        0.0,
        dst=image,
    )


def _draw_panel_header(image, title: str, frame: int) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    label = f"{title}   FRAME {frame}"
    scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, scale, thickness
    )
    cv2.rectangle(
        image,
        (8, 8),
        (min(image.shape[1] - 1, text_width + 26), text_height + baseline + 20),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        image,
        label,
        (16, text_height + 13),
        font,
        scale,
        (245, 245, 245),
        thickness,
        cv2.LINE_AA,
    )


def _gt_boxes_for_panel(
    sequence: SequenceData,
    frame: int,
    event: ErrorEvent,
    config: PipelineConfig,
) -> list[GTBox]:
    """Return evaluated pedestrian tracks plus non-target event partners."""
    target_classes = set(config.classes["TARGET_CLASSES"])
    selected: dict[int, GTBox] = {
        box.gt_id: box
        for box in sequence.gt_by_frame.get(frame, [])
        if box.flag != 0 and box.cls in target_classes
    }
    for gt_id in (event.focal_gt_id, event.partner_gt_id):
        box = _find_box(sequence, frame, gt_id)
        if box is not None:
            selected[box.gt_id] = box
    return [selected[gt_id] for gt_id in sorted(selected)]


def _render_side_by_side_frame(
    image,
    frame: int,
    event: ErrorEvent,
    sequence: SequenceData,
    matching: MatchingResult,
    config: PipelineConfig,
):
    import cv2

    tracker_panel = image.copy()
    gt_panel = image.copy()
    target_tracker_ids: set[int] = set()
    matched_tracker_id = matching.match.get(frame, {}).get(event.focal_gt_id)
    if matched_tracker_id is not None:
        target_tracker_ids.add(matched_tracker_id)
    # A hijacked tracker remains the event target even while TrackEval leaves it
    # unmatched after it locks onto a static-person annotation.
    if event.error_type == "FP_ID_HIJACK" and event.prev_trk_id is not None:
        target_tracker_ids.add(event.prev_trk_id)

    tracker_boxes = sorted(
        sequence.tracker_by_frame.get(frame, []), key=lambda item: item.trk_id
    )
    _draw_context_boxes(
        tracker_panel,
        [box.bbox for box in tracker_boxes if box.trk_id not in target_tracker_ids],
        TRACKER_COLOR,
    )
    for box in tracker_boxes:
        if box.trk_id not in target_tracker_ids:
            continue
        matched_gt_id = matching.reverse.get(frame, {}).get(box.trk_id)
        visibility_box = _find_box(sequence, frame, matched_gt_id)
        if visibility_box is None and event.error_type == "FP_ID_HIJACK":
            visibility_box = _find_box(sequence, frame, event.partner_gt_id)
        if visibility_box is None:
            visibility_box = _find_box(sequence, frame, event.focal_gt_id)
        _draw_labeled_box(
            tracker_panel,
            box.bbox,
            _target_label("T", box.trk_id, visibility_box),
            FOCAL_COLOR,
            thickness=3,
        )

    target_gt_ids = {event.focal_gt_id}
    for tracker_id in target_tracker_ids:
        matched_gt_id = matching.reverse.get(frame, {}).get(tracker_id)
        if matched_gt_id is not None:
            target_gt_ids.add(matched_gt_id)
    if event.error_type == "FP_ID_HIJACK" and event.partner_gt_id is not None:
        target_gt_ids.add(event.partner_gt_id)

    gt_boxes = _gt_boxes_for_panel(sequence, frame, event, config)
    _draw_context_boxes(
        gt_panel,
        [box.bbox for box in gt_boxes if box.gt_id not in target_gt_ids],
        GT_COLOR,
    )
    for box in gt_boxes:
        if box.gt_id not in target_gt_ids:
            continue
        _draw_labeled_box(
            gt_panel,
            box.bbox,
            _target_label("G", box.gt_id, box),
            FOCAL_COLOR,
            thickness=3,
        )

    _draw_panel_header(tracker_panel, "TRACKER OUTPUT", frame)
    _draw_panel_header(gt_panel, "GROUND TRUTH", frame)
    combined = cv2.hconcat([tracker_panel, gt_panel])
    cv2.line(
        combined,
        (sequence.width - 1, 0),
        (sequence.width - 1, combined.shape[0] - 1),
        (245, 245, 245),
        2,
    )
    return combined


def render_event_clip(
    event: ErrorEvent,
    sequence: SequenceData,
    matching: MatchingResult,
    output_dir: Path,
    config: PipelineConfig,
) -> None:
    import cv2

    radius = int(config.thresholds["EVENT_WINDOW"])
    frames = [
        frame
        for frame in sequence.selected_frames
        if event.frame_start - radius <= frame <= event.frame_end + radius
    ]
    if not frames:
        raise ValueError(f"No tracker-authorized frames available for event {event.event_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_path = output_dir / f"{event.event_id}.mp4"
    sidecar_path = output_dir / f"{event.event_id}.json"
    encoded_width = sequence.width * 2
    encoded_height = sequence.height + sequence.height % 2
    if clip_path.is_file() and clip_path.stat().st_size > 0 and sidecar_path.is_file():
        try:
            existing = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = None
        if (
            isinstance(existing, dict)
            and existing.get("render_layout") == RENDER_LAYOUT
            and existing.get("event") == event.to_dict()
            and existing.get("rendered_frames") == frames
            and existing.get("encoded_frame_size") == [encoded_width, encoded_height]
        ):
            return
    missing = [frame for frame in frames if not Path(sequence.image_paths[frame]).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Cannot render {event.event_id}; missing source frames {missing[:10]}"
        )
    temporary_clip_path = clip_path.with_name(f".{clip_path.stem}.rendering.mp4")
    if temporary_clip_path.exists():
        temporary_clip_path.unlink()
    writer = cv2.VideoWriter(
        str(temporary_clip_path),
        cv2.VideoWriter_fourcc(*config.clip_codec),
        sequence.fps,
        (encoded_width, encoded_height),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"OpenCV could not create {temporary_clip_path} with codec {config.clip_codec}"
        )
    render_succeeded = False
    try:
        for frame in frames:
            image = cv2.imread(sequence.image_paths[frame])
            if image is None:
                raise RuntimeError(f"OpenCV could not read {sequence.image_paths[frame]}")
            combined = _render_side_by_side_frame(
                image, frame, event, sequence, matching, config
            )
            if combined.shape[1] != encoded_width or combined.shape[0] != encoded_height:
                combined = cv2.copyMakeBorder(
                    combined,
                    0,
                    encoded_height - combined.shape[0],
                    0,
                    encoded_width - combined.shape[1],
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
            writer.write(combined)
        render_succeeded = True
    finally:
        writer.release()
        if not render_succeeded and temporary_clip_path.exists():
            temporary_clip_path.unlink()
    os.replace(temporary_clip_path, clip_path)
    sidecar = {
        "render_layout": RENDER_LAYOUT,
        "event": event.to_dict(),
        "rendered_frames": frames,
        "source_frame_size": [sequence.width, sequence.height],
        "panel_frame_size": [sequence.width, encoded_height],
        "encoded_frame_size": [encoded_width, encoded_height],
        "panels": {
            "left": {
                "label": "TRACKER OUTPUT",
                "annotations": (
                    "focal GT's per-frame TrackEval-matched tracker box is fully "
                    "opaque and labeled T:<track_id> V:<GT_visibility>; all other "
                    "tracker boxes are unlabeled at 10% opacity"
                ),
            },
            "right": {
                "label": "GROUND TRUTH",
                "annotations": (
                    "focal/matching GT boxes are fully opaque and labeled "
                    "G:<gt_id> V:<visibility>; all other displayed GT boxes are "
                    "unlabeled at 10% opacity"
                ),
            },
        },
        "focus_style": {
            "target_box_opacity": 1.0,
            "context_box_opacity": CONTEXT_OPACITY,
            "context_labels": False,
            "target_label_background_opacity": 0.0,
            "target_labels_include_visibility": True,
            "visibility_format": "V:<GT_visibility_2dp>",
            "focal_gt_id": event.focal_gt_id,
        },
        "note": (
            "Withheld scoring labels; do not provide this sidecar to annotators "
            "or the VLM. Video overlays show target IDs and GT visibility, but do "
            "not show cause, role, class, detector confidence, or cause confidence."
        ),
    }
    temporary_sidecar_path = sidecar_path.with_name(f".{sidecar_path.name}.rendering")
    with temporary_sidecar_path.open("w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_sidecar_path, sidecar_path)


def render_class_audit(sequence: SequenceData, output_dir: Path) -> list[Path]:
    """Create deterministic contact sheets for the required class-ID eyeball audit."""
    import cv2
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    by_class: dict[int, list[GTBox]] = {}
    for frame in sequence.selected_frames:
        for box in sequence.gt_by_frame.get(frame, []):
            by_class.setdefault(box.cls, []).append(box)
    outputs: list[Path] = []
    for class_id, boxes in sorted(by_class.items()):
        if not boxes:
            continue
        sample_count = min(6, len(boxes))
        sample_indices = sorted(
            {
                round(index * (len(boxes) - 1) / max(1, sample_count - 1))
                for index in range(sample_count)
            }
        )
        tiles = []
        for index in sample_indices:
            box = boxes[index]
            image = cv2.imread(sequence.image_paths[box.frame])
            if image is None:
                raise RuntimeError(
                    f"OpenCV could not read class-audit image {sequence.image_paths[box.frame]}"
                )
            x1, y1, x2, y2 = (int(round(value)) for value in box.bbox)
            pad_x = max(4, int((x2 - x1) * 0.1))
            pad_y = max(4, int((y2 - y1) * 0.1))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(image.shape[1], x2 + pad_x)
            y2 = min(image.shape[0], y2 + pad_y)
            crop = image[y1:y2, x1:x2]
            tile = np.full((260, 240, 3), 32, dtype=np.uint8)
            if crop.size:
                scale = min(240 / crop.shape[1], 220 / crop.shape[0])
                resized = cv2.resize(
                    crop,
                    (
                        max(1, int(round(crop.shape[1] * scale))),
                        max(1, int(round(crop.shape[0] * scale))),
                    ),
                )
                left = (240 - resized.shape[1]) // 2
                top = (220 - resized.shape[0]) // 2
                tile[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
            text = f"f={box.frame} id={box.gt_id} flag={box.flag} vis={box.visibility:.2f}"
            cv2.putText(
                tile,
                text,
                (5, 247),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            tiles.append(tile)
        blank = np.full_like(tiles[0], 32)
        while len(tiles) < 6:
            tiles.append(blank.copy())
        sheet = np.vstack((np.hstack(tiles[:3]), np.hstack(tiles[3:6])))
        cv2.putText(
            sheet,
            f"{sequence.name} observed GT class {class_id}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        path = output_dir / f"{sequence.name}__class_{class_id}.jpg"
        if not cv2.imwrite(str(path), sheet):
            raise RuntimeError(f"OpenCV could not write {path}")
        outputs.append(path)
    return outputs
