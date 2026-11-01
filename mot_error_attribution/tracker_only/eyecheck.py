"""Regime-B eye-check renders: instruction overlays, sampled sets, apply/release."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from mot_error_attribution.models import TrkBox
from mot_error_attribution.probe_clips import FPS, encode_h264
from mot_error_attribution.vlm_probe.summarize import write_csv
from vlm_clip_style_tests.render_annotation_styles import (
    STYLE_TOP_LEFT,
    _prediction_boxes,
    render_style_frame,
)

from .derive import highlight_id_for_frame, pinned_lifetime
from .io import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STYLE_ROOT,
    load_clip_bundle,
    load_json,
    load_manifest,
    write_atomic_json,
)
from .preview import PREVIEW_STYLE


REJOINED_NEGATIVES = (
    "C006",
    "C045",
    "C052",
    "C091",
    "C138",
    "C186",
    "C223",
)
LATE_MATCH_CLIPS = ("C028", "C067", "C149", "C180", "C181", "C219")
STANDEE_PRIMARY = "C063"
STANDEE_BACKUP = "C002"
STANDEE_TRACK_ID = 8
STANDEE_CLAIM_GT_IDS = (20, 21, 22, 23, 29, 35, 36)
E2_SEED = 20260904
E1_N = 10
E1_MIN_MOT20_03 = 4
E1_MIN_SMALL = 2
E1_CALIBRATION_N = 3
E2_OTHER_N = 8
E2_DECOY_N = 5
ROI_PAD = 40
SHADE_BLACK = 0.6
GHOST_BOX_COLOR = (180, 0, 255)
GT_BOX_COLOR = (80, 220, 80)
DASH_COLOR = (0, 140, 255)
BAND_BG = (18, 18, 18)
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.48
FONT_THICKNESS = 1
LINE_GAP = 6
BAND_MARGIN = 10
ANSWERS_FIELDS = ("clip_id", "set", "verdict", "note")
KEY_FIELDS = ("file_id", "clip_id", "role", "primary_error")
NEGATIVE_EXCLUDE_REASON = "eyecheck_negative_not_certified"
LATE_UNANSWERABLE_REASON = "late_match_unanswerable"

SET_E1 = "E1_fragmentation"
SET_E2 = "E2_negatives"
SET_E3 = "E3_late_match"
SET_E4 = "E4_standee"


def source_clip_id(clip_id: str) -> str:
    text = str(clip_id)
    if text.endswith("-T") or text.endswith("-G"):
        return text.rsplit("-", 1)[0]
    return text


def _even(value: int) -> int:
    return int(value) if int(value) % 2 == 0 else int(value) + 1


def _percentile(values: Sequence[float], pct: float) -> float:
    xs = sorted(float(item) for item in values)
    if not xs:
        return 0.0
    rank = (len(xs) - 1) * float(pct) / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def scored_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in payload.get("records") or () if row.get("scored_primary")]


def record_by_source(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in records:
        out[source_clip_id(str(row["clip_id"]))] = dict(row)
    return out


def box_heights_for_track(
    annotation: Mapping[str, Any],
    track_id: int,
    clip_frames: Sequence[int],
) -> list[float]:
    wanted = {int(frame) for frame in clip_frames}
    heights: list[float] = []
    for row in annotation.get("prediction_annotations") or ():
        if int(row["track_id"]) != int(track_id):
            continue
        if int(row["source_frame"]) not in wanted:
            continue
        x1, y1, x2, y2 = (float(item) for item in row["bbox_xyxy"])
        heights.append(max(0.0, y2 - y1))
    return heights


def median_height(heights: Sequence[float]) -> float:
    xs = sorted(float(item) for item in heights)
    if not xs:
        return 0.0
    mid = len(xs) // 2
    if len(xs) % 2:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def reacquisition_track_id(
    matching_by_frame: Sequence[Mapping[str, Any]],
    pin_id: int | None,
    pin_last_frame: int | None,
) -> int | None:
    if pin_id is None or pin_last_frame is None:
        return None
    for row in matching_by_frame:
        if int(row["source_frame"]) <= int(pin_last_frame):
            continue
        track_id = _optional_int(row.get("matched_prediction_track_id"))
        if track_id is None or int(track_id) == int(pin_id):
            continue
        return int(track_id)
    return None


def fragmentation_height(
    record: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> tuple[float, int | None]:
    pin = _optional_int(record.get("highlight_track_id"))
    clip_frames = list(bundle.get("clip_frames") or record.get("clip_frames") or ())
    lifetime = pinned_lifetime(pin, clip_frames, bundle.get("tracker_by_frame") or {})
    pin_last = max(lifetime) if lifetime else None
    successor = reacquisition_track_id(
        bundle.get("matching_by_frame") or (), pin, pin_last
    )
    track_for_height = successor if successor is not None else pin
    heights = (
        box_heights_for_track(bundle["annotation"], int(track_for_height), clip_frames)
        if track_for_height is not None
        else []
    )
    return median_height(heights), successor


def select_e1_fragmentation(
    records: Sequence[Mapping[str, Any]],
    height_by_clip: Mapping[str, float],
    *,
    n: int = E1_N,
    min_mot20_03: int = E1_MIN_MOT20_03,
    min_small: int = E1_MIN_SMALL,
    calibration_n: int = E1_CALIBRATION_N,
) -> list[dict[str, Any]]:
    pool = [
        dict(row)
        for row in records
        if row.get("scored_primary")
        and row.get("identity_change_subtype") == "fragmentation"
    ]
    if len(pool) < n:
        raise ValueError(f"Need {n} scored fragmentation clips, found {len(pool)}")
    heights = [float(height_by_clip[source_clip_id(str(row["clip_id"]))]) for row in pool]
    small_cut = _percentile(heights, 25)
    small = sorted(
        (
            row
            for row in pool
            if float(height_by_clip[source_clip_id(str(row["clip_id"]))]) <= small_cut
        ),
        key=lambda row: (
            float(height_by_clip[source_clip_id(str(row["clip_id"]))]),
            source_clip_id(str(row["clip_id"])),
        ),
    )
    mot03 = sorted(
        (row for row in pool if str(row.get("sequence")) == "MOT20-03"),
        key=lambda row: source_clip_id(str(row["clip_id"])),
    )
    chosen: dict[str, dict[str, Any]] = {}
    for row in small[:min_small]:
        chosen[source_clip_id(str(row["clip_id"]))] = row
    for row in mot03:
        n_mot = sum(1 for item in chosen.values() if item.get("sequence") == "MOT20-03")
        if n_mot >= min_mot20_03:
            break
        chosen[source_clip_id(str(row["clip_id"]))] = row
    for row in sorted(pool, key=lambda item: source_clip_id(str(item["clip_id"]))):
        if len(chosen) >= n:
            break
        chosen[source_clip_id(str(row["clip_id"]))] = row
    if len(chosen) < n:
        raise ValueError(f"Could not select {n} E1 clips")
    selected = list(chosen.values())
    n_mot = sum(1 for row in selected if row.get("sequence") == "MOT20-03")
    n_small = sum(
        1
        for row in selected
        if float(height_by_clip[source_clip_id(str(row["clip_id"]))]) <= small_cut
    )
    if n_mot < min_mot20_03 or n_small < min_small:
        raise ValueError(
            f"E1 strata unmet: MOT20-03={n_mot} (need {min_mot20_03}), "
            f"small={n_small} (need {min_small})"
        )
    by_large = sorted(
        selected,
        key=lambda row: (
            -float(height_by_clip[source_clip_id(str(row["clip_id"]))]),
            source_clip_id(str(row["clip_id"])),
        ),
    )
    calibration_ids = {
        source_clip_id(str(row["clip_id"])) for row in by_large[:calibration_n]
    }
    ordered = [row for row in by_large if source_clip_id(str(row["clip_id"])) in calibration_ids]
    ordered.extend(
        sorted(
            (
                row
                for row in selected
                if source_clip_id(str(row["clip_id"])) not in calibration_ids
            ),
            key=lambda row: source_clip_id(str(row["clip_id"])),
        )
    )
    items = []
    for index, row in enumerate(ordered, start=1):
        clip_id = source_clip_id(str(row["clip_id"]))
        items.append(
            {
                "clip_id": clip_id,
                "set": SET_E1,
                "sequence": row.get("sequence"),
                "highlight_track_id": row.get("highlight_track_id"),
                "median_box_height": float(height_by_clip[clip_id]),
                "small_box": float(height_by_clip[clip_id]) <= small_cut,
                "calibration": clip_id in calibration_ids,
                "index": index,
                "n": n,
                "first_match_clip_frame": row.get("first_match_clip_frame"),
                "n_clip_frames": len(row.get("clip_frames") or ()),
            }
        )
    return items


def select_e2_blind(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = E2_SEED,
    other_n: int = E2_OTHER_N,
    decoy_n: int = E2_DECOY_N,
    rejoined: Sequence[str] = REJOINED_NEGATIVES,
) -> list[dict[str, Any]]:
    by_source = record_by_source(
        [row for row in records if row.get("scored_primary")]
    )
    negatives = [
        source_clip_id(str(row["clip_id"]))
        for row in records
        if row.get("scored_primary") and row.get("primary_error") == "no_error"
    ]
    misses = [
        source_clip_id(str(row["clip_id"]))
        for row in records
        if row.get("scored_primary") and row.get("primary_error") == "miss"
    ]
    missing = [clip_id for clip_id in rejoined if clip_id not in negatives]
    if missing:
        raise ValueError(f"Rejoined negatives are not scored no_error: {missing}")
    other = sorted(clip_id for clip_id in negatives if clip_id not in set(rejoined))
    rng = random.Random(int(seed))
    sampled = sorted(rng.sample(other, other_n)) if other_n else []
    decoy_pool = [clip_id for clip_id in misses if clip_id not in set(rejoined)]
    decoys = sorted(rng.sample(decoy_pool, decoy_n))
    items: list[dict[str, Any]] = []
    for clip_id in rejoined:
        items.append(
            {
                "clip_id": clip_id,
                "role": "rejoined_negative",
                "primary_error": "no_error",
                "highlight_track_id": by_source[clip_id]["highlight_track_id"],
            }
        )
    for clip_id in sampled:
        items.append(
            {
                "clip_id": clip_id,
                "role": "sampled_negative",
                "primary_error": "no_error",
                "highlight_track_id": by_source[clip_id]["highlight_track_id"],
            }
        )
    for clip_id in decoys:
        items.append(
            {
                "clip_id": clip_id,
                "role": "decoy_miss",
                "primary_error": "miss",
                "highlight_track_id": by_source[clip_id]["highlight_track_id"],
            }
        )
    order = list(range(len(items)))
    rng.shuffle(order)
    blinded: list[dict[str, Any]] = []
    for index, source_index in enumerate(order, start=1):
        row = dict(items[source_index])
        row["file_id"] = f"E2_{index:02d}"
        row["set"] = SET_E2
        row["index"] = index
        row["n"] = len(items)
        blinded.append(row)
    return blinded


def select_e3_late(
    records: Sequence[Mapping[str, Any]],
    clip_ids: Sequence[str] = LATE_MATCH_CLIPS,
) -> list[dict[str, Any]]:
    by_source = record_by_source(records)
    items = []
    for index, clip_id in enumerate(clip_ids, start=1):
        row = by_source.get(clip_id)
        if row is None:
            raise ValueError(f"Late-match clip {clip_id} missing from freeze")
        n_frames = len(row.get("clip_frames") or ())
        items.append(
            {
                "clip_id": clip_id,
                "set": SET_E3,
                "highlight_track_id": row.get("highlight_track_id"),
                "first_match_clip_frame": row.get("first_match_clip_frame"),
                "n_clip_frames": n_frames,
                "index": index,
                "n": len(clip_ids),
            }
        )
    return items


def parse_xyxy(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    parts = [float(item) for item in str(value).split(";") if str(item).strip()]
    if len(parts) != 4:
        raise ValueError(f"Bad bbox {value!r}")
    return parts[0], parts[1], parts[2], parts[3]


def standee_roi(
    disagreement_rows: Sequence[Mapping[str, Any]],
    *,
    pad: int = ROI_PAD,
) -> tuple[int, int, int, int]:
    xs1: list[float] = []
    ys1: list[float] = []
    xs2: list[float] = []
    ys2: list[float] = []
    for row in disagreement_rows:
        x1, y1, x2, y2 = parse_xyxy(row.get("example_bbox_xyxy"))
        xs1.append(x1)
        ys1.append(y1)
        xs2.append(x2)
        ys2.append(y2)
    if not xs1:
        raise ValueError("No ghost disagreement boxes to union")
    return (
        int(math.floor(min(xs1) - pad)),
        int(math.floor(min(ys1) - pad)),
        int(math.ceil(max(xs2) + pad)),
        int(math.ceil(max(ys2) + pad)),
    )


def clamp_roi(
    roi: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    x1 = max(0, min(int(x1), int(width) - 1))
    y1 = max(0, min(int(y1), int(height) - 1))
    x2 = max(x1 + 1, min(int(x2), int(width)))
    y2 = max(y1 + 1, min(int(y2), int(height)))
    return x1, y1, x2, y2


def e1_overlay_lines(item: Mapping[str, Any]) -> list[str]:
    mode = (
        "calibration (reacquisition dashed)"
        if item.get("calibration")
        else "unaided (no extra marker)"
    )
    return [
        f"[E1-FRAG]  {item['clip_id']}   |  EXPECTED: fragmentation",
        "WATCH FOR: highlighted box ends, then an UNHIGHLIGHTED white box",
        "           with a NEW ID appears on the SAME person",
        "ANSWER:    Is that unhighlighted box clearly legible?   PASS / FAIL",
        f"MODE:      {mode}",
    ]


def e2_overlay_lines(item: Mapping[str, Any]) -> list[str]:
    label = str(item.get("file_id") or item["clip_id"])
    return [
        f"[E2-CLEAN?]  {label}",
        "WATCH FOR: anything that looks like a tracker failure on the highlighted track",
        "ANSWER:    Does this track look correct throughout?   YES / NO",
    ]


def e3_overlay_lines(item: Mapping[str, Any]) -> list[str]:
    first = int(item.get("first_match_clip_frame") or 0)
    n_frames = int(item.get("n_clip_frames") or 0)
    return [
        f"[E3-LATE]  {item['clip_id']}   |  pinned track first appears at clip frame {first} of {n_frames}",
        "WATCH FOR: whether there is enough highlighted track to judge anything",
        "ANSWER:    Is this clip answerable?   YES / NO",
    ]


def e4_overlay_lines(item: Mapping[str, Any]) -> list[str]:
    gt_ids = item.get("claim_gt_ids") or STANDEE_CLAIM_GT_IDS
    gt_text = ", ".join(str(item) for item in gt_ids)
    return [
        f"[E4-STANDEE]  {item['clip_id']}",
        "CLAIM: tracker ID 8 is a stationary false positive on the standee.",
        f"       TrackEval credits it against GT {gt_text}",
        "       across 258 frames in 12 clips.",
        "WATCH FOR: is the ID-8 box stationary and person-sized with NO person in it?",
        "           does a GT box get matched to it as someone walks past?",
        "ANSWER:    Confirmed / Not confirmed",
    ]


def bottom_band_text(item: Mapping[str, Any], *, extra: str = "") -> str:
    pin = item.get("highlight_track_id")
    pin_text = "none" if pin in (None, "") else str(int(pin))
    index = int(item.get("index") or 1)
    n_set = int(item.get("n") or 1)
    text = f"pinned track ID = {pin_text}  |  clip {index} of {n_set} in this set"
    if extra:
        text = f"{text}  |  {extra}"
    return text


def overlay_leaks_primary_error(lines: Sequence[str]) -> bool:
    blob = " ".join(lines).lower()
    for token in ("no_error", "primary_error", "decoy", "miss", "identity_change"):
        if token in blob:
            return True
    return False


def _cv2():
    import cv2

    return cv2


def wrap_line(text: str, max_width: int, *, font_scale: float = FONT_SCALE) -> list[str]:
    cv2 = _cv2()
    text = str(text)
    if not text:
        return [""]
    if cv2.getTextSize(text, FONT, font_scale, FONT_THICKNESS)[0][0] <= max_width:
        return [text]
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        width = cv2.getTextSize(trial, FONT, font_scale, FONT_THICKNESS)[0][0]
        if width <= max_width:
            current = trial
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines or [text]


def line_height(font_scale: float = FONT_SCALE) -> int:
    cv2 = _cv2()
    size, baseline = cv2.getTextSize("Ag", FONT, font_scale, FONT_THICKNESS)
    return int(size[1] + baseline + LINE_GAP)


def band_height_for(lines: Sequence[str], width: int, *, font_scale: float = FONT_SCALE) -> int:
    wrapped: list[str] = []
    max_width = max(20, int(width) - 2 * BAND_MARGIN)
    for line in lines:
        wrapped.extend(wrap_line(line, max_width, font_scale=font_scale))
    raw = 2 * BAND_MARGIN + max(1, len(wrapped)) * line_height(font_scale)
    return _even(max(40, raw))


def draw_text_block(
    canvas: np.ndarray,
    lines: Sequence[str],
    *,
    origin_y: int,
    width: int,
    font_scale: float = FONT_SCALE,
    color: tuple[int, int, int] = (240, 240, 240),
) -> None:
    cv2 = _cv2()
    max_width = max(20, int(width) - 2 * BAND_MARGIN)
    y = int(origin_y)
    for line in lines:
        for wrapped in wrap_line(line, max_width, font_scale=font_scale):
            y += line_height(font_scale)
            cv2.putText(
                canvas,
                wrapped,
                (BAND_MARGIN, y - 4),
                FONT,
                font_scale,
                color,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )


def compose_banded_frame(
    image: np.ndarray,
    *,
    top_lines: Sequence[str],
    bottom_text: str,
    frame_label: str,
) -> np.ndarray:
    cv2 = _cv2()
    height, width = image.shape[:2]
    top_h = band_height_for(top_lines, width)
    bottom_lines = [bottom_text, frame_label]
    bottom_h = band_height_for(bottom_lines, width)
    canvas = np.zeros((height + top_h + bottom_h, width, 3), dtype=np.uint8)
    canvas[:] = BAND_BG
    canvas[top_h : top_h + height] = image
    draw_text_block(canvas, top_lines, origin_y=0, width=width)
    cv2.rectangle(canvas, (0, top_h + height), (width, height + top_h + bottom_h), BAND_BG, -1)
    draw_text_block(
        canvas,
        bottom_lines,
        origin_y=top_h + height,
        width=width,
    )
    return canvas


def draw_dashed_rect(
    image: np.ndarray,
    xyxy: Sequence[float],
    color: tuple[int, int, int] = DASH_COLOR,
    *,
    thickness: int = 2,
    dash: int = 8,
    gap: int = 6,
) -> None:
    cv2 = _cv2()
    x1, y1, x2, y2 = (int(round(item)) for item in xyxy)
    points = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    for start, end in zip(points, points[1:] + points[:1]):
        x0, y0 = start
        x3, y3 = end
        length = math.hypot(x3 - x0, y3 - y0)
        if length <= 0:
            continue
        dx = (x3 - x0) / length
        dy = (y3 - y0) / length
        pos = 0.0
        draw = True
        while pos < length:
            span = dash if draw else gap
            nxt = min(length, pos + span)
            if draw:
                p1 = (int(round(x0 + dx * pos)), int(round(y0 + dy * pos)))
                p2 = (int(round(x0 + dx * nxt)), int(round(y0 + dy * nxt)))
                cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)
            pos = nxt
            draw = not draw


def _read_source_frame(
    *,
    dataset_root: Path,
    sequence: str,
    frame: int,
    expected_width: int,
    expected_height: int,
) -> np.ndarray:
    cv2 = _cv2()
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
    return image


def _boxes_for_track(boxes: Sequence[TrkBox], track_id: int | None) -> list[TrkBox]:
    if track_id is None:
        return []
    wanted = int(track_id)
    return [box for box in boxes if int(box.trk_id) == wanted]


def _gt_boxes_by_frame(annotation: Mapping[str, Any]) -> dict[int, list[TrkBox]]:
    by_frame: dict[int, list[TrkBox]] = {}
    for row in annotation.get("gt_annotations") or ():
        frame = int(row["source_frame"])
        by_frame.setdefault(frame, []).append(
            TrkBox(
                frame=frame,
                trk_id=int(row["track_id"]),
                bbox=tuple(map(float, row["bbox_xyxy"])),
            )
        )
    return by_frame


def _matched_gt_ids(
    annotation: Mapping[str, Any],
    frame: int,
    track_id: int,
) -> list[int]:
    ids: list[int] = []
    for row in annotation.get("prediction_annotations") or ():
        if int(row["source_frame"]) != int(frame):
            continue
        if int(row["track_id"]) != int(track_id):
            continue
        matched = _optional_int(row.get("matched_gt_id"))
        if matched is not None and matched not in ids:
            ids.append(matched)
    return ids


def _put_box_label(
    image: np.ndarray,
    xyxy: Sequence[float],
    text: str,
    color: tuple[int, int, int],
) -> None:
    cv2 = _cv2()
    x1, y1, _, _ = (int(round(item)) for item in xyxy)
    cv2.putText(
        image,
        text,
        (x1, max(14, y1 - 6)),
        FONT,
        FONT_SCALE,
        color,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )


def _draw_solid_box(
    image: np.ndarray,
    xyxy: Sequence[float],
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
) -> None:
    cv2 = _cv2()
    x1, y1, x2, y2 = (int(round(item)) for item in xyxy)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def shade_outside_roi(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    shade: float = SHADE_BLACK,
) -> np.ndarray:
    x1, y1, x2, y2 = roi
    keep = 1.0 - float(shade)
    shaded = np.clip(image.astype(np.float32) * keep, 0, 255).astype(np.uint8)
    shaded[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return shaded


def draw_inset(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    scale: float = 2.0,
    margin: int = 12,
) -> np.ndarray:
    cv2 = _cv2()
    x1, y1, x2, y2 = roi
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return image
    height, width = image.shape[:2]
    inset_w = min(int(round((x2 - x1) * scale)), width - 2 * margin)
    inset_h = min(int(round((y2 - y1) * scale)), height - 2 * margin)
    inset_w = max(2, _even(inset_w))
    inset_h = max(2, _even(inset_h))
    inset = cv2.resize(crop, (inset_w, inset_h), interpolation=cv2.INTER_NEAREST)
    x0 = width - margin - inset_w
    y0 = margin
    image[y0 : y0 + inset_h, x0 : x0 + inset_w] = inset
    cv2.rectangle(
        image,
        (x0, y0),
        (x0 + inset_w, y0 + inset_h),
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    return image


def render_style_with_optional_dash(
    image: np.ndarray,
    boxes: Sequence[TrkBox],
    highlight_id: int | None,
    *,
    dash_track_id: int | None,
) -> np.ndarray:
    canvas = render_style_frame(image, boxes, highlight_id, PREVIEW_STYLE)
    if dash_track_id is None:
        return canvas
    for box in _boxes_for_track(boxes, dash_track_id):
        draw_dashed_rect(canvas, box.bbox)
    return canvas


def iter_regime_b_frames(
    *,
    bundle: Mapping[str, Any],
    dataset_root: Path,
    highlight_track_id: int | None,
    dash_track_id: int | None = None,
) -> Iterable[tuple[int, np.ndarray]]:
    annotation = bundle["annotation"]
    sequence = str(annotation["sequence"])
    by_frame = _prediction_boxes(dict(annotation))
    tracker_ids_by_frame = {
        frame: [int(box.trk_id) for box in boxes] for frame, boxes in by_frame.items()
    }
    width = int(annotation["video"]["width"])
    height = int(annotation["video"]["height"])
    for raw_frame in annotation["clip_frames"]:
        frame = int(raw_frame)
        image = _read_source_frame(
            dataset_root=dataset_root,
            sequence=sequence,
            frame=frame,
            expected_width=width,
            expected_height=height,
        )
        highlight_id = highlight_id_for_frame(
            highlight_track_id, frame, tracker_ids_by_frame
        )
        yield frame, render_style_with_optional_dash(
            image,
            by_frame.get(frame, []),
            highlight_id,
            dash_track_id=dash_track_id,
        )


def iter_standee_frames(
    *,
    bundle: Mapping[str, Any],
    dataset_root: Path,
    roi: tuple[int, int, int, int],
    ghost_track_id: int = STANDEE_TRACK_ID,
) -> Iterable[tuple[int, np.ndarray]]:
    cv2 = _cv2()
    annotation = bundle["annotation"]
    sequence = str(annotation["sequence"])
    by_frame = _prediction_boxes(dict(annotation))
    gt_by_frame = _gt_boxes_by_frame(annotation)
    width = int(annotation["video"]["width"])
    height = int(annotation["video"]["height"])
    clamped = clamp_roi(roi, width, height)
    for raw_frame in annotation["clip_frames"]:
        frame = int(raw_frame)
        image = _read_source_frame(
            dataset_root=dataset_root,
            sequence=sequence,
            frame=frame,
            expected_width=width,
            expected_height=height,
        )
        canvas = shade_outside_roi(image, clamped)
        ghost_boxes = _boxes_for_track(by_frame.get(frame, []), ghost_track_id)
        for box in ghost_boxes:
            _draw_solid_box(canvas, box.bbox, GHOST_BOX_COLOR, thickness=2)
            _put_box_label(canvas, box.bbox, "TRACKER ID 8 (alleged ghost)", GHOST_BOX_COLOR)
        matched_ids = _matched_gt_ids(annotation, frame, ghost_track_id)
        for gt_box in gt_by_frame.get(frame, []):
            if int(gt_box.trk_id) not in set(matched_ids):
                continue
            _draw_solid_box(canvas, gt_box.bbox, GT_BOX_COLOR, thickness=2)
            _put_box_label(canvas, gt_box.bbox, f"GT {int(gt_box.trk_id)}", GT_BOX_COLOR)
        canvas = draw_inset(canvas, clamped)
        cv2.rectangle(
            canvas,
            (clamped[0], clamped[1]),
            (clamped[2], clamped[3]),
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        yield frame, canvas


def encode_overlay_clip(
    *,
    frames: Iterable[tuple[int, np.ndarray]],
    top_lines: Sequence[str],
    bottom_text: str,
    output_path: Path,
    n_frames: int,
    source_width: int,
    source_height: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample = compose_banded_frame(
        np.zeros((source_height, source_width, 3), dtype=np.uint8),
        top_lines=top_lines,
        bottom_text=bottom_text,
        frame_label="frame 999/999  source 99999",
    )
    height, width = sample.shape[:2]
    width = _even(width)
    height = _even(height)

    def banded() -> Iterable[np.ndarray]:
        for index, (source_frame, frame) in enumerate(frames, start=1):
            labeled = compose_banded_frame(
                frame,
                top_lines=top_lines,
                bottom_text=bottom_text,
                frame_label=f"frame {index}/{n_frames}  source {int(source_frame)}",
            )
            if labeled.shape[1] != width or labeled.shape[0] != height:
                pad_w = width - labeled.shape[1]
                pad_h = height - labeled.shape[0]
                labeled = np.pad(
                    labeled,
                    ((0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)),
                )
                labeled = labeled[:height, :width]
            yield labeled

    encode_h264(banded(), output_path, width, height, FPS)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return {
        "video": str(output_path),
        "width": width,
        "height": height,
        "n_frames": n_frames,
        "sha256": digest,
    }


def blank_answers(rows: Sequence[Mapping[str, Any]], set_name: str) -> list[dict[str, str]]:
    answers = []
    for row in rows:
        answers.append(
            {
                "clip_id": str(row.get("file_id") or row["clip_id"]),
                "set": set_name,
                "verdict": "",
                "note": "",
            }
        )
    return answers


def _write_answers(path: Path, rows: Sequence[Mapping[str, Any]], set_name: str) -> None:
    write_csv(path, blank_answers(rows, set_name), ANSWERS_FIELDS)


def build_selection(
    *,
    payload: Mapping[str, Any],
    bundles: Mapping[str, Mapping[str, Any]],
    disagreement_rows: Sequence[Mapping[str, Any]],
    seed: int = E2_SEED,
) -> dict[str, Any]:
    records = list(payload.get("records") or [])
    height_by_clip: dict[str, float] = {}
    successor_by_clip: dict[str, int | None] = {}
    for row in records:
        if not row.get("scored_primary"):
            continue
        if row.get("identity_change_subtype") != "fragmentation":
            continue
        clip_id = source_clip_id(str(row["clip_id"]))
        height, successor = fragmentation_height(row, bundles[clip_id])
        height_by_clip[clip_id] = height
        successor_by_clip[clip_id] = successor
    e1 = select_e1_fragmentation(records, height_by_clip)
    for item in e1:
        item["reacquisition_track_id"] = successor_by_clip.get(item["clip_id"])
    e2 = select_e2_blind(records, seed=seed)
    e3 = select_e3_late(records)
    roi = standee_roi(disagreement_rows)
    e4 = [
        {
            "clip_id": STANDEE_PRIMARY,
            "set": SET_E4,
            "role": "primary",
            "highlight_track_id": STANDEE_TRACK_ID,
            "index": 1,
            "n": 2,
            "claim_gt_ids": list(STANDEE_CLAIM_GT_IDS),
            "roi_xyxy": list(roi),
        },
        {
            "clip_id": STANDEE_BACKUP,
            "set": SET_E4,
            "role": "backup",
            "highlight_track_id": STANDEE_TRACK_ID,
            "index": 2,
            "n": 2,
            "claim_gt_ids": list(STANDEE_CLAIM_GT_IDS),
            "roi_xyxy": list(roi),
        },
    ]
    return {
        "schema_version": "eyecheck_selection_v1",
        "seed": int(seed),
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "e4": e4,
        "standee_roi_xyxy": list(roi),
        "e1_small_height_cutoff": _percentile(list(height_by_clip.values()), 25)
        if height_by_clip
        else None,
    }


def render_set_e1(
    *,
    items: Sequence[Mapping[str, Any]],
    bundles: Mapping[str, Mapping[str, Any]],
    dataset_root: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    rendered = []
    n_set = len(items)
    for item in items:
        clip_id = str(item["clip_id"])
        bundle = bundles[clip_id]
        dash_id = (
            item.get("reacquisition_track_id") if item.get("calibration") else None
        )
        top = e1_overlay_lines(item)
        bottom = bottom_band_text(item)
        path = output_dir / f"{clip_id}.mp4"
        print(f"  [E1 {item['index']}/{n_set}] {clip_id}", flush=True)
        info = encode_overlay_clip(
            frames=iter_regime_b_frames(
                bundle=bundle,
                dataset_root=dataset_root,
                highlight_track_id=item.get("highlight_track_id"),
                dash_track_id=None if dash_id in (None, "") else int(dash_id),
            ),
            top_lines=top,
            bottom_text=bottom,
            output_path=path,
            n_frames=len(bundle["annotation"]["clip_frames"]),
            source_width=int(bundle["annotation"]["video"]["width"]),
            source_height=int(bundle["annotation"]["video"]["height"]),
        )
        rendered.append({**dict(item), **info})
    return rendered


def render_set_e2(
    *,
    items: Sequence[Mapping[str, Any]],
    bundles: Mapping[str, Mapping[str, Any]],
    dataset_root: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    rendered = []
    n_set = len(items)
    for item in items:
        if overlay_leaks_primary_error(e2_overlay_lines(item)):
            raise ValueError("E2 overlay leaked a class label")
        clip_id = str(item["clip_id"])
        file_id = str(item["file_id"])
        bundle = bundles[clip_id]
        path = output_dir / f"{file_id}.mp4"
        print(f"  [E2 {item['index']}/{n_set}] {file_id}", flush=True)
        info = encode_overlay_clip(
            frames=iter_regime_b_frames(
                bundle=bundle,
                dataset_root=dataset_root,
                highlight_track_id=item.get("highlight_track_id"),
            ),
            top_lines=e2_overlay_lines(item),
            bottom_text=bottom_band_text(item),
            output_path=path,
            n_frames=len(bundle["annotation"]["clip_frames"]),
            source_width=int(bundle["annotation"]["video"]["width"]),
            source_height=int(bundle["annotation"]["video"]["height"]),
        )
        rendered.append({**dict(item), **info})
    return rendered


def render_set_e3(
    *,
    items: Sequence[Mapping[str, Any]],
    bundles: Mapping[str, Mapping[str, Any]],
    dataset_root: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    rendered = []
    n_set = len(items)
    for item in items:
        clip_id = str(item["clip_id"])
        bundle = bundles[clip_id]
        first = int(item.get("first_match_clip_frame") or 0)
        n_frames = int(item.get("n_clip_frames") or len(bundle["annotation"]["clip_frames"]))
        extra = f"first match at clip frame {first} of {n_frames}"
        path = output_dir / f"{clip_id}.mp4"
        print(f"  [E3 {item['index']}/{n_set}] {clip_id}", flush=True)
        info = encode_overlay_clip(
            frames=iter_regime_b_frames(
                bundle=bundle,
                dataset_root=dataset_root,
                highlight_track_id=item.get("highlight_track_id"),
            ),
            top_lines=e3_overlay_lines(item),
            bottom_text=bottom_band_text(item, extra=extra),
            output_path=path,
            n_frames=n_frames,
            source_width=int(bundle["annotation"]["video"]["width"]),
            source_height=int(bundle["annotation"]["video"]["height"]),
        )
        rendered.append({**dict(item), **info})
    return rendered


def render_set_e4(
    *,
    items: Sequence[Mapping[str, Any]],
    bundles: Mapping[str, Mapping[str, Any]],
    dataset_root: Path,
    output_dir: Path,
    roi: tuple[int, int, int, int],
) -> list[dict[str, Any]]:
    rendered = []
    n_set = len(items)
    for item in items:
        clip_id = str(item["clip_id"])
        bundle = bundles[clip_id]
        path = output_dir / f"{clip_id}.mp4"
        print(f"  [E4 {item['index']}/{n_set}] {clip_id}", flush=True)
        width = int(bundle["annotation"]["video"]["width"])
        height = int(bundle["annotation"]["video"]["height"])
        extra = "TRACKER ID 8 (alleged ghost)  |  full frame + 2x ROI inset"
        info = encode_overlay_clip(
            frames=iter_standee_frames(
                bundle=bundle,
                dataset_root=dataset_root,
                roi=roi,
            ),
            top_lines=e4_overlay_lines(item),
            bottom_text=bottom_band_text(item, extra=extra),
            output_path=path,
            n_frames=len(bundle["annotation"]["clip_frames"]),
            source_width=width,
            source_height=height,
        )
        rendered.append({**dict(item), **info})
    return rendered


def load_disagreement_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def dataset_root_from_summary(style_root: Path) -> Path:
    summary = load_json(style_root / "dataset_summary.json")
    return Path(summary["source"]["dataset_root"])


def needed_clip_ids(selection: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in ("e1", "e2", "e3", "e4"):
        for row in selection.get(key) or ():
            clip_id = str(row["clip_id"])
            if clip_id not in ids:
                ids.append(clip_id)
    return ids


def render_eyecheck(
    *,
    frozen_path: Path,
    output_root: Path,
    style_root: Path = DEFAULT_STYLE_ROOT,
    dataset_root: Path | None = None,
    seed: int = E2_SEED,
) -> dict[str, Any]:
    payload = load_json(frozen_path)
    disagreement_path = output_root / "gt_ghost_disagreement.csv"
    disagreements = load_disagreement_csv(disagreement_path)
    manifest = load_manifest(style_root)
    data_root = Path(dataset_root) if dataset_root else dataset_root_from_summary(style_root)
    height_ids = [
        source_clip_id(str(row["clip_id"]))
        for row in payload.get("records") or ()
        if row.get("scored_primary")
        and row.get("identity_change_subtype") == "fragmentation"
    ]
    print(f"Loading {len(height_ids)} fragmentation bundles for E1 strata", flush=True)
    height_bundles = {
        clip_id: load_clip_bundle(style_root, clip_id, manifest) for clip_id in height_ids
    }
    selection = build_selection(
        payload=payload,
        bundles=height_bundles,
        disagreement_rows=disagreements,
        seed=seed,
    )
    load_ids = needed_clip_ids(selection)
    bundles = dict(height_bundles)
    for clip_id in load_ids:
        if clip_id not in bundles:
            bundles[clip_id] = load_clip_bundle(style_root, clip_id, manifest)
    eye_root = output_root / "preview" / "eyecheck"
    e1_dir = eye_root / SET_E1
    e2_dir = eye_root / SET_E2
    e3_dir = eye_root / SET_E3
    e4_dir = eye_root / SET_E4
    roi = tuple(int(item) for item in selection["standee_roi_xyxy"])
    print(f"Rendering eye-check clips to {eye_root}", flush=True)
    e1_rendered = render_set_e1(
        items=selection["e1"],
        bundles=bundles,
        dataset_root=data_root,
        output_dir=e1_dir,
    )
    e2_rendered = render_set_e2(
        items=selection["e2"],
        bundles=bundles,
        dataset_root=data_root,
        output_dir=e2_dir,
    )
    e3_rendered = render_set_e3(
        items=selection["e3"],
        bundles=bundles,
        dataset_root=data_root,
        output_dir=e3_dir,
    )
    e4_rendered = render_set_e4(
        items=selection["e4"],
        bundles=bundles,
        dataset_root=data_root,
        output_dir=e4_dir,
        roi=(roi[0], roi[1], roi[2], roi[3]),
    )
    _write_answers(e1_dir / "ANSWERS.csv", e1_rendered, SET_E1)
    _write_answers(e2_dir / "ANSWERS.csv", e2_rendered, SET_E2)
    _write_answers(e3_dir / "ANSWERS.csv", e3_rendered, SET_E3)
    _write_answers(e4_dir / "ANSWERS.csv", e4_rendered, SET_E4)
    write_csv(
        e2_dir / "KEY.csv",
        [
            {
                "file_id": row["file_id"],
                "clip_id": row["clip_id"],
                "role": row["role"],
                "primary_error": row["primary_error"],
            }
            for row in e2_rendered
        ],
        KEY_FIELDS,
    )
    selection_out = {
        **selection,
        "e1": e1_rendered,
        "e2": e2_rendered,
        "e3": e3_rendered,
        "e4": e4_rendered,
        "style": STYLE_TOP_LEFT,
        "frozen": str(frozen_path),
        "note": "E2 class labels live in E2_negatives/KEY.csv. Do not open KEY until after scoring ANSWERS.csv.",
    }
    write_atomic_json(eye_root / "selection.json", selection_out)
    report = {
        "eye_root": str(eye_root),
        "n_e1": len(e1_rendered),
        "n_e2": len(e2_rendered),
        "n_e3": len(e3_rendered),
        "n_e4": len(e4_rendered),
        "standee_roi_xyxy": list(roi),
        "preview_pending_eye_check": True,
    }
    write_atomic_json(eye_root / "render_report.json", report)
    return report


def _norm_verdict(value: str) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    text = text.replace("'", "")
    aliases = {
        "pass": "pass",
        "fail": "fail",
        "dont know": "dont_know",
        "do not know": "dont_know",
        "uncertain": "dont_know",
        "yes": "yes",
        "no": "no",
        "fine": "yes",
        "clean": "yes",
        "ok": "yes",
        "miss": "no",
        "broken": "no",
        "clip is answerable": "yes",
        "answerable": "yes",
        "unanswerable": "no",
        "clip is unanswerable": "no",
        "confirmed": "confirmed",
        "not confirmed": "not_confirmed",
        "not_confirmed": "not_confirmed",
    }
    if text not in aliases:
        return text
    return aliases[text]


def read_answers_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def score_e2(
    answers: Sequence[Mapping[str, str]],
    key_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    key = {str(row["file_id"]): row for row in key_rows}
    marked_no: list[str] = []
    decoy_yes: list[str] = []
    rows = []
    for answer in answers:
        file_id = str(answer.get("clip_id") or "")
        meta = key[file_id]
        verdict = _norm_verdict(str(answer.get("verdict") or ""))
        role = str(meta["role"])
        true_label = str(meta["primary_error"])
        if role != "decoy_miss" and verdict == "no":
            marked_no.append(str(meta["clip_id"]))
        if role == "decoy_miss" and verdict == "yes":
            decoy_yes.append(str(meta["clip_id"]))
        rows.append(
            {
                "file_id": file_id,
                "clip_id": meta["clip_id"],
                "role": role,
                "true_primary_error": true_label,
                "verdict": verdict,
                "note": answer.get("note", ""),
            }
        )
    return {
        "rows": rows,
        "exclude_negatives": marked_no,
        "decoy_marked_yes": decoy_yes,
    }


def collect_eyecheck_results(eye_root: Path) -> dict[str, Any]:
    e1 = read_answers_csv(eye_root / SET_E1 / "ANSWERS.csv")
    e2_answers = read_answers_csv(eye_root / SET_E2 / "ANSWERS.csv")
    e2_key = read_answers_csv(eye_root / SET_E2 / "KEY.csv")
    e3 = read_answers_csv(eye_root / SET_E3 / "ANSWERS.csv")
    e4 = read_answers_csv(eye_root / SET_E4 / "ANSWERS.csv")
    for name, rows in (
        ("E1", e1),
        ("E2", e2_answers),
        ("E3", e3),
        ("E4", e4),
    ):
        missing = [row.get("clip_id") for row in rows if not str(row.get("verdict") or "").strip()]
        if missing:
            raise ValueError(f"{name} ANSWERS.csv has blank verdicts: {missing}")
    e1_fail = [str(row["clip_id"]) for row in e1 if _norm_verdict(row.get("verdict", "")) == "fail"]
    e2_score = score_e2(e2_answers, e2_key)
    e3_no = [
        str(row["clip_id"])
        for row in e3
        if _norm_verdict(row.get("verdict", "")) == "no"
    ]
    e4_primary = next((row for row in e4 if str(row.get("clip_id")) == STANDEE_PRIMARY), e4[0] if e4 else {})
    n_fail = len(e1_fail)
    e1_uncertain = [
        str(row["clip_id"])
        for row in e1
        if _norm_verdict(row.get("verdict", "")) == "dont_know"
    ]
    e4_verdict = _norm_verdict(str(e4_primary.get("verdict") or ""))
    if e4_verdict == "yes":
        e4_verdict = "confirmed"
    return {
        "schema_version": "eyecheck_results_v1",
        "e1": {
            "n": len(e1),
            "n_fail": n_fail,
            "n_uncertain": len(e1_uncertain),
            "passed": n_fail < 3,
            "fail_clip_ids": e1_fail,
            "uncertain_clip_ids": e1_uncertain,
            "answers": e1,
        },
        "e2": e2_score,
        "e3": {"unanswerable": e3_no, "answers": e3},
        "e4": {
            "verdict": e4_verdict,
            "answers": e4,
            "gates_release": False,
        },
        "unscore": {
            clip_id: NEGATIVE_EXCLUDE_REASON
            for clip_id in e2_score["exclude_negatives"]
        }
        | {clip_id: LATE_UNANSWERABLE_REASON for clip_id in e3_no},
    }


def apply_unscore(
    payload: Mapping[str, Any],
    unscore: Mapping[str, str],
) -> dict[str, Any]:
    updated = json.loads(json.dumps(payload))
    for row in updated.get("records") or ():
        clip_id = source_clip_id(str(row["clip_id"]))
        if clip_id not in unscore:
            continue
        reason = unscore[clip_id]
        existing = [item for item in str(row.get("unscored_reason") or "").split(";") if item]
        if reason not in existing:
            existing.append(reason)
        row["unscored_reason"] = ";".join(existing)
        row["scored_primary"] = False
    n_neg = sum(
        1
        for row in updated.get("records") or ()
        if row.get("scored_primary") and row.get("primary_error") == "no_error"
    )
    updated["eyecheck"] = {
        "applied": True,
        "n_scored_no_error": n_neg,
        "negative_set_below_40": n_neg < 40,
    }
    return updated


def apply_eyecheck(
    *,
    frozen_path: Path,
    eye_root: Path,
    results_path: Path | None = None,
    release_path: Path | None = None,
) -> dict[str, Any]:
    results = collect_eyecheck_results(eye_root)
    out_results = results_path or eye_root / "eyecheck_results.json"
    write_atomic_json(out_results, results)
    if not results["e1"]["passed"]:
        raise ValueError(
            "E1 failed (3 or more of 10 FAIL). Do not release. "
            "Increase unhighlighted stroke weight or collapse identity_change with miss."
        )
    if release_path is None:
        return results
    payload = load_json(frozen_path)
    released = apply_unscore(payload, results["unscore"])
    released["preview_pending_eye_check"] = False
    released["unreleased"] = False
    released["frozen"] = True
    released["eyecheck_results"] = str(out_results)
    n_neg = int(released["eyecheck"]["n_scored_no_error"])
    if n_neg < 40:
        released["eyecheck"]["note"] = (
            f"Negative set is {n_neg} (<40); d-prime is still computable but the interval widens."
        )
    write_atomic_json(release_path, released)
    return results


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Render or apply the regime-B eye check")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--style-root", type=Path, default=DEFAULT_STYLE_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--frozen",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "gt_tracker_only_frozen_v2.json",
    )
    parser.add_argument("--seed", type=int, default=E2_SEED)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--release", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root).resolve()
    eye_root = output_root / "preview" / "eyecheck"
    if args.apply:
        release_path = (
            output_root / "gt_tracker_only_frozen_released.json" if args.release else None
        )
        results = apply_eyecheck(
            frozen_path=Path(args.frozen),
            eye_root=eye_root,
            release_path=release_path,
        )
        print(json.dumps({"applied": True, "e1_passed": results["e1"]["passed"]}, indent=2))
        return 0
    report = render_eyecheck(
        frozen_path=Path(args.frozen),
        output_root=output_root,
        style_root=Path(args.style_root),
        dataset_root=args.dataset_root,
        seed=int(args.seed),
    )
    print(json.dumps(report, indent=2))
    return 0
