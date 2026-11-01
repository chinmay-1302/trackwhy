"""Post-pass: custom occluder polygons → masks → visible_miss_unconfirmed.

MOT20 classes 9/10/11 stay in ``gt.txt`` and are never merged into these masks.
The canonical v4 corpus is not modified; this writes a sidecar output tree.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
import os
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .config import DEFAULT_CLASSES, EVALUATION_SEQUENCES
from .geometry import xywh_to_xyxy, in_frame_fraction, border_edges
from .io import _read_gt
from .models import BBox, GTBox

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mot-error-attribution-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import median_filter


REF_FRAME = {"MOT20-01": 1, "MOT20-03": 1}
ORB_FEATURES = 2000
RANSAC_THRESH = 3.0
MIN_INLIER_RATIO = 0.3
STATIC_MAX_SHIFT = 10.0
STATIC_MAX_ROTATION_DEG = 0.5
MEDIAN_WINDOW = 5
IOA_MASK_THRESH = 0.10
SHAKE_SAMPLES = 50
CUSTOM_MIN_VERTICES = 5
MASK_MEAN_MIN = 0.001
MASK_MEAN_MAX = 0.25
MIN_REMAINDER_PX = 500
MOT20_03_MASK_MEAN_MIN = 0.007
CMC_MODE = "static_dilate"
DILATE_MARGIN = 40
UNCONFIRMED_WARN_FRACTION = 0.5
UNCONFIRMED_ROBUST_FRACTION = 0.15
UNCONFIRMED_MATERIAL_FRACTION = 0.40
TRUNC_THRESH = 0.85
TRUNC_BODY_FRACTION = 0.15
MARGIN_FLOOR_PX = 4.0
BORDER_MARGIN_PX = 15
TRUNCATION_WARN_FRACTION = 0.50
POST_CANOPY_VISIBLE_MISS = 270
POST_CANOPY_UNCONFIRMED = 12
OCCLUDER_CLASSES = tuple(DEFAULT_CLASSES["OCCLUDER_CLASSES"])

IDENTITY_AFFINE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


@dataclass(frozen=True)
class CocoPolygon:
    annotation_id: int
    image_id: int
    sequence: str
    category_name: str
    points: np.ndarray
    bbox: BBox

    @property
    def vertex_count(self) -> int:
        return int(self.points.shape[0])


@dataclass
class SequenceMeta:
    name: str
    num_frames: int
    width: int
    height: int
    im_dir: str
    im_ext: str
    dataset_root: Path

    def image_path(self, frame: int) -> Path:
        return (
            self.dataset_root
            / "train"
            / self.name
            / self.im_dir
            / f"{frame:06d}{self.im_ext}"
        )


@dataclass
class SplitResult:
    custom: list[CocoPolygon]
    mot20: list[CocoPolygon]
    dropped_overlap: list[CocoPolygon]


@dataclass
class SequenceMasks:
    sequence: str
    mode: str
    ref_mask: np.ndarray
    width: int
    height: int
    static_mask: np.ndarray | None = None
    matrices: dict[int, np.ndarray] | None = None
    dilate_margin: int | None = None

    def for_frame(self, frame: int) -> np.ndarray:
        if self.static_mask is not None:
            return self.static_mask
        return self.ref_mask


@dataclass
class PolygonResolution:
    polygon: CocoPolygon
    original_px: int
    remaining_px: int
    kept: bool
    reason: str | None = None


@dataclass
class ShakeRow:
    frame: int
    dx: float
    dy: float
    rotation_deg: float
    scale: float
    inlier_ratio: float
    used_static_region: bool


def segmentation_vertex_count(annotation: dict[str, Any]) -> int:
    segmentation = annotation.get("segmentation") or []
    if not segmentation:
        return 0
    return len(segmentation[0]) // 2


def points_from_segmentation(segmentation: list[list[float]]) -> np.ndarray:
    coords = np.asarray(segmentation[0], dtype=np.float32)
    if coords.size % 2:
        raise ValueError("COCO segmentation length is not even")
    return coords.reshape(-1, 2)


def sequence_from_file_name(file_name: str) -> str:
    stem = Path(file_name).stem
    if stem.startswith("MOT20-"):
        return stem
    for part in Path(file_name).parts:
        if part.startswith("MOT20-"):
            return part
    raise ValueError(f"Cannot find a MOT20 sequence name in image path: {file_name}")


def load_coco_export(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_coco_polygons(
    payload: dict[str, Any],
    sequences: Iterable[str] = EVALUATION_SEQUENCES,
) -> dict[str, SplitResult]:
    """Split COCO polygons by vertex count. MOT20 boxes are 4-vertex rectangles."""
    wanted = set(sequences)
    images = {image["id"]: image for image in payload["images"]}
    categories = {item["id"]: item["name"] for item in payload["categories"]}
    grouped: dict[str, SplitResult] = {
        sequence: SplitResult(custom=[], mot20=[], dropped_overlap=[])
        for sequence in wanted
    }
    for annotation in payload["annotations"]:
        image = images[annotation["image_id"]]
        sequence = sequence_from_file_name(str(image["file_name"]))
        if sequence not in grouped:
            continue
        points = points_from_segmentation(annotation["segmentation"])
        bbox = xywh_to_xyxy(tuple(map(float, annotation["bbox"])))
        polygon = CocoPolygon(
            annotation_id=int(annotation["id"]),
            image_id=int(annotation["image_id"]),
            sequence=sequence,
            category_name=categories[annotation["category_id"]],
            points=points,
            bbox=bbox,
        )
        if polygon.vertex_count >= CUSTOM_MIN_VERTICES:
            grouped[sequence].custom.append(polygon)
        else:
            grouped[sequence].mot20.append(polygon)
    return grouped


def boxes_to_mask(boxes: list[BBox], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        left = max(0, int(round(x1)))
        top = max(0, int(round(y1)))
        right = min(width, int(round(x2)))
        bottom = min(height, int(round(y2)))
        if right > left and bottom > top:
            mask[top:bottom, left:right] = 1
    return mask


def resolve_custom_region(
    poly_mask: np.ndarray, mot20_occluder_mask: np.ndarray
) -> tuple[np.ndarray, int]:
    """Keep only the part of a custom polygon not already covered by MOT20."""
    remainder = np.where(mot20_occluder_mask == 0, poly_mask, 0).astype(np.uint8)
    return remainder, int(remainder.sum())


def compose_custom_mask(
    polygons: list[CocoPolygon],
    mot20_occluder_mask: np.ndarray,
    height: int,
    width: int,
    min_remainder_px: int = MIN_REMAINDER_PX,
) -> tuple[np.ndarray, list[PolygonResolution]]:
    custom_mask = np.zeros((height, width), dtype=np.uint8)
    resolutions: list[PolygonResolution] = []
    for polygon in polygons:
        poly_mask = polygons_to_mask([polygon], height, width)
        remainder, remaining_px = resolve_custom_region(poly_mask, mot20_occluder_mask)
        original_px = int(poly_mask.sum())
        if remaining_px < min_remainder_px:
            resolutions.append(
                PolygonResolution(
                    polygon=polygon,
                    original_px=original_px,
                    remaining_px=remaining_px,
                    kept=False,
                    reason="fully_covered_by_mot20",
                )
            )
            continue
        custom_mask |= remainder
        resolutions.append(
            PolygonResolution(
                polygon=polygon,
                original_px=original_px,
                remaining_px=remaining_px,
                kept=True,
            )
        )
    return custom_mask, resolutions


def polygons_to_mask(
    polygons: list[np.ndarray] | list[CocoPolygon], height: int, width: int
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for item in polygons:
        points = item.points if isinstance(item, CocoPolygon) else item
        if len(points) < 3:
            continue
        cv2.fillPoly(mask, [np.int32(np.round(points))], 1)
    return mask


def ioa_mask(bbox: BBox, mask: np.ndarray) -> float:
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(mask.shape[1], x2)
    y2 = min(mask.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = mask[y1:y2, x1:x2]
    return float(crop.sum()) / float(crop.size)


def frames_for_event(event: dict[str, Any]) -> list[int]:
    evidence = event.get("evidence") or {}
    fn_frames = evidence.get("fn_frames")
    if fn_frames:
        return [int(frame) for frame in fn_frames]
    start = int(event["frame_start"])
    end = int(event["frame_end"])
    return list(range(start, end + 1))


def event_ioa_unannotated_structure(
    event: dict[str, Any],
    gt_by_id: dict[tuple[int, int], BBox],
    mask_for_frame: Callable[[int], np.ndarray],
) -> float:
    scores: list[float] = []
    focal_id = int(event["focal_gt_id"])
    for frame in frames_for_event(event):
        bbox = gt_by_id.get((frame, focal_id))
        if bbox is None:
            continue
        scores.append(ioa_mask(bbox, mask_for_frame(frame)))
    return max(scores) if scores else 0.0


def apply_visible_miss_postpass(
    events: list[dict[str, Any]],
    scores: list[float],
    threshold: float = IOA_MASK_THRESH,
) -> list[dict[str, Any]]:
    if len(events) != len(scores):
        raise ValueError("Event and score counts must match")
    updated: list[dict[str, Any]] = []
    for event, score in zip(events, scores):
        copy = deepcopy(event)
        evidence = dict(copy.get("evidence") or {})
        evidence["ioa_unannotated_structure"] = float(score)
        copy["evidence"] = evidence
        if copy.get("cause") == "visible_miss" and score >= threshold:
            copy["cause"] = "visible_miss_unconfirmed"
        updated.append(copy)
    return updated


def conservation_report(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> dict[str, Any]:
    def count(events: list[dict[str, Any]], cause: str) -> int:
        return sum(event.get("cause") == cause for event in events)

    old_visible = count(before, "visible_miss")
    new_visible = count(after, "visible_miss")
    unconfirmed = count(after, "visible_miss_unconfirmed")
    full_before = count(before, "full_occl_static")
    full_after = count(after, "full_occl_static")
    other_before = Counter(
        event.get("cause")
        for event in before
        if event.get("cause") not in {"visible_miss", "visible_miss_unconfirmed"}
    )
    other_after = Counter(
        event.get("cause")
        for event in after
        if event.get("cause") not in {"visible_miss", "visible_miss_unconfirmed"}
    )
    missing_ioa = [
        event.get("event_id")
        for event in after
        if "ioa_unannotated_structure" not in (event.get("evidence") or {})
    ]
    fraction = (unconfirmed / old_visible) if old_visible else 0.0
    return {
        "visible_miss_before": old_visible,
        "visible_miss_after": new_visible,
        "visible_miss_unconfirmed": unconfirmed,
        "visible_miss_conserved": new_visible + unconfirmed == old_visible,
        "full_occl_static_before": full_before,
        "full_occl_static_after": full_after,
        "full_occl_static_unchanged": full_before == full_after,
        "other_causes_unchanged": other_before == other_after,
        "missing_ioa_event_ids": missing_ioa,
        "unconfirmed_fraction": fraction,
        "warn_high_contamination": fraction > UNCONFIRMED_WARN_FRACTION,
    }


def margin_for_box(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    margin_x = max(MARGIN_FLOOR_PX, TRUNC_BODY_FRACTION * (x2 - x1))
    margin_y = max(MARGIN_FLOOR_PX, TRUNC_BODY_FRACTION * (y2 - y1))
    return margin_x, margin_y


def hybrid_truncation_edges(
    box: BBox,
    width: int,
    height: int,
    *,
    hybrid: bool,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Left/top use proximity when clipped; right/bottom use real overflow."""
    x1, y1, x2, y2 = (float(value) for value in box)
    margin_x, margin_y = margin_for_box(box)
    edges: list[str] = []
    tests: dict[str, str] = {}
    if x1 <= margin_x:
        edges.append("left")
        tests["left"] = "proximity"
    if y1 <= margin_y:
        edges.append("top")
        tests["top"] = "proximity"
    if hybrid:
        if x2 > width:
            edges.append("right")
            tests["right"] = "overflow"
        if y2 > height:
            edges.append("bottom")
            tests["bottom"] = "overflow"
    else:
        if x2 >= width - margin_x:
            edges.append("right")
            tests["right"] = "proximity"
        if y2 >= height - margin_y:
            edges.append("bottom")
            tests["bottom"] = "proximity"
    return tuple(edges), tests


def event_focal_boxes(
    event: dict[str, Any],
    gt_by_id: dict[tuple[int, int], BBox],
) -> list[BBox]:
    focal_id = int(event["focal_gt_id"])
    boxes: list[BBox] = []
    for frame in frames_for_event(event):
        bbox = gt_by_id.get((frame, focal_id))
        if bbox is not None:
            boxes.append(bbox)
    return boxes


def event_in_frame_fraction(
    event: dict[str, Any],
    gt_by_id: dict[tuple[int, int], BBox],
    width: int,
    height: int,
) -> float:
    boxes = event_focal_boxes(event, gt_by_id)
    if not boxes:
        return 1.0
    return min(in_frame_fraction(box, width, height) for box in boxes)


def event_hybrid_truncation(
    event: dict[str, Any],
    gt_by_id: dict[tuple[int, int], BBox],
    width: int,
    height: int,
    *,
    hybrid: bool,
) -> tuple[tuple[str, ...], dict[str, str]]:
    found: list[str] = []
    tests: dict[str, str] = {}
    for box in event_focal_boxes(event, gt_by_id):
        edges, edge_tests = hybrid_truncation_edges(
            box, width, height, hybrid=hybrid
        )
        for edge in edges:
            if edge not in found:
                found.append(edge)
            tests.setdefault(edge, edge_tests[edge])
    return tuple(found), tests


def event_legacy_t1b_edges(
    event: dict[str, Any],
    gt_by_id: dict[tuple[int, int], BBox],
    width: int,
    height: int,
    margin: float = BORDER_MARGIN_PX,
) -> tuple[str, ...]:
    found: list[str] = []
    for box in event_focal_boxes(event, gt_by_id):
        for edge in border_edges(box, width, height, margin):
            if edge not in found:
                found.append(edge)
    return tuple(found)


def apply_truncation_postpass(
    events: list[dict[str, Any]],
    gt_by_id: dict[tuple[int, int], BBox],
    width: int,
    height: int,
    *,
    hybrid: bool = True,
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for event in events:
        copy = deepcopy(event)
        evidence = dict(copy.get("evidence") or {})
        frac = event_in_frame_fraction(copy, gt_by_id, width, height)
        edges, tests = event_hybrid_truncation(
            copy, gt_by_id, width, height, hybrid=hybrid
        )
        evidence["in_frame_fraction"] = float(frac)
        evidence["trunc_edges"] = list(edges)
        evidence["truncation_edges"] = list(edges)
        evidence["trunc_test_by_edge"] = tests
        evidence["truncation_test"] = "hybrid" if hybrid else "proximity"
        evidence["touches_border"] = bool(edges)
        copy["evidence"] = evidence
        if copy.get("cause") in {"visible_miss", "visible_miss_truncated"}:
            copy["cause"] = (
                "visible_miss_truncated" if edges else "visible_miss"
            )
        updated.append(copy)
    return updated


def truncation_report(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> dict[str, Any]:
    def count(events: list[dict[str, Any]], cause: str) -> int:
        return sum(event.get("cause") == cause for event in events)

    before_visible = count(before, "visible_miss")
    before_truncated = count(before, "visible_miss_truncated")
    after_visible = count(after, "visible_miss")
    truncated = count(after, "visible_miss_truncated")
    unconfirmed_before = count(before, "visible_miss_unconfirmed")
    unconfirmed_after = count(after, "visible_miss_unconfirmed")
    pool = before_visible + before_truncated
    truncated_fracs = [
        float((event.get("evidence") or {}).get("in_frame_fraction", 1.0))
        for event in after
        if event.get("cause") == "visible_miss_truncated"
    ]
    edge_counts = Counter()
    for event in after:
        if event.get("cause") != "visible_miss_truncated":
            continue
        for edge in (event.get("evidence") or {}).get("trunc_edges") or (
            (event.get("evidence") or {}).get("truncation_edges") or []
        ):
            edge_counts[edge] += 1
    missing_frac = [
        event.get("event_id")
        for event in after
        if "in_frame_fraction" not in (event.get("evidence") or {})
    ]
    remaining_ok = all(
        not (event.get("evidence") or {}).get("trunc_edges")
        for event in after
        if event.get("cause") == "visible_miss"
    )
    truncated_ok = all(
        bool((event.get("evidence") or {}).get("trunc_edges"))
        for event in after
        if event.get("cause") == "visible_miss_truncated"
    )
    false_bottom = [
        event.get("event_id")
        for event in after
        if event.get("cause") == "visible_miss_truncated"
        and "bottom" in ((event.get("evidence") or {}).get("trunc_edges") or [])
        and float((event.get("evidence") or {}).get("in_frame_fraction", 0.0)) >= 0.999
    ]
    fraction = (truncated / pool) if pool else 0.0
    return {
        "visible_miss_before": before_visible,
        "visible_miss_after": after_visible,
        "visible_miss_truncated": truncated,
        "visible_miss_conserved": after_visible + truncated == pool,
        "visible_miss_unconfirmed_unchanged": unconfirmed_before == unconfirmed_after,
        "visible_miss_unconfirmed": unconfirmed_after,
        "median_in_frame_fraction_truncated": (
            float(np.median(truncated_fracs)) if truncated_fracs else None
        ),
        "edge_counts": dict(edge_counts),
        "missing_in_frame_fraction": missing_frac,
        "remaining_not_on_border": remaining_ok,
        "truncated_on_border": truncated_ok,
        "no_inframe_bottom_truncation": not false_bottom,
        "inframe_bottom_event_ids": false_bottom,
        "truncated_fraction": fraction,
        "warn_high_truncation": fraction > TRUNCATION_WARN_FRACTION,
        "truncation_test": "hybrid",
    }


def diagnose_gt_clipping(
    gt_by_frame: dict[int, list[GTBox]], width: int, height: int
) -> dict[str, Any]:
    x1s: list[float] = []
    y1s: list[float] = []
    x2s: list[float] = []
    y2s: list[float] = []
    neg_x1 = neg_y1 = past_w = past_h = 0
    for boxes in gt_by_frame.values():
        for box in boxes:
            if box.cls != 1:
                continue
            x1, y1, x2, y2 = box.bbox
            x1s.append(x1)
            y1s.append(y1)
            x2s.append(x2)
            y2s.append(y2)
            if x1 < 1:
                neg_x1 += 1
            if y1 < 1:
                neg_y1 += 1
            if x2 > width:
                past_w += 1
            if y2 > height:
                past_h += 1
    left_top_clipped = neg_x1 == 0 and neg_y1 == 0
    right_bottom_unclipped = past_w > 0 or past_h > 0
    hybrid = left_top_clipped and right_bottom_unclipped
    return {
        "n_pedestrian_boxes": len(x1s),
        "x1_lt_1": neg_x1,
        "y1_lt_1": neg_y1,
        "x2_gt_width": past_w,
        "y2_gt_height": past_h,
        "x1_min": min(x1s) if x1s else None,
        "y1_min": min(y1s) if y1s else None,
        "x2_max": max(x2s) if x2s else None,
        "y2_max": max(y2s) if y2s else None,
        "left_top_clipped": left_top_clipped,
        "right_bottom_unclipped": right_bottom_unclipped,
        "hybrid": hybrid,
        "branch": "hybrid" if hybrid else "proximity_all",
        "reason": (
            "left/top never go below 1 (clipped); right/bottom overflow exists "
            "(unclipped). Hybrid per-edge test applies."
            if hybrid
            else "right/bottom also appear clipped; falling back to scaled "
            "proximity on all four edges."
        ),
    }


def read_sequence_meta(dataset_root: Path, sequence: str) -> SequenceMeta:
    path = dataset_root / "train" / sequence / "seqinfo.ini"
    if not path.is_file():
        raise FileNotFoundError(f"Missing seqinfo.ini for {sequence}: {path}")
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(path)
    section = parser["Sequence"]
    return SequenceMeta(
        name=sequence,
        num_frames=int(section["seqLength"]),
        width=int(section["imWidth"]),
        height=int(section["imHeight"]),
        im_dir=section.get("imDir", "img1"),
        im_ext=section.get("imExt", ".jpg"),
        dataset_root=dataset_root,
    )


def load_gt(
    dataset_root: Path, sequence: str, num_frames: int
) -> dict[int, list[GTBox]]:
    by_frame, _, _, _ = _read_gt(
        dataset_root, sequence, set(range(1, num_frames + 1))
    )
    return by_frame


def occluder_boxes_on_frame(
    gt_by_frame: dict[int, list[GTBox]], frame: int
) -> list[BBox]:
    return [
        box.bbox
        for box in gt_by_frame.get(frame, [])
        if box.cls in OCCLUDER_CLASSES
    ]


def gt_lookup(gt_by_frame: dict[int, list[GTBox]]) -> dict[tuple[int, int], BBox]:
    return {
        (box.frame, box.gt_id): box.bbox
        for boxes in gt_by_frame.values()
        for box in boxes
    }


def validate_ref_mask(
    mask: np.ndarray, height: int, width: int, *, allow_empty: bool
) -> None:
    if mask.shape != (height, width):
        raise AssertionError(
            f"Mask shape {mask.shape} != expected {(height, width)}"
        )
    if allow_empty:
        return
    mean = float(mask.mean())
    if not MASK_MEAN_MIN < mean < MASK_MEAN_MAX:
        raise AssertionError(
            f"Custom mask mean {mean:.6f} is outside ({MASK_MEAN_MIN}, {MASK_MEAN_MAX})"
        )


def _bgr_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def render_split_audit(
    frame_bgr: np.ndarray,
    split: SplitResult,
    output_path: Path,
) -> None:
    vis = frame_bgr.copy()
    for polygon in split.mot20:
        cv2.polylines(vis, [np.int32(np.round(polygon.points))], True, (0, 0, 255), 2)
    for polygon in split.dropped_overlap:
        cv2.polylines(vis, [np.int32(np.round(polygon.points))], True, (0, 255, 255), 2)
    for polygon in split.custom:
        overlay = vis.copy()
        cv2.fillPoly(overlay, [np.int32(np.round(polygon.points))], (0, 180, 0))
        vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
        cv2.polylines(vis, [np.int32(np.round(polygon.points))], True, (0, 255, 0), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def render_mask_overlay(
    frame_bgr: np.ndarray, mask: np.ndarray, output_path: Path
) -> None:
    overlay = frame_bgr.astype(np.float32) * 0.5
    overlay += mask[..., None].astype(np.float32) * 128.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.clip(overlay, 0, 255).astype(np.uint8))


def static_region_mask(height: int, width: int, sequence: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if sequence == "MOT20-03":
        mask[: height // 3, :] = 255
    else:
        mask[: height // 3, :] = 255
        mask[:, : width // 8] = 255
        mask[:, -width // 8 :] = 255
    return mask


def estimate_shift(
    ref_gray: np.ndarray,
    cur_gray: np.ndarray,
    detection_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float]:
    orb = cv2.ORB_create(ORB_FEATURES)
    keypoints_ref, desc_ref = orb.detectAndCompute(ref_gray, detection_mask)
    keypoints_cur, desc_cur = orb.detectAndCompute(cur_gray, detection_mask)
    if (
        desc_ref is None
        or desc_cur is None
        or len(keypoints_ref) < 4
        or len(keypoints_cur) < 4
    ):
        return None, 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_ref, desc_cur)
    if len(matches) < 4:
        return None, 0.0
    matches = sorted(matches, key=lambda item: item.distance)[:500]
    source = np.float32(
        [keypoints_ref[item.queryIdx].pt for item in matches]
    ).reshape(-1, 1, 2)
    dest = np.float32(
        [keypoints_cur[item.trainIdx].pt for item in matches]
    ).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(
        source, dest, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH
    )
    if matrix is None or inliers is None or len(inliers) == 0:
        return None, 0.0
    return matrix, float(inliers.mean())


def decompose_affine(matrix: np.ndarray | None) -> tuple[float, float, float, float]:
    if matrix is None:
        return 0.0, 0.0, 0.0, 1.0
    dx = float(matrix[0, 2])
    dy = float(matrix[1, 2])
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    rotation_deg = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    return dx, dy, rotation_deg, scale if scale > 0 else 1.0


def sample_frame_ids(num_frames: int, count: int = SHAKE_SAMPLES) -> list[int]:
    if num_frames <= count:
        return list(range(1, num_frames + 1))
    samples = np.unique(np.linspace(1, num_frames, count, dtype=int))
    return [int(frame) for frame in samples]


def measure_shift_on_frame(
    ref_gray: np.ndarray,
    cur_gray: np.ndarray,
    sequence: str,
) -> tuple[np.ndarray, float, bool]:
    matrix, inliers = estimate_shift(ref_gray, cur_gray)
    used_static = False
    if inliers < MIN_INLIER_RATIO:
        region = static_region_mask(*ref_gray.shape, sequence)
        retry, retry_inliers = estimate_shift(ref_gray, cur_gray, region)
        if retry is not None and retry_inliers >= inliers:
            matrix, inliers = retry, retry_inliers
            used_static = True
    if matrix is None:
        return IDENTITY_AFFINE.copy(), inliers, used_static
    return matrix, inliers, used_static


def measure_camera_shake(
    meta: SequenceMeta,
    ref_frame: int,
    sample_frames: list[int],
) -> list[ShakeRow]:
    ref_bgr = _bgr_image(meta.image_path(ref_frame))
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    rows: list[ShakeRow] = []
    for frame in sample_frames:
        if frame == ref_frame:
            rows.append(
                ShakeRow(frame, 0.0, 0.0, 0.0, 1.0, 1.0, False)
            )
            continue
        cur_gray = cv2.cvtColor(_bgr_image(meta.image_path(frame)), cv2.COLOR_BGR2GRAY)
        matrix, inliers, used_static = measure_shift_on_frame(
            ref_gray, cur_gray, meta.name
        )
        dx, dy, rotation, scale = decompose_affine(matrix)
        rows.append(
            ShakeRow(frame, dx, dy, rotation, scale, inliers, used_static)
        )
    return rows


def write_shake_csv(path: Path, rows: list[ShakeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame",
                "dx",
                "dy",
                "rotation_deg",
                "scale",
                "inlier_ratio",
                "used_static_region",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "frame": row.frame,
                    "dx": row.dx,
                    "dy": row.dy,
                    "rotation_deg": row.rotation_deg,
                    "scale": row.scale,
                    "inlier_ratio": row.inlier_ratio,
                    "used_static_region": row.used_static_region,
                }
            )


def write_shake_plot(path: Path, sequence: str, rows: list[ShakeRow]) -> None:
    frames = [row.frame for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(frames, [row.dx for row in rows], label="dx")
    axes[0].plot(frames, [row.dy for row in rows], label="dy")
    axes[0].set_ylabel("pixels")
    axes[0].legend()
    axes[0].set_title(f"{sequence} camera translation")
    axes[1].plot(frames, [row.inlier_ratio for row in rows], color="tab:green")
    axes[1].axhline(MIN_INLIER_RATIO, color="tab:red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("inlier ratio")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def shake_decision(rows: list[ShakeRow]) -> dict[str, float | bool | int]:
    """Report-only ORB summary. Never chooses warping; fallback fires below 50% trust."""
    n_samples = len(rows)
    trusted = [row for row in rows if row.inlier_ratio >= MIN_INLIER_RATIO]
    n_trusted = len(trusted)
    untrusted_fallback = True if n_samples == 0 else n_trusted < 0.5 * n_samples
    if trusted:
        max_shift = max(math.hypot(row.dx, row.dy) for row in trusted)
        max_rotation = max(abs(row.rotation_deg) for row in trusted)
    else:
        max_shift = 0.0
        max_rotation = 0.0
    return {
        "max_shift_px": max_shift,
        "max_rotation_deg": max_rotation,
        "use_static_dilation": True,
        "n_trusted_samples": n_trusted,
        "n_samples": n_samples,
        "untrusted_fallback": untrusted_fallback,
    }


def dilate_mask(mask: np.ndarray, margin: int = DILATE_MARGIN) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)
    )
    return cv2.dilate(mask, kernel)


def estimate_all_frame_transforms(
    meta: SequenceMeta,
    ref_frame: int,
) -> dict[int, np.ndarray]:
    ref_gray = cv2.cvtColor(_bgr_image(meta.image_path(ref_frame)), cv2.COLOR_BGR2GRAY)
    matrices: dict[int, np.ndarray] = {}
    previous = IDENTITY_AFFINE.copy()
    dx_values = np.zeros(meta.num_frames, dtype=np.float64)
    dy_values = np.zeros(meta.num_frames, dtype=np.float64)
    raw: list[np.ndarray] = []
    inliers_list: list[float] = []
    for index, frame in enumerate(range(1, meta.num_frames + 1)):
        if frame == ref_frame:
            matrix = IDENTITY_AFFINE.copy()
            inliers = 1.0
        else:
            cur_gray = cv2.cvtColor(
                _bgr_image(meta.image_path(frame)), cv2.COLOR_BGR2GRAY
            )
            matrix, inliers, _ = measure_shift_on_frame(ref_gray, cur_gray, meta.name)
            if inliers < MIN_INLIER_RATIO:
                matrix = previous.copy()
        dx, dy, _, _ = decompose_affine(matrix)
        dx_values[index] = dx
        dy_values[index] = dy
        raw.append(matrix)
        inliers_list.append(inliers)
        previous = matrix
    dx_smooth = median_filter(dx_values, size=MEDIAN_WINDOW, mode="nearest")
    dy_smooth = median_filter(dy_values, size=MEDIAN_WINDOW, mode="nearest")
    for index, frame in enumerate(range(1, meta.num_frames + 1)):
        matrix = raw[index].copy()
        matrix[0, 2] = dx_smooth[index]
        matrix[1, 2] = dy_smooth[index]
        matrices[frame] = matrix
    return matrices


def load_events_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def write_events_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


def write_before_after_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sequence",
        "visible_miss_before",
        "visible_miss_after",
        "visible_miss_unconfirmed",
        "full_occl_static_before",
        "full_occl_static_after",
        "unconfirmed_fraction",
        "warn_high_contamination",
        "visible_miss_conserved",
        "full_occl_static_unchanged",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_truncation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sequence",
        "visible_miss_before",
        "visible_miss_truncated",
        "visible_miss_after",
        "median_in_frame_fraction_truncated",
        "n_left",
        "n_right",
        "n_top",
        "n_bottom",
        "truncation_test",
        "truncated_fraction",
        "warn_high_truncation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            edges = row.get("edge_counts") or {}
            writer.writerow(
                {
                    "sequence": row["sequence"],
                    "visible_miss_before": row["visible_miss_before"],
                    "visible_miss_truncated": row["visible_miss_truncated"],
                    "visible_miss_after": row["visible_miss_after"],
                    "median_in_frame_fraction_truncated": row[
                        "median_in_frame_fraction_truncated"
                    ],
                    "n_left": edges.get("left", 0),
                    "n_right": edges.get("right", 0),
                    "n_top": edges.get("top", 0),
                    "n_bottom": edges.get("bottom", 0),
                    "truncation_test": row["truncation_test"],
                    "truncated_fraction": row["truncated_fraction"],
                    "warn_high_truncation": row["warn_high_truncation"],
                }
            )


def write_truncation_by_edge_csv(
    path: Path,
    before_counts: dict[str, int],
    after_rows: list[dict[str, Any]],
) -> None:
    after_counts: dict[str, int] = Counter()
    after_tests: dict[str, str] = {
        "left": "proximity",
        "top": "proximity",
        "right": "overflow",
        "bottom": "overflow",
    }
    for row in after_rows:
        edges = row.get("edge_counts") or {}
        for edge, count in edges.items():
            after_counts[edge] += int(count)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["edge", "test_used", "count_t1b_before", "count_hybrid_after"],
        )
        writer.writeheader()
        for edge in ("left", "bottom", "right", "top"):
            writer.writerow(
                {
                    "edge": edge,
                    "test_used": after_tests[edge],
                    "count_t1b_before": int(before_counts.get(edge, 0)),
                    "count_hybrid_after": int(after_counts.get(edge, 0)),
                }
            )


def write_funnel_csv(
    path: Path,
    raw: int,
    unconfirmed: int,
    truncated: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    final = raw - unconfirmed - truncated
    rows = [
        ("visible_miss (raw)", raw, ""),
        ("within 40px of unannotated canopy", -unconfirmed, "visible_miss_unconfirmed"),
        ("truncated by frame boundary", -truncated, "visible_miss_truncated"),
        ("visible_miss (final)", final, "headline lower bound"),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["stage", "delta_or_count", "label"]
        )
        writer.writeheader()
        for stage, value, label in rows:
            writer.writerow(
                {"stage": stage, "delta_or_count": value, "label": label}
            )


def write_edge_histogram(path: Path, edge_counts: dict[str, int]) -> None:
    order = ("left", "bottom", "right", "top")
    values = [int(edge_counts.get(edge, 0)) for edge in order]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(order, values, color="#4f81bd")
    axis.set_ylabel("truncated visible_miss events")
    axis.set_title("Truncation edge histogram")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def remaining_visible_miss_centres(
    events: list[dict[str, Any]],
    gt_by_id: dict[tuple[int, int], BBox],
) -> tuple[list[float], list[float]]:
    """One centre per remaining visible_miss event (median over event frames)."""
    xs: list[float] = []
    ys: list[float] = []
    for event in events:
        if event.get("cause") != "visible_miss":
            continue
        focal_id = int(event["focal_gt_id"])
        event_x: list[float] = []
        event_y: list[float] = []
        for frame in frames_for_event(event):
            bbox = gt_by_id.get((frame, focal_id))
            if bbox is None:
                continue
            event_x.append((bbox[0] + bbox[2]) / 2.0)
            event_y.append((bbox[1] + bbox[3]) / 2.0)
        if not event_x:
            continue
        xs.append(float(np.median(event_x)))
        ys.append(float(np.median(event_y)))
    return xs, ys


def write_visible_miss_heatmap(
    output_path: Path,
    sequence_frames: dict[str, np.ndarray],
    sequence_masks: dict[str, SequenceMasks],
    sequence_centres: dict[str, tuple[list[float], list[float]]],
    sequence_mot20_boxes: dict[str, list[BBox]],
) -> None:
    names = list(sequence_frames)
    figure, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 6))
    if len(names) == 1:
        axes = [axes]
    for axis, sequence in zip(axes, names):
        frame = cv2.cvtColor(sequence_frames[sequence], cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        axis.imshow(frame)
        for box in sequence_mot20_boxes.get(sequence, []):
            x1, y1, x2, y2 = box
            axis.add_patch(
                plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="red",
                    linewidth=0.8,
                    alpha=0.85,
                )
            )
        mask = sequence_masks[sequence].static_mask
        if mask is None:
            mask = sequence_masks[sequence].ref_mask
        if mask.any():
            axis.contour(mask, levels=[0.5], colors=["lime"], linewidths=1.6)
        xs, ys = sequence_centres.get(sequence, ([], []))
        if xs:
            axis.scatter(
                xs,
                ys,
                s=42,
                c="gold",
                edgecolors="black",
                linewidths=0.45,
                zorder=5,
            )
        axis.set_title(f"{sequence} remaining visible_miss (n={len(xs)})")
        axis.set_xlim(0, width)
        axis.set_ylim(height, 0)
        axis.axis("off")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def contamination_verdict(unconfirmed_fraction: float) -> str:
    if unconfirmed_fraction < UNCONFIRMED_ROBUST_FRACTION:
        return "headline_robust"
    if unconfirmed_fraction <= UNCONFIRMED_MATERIAL_FRACTION:
        return "report_both_numbers"
    return "category_contaminated"


def _resolution_summary(resolutions: list[PolygonResolution]) -> dict[str, Any]:
    return {
        "custom_count": sum(item.kept for item in resolutions),
        "dropped_overlap_count": sum(not item.kept for item in resolutions),
        "polygons": [
            {
                "id": item.polygon.annotation_id,
                "kept": item.kept,
                "original_px": item.original_px,
                "remaining_px": item.remaining_px,
                "reason": item.reason,
                "vertex_count": item.polygon.vertex_count,
            }
            for item in resolutions
        ],
    }


def _split_summary(split: SplitResult) -> dict[str, Any]:
    return {
        "custom_count": len(split.custom),
        "mot20_count": len(split.mot20),
        "dropped_overlap_count": len(split.dropped_overlap),
        "custom_ids": [item.annotation_id for item in split.custom],
        "custom_vertex_counts": [item.vertex_count for item in split.custom],
        "dropped_overlap_ids": [item.annotation_id for item in split.dropped_overlap],
        "mot20_vertex_counts": sorted(item.vertex_count for item in split.mot20),
    }


def run_occluder_mask_postpass(
    export_path: Path,
    dataset_root: Path,
    events_root: Path,
    output_root: Path,
    sequences: tuple[str, ...] = EVALUATION_SEQUENCES,
) -> dict[str, Any]:
    payload = load_coco_export(export_path)
    splits = split_coco_polygons(payload, sequences)
    occluders_dir = output_root / "occluders"
    summary_dir = output_root / "summary"
    events_out_dir = output_root / "events"
    occluders_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    events_out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "export": str(export_path.resolve()),
        "sequences": {},
        "gates": {},
        "warnings": [],
    }
    heatmap_frames: dict[str, np.ndarray] = {}
    heatmap_masks: dict[str, SequenceMasks] = {}
    heatmap_centres: dict[str, tuple[list[float], list[float]]] = {}
    heatmap_mot20: dict[str, list[BBox]] = {}
    heatmap_centres_pre_trunc: dict[str, tuple[list[float], list[float]]] = {}
    truncation_rows: list[dict[str, Any]] = []
    t1b_edge_counts: Counter[str] = Counter()
    before_after_rows: list[dict[str, Any]] = []
    all_before: list[dict[str, Any]] = []
    all_canopy: list[dict[str, Any]] = []
    all_after: list[dict[str, Any]] = []

    for sequence in sequences:
        meta = read_sequence_meta(dataset_root, sequence)
        ref_frame = REF_FRAME[sequence]
        gt_by_frame = load_gt(dataset_root, sequence, meta.num_frames)
        split = splits[sequence]
        mot20_boxes = occluder_boxes_on_frame(gt_by_frame, ref_frame)
        mot20_mask = boxes_to_mask(mot20_boxes, meta.height, meta.width)
        ref_mask, resolutions = compose_custom_mask(
            split.custom, mot20_mask, meta.height, meta.width
        )
        kept = [item.polygon for item in resolutions if item.kept]
        dropped = [item.polygon for item in resolutions if not item.kept]
        split.custom = kept
        split.dropped_overlap = dropped

        ref_bgr = _bgr_image(meta.image_path(ref_frame))
        if ref_bgr.shape[0] != meta.height or ref_bgr.shape[1] != meta.width:
            raise AssertionError(
                f"{sequence} frame {ref_frame} shape {ref_bgr.shape[:2]} "
                f"!= seqinfo {(meta.height, meta.width)}"
            )
        render_split_audit(
            ref_bgr, split, summary_dir / f"split_audit_{sequence}.jpg"
        )
        validate_ref_mask(
            ref_mask, meta.height, meta.width, allow_empty=not bool(kept)
        )
        if sequence == "MOT20-03" and float(ref_mask.mean()) <= MOT20_03_MASK_MEAN_MIN:
            raise AssertionError(
                f"MOT20-03 mask_mean {float(ref_mask.mean()):.6f} "
                f"<= {MOT20_03_MASK_MEAN_MIN}; both canopies should be present"
            )
        np.save(occluders_dir / f"{sequence}__custom_ref_mask.npy", ref_mask)
        render_mask_overlay(
            ref_bgr, ref_mask, summary_dir / f"mask_overlay_{sequence}.jpg"
        )

        sequence_report: dict[str, Any] = {
            "ref_frame": ref_frame,
            "split": _split_summary(split),
            "resolutions": _resolution_summary(resolutions),
            "mask_mean": float(ref_mask.mean()),
            "mask_shape": list(ref_mask.shape),
        }

        if not kept:
            masks = SequenceMasks(
                sequence=sequence,
                mode="empty",
                ref_mask=ref_mask,
                width=meta.width,
                height=meta.height,
                static_mask=ref_mask,
                dilate_margin=0,
            )
            np.save(occluders_dir / f"{sequence}__mask_static.npy", ref_mask)
            sequence_report["cmc"] = {
                "skipped": True,
                "reason": "empty_custom_mask",
                "mode": "empty",
            }
        else:
            sample_frames = sample_frame_ids(meta.num_frames, SHAKE_SAMPLES)
            shake_rows = measure_camera_shake(meta, ref_frame, sample_frames)
            write_shake_csv(summary_dir / f"camera_shake_{sequence}.csv", shake_rows)
            write_shake_plot(
                summary_dir / f"camera_shake_{sequence}.png", sequence, shake_rows
            )
            decision = shake_decision(shake_rows)
            dilated = dilate_mask(ref_mask, DILATE_MARGIN)
            np.save(occluders_dir / f"{sequence}__mask_static.npy", dilated)
            masks = SequenceMasks(
                sequence=sequence,
                mode=CMC_MODE,
                ref_mask=ref_mask,
                width=meta.width,
                height=meta.height,
                static_mask=dilated,
                dilate_margin=DILATE_MARGIN,
            )
            min_inlier = min(row.inlier_ratio for row in shake_rows)
            sequence_report["cmc"] = {
                "skipped": False,
                **decision,
                "min_inlier_ratio": min_inlier,
                "mode": CMC_MODE,
                "dilate_margin": DILATE_MARGIN,
                "estimate_reliable": not bool(decision["untrusted_fallback"]),
            }
            if "warped" in str(sequence_report["cmc"]["mode"]):
                raise AssertionError("CMC mode must not be warped")
            n_samples = int(decision["n_samples"])
            n_trusted = int(decision["n_trusted_samples"])
            if n_samples and n_trusted < 0.5 * n_samples and not decision["untrusted_fallback"]:
                raise AssertionError(
                    "untrusted_fallback must be True when fewer than half the "
                    "ORB samples are trusted"
                )

        if sequence == "MOT20-03":
            if sequence_report["resolutions"]["custom_count"] != 2:
                raise AssertionError(
                    "MOT20-03 must keep both custom polygons after remainder subtraction"
                )
            for item in resolutions:
                if not item.kept and item.remaining_px >= MIN_REMAINDER_PX:
                    raise AssertionError(
                        f"Dropped polygon {item.polygon.annotation_id} still has "
                        f"{item.remaining_px} px remaining"
                    )

        event_path = events_root / f"{sequence}__tracktrack.jsonl"
        if not event_path.is_file():
            matches = sorted(events_root.glob(f"{sequence}__*.jsonl"))
            if not matches:
                raise FileNotFoundError(f"No events jsonl for {sequence} in {events_root}")
            event_path = matches[0]
        before = load_events_jsonl(event_path)
        lookup = gt_lookup(gt_by_frame)
        scores = [
            event_ioa_unannotated_structure(
                event, lookup, masks.for_frame
            )
            for event in before
        ]
        after_canopy = apply_visible_miss_postpass(before, scores, IOA_MASK_THRESH)
        clipping = diagnose_gt_clipping(gt_by_frame, meta.width, meta.height)
        after = apply_truncation_postpass(
            after_canopy,
            lookup,
            meta.width,
            meta.height,
            hybrid=bool(clipping["hybrid"]),
        )
        write_events_jsonl(events_out_dir / event_path.name, after)
        conservation = conservation_report(before, after_canopy)
        trunc = truncation_report(after_canopy, after)
        for event in after_canopy:
            if event.get("cause") != "visible_miss":
                continue
            for edge in event_legacy_t1b_edges(
                event, lookup, meta.width, meta.height
            ):
                t1b_edge_counts[edge] += 1
        if not conservation["visible_miss_conserved"]:
            raise AssertionError(f"{sequence}: visible_miss counts are not conserved")
        if not conservation["full_occl_static_unchanged"]:
            raise AssertionError(f"{sequence}: full_occl_static counts changed")
        if conservation["missing_ioa_event_ids"]:
            raise AssertionError(
                f"{sequence}: missing ioa_unannotated_structure on "
                f"{conservation['missing_ioa_event_ids'][:5]}"
            )
        if not trunc["visible_miss_conserved"]:
            raise AssertionError(f"{sequence}: truncation does not conserve visible_miss")
        if not trunc["visible_miss_unconfirmed_unchanged"]:
            raise AssertionError(f"{sequence}: truncation changed visible_miss_unconfirmed")
        if trunc["missing_in_frame_fraction"]:
            raise AssertionError(
                f"{sequence}: missing in_frame_fraction on "
                f"{trunc['missing_in_frame_fraction'][:5]}"
            )
        if not trunc["remaining_not_on_border"] or not trunc["truncated_on_border"]:
            raise AssertionError(f"{sequence}: truncation soundness gates failed")
        if not trunc["no_inframe_bottom_truncation"]:
            raise AssertionError(
                f"{sequence}: fully in-frame boxes flagged as bottom-truncated: "
                f"{trunc['inframe_bottom_event_ids'][:5]}"
            )
        if conservation["warn_high_contamination"]:
            warning = (
                f"{sequence}: visible_miss_unconfirmed / old visible_miss = "
                f"{conservation['unconfirmed_fraction']:.3f} exceeds "
                f"{UNCONFIRMED_WARN_FRACTION}"
            )
            report["warnings"].append(warning)
        if trunc["warn_high_truncation"]:
            report["warnings"].append(
                f"{sequence}: visible_miss_truncated / visible_miss_before = "
                f"{trunc['truncated_fraction']:.3f} exceeds {TRUNCATION_WARN_FRACTION}"
            )
        sequence_report["conservation"] = conservation
        sequence_report["gt_clipping"] = clipping
        sequence_report["truncation"] = trunc
        report["sequences"][sequence] = sequence_report
        before_after_rows.append({"sequence": sequence, **conservation})
        truncation_rows.append({"sequence": sequence, **trunc})
        heatmap_frames[sequence] = ref_bgr
        heatmap_masks[sequence] = masks
        heatmap_centres_pre_trunc[sequence] = remaining_visible_miss_centres(
            after_canopy, lookup
        )
        heatmap_centres[sequence] = remaining_visible_miss_centres(after, lookup)
        heatmap_mot20[sequence] = mot20_boxes
        all_before.extend(before)
        all_canopy.extend(after_canopy)
        all_after.extend(after)

    write_before_after_csv(
        summary_dir / "visible_miss_before_after.csv", before_after_rows
    )
    write_truncation_csv(summary_dir / "truncation_analysis.csv", truncation_rows)
    write_truncation_by_edge_csv(
        summary_dir / "truncation_by_edge.csv",
        dict(t1b_edge_counts),
        truncation_rows,
    )
    write_visible_miss_heatmap(
        summary_dir / "visible_miss_heatmap_pre_truncation.png",
        heatmap_frames,
        heatmap_masks,
        heatmap_centres_pre_trunc,
        heatmap_mot20,
    )
    write_visible_miss_heatmap(
        summary_dir / "visible_miss_heatmap.png",
        heatmap_frames,
        heatmap_masks,
        heatmap_centres,
        heatmap_mot20,
    )
    overall = conservation_report(all_before, all_canopy)
    overall_trunc = truncation_report(all_canopy, all_after)
    write_funnel_csv(
        summary_dir / "visible_miss_funnel.csv",
        int(overall["visible_miss_before"]),
        int(overall["visible_miss_unconfirmed"]),
        int(overall_trunc["visible_miss_truncated"]),
    )
    write_edge_histogram(
        summary_dir / "truncation_edge_histogram.png",
        overall_trunc.get("edge_counts") or {},
    )
    verdict = contamination_verdict(float(overall["unconfirmed_fraction"]))
    mot20_03 = report["sequences"].get("MOT20-03", {})
    mot20_03_cmc = mot20_03.get("cmc", {})
    mot20_03_res = mot20_03.get("resolutions", {})
    mot20_03_trunc = mot20_03.get("truncation", {})
    report["gates"] = {
        "mask_shapes_ok": True,
        "all_events_have_ioa": not overall["missing_ioa_event_ids"],
        "visible_miss_conserved": overall["visible_miss_conserved"],
        "full_occl_static_unchanged": overall["full_occl_static_unchanged"],
        "gf1_subtraction_not_drop": mot20_03_res.get("custom_count") == 2,
        "gf1_mask_mean": float(mot20_03.get("mask_mean", 0.0)) > MOT20_03_MASK_MEAN_MIN,
        "gf2_static_dilate": mot20_03_cmc.get("mode") == CMC_MODE,
        "gf2_not_warped": "warped" not in str(mot20_03_cmc.get("mode", "")),
        "gf3_untrusted_fallback": (
            int(mot20_03_cmc.get("n_trusted_samples", 0))
            >= 0.5 * int(mot20_03_cmc.get("n_samples", 0) or 1)
            or bool(mot20_03_cmc.get("untrusted_fallback"))
        ),
        "truncation_conserved": overall_trunc["visible_miss_conserved"],
        "truncation_unconfirmed_unchanged": overall_trunc[
            "visible_miss_unconfirmed_unchanged"
        ],
        "truncation_all_have_in_frame_fraction": not overall_trunc[
            "missing_in_frame_fraction"
        ],
        "truncation_remaining_not_on_border": overall_trunc["remaining_not_on_border"],
        "truncation_truncated_on_border": overall_trunc["truncated_on_border"],
        "truncation_no_inframe_bottom": overall_trunc["no_inframe_bottom_truncation"],
        "truncation_post_canopy_total": (
            int(overall_trunc["visible_miss_after"])
            + int(overall_trunc["visible_miss_truncated"])
            == POST_CANOPY_VISIBLE_MISS
        ),
        "truncation_unconfirmed_count": (
            int(overall_trunc["visible_miss_unconfirmed"]) == POST_CANOPY_UNCONFIRMED
        ),
        "contamination_verdict": verdict,
        "overall": overall,
        "overall_truncation": overall_trunc,
    }
    trunc_gate_keys = (
        "truncation_conserved",
        "truncation_unconfirmed_unchanged",
        "truncation_all_have_in_frame_fraction",
        "truncation_remaining_not_on_border",
        "truncation_truncated_on_border",
        "truncation_no_inframe_bottom",
        "truncation_post_canopy_total",
        "truncation_unconfirmed_count",
    )
    if not all(report["gates"][key] for key in trunc_gate_keys):
        raise AssertionError(f"Truncation gates failed: {report['gates']}")
    if not all(
        report["gates"][key]
        for key in (
            "gf1_subtraction_not_drop",
            "gf1_mask_mean",
            "gf2_static_dilate",
            "gf2_not_warped",
            "gf3_untrusted_fallback",
        )
        if mot20_03
    ):
        raise AssertionError(f"CMC fix gates failed: {report['gates']}")
    (summary_dir / "occluder_mask_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rasterize custom occluder polygons and flag visible_miss events "
            "that overlap unannotated static structure"
        )
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=Path("mot20-correction/mot20-correction-firstframe/result.json"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/MOT20"))
    parser.add_argument(
        "--events-root", type=Path, default=Path("out/tracktrack_v4_full/events")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("out/occluder_masks")
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_occluder_mask_postpass(
        export_path=args.export,
        dataset_root=args.dataset_root,
        events_root=args.events_root,
        output_root=args.output_root,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
