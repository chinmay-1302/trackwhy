"""Stratified clip sample: tracker-only annotator clips plus private audit renders.

Annotators and the VLM see identical tracker-only MP4s. Ground truth, visibility,
class names, and assigned cause never appear on those clips or in ``clips_meta``.
Audit side-by-sides are written to a separate directory and are not imported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import random
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import DEFAULT_CLASSES, EVALUATION_SEQUENCES, PipelineConfig
from .geometry import iou
from .io import MOTCHALLENGE_CLASS_NAMES, _read_seqinfo, load_sequences
from .models import BBox, SequenceData, TrkBox
from .occluder_masks import load_events_jsonl


SEED = 2026
TARGET_PER_CAUSE = 30
MIN_PER_CAUSE = 8
MAX_CLIPS = 300
PRE_FRAMES = 40
POST_FRAMES = 40
ONSET_FRAMES = 60
END_FRAMES = 60
SEGMENT_FRAMES = 60
NEVER_RESUMES_GAP_FRAMES = 60
MAX_CLIP_FRAMES = 200
MIN_FN_RUN_FRAMES = 3
MIN_CLIP_FRAMES = MIN_FN_RUN_FRAMES + PRE_FRAMES + POST_FRAMES
MIN_NON_FN_CLIP_FRAMES = PRE_FRAMES + POST_FRAMES + 1
TOP_UP_PRIORITY = (
    "unattributed_idsw",
    "track_drift_to_static",
    "unattributed_fn",
)
FPS = 10
SOURCE_FPS = 25.0
CUT_MARKER_HOLD_SECONDS = 1.0
CUT_MARKER_FRAMES = int(round(FPS * CUT_MARKER_HOLD_SECONDS))
PROBE_WORKERS = 4
HIGHLIGHT_IOU = 0.30
AUDIT_EDGE_MARGIN = 15

FAINT_COLOR = (170, 170, 170)
HIGHLIGHT_COLOR = (0, 255, 255)
AUDIT_GT_COLOR = (70, 220, 90)
AUDIT_FOCAL_COLOR = (0, 215, 255)
AUDIT_OCCLUDER_COLOR = (40, 80, 220)
SKIP_LABEL = "— skip —"

LEAK_PATTERN = re.compile(
    r"occlud|static|crowd|visib|truncat|miss|target|occludee|hijack",
    re.IGNORECASE,
)
ANNOTATOR_GT_LEAK_TERMS = ("gt_by_frame", "GTBox", "visibility", "occlud")

AVOIDABILITY = {
    "full_occl_static": "unavoidable",
    "full_occl_pedestrian": "unavoidable",
    "full_occl_crowd_aggregate": "unavoidable",
    "partial_occl_pedestrian": "partially_avoidable",
    "partial_occl_static": "partially_avoidable",
    "track_drift_to_static": "avoidable",
    "reentry": "ambiguous",
    "gt_annotation_gap": "unknown",
    "visible_miss": "avoidable",
    "visible_miss_truncated": "boundary",
    "visible_miss_unconfirmed": "unconfirmed",
    "unattributed_fn": "unknown",
    "unattributed_idsw": "unknown",
}

TARGET_CLASSES = set(DEFAULT_CLASSES["TARGET_CLASSES"])
OCCLUDER_CLASSES = set(DEFAULT_CLASSES["OCCLUDER_CLASSES"]) | set(
    DEFAULT_CLASSES["CROWD"]
) | set(DEFAULT_CLASSES["STATIC_PERSON"])


def event_n_frames(event: dict[str, Any]) -> int:
    return int(event["frame_end"]) - int(event["frame_start"]) + 1


def min_clip_frames_for(event: dict[str, Any]) -> int:
    """FN/FRAG use the micro-gap floor. IDSW/hijack keep PRE+POST+1."""
    if event.get("error_type") in {"FN", "FRAG_NO_SWITCH"}:
        return MIN_CLIP_FRAMES
    return MIN_NON_FN_CLIP_FRAMES


def cause_seed(cause: str, seed: int = SEED) -> int:
    digest = hashlib.sha1(f"{seed}:{cause}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def stratified_draw(
    events: list[dict[str, Any]],
    n: int,
    seed: int,
    strata: tuple[str, ...] = ("sequence", "error_type"),
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        key = tuple(str(event.get(name, "")) for name in strata)
        buckets[key].append(event)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    sample: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while len(sample) < n:
        progressed = False
        for key in keys:
            if buckets[key]:
                sample.append(buckets[key].pop())
                progressed = True
                if len(sample) >= n:
                    break
        if not progressed:
            break
    return sample


def sequence_lengths(dataset_root: Path) -> dict[str, int]:
    return {
        sequence: _read_seqinfo(Path(dataset_root), sequence)[0]
        for sequence in EVALUATION_SEQUENCES
    }


def event_clip_length(event: dict[str, Any], num_frames: int) -> int:
    return len(clip_source_frames(clip_segments(event, num_frames)))


def sample_events(
    events: list[dict[str, Any]],
    *,
    seed: int = SEED,
    target_per_cause: int = TARGET_PER_CAUSE,
    max_clips: int = MAX_CLIPS,
    lengths: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    by_cause: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_cause[str(event["cause"])].append(event)
    sample: list[dict[str, Any]] = []
    for cause in sorted(by_cause):
        pool = by_cause[cause]
        if lengths is not None:
            qualified = [
                event
                for event in pool
                if event_clip_length(event, lengths[str(event["sequence"])])
                >= min_clip_frames_for(event)
            ]
            pool = qualified or pool
        take = min(target_per_cause, len(pool))
        drawn = stratified_draw(pool, take, cause_seed(cause, seed))
        remaining = max_clips - len(sample)
        if remaining <= 0:
            break
        sample.extend(drawn[:remaining])
    return sample


def top_up_sample(
    sample: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    *,
    extra_slots: int,
    lengths: dict[str, int],
    seed: int = SEED,
    priority: tuple[str, ...] = TOP_UP_PRIORITY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fill leftover cap from under-powered causes that still have depth."""
    if extra_slots <= 0:
        return sample, []
    have = {event["event_id"] for event in sample}
    leftover: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in pool:
        if event["event_id"] in have:
            continue
        if (
            event_clip_length(event, lengths[str(event["sequence"])])
            < min_clip_frames_for(event)
        ):
            continue
        leftover[str(event["cause"])].append(event)
    added: list[dict[str, Any]] = []
    remaining = extra_slots
    for cause in priority:
        if remaining <= 0:
            break
        drawn = stratified_draw(
            leftover.get(cause, []), remaining, cause_seed(f"topup:{cause}", seed)
        )
        added.extend(drawn)
        remaining -= len(drawn)
    return sample + added, added


def load_sidecar_events(events_root: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for sequence in EVALUATION_SEQUENCES:
        matches = sorted(events_root.glob(f"{sequence}__*.jsonl"))
        if not matches:
            raise FileNotFoundError(f"No sidecar events for {sequence} in {events_root}")
        events.extend(load_events_jsonl(matches[0]))
    return events


def clip_stage1_bucket(event: Mapping[str, Any] | dict[str, Any]) -> str:
    error_type = str(event.get("error_type") or "")
    if error_type == "IDSW":
        return "IDSW"
    if error_type == "FP_ID_HIJACK":
        return "FP_ID_HIJACK"
    if error_type == "FRAG_NO_SWITCH":
        return "FN_resumes_same_id"
    if error_type == "FN":
        after = event.get("new_trk_id")
        if after is None or after == "":
            highlighted = event.get("highlighted_track_ids") or {}
            if isinstance(highlighted, Mapping):
                after = highlighted.get("after")
        if after is None or after == "":
            return "FN_never_resumes"
        return "FN_resumes_same_id"
    return error_type


def event_with_highlight_after(
    event: Mapping[str, Any] | dict[str, Any],
    after: int | None,
) -> dict[str, Any]:
    tagged = dict(event)
    if tagged.get("new_trk_id") in (None, "") and after is not None:
        tagged["new_trk_id"] = after
    return tagged


def fn_never_resumes(event: Mapping[str, Any] | dict[str, Any]) -> bool:
    return clip_stage1_bucket(event) == "FN_never_resumes"


def _clamped_frames(lo: int, hi: int, num_frames: int) -> list[int]:
    start = max(1, int(lo))
    end = min(int(num_frames), int(hi))
    if end < start:
        return []
    return list(range(start, end + 1))


def frames_skipped_between(previous: Sequence[int], nxt: Sequence[int]) -> int:
    if not previous or not nxt:
        return 0
    return max(0, int(nxt[0]) - int(previous[-1]) - 1)


def skip_caption(n_skipped: int, source_fps: float = SOURCE_FPS) -> str:
    rate = float(source_fps) if source_fps else SOURCE_FPS
    seconds = n_skipped / rate
    return f"--- skipping {int(n_skipped)} frames ({seconds:.1f}s) ---"


def clip_segments(
    event: dict[str, Any],
    num_frames: int,
    *,
    pre: int = PRE_FRAMES,
    post: int = POST_FRAMES,
    onset: int = SEGMENT_FRAMES,
    end_span: int = SEGMENT_FRAMES,
    max_frames: int = MAX_CLIP_FRAMES,
    never_resumes_gap: int = NEVER_RESUMES_GAP_FRAMES,
) -> list[list[int]]:
    start = int(event["frame_start"])
    end = int(event["frame_end"])
    full_lo = max(1, start - pre)
    full_hi = min(num_frames, end + post)
    contiguous = list(range(full_lo, full_hi + 1)) if full_hi >= full_lo else []
    if fn_never_resumes(event) and len(contiguous) > max_frames:
        gap_end = min(end, start + int(never_resumes_gap))
        truncated = _clamped_frames(start - pre, gap_end, num_frames)
        return [truncated] if truncated else []
    if len(contiguous) <= max_frames:
        return [contiguous]
    seg_a = _clamped_frames(start - pre, start + int(onset) - 1, num_frames)
    seg_b = _clamped_frames(end - int(end_span), end + post, num_frames)
    if seg_a and seg_b and seg_b[0] <= seg_a[-1] + 1:
        merged = _clamped_frames(seg_a[0], max(seg_a[-1], seg_b[-1]), num_frames)
        return [merged] if merged else []
    return [segment for segment in (seg_a, seg_b) if segment]


def clip_source_frames(segments: list[list[int]]) -> list[int]:
    frames: list[int] = []
    for segment in segments:
        frames.extend(segment)
    return frames


def even_size(width: int, height: int) -> tuple[int, int]:
    return width + width % 2, height + height % 2


def find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in (
        "/home/linuxbrew/.linuxbrew/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ):
        if Path(candidate).is_file():
            return candidate
    raise RuntimeError("ffmpeg with libx264 is required to encode H.264 probe clips")


def find_ffprobe() -> str:
    found = shutil.which("ffprobe")
    if found:
        return found
    ffmpeg = Path(find_ffmpeg())
    candidate = ffmpeg.with_name("ffprobe")
    if candidate.is_file():
        return str(candidate)
    raise RuntimeError("ffprobe is required to verify browser-ready clips")


def _draw_labeled_box(
    image: np.ndarray,
    bbox: BBox,
    label: str,
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    scale: float = 0.7,
) -> str:
    import cv2

    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    x1 = min(max(x1, 0), max(0, width - 1))
    x2 = min(max(x2, 0), max(0, width - 1))
    y1 = min(max(y1, 0), max(0, height - 1))
    y2 = min(max(y2, 0), max(0, height - 1))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if not label:
        return label
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(
        label, font, scale, text_thickness
    )
    label_left = min(x1, max(0, width - text_width - 8))
    text_y = max(text_height + 2, y1 - 4)
    cv2.putText(
        image, label, (label_left, text_y), font, scale, (10, 10, 10), 3, cv2.LINE_AA
    )
    cv2.putText(
        image, label, (label_left, text_y), font, scale, color, text_thickness, cv2.LINE_AA
    )
    return label


def cut_marker_frame(
    width: int,
    height: int,
    n_skipped: int = 0,
    source_fps: float = SOURCE_FPS,
) -> np.ndarray:
    import cv2

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    text = skip_caption(n_skipped, source_fps)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    while text_width > max(8, width - 24) and scale > 0.35:
        scale -= 0.08
        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
    origin = ((width - text_width) // 2, (height + text_height) // 2)
    cv2.putText(frame, text, origin, font, scale, (240, 240, 240), thickness, cv2.LINE_AA)
    return frame


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _gt_box(sequence: SequenceData, frame: int, gt_id: int | None):
    if gt_id is None:
        return None
    return next(
        (box for box in sequence.gt_by_frame.get(frame, []) if box.gt_id == int(gt_id)),
        None,
    )


def _best_tracker_id_for_gt(
    sequence: SequenceData, frame: int, gt_id: int | None
) -> int | None:
    box = _gt_box(sequence, frame, gt_id)
    if box is None:
        return None
    best_id: int | None = None
    best = HIGHLIGHT_IOU
    for tracker in sequence.tracker_by_frame.get(frame, []):
        overlap = iou(tracker.bbox, box.bbox)
        if overlap > best:
            best = overlap
            best_id = int(tracker.trk_id)
    return best_id


def _search_track_id(
    sequence: SequenceData,
    frames: Iterable[int],
    gt_id: int | None,
) -> int | None:
    for frame in frames:
        found = _best_tracker_id_for_gt(sequence, frame, gt_id)
        if found is not None:
            return found
    return None


@dataclass(frozen=True)
class HighlightPlan:
    before: int | None
    after: int | None
    failure_start: int
    failure_end: int
    error_type: str

    def id_for_frame(self, frame: int) -> int | None:
        if self.error_type == "FP_ID_HIJACK":
            return self.before
        if self.failure_start <= frame <= self.failure_end:
            if self.error_type == "IDSW":
                if frame == self.failure_start:
                    return self.before
                if frame == self.failure_end:
                    return self.after
                return None
            return None
        if frame < self.failure_start:
            return self.before
        return self.after


def resolve_highlight_plan(
    event: dict[str, Any], sequence: SequenceData
) -> HighlightPlan:
    start = int(event["frame_start"])
    end = int(event["frame_end"])
    focal = int(event["focal_gt_id"])
    before = _optional_int(event.get("prev_trk_id"))
    after = _optional_int(event.get("new_trk_id"))
    if before is None:
        before = _search_track_id(
            sequence,
            range(start - 1, max(0, start - PRE_FRAMES - 1), -1),
            focal,
        )
    if after is None:
        after = _search_track_id(
            sequence,
            range(end + 1, min(sequence.num_frames, end + POST_FRAMES) + 1),
            focal,
        )
    return HighlightPlan(
        before=before,
        after=after,
        failure_start=start,
        failure_end=end,
        error_type=str(event["error_type"]),
    )


def event_is_verifiable(event: dict[str, Any], sequence: SequenceData) -> bool:
    plan = resolve_highlight_plan(event, sequence)
    wanted = {track for track in (plan.before, plan.after) if track is not None}
    if not wanted:
        return False
    frames = clip_source_frames(
        clip_segments(event_with_highlight_after(event, plan.after), sequence.num_frames)
    )
    present = {
        int(box.trk_id)
        for frame in frames
        for box in sequence.tracker_by_frame.get(frame, [])
    }
    return bool(wanted & present)


def render_annotator_frame(
    image: np.ndarray,
    tracker_boxes: list[TrkBox],
    highlight_id: int | None,
) -> tuple[np.ndarray, list[str]]:
    """Tracker-only frame. Must not receive GT geometry."""
    canvas = image.copy()
    labels: list[str] = []
    for box in tracker_boxes:
        track_id = int(box.trk_id)
        highlighted = highlight_id is not None and track_id == int(highlight_id)
        labels.append(
            _draw_labeled_box(
                canvas,
                box.bbox,
                str(track_id),
                HIGHLIGHT_COLOR if highlighted else FAINT_COLOR,
                thickness=3 if highlighted else 1,
                scale=0.7 if highlighted else 0.4,
            )
        )
    return canvas, labels


def annotator_renderer_is_gt_free() -> bool:
    source = inspect.getsource(render_annotator_frame)
    return not any(term in source for term in ANNOTATOR_GT_LEAK_TERMS)


def _audit_header_lines(event: dict[str, Any], frame: int) -> list[str]:
    evidence = event.get("evidence") or {}
    bits = []
    for key in (
        "ioa_focal",
        "in_frame_fraction",
        "ioa_unannotated_structure",
        "vote_fraction",
        "trunc_edges",
    ):
        if key in evidence:
            bits.append(f"{key}={evidence[key]}")
    cause = str(event.get("cause"))
    return [
        f"{event['event_id']}  f={frame}  {cause}  {AVOIDABILITY.get(cause, 'unknown')}",
        "  ".join(str(item) for item in bits)[:160],
    ]


def render_audit_gt_frame(
    image: np.ndarray,
    frame: int,
    event: dict[str, Any],
    sequence: SequenceData,
    occluder_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Maximally leaky GT panel. Never shown to annotators or the VLM."""
    import cv2

    canvas = image.copy()
    if occluder_mask is not None and occluder_mask.shape[:2] == canvas.shape[:2]:
        tint = canvas.copy()
        tint[occluder_mask > 0] = (40, 180, 40)
        cv2.addWeighted(tint, 0.28, canvas, 0.72, 0.0, dst=canvas)
    height, width = canvas.shape[:2]
    cv2.rectangle(
        canvas,
        (AUDIT_EDGE_MARGIN, AUDIT_EDGE_MARGIN),
        (width - 1 - AUDIT_EDGE_MARGIN, height - 1 - AUDIT_EDGE_MARGIN),
        (0, 200, 255),
        1,
    )
    focal = int(event["focal_gt_id"])
    for box in sequence.gt_by_frame.get(frame, []):
        if box.cls in OCCLUDER_CLASSES:
            name = MOTCHALLENGE_CLASS_NAMES.get(box.cls, str(box.cls))
            _draw_labeled_box(
                canvas, box.bbox, name, AUDIT_OCCLUDER_COLOR, thickness=2, scale=0.45
            )
        elif box.cls in TARGET_CLASSES or box.gt_id == focal:
            if box.gt_id == focal:
                label = f"{box.gt_id} V:{box.visibility:.2f}"
                _draw_labeled_box(
                    canvas, box.bbox, label, AUDIT_FOCAL_COLOR, thickness=3, scale=0.55
                )
            else:
                _draw_labeled_box(
                    canvas,
                    box.bbox,
                    str(box.gt_id),
                    AUDIT_GT_COLOR,
                    thickness=1,
                    scale=0.4,
                )
    y = 22
    for line in _audit_header_lines(event, frame):
        if not line:
            continue
        cv2.putText(
            canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 10, 10), 3, cv2.LINE_AA
        )
        cv2.putText(
            canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA
        )
        y += 18
    return canvas


def pad_even(image: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    pad_x = width - image.shape[1]
    pad_y = height - image.shape[0]
    if pad_x == 0 and pad_y == 0:
        return image
    return cv2.copyMakeBorder(
        image, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )


def _ffmpeg_command(path: Path, width: int, height: int, fps: int) -> list[str]:
    return [
        find_ffmpeg(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "4.0",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-movflags",
        "+faststart",
        str(path),
    ]


def _start_encoder(path: Path, width: int, height: int, fps: int) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    process = subprocess.Popen(
        _ffmpeg_command(path, width, height, fps),
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError(f"ffmpeg stdin missing for {path}")
    return process


def _finish_encoder(process: subprocess.Popen, path: Path) -> None:
    _stdout, stderr = process.communicate()
    if process.returncode != 0 or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg failed for {path}: {stderr.decode('utf-8', 'replace')}"
        )


def encode_h264(
    frames: Iterable[np.ndarray],
    output_path: Path,
    width: int,
    height: int,
    fps: int = FPS,
) -> None:
    temporary = output_path.with_name(f".{output_path.stem}.rendering.mp4")
    if temporary.exists():
        temporary.unlink()
    process = _start_encoder(temporary, width, height, fps)
    try:
        assert process.stdin is not None
        for frame in frames:
            if frame.shape[1] != width or frame.shape[0] != height:
                raise ValueError(
                    f"Frame size {frame.shape[1]}x{frame.shape[0]} != {width}x{height}"
                )
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        _finish_encoder(process, temporary)
        os.replace(temporary, output_path)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if temporary.exists() and not output_path.exists():
            temporary.unlink()


def withheld_payload(event: dict[str, Any]) -> dict[str, Any]:
    cause = str(event["cause"])
    return {
        "cause": cause,
        "sequence": event["sequence"],
        "avoidability": AVOIDABILITY.get(cause, "unknown"),
        "evidence": deepcopy(event.get("evidence") or {}),
        "error_type": event["error_type"],
        "focal_gt_id": event["focal_gt_id"],
        "partner_gt_id": event.get("partner_gt_id"),
        "role": event.get("role"),
        "frame_start": event["frame_start"],
        "frame_end": event["frame_end"],
        "cause_confidence": event.get("cause_confidence"),
    }


def write_manifest(path: Path, sample: list[dict[str, Any]], available: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sampled = Counter(str(event["cause"]) for event in sample)
    fields = [
        "event_id",
        "cause",
        "sequence",
        "error_type",
        "n_frames",
        "underpowered",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in sample:
            cause = str(event["cause"])
            writer.writerow(
                {
                    "event_id": event["event_id"],
                    "cause": cause,
                    "sequence": event["sequence"],
                    "error_type": event["error_type"],
                    "n_frames": event_n_frames(event),
                    "underpowered": str(
                        sampled[cause] < TARGET_PER_CAUSE
                        or available[cause] < TARGET_PER_CAUSE
                    ).lower(),
                }
            )


def write_unverifiable_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["event_id", "cause", "sequence", "error_type", "reason"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_labelstudio_import_notes(path: Path) -> None:
    path.write_text(
        "Label Studio import — clips/ ONLY\n"
        "\n"
        "Import every MP4 under clips/. Do not add audit/, clips_meta/, withheld/,\n"
        "or summary/ to the project storage or task JSON.\n"
        "\n"
        "A filename collision with audit/ leaks full ground truth and is not recoverable.\n",
        encoding="utf-8",
    )


def write_annotator_guide_snippet(path: Path) -> None:
    path.write_text(
        "A highlighted box with a number is the track the program was following. "
        "At some point something goes wrong with it — it disappears, its number "
        "changes, or it slides onto somebody else. When it disappears, you will "
        "have to find the person yourself; that is part of the task.\n",
        encoding="utf-8",
    )


def leak_hits(texts: Iterable[str]) -> list[str]:
    return [text for text in texts if LEAK_PATTERN.search(text)]


def load_audit_masks(mask_root: Path | None) -> dict[str, np.ndarray]:
    if mask_root is None or not mask_root.is_dir():
        return {}
    masks: dict[str, np.ndarray] = {}
    for sequence in EVALUATION_SEQUENCES:
        path = mask_root / f"{sequence}__mask_static.npy"
        if path.is_file():
            masks[sequence] = np.load(path)
    return masks


def render_event_clips(
    event: dict[str, Any],
    sequence: SequenceData,
    clips_dir: Path,
    audit_dir: Path,
    meta_dir: Path,
    withheld_dir: Path,
    occluder_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    import cv2

    if not annotator_renderer_is_gt_free():
        raise AssertionError("render_annotator_frame must not mention GT")
    plan = resolve_highlight_plan(event, sequence)
    segments = clip_segments(
        event_with_highlight_after(event, plan.after), sequence.num_frames
    )
    source_frames = clip_source_frames(segments)
    if not source_frames:
        raise ValueError(f"No frames for {event['event_id']}")
    missing = [
        frame
        for frame in source_frames
        if not Path(sequence.image_paths[frame]).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Cannot render {event['event_id']}; missing frames {missing[:10]}"
        )
    encode_w, encode_h = even_size(sequence.width, sequence.height)
    overlay_strings: list[str] = []
    clip_path = clips_dir / f"{event['event_id']}.mp4"
    audit_path = audit_dir / f"{event['event_id']}.mp4"
    left_tmp = clip_path.with_name(f".{clip_path.stem}.rendering.mp4")
    audit_tmp = audit_path.with_name(f".{audit_path.stem}.rendering.mp4")
    left_enc = _start_encoder(left_tmp, encode_w, encode_h, FPS)
    audit_enc = _start_encoder(audit_tmp, encode_w * 2, encode_h, FPS)
    try:
        assert left_enc.stdin is not None and audit_enc.stdin is not None
        for index, segment in enumerate(segments):
            if index:
                skipped = frames_skipped_between(segments[index - 1], segment)
                source_fps = float(getattr(sequence, "fps", SOURCE_FPS) or SOURCE_FPS)
                marker = cut_marker_frame(
                    encode_w, encode_h, n_skipped=skipped, source_fps=source_fps
                )
                caption = skip_caption(skipped, source_fps)
                overlay_strings.append(caption)
                hold = CUT_MARKER_FRAMES
                for _ in range(hold):
                    left_enc.stdin.write(np.ascontiguousarray(marker).tobytes())
                    stacked = np.concatenate([marker, marker], axis=1)
                    audit_enc.stdin.write(np.ascontiguousarray(stacked).tobytes())
            for frame in segment:
                image = cv2.imread(sequence.image_paths[frame])
                if image is None:
                    raise RuntimeError(
                        f"OpenCV could not read {sequence.image_paths[frame]}"
                    )
                left, labels = render_annotator_frame(
                    image,
                    list(sequence.tracker_by_frame.get(frame, [])),
                    plan.id_for_frame(frame),
                )
                overlay_strings.extend(labels)
                left = pad_even(left, encode_w, encode_h)
                right = pad_even(
                    render_audit_gt_frame(
                        image, frame, event, sequence, occluder_mask
                    ),
                    encode_w,
                    encode_h,
                )
                left_enc.stdin.write(np.ascontiguousarray(left).tobytes())
                audit_enc.stdin.write(
                    np.ascontiguousarray(np.concatenate([left, right], axis=1)).tobytes()
                )
        _finish_encoder(left_enc, left_tmp)
        _finish_encoder(audit_enc, audit_tmp)
        os.replace(left_tmp, clip_path)
        os.replace(audit_tmp, audit_path)
    finally:
        for process, temporary, final in (
            (left_enc, left_tmp, clip_path),
            (audit_enc, audit_tmp, audit_path),
        ):
            if process.poll() is None:
                process.kill()
                process.wait()
            if temporary.exists() and not final.exists():
                temporary.unlink()
    unique_labels = sorted(set(overlay_strings))
    leaks = leak_hits(unique_labels)
    if leaks:
        raise AssertionError(
            f"{event['event_id']} annotator overlay leaked withheld terms: {leaks[:8]}"
        )
    meta = {
        "event_id": event["event_id"],
        "clip_frames": source_frames,
        "segments": segments,
        "cut_between_segments": len(segments) > 1,
        "cut_marker_frames": CUT_MARKER_FRAMES if len(segments) > 1 else 0,
        "highlighted_track_ids": {"before": plan.before, "after": plan.after},
        "failure_window": [plan.failure_start, plan.failure_end],
        "n_encoded_frames": (
            len(source_frames) + max(0, len(segments) - 1) * CUT_MARKER_FRAMES
        ),
        "overlay_strings": unique_labels,
        "encoded_frame_size": [encode_w, encode_h],
        "fps": FPS,
    }
    meta_dir.mkdir(parents=True, exist_ok=True)
    withheld_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / f"{event['event_id']}.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    (withheld_dir / f"{event['event_id']}.json").write_text(
        json.dumps({"event_id": event["event_id"], **withheld_payload(event)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return {
        "event_id": event["event_id"],
        "cause": event["cause"],
        "n_source_frames": len(source_frames),
        "n_encoded_frames": meta["n_encoded_frames"],
        "n_segments": len(segments),
        "cut_marker_frames": meta["cut_marker_frames"],
        "short_of_min": len(source_frames) < min_clip_frames_for(event),
        "has_cut": len(segments) > 1,
        "overlay_strings": unique_labels,
        "highlighted_track_ids": meta["highlighted_track_ids"],
    }


def verify_browser_ready(clip_path: Path) -> dict[str, Any]:
    ffprobe = find_ffprobe()
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,profile,pix_fmt",
            "-show_entries",
            "format=nb_streams",
            "-of",
            "json",
            str(clip_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(probe.stdout)
    stream = payload["streams"][0]
    audio = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(clip_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    seek = subprocess.run(
        [
            find_ffmpeg(),
            "-v",
            "error",
            "-ss",
            "1.5",
            "-i",
            str(clip_path),
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    return {
        "codec": stream.get("codec_name"),
        "profile": stream.get("profile"),
        "pix_fmt": stream.get("pix_fmt"),
        "h264_yuv420p": stream.get("codec_name") == "h264"
        and stream.get("pix_fmt") == "yuv420p",
        "no_audio": not audio.stdout.strip(),
        "mid_clip_seek_ok": seek.returncode == 0,
        "label_studio_ui_tested": False,
        "label_studio_ui_note": (
            "Codec/seek verified locally. Import clips/ only and confirm "
            "scrubbing in the Label Studio video tag before the annotation round."
        ),
    }


def write_s4_review_stills(
    sample: list[dict[str, Any]],
    audit_dir: Path,
    review_dir: Path,
) -> list[dict[str, Any]]:
    import cv2

    review_dir.mkdir(parents=True, exist_ok=True)
    chosen: dict[str, dict[str, Any]] = {}
    for event in sample:
        cause = str(event["cause"])
        if cause not in chosen:
            chosen[cause] = event
    rows: list[dict[str, Any]] = []
    for cause, event in sorted(chosen.items()):
        clip_path = audit_dir / f"{event['event_id']}.mp4"
        capture = cv2.VideoCapture(str(clip_path))
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            index = max(0, total // 2)
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read review frame from {clip_path}")
        still_path = review_dir / f"{cause}.jpg"
        if not cv2.imwrite(str(still_path), frame):
            raise RuntimeError(f"Could not write {still_path}")
        rows.append(
            {
                "cause": cause,
                "event_id": event["event_id"],
                "still": str(still_path),
                "review_frame_index": index,
            }
        )
    return rows


def run_s4_checks(
    sample: list[dict[str, Any]],
    render_rows: list[dict[str, Any]],
    clips_dir: Path,
    audit_dir: Path,
    meta_dir: Path,
    withheld_dir: Path,
    import_notes: Path,
) -> dict[str, Any]:
    overlay_texts = [
        text
        for row in render_rows
        for text in row.get("overlay_strings") or []
    ]
    leaks = leak_hits(overlay_texts)
    short = [row for row in render_rows if row["short_of_min"]]
    missing_clips = [
        event["event_id"]
        for event in sample
        if not (clips_dir / f"{event['event_id']}.mp4").is_file()
    ]
    missing_audit = [
        event["event_id"]
        for event in sample
        if not (audit_dir / f"{event['event_id']}.mp4").is_file()
    ]
    import_text = import_notes.read_text(encoding="utf-8") if import_notes.is_file() else ""
    return {
        "n_clips": len(render_rows),
        "overlay_leak_hits": leaks,
        "overlay_leak_free": not leaks,
        "short_clips": [row["event_id"] for row in short],
        "all_meet_min_length": not short,
        "min_clip_frames": MIN_CLIP_FRAMES,
        "min_non_fn_clip_frames": MIN_NON_FN_CLIP_FRAMES,
        "missing_clips": missing_clips,
        "missing_audit": missing_audit,
        "withheld_json_in_clips_dir": [path.name for path in clips_dir.glob("*.json")],
        "audit_json_in_clips_dir": [path.name for path in clips_dir.glob("*.json")],
        "annotator_renderer_gt_free": annotator_renderer_is_gt_free(),
        "audit_excluded_from_labelstudio_import": (
            "clips/" in import_text
            and "Do not add audit/" in import_text
            and "import audit" not in import_text.lower()
        ),
        "withheld_count": len(list(withheld_dir.glob("*.json"))),
        "meta_overlay_leaks": [
            path.name
            for path in meta_dir.glob("*.json")
            if leak_hits(json.loads(path.read_text(encoding="utf-8")).get("overlay_strings") or [])
        ],
    }


def run_probe_clip_sample(
    events_root: Path,
    tracker_output: Path,
    dataset_root: Path,
    output_root: Path,
    *,
    seed: int = SEED,
    workers: int = PROBE_WORKERS,
    sample_only: bool = False,
    mask_root: Path | None = None,
    keep_existing_manifest: bool = False,
    top_up_to: int | None = None,
) -> dict[str, Any]:
    events = load_sidecar_events(events_root)
    available = Counter(str(event["cause"]) for event in events)
    lengths = sequence_lengths(dataset_root)
    sequences, _coverage = load_sequences(
        tracker_output, dataset_root, PipelineConfig(render_clips=False)
    )
    unverifiable_rows: list[dict[str, Any]] = []
    verifiable: list[dict[str, Any]] = []
    for event in events:
        sequence = sequences[str(event["sequence"])]
        if event_is_verifiable(event, sequence):
            verifiable.append(event)
        else:
            unverifiable_rows.append(
                {
                    "event_id": event["event_id"],
                    "cause": event["cause"],
                    "sequence": event["sequence"],
                    "error_type": event["error_type"],
                    "reason": "no_target_tracker_box_in_clip_window",
                }
            )
    sample = sample_events(verifiable, seed=seed, lengths=lengths)
    manifest_path = output_root / "summary" / "clip_sample_manifest.csv"
    if keep_existing_manifest and manifest_path.is_file():
        by_id = {event["event_id"]: event for event in verifiable}
        kept = []
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                event = by_id.get(row["event_id"])
                if event is not None:
                    kept.append(event)
        if kept:
            sample = kept
    added: list[dict[str, Any]] = []
    if top_up_to is not None and len(sample) < top_up_to:
        sample, added = top_up_sample(
            sample,
            verifiable,
            extra_slots=min(top_up_to, MAX_CLIPS) - len(sample),
            lengths=lengths,
            seed=seed,
        )
    if len(sample) > MAX_CLIPS:
        raise AssertionError(f"Sample exceeded cap: {len(sample)} > {MAX_CLIPS}")
    clips_dir = output_root / "clips"
    audit_dir = output_root / "audit"
    meta_dir = output_root / "clips_meta"
    withheld_dir = output_root / "withheld"
    summary_dir = output_root / "summary"
    for directory in (clips_dir, audit_dir, meta_dir, withheld_dir, summary_dir):
        directory.mkdir(parents=True, exist_ok=True)
    write_manifest(summary_dir / "clip_sample_manifest.csv", sample, available)
    write_unverifiable_csv(summary_dir / "unverifiable_events.csv", unverifiable_rows)
    write_labelstudio_import_notes(summary_dir / "labelstudio_import.txt")
    write_annotator_guide_snippet(summary_dir / "annotator_guide_snippet.txt")
    sampled_causes = Counter(str(event["cause"]) for event in sample)
    report: dict[str, Any] = {
        "seed": seed,
        "target_per_cause": TARGET_PER_CAUSE,
        "min_per_cause": MIN_PER_CAUSE,
        "available_by_cause": dict(available),
        "verifiable_by_cause": dict(Counter(str(event["cause"]) for event in verifiable)),
        "sampled_by_cause": dict(sampled_causes),
        "n_sample": len(sample),
        "top_up_added": [event["event_id"] for event in added],
        "top_up_by_cause": dict(Counter(str(event["cause"]) for event in added)),
        "n_unverifiable": len(unverifiable_rows),
        "underpowered_causes": [
            cause
            for cause, count in sorted(available.items())
            if count < TARGET_PER_CAUSE
        ],
        "below_min_per_cause": [
            cause
            for cause, count in sorted(Counter(str(event["cause"]) for event in verifiable).items())
            if count < MIN_PER_CAUSE
        ],
        "sampled_under_15": [
            cause for cause, count in sorted(sampled_causes.items()) if count < 15
        ],
        "sampled_under_30": [
            cause for cause, count in sorted(sampled_causes.items()) if count < TARGET_PER_CAUSE
        ],
        "limitation": (
            f"{len(unverifiable_rows)} events could not be blind-verified because "
            "the tracker produced no box for the target."
        ),
    }
    if sample_only:
        (summary_dir / "probe_clip_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return report

    for directory in (clips_dir, audit_dir, meta_dir, withheld_dir):
        directory.mkdir(parents=True, exist_ok=True)
    masks = load_audit_masks(mask_root or Path("out/occluder_masks/occluders"))
    keep = {event["event_id"] for event in sample}
    for directory in (clips_dir, audit_dir, meta_dir, withheld_dir):
        for path in directory.iterdir():
            if path.stem not in keep and not path.name.startswith("."):
                path.unlink()

    def _already_rendered(event: dict[str, Any]) -> bool:
        event_id = event["event_id"]
        return all(
            path.is_file() and path.stat().st_size > 0
            for path in (
                clips_dir / f"{event_id}.mp4",
                audit_dir / f"{event_id}.mp4",
                meta_dir / f"{event_id}.json",
                withheld_dir / f"{event_id}.json",
            )
        )

    def _render(event: dict[str, Any]) -> dict[str, Any]:
        return render_event_clips(
            event,
            sequences[event["sequence"]],
            clips_dir,
            audit_dir,
            meta_dir,
            withheld_dir,
            masks.get(str(event["sequence"])),
        )

    def _row_from_meta(event: dict[str, Any]) -> dict[str, Any]:
        meta = json.loads(
            (meta_dir / f"{event['event_id']}.json").read_text(encoding="utf-8")
        )
        frames = meta.get("clip_frames") or []
        return {
            "event_id": event["event_id"],
            "cause": event["cause"],
            "n_source_frames": len(frames),
            "n_encoded_frames": meta.get("n_encoded_frames", len(frames)),
            "short_of_min": len(frames) < min_clip_frames_for(event),
            "has_cut": bool(meta.get("cut_between_segments")),
            "overlay_strings": meta.get("overlay_strings") or [],
            "highlighted_track_ids": meta.get("highlighted_track_ids"),
        }

    pending = [event for event in sample if not _already_rendered(event)]
    render_rows = [_row_from_meta(event) for event in sample if _already_rendered(event)]
    if pending:
        render_rows.append(_render(pending[0]))
        rest = pending[1:]
        if rest:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_render, event) for event in rest]
                for future in as_completed(futures):
                    render_rows.append(future.result())
    report["browser_ready"] = verify_browser_ready(
        clips_dir / f"{sample[0]['event_id']}.mp4"
    )
    if not report["browser_ready"]["h264_yuv420p"] or not report["browser_ready"]["mid_clip_seek_ok"]:
        raise AssertionError(f"Smoke clip is not browser-ready: {report['browser_ready']}")
    render_rows.sort(key=lambda row: row["event_id"])
    review = write_s4_review_stills(sample, audit_dir, summary_dir / "s4_review")
    checks = run_s4_checks(
        sample,
        render_rows,
        clips_dir,
        audit_dir,
        meta_dir,
        withheld_dir,
        summary_dir / "labelstudio_import.txt",
    )
    report["s4"] = checks
    report["s4_review"] = review
    report["n_cut_clips"] = sum(row["has_cut"] for row in render_rows)
    report["n_short_boundary_clips"] = sum(row["short_of_min"] for row in render_rows)
    failed = (
        checks["overlay_leak_hits"]
        or checks["meta_overlay_leaks"]
        or checks["missing_clips"]
        or checks["missing_audit"]
        or not checks["all_meet_min_length"]
        or not checks["annotator_renderer_gt_free"]
        or not checks["audit_excluded_from_labelstudio_import"]
        or checks["withheld_json_in_clips_dir"]
    )
    if failed:
        raise AssertionError(f"S4 gates failed: {checks}")
    (summary_dir / "probe_clip_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


SEGMENT_DIAG_FIELDS = (
    "clip_id",
    "event_id",
    "error_type",
    "cause",
    "span",
    "total",
    "n_segments",
    "resumes",
)


def segmentation_action(n_segmented: int) -> str:
    if n_segmented <= 15:
        return "fix_renderer_no_rerender"
    if n_segmented <= 30:
        return "fix_renderer_rerender_all_long"
    return "stop_and_reassess"


def currently_cut_event_ids(clip_map: Path, meta_dir: Path) -> list[str]:
    ids: list[str] = []
    with Path(clip_map).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            event_id = (row.get("event_id") or "").strip()
            if not event_id:
                continue
            meta_path = Path(meta_dir) / f"{event_id}.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            segments = meta.get("segments") or []
            if meta.get("cut_between_segments") and len(segments) > 1:
                ids.append(event_id)
    return ids


def clip_set_frozen_path(output_root: Path) -> Path:
    return Path(output_root) / "summary" / "clip_set_frozen.json"


def clip_set_is_frozen(output_root: Path) -> bool:
    path = clip_set_frozen_path(output_root)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(payload.get("frozen"))


def assert_clip_set_not_frozen(output_root: Path) -> None:
    if clip_set_is_frozen(output_root):
        raise SystemExit(
            f"Clip set is frozen ({clip_set_frozen_path(output_root)}). "
            "No further render changes."
        )


def y1_long_clip_rows(csv_path: Path) -> list[dict[str, str]]:
    if not Path(csv_path).is_file():
        return []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_void_a1(
    path: Path,
    clip_ids: Sequence[str],
    event_ids: Sequence[str],
) -> None:
    Path(path).write_text(
        json.dumps(
            {
                "clip_ids": list(clip_ids),
                "event_ids": list(event_ids),
                "n": len(clip_ids),
                "note": (
                    "A1 answers on these clips are void; they saw a different "
                    "stimulus. Exclude from A1 pairwise κ."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def copy_summary_to_vlm(output_root: Path, *names: str) -> None:
    vlm_summary = Path("out/vlm_probe/summary")
    if not vlm_summary.is_dir():
        return
    for name in names:
        source = Path(output_root) / "summary" / name
        if source.is_file():
            shutil.copy2(source, vlm_summary / name)


def diagnose_segmented_clips(
    clip_map: Path,
    meta_dir: Path,
    *,
    summary_dir: Path,
    pre: int = PRE_FRAMES,
    post: int = POST_FRAMES,
    max_frames: int = MAX_CLIP_FRAMES,
) -> dict[str, Any]:
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with Path(clip_map).open(newline="", encoding="utf-8") as handle:
        catalog = list(csv.DictReader(handle))
    for row in catalog:
        event_id = (row.get("event_id") or "").strip()
        meta_path = Path(meta_dir) / f"{event_id}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        window = meta.get("failure_window") or []
        if len(window) != 2:
            continue
        fs, fe = int(window[0]), int(window[1])
        span = fe - fs + 1
        total = span + pre + post
        segments = meta.get("segments") or []
        n_segments = len(segments) if segments else 1
        cut = bool(meta.get("cut_between_segments")) or total > max_frames
        if not cut and n_segments <= 1 and total <= max_frames:
            continue
        after = (meta.get("highlighted_track_ids") or {}).get("after")
        bucket = clip_stage1_bucket(
            {"error_type": row.get("error_type"), "new_trk_id": after}
        )
        rows.append(
            {
                "clip_id": row.get("clip_id") or "",
                "event_id": event_id,
                "error_type": bucket,
                "cause": row.get("cause") or "",
                "span": span,
                "total": total,
                "n_segments": n_segments,
                "resumes": bucket != "FN_never_resumes",
            }
        )
    csv_path = summary_dir / "segmented_clips.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SEGMENT_DIAG_FIELDS))
        writer.writeheader()
        for item in rows:
            writer.writerow(item)
    never = [item for item in rows if item["error_type"] == "FN_never_resumes"]
    n = len(rows)
    action = segmentation_action(n)
    report = {
        "n_catalog": len(catalog),
        "n_segmented": n,
        "action": action,
        "by_error_type": dict(Counter(item["error_type"] for item in rows)),
        "by_cause": dict(Counter(item["cause"] for item in rows)),
        "fn_never_resumes_clip_ids": [item["clip_id"] for item in never],
        "fn_never_resumes_event_ids": [item["event_id"] for item in never],
        "all_segmented_clip_ids": [item["clip_id"] for item in rows],
        "csv": str(csv_path),
        "limitation": (
            f"{n} of {len(catalog)} clips exceeded the maximum length and were "
            "presented as two segments separated by an explicit frame-count marker. "
            "Clips for failures where the track never resumed were truncated rather "
            "than segmented, since no return exists to display."
        ),
    }
    (summary_dir / "segmentation_fix.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def rerender_probe_events(
    event_ids: Sequence[str],
    *,
    events_root: Path,
    tracker_output: Path,
    dataset_root: Path,
    output_root: Path,
    mask_root: Path | None = None,
    clip_map: Path | None = None,
    hashed_video_dirs: Sequence[Path] | None = None,
) -> dict[str, Any]:
    wanted = {item.strip() for item in event_ids if item.strip()}
    if not wanted:
        return {"n_rendered": 0, "event_ids": []}
    events = [
        event
        for event in load_sidecar_events(events_root)
        if event["event_id"] in wanted
    ]
    missing = wanted - {event["event_id"] for event in events}
    if missing:
        raise ValueError(f"Events not in sidecar: {sorted(missing)}")
    sequences, _coverage = load_sequences(
        tracker_output, dataset_root, PipelineConfig(render_clips=False)
    )
    masks = load_audit_masks(mask_root or Path("out/occluder_masks/occluders"))
    clips_dir = output_root / "clips"
    audit_dir = output_root / "audit"
    meta_dir = output_root / "clips_meta"
    withheld_dir = output_root / "withheld"
    rendered = []
    for event in events:
        rendered.append(
            render_event_clips(
                event,
                sequences[str(event["sequence"])],
                clips_dir,
                audit_dir,
                meta_dir,
                withheld_dir,
                masks.get(str(event["sequence"])),
            )
        )
    hashed_counts: dict[str, int] = {}
    void_clip_ids: list[str] = []
    if clip_map and Path(clip_map).is_file():
        from mot_error_attribution.probe_clip_ls import place_video

        by_event = {
            row["event_id"]: row
            for row in csv.DictReader(Path(clip_map).open(newline="", encoding="utf-8"))
        }
        dest_roots = [Path(item) for item in (hashed_video_dirs or []) if Path(item).is_dir()]
        for event in events:
            row = by_event.get(event["event_id"])
            if not row:
                continue
            void_clip_ids.append(row.get("clip_id") or "")
            source = clips_dir / f"{event['event_id']}.mp4"
            hashed = (row.get("hashed_filename") or "").strip()
            if not hashed or not source.is_file():
                continue
            for dest_root in dest_roots:
                dest = dest_root / hashed
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                action = place_video(source, dest)
                hashed_counts[action] = hashed_counts.get(action, 0) + 1
    return {
        "n_rendered": len(rendered),
        "event_ids": [event["event_id"] for event in events],
        "void_a1_clip_ids": [item for item in void_clip_ids if item],
        "hashed_video_placement": hashed_counts,
        "has_cut": [row["has_cut"] for row in rendered],
        "n_segments": [row.get("n_segments") for row in rendered],
        "cut_marker_frames": [row.get("cut_marker_frames") for row in rendered],
        "n_source_frames": [row["n_source_frames"] for row in rendered],
        "skip_captions": [
            item
            for row in rendered
            for item in row.get("overlay_strings") or []
            if str(item).startswith("--- skipping ")
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample and render the shared VLM / human probe clip set"
    )
    parser.add_argument(
        "--events-root",
        type=Path,
        default=Path("out/min_fn3/events"),
    )
    parser.add_argument(
        "--tracker-output",
        type=Path,
        default=Path("tracker_output/tracktrack"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/MOT20"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("out/probe_clips")
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=Path("out/occluder_masks/occluders"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=PROBE_WORKERS)
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Write the manifest without rendering clips",
    )
    parser.add_argument(
        "--keep-existing-manifest",
        action="store_true",
        help="Keep the current clip_sample_manifest.csv and only add top-up events",
    )
    parser.add_argument(
        "--top-up-to",
        type=int,
        default=None,
        help="Add leftover slots up to this size from under-powered causes with depth",
    )
    parser.add_argument(
        "--diagnose-segments",
        action="store_true",
        help="Count currently segmented clips and write segmented_clips.csv; do not render",
    )
    parser.add_argument(
        "--fix-segmentation",
        action="store_true",
        help="Diagnose, then re-render all long clips still using a cut",
    )
    parser.add_argument(
        "--rerender-cut-clips",
        action="store_true",
        help="Re-render clips whose current meta still has a cut; do not rewrite Y1 CSV",
    )
    parser.add_argument(
        "--clip-map",
        type=Path,
        default=Path("out/labelstudio_probe_clips/clip_map.csv"),
    )
    return parser


HASHED_VIDEO_DIRS = (
    Path("out/labelstudio_probe_clips/videos"),
)


def finalize_long_clip_set(
    output_root: Path,
    rendered: dict[str, Any],
    *,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary_dir = Path(output_root) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    report = dict(previous or {})
    y1_rows = y1_long_clip_rows(summary_dir / "segmented_clips.csv")
    clip_ids = sorted(
        {row.get("clip_id") or "" for row in y1_rows if row.get("clip_id")}
        | set(report.get("all_segmented_clip_ids") or [])
        | set(report.get("void_a1_clip_ids") or [])
        | set(rendered.get("void_a1_clip_ids") or [])
    )
    event_by_clip = {
        row["clip_id"]: row["event_id"] for row in y1_rows if row.get("clip_id")
    }
    event_ids = [event_by_clip[cid] for cid in clip_ids if cid in event_by_clip]
    if not event_ids:
        event_ids = list(
            dict.fromkeys(
                list(report.get("fn_never_resumes_event_ids") or [])
                + list(rendered.get("event_ids") or [])
            )
        )
    n = int(report.get("n_segmented") or len(clip_ids))
    catalog = int(report.get("n_catalog") or 284)
    report["action"] = "fix_renderer_rerender_all_long"
    report["rerender_remaining"] = rendered
    report["void_a1_clip_ids"] = clip_ids
    report["remaining_four_segment_clip_ids"] = []
    report["clip_set_frozen"] = True
    report["limitation"] = (
        f"{n} of {catalog} clips exceeded the maximum length and were "
        "presented as two segments separated by an explicit frame-count marker. "
        "Clips for failures where the track never resumed were truncated rather "
        "than segmented, since no return exists to display."
    )
    write_void_a1(summary_dir / "void_a1_clip_ids.json", clip_ids, event_ids)
    (summary_dir / "segmentation_fix.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    freeze = {
        "frozen": True,
        "n_clips": catalog,
        "n_long": n,
        "void_a1_n": len(clip_ids),
        "void_a1_clip_ids": clip_ids,
        "note": (
            "Final annotator stimulus. No further render changes. "
            "A2–A4 use this version. Exclude the void A1 IDs from A1 pairwise κ."
        ),
    }
    clip_set_frozen_path(output_root).write_text(
        json.dumps(freeze, indent=2) + "\n", encoding="utf-8"
    )
    copy_summary_to_vlm(
        output_root,
        "segmentation_fix.json",
        "void_a1_clip_ids.json",
        "clip_set_frozen.json",
    )
    return report


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.diagnose_segments and not args.fix_segmentation and not args.rerender_cut_clips:
        if clip_set_is_frozen(args.output_root):
            raise SystemExit(
                f"Clip set is frozen ({clip_set_frozen_path(args.output_root)}). "
                "Not rewriting segmented_clips.csv."
            )
        diagnosis = diagnose_segmented_clips(
            args.clip_map,
            args.output_root / "clips_meta",
            summary_dir=args.output_root / "summary",
        )
        copy_summary_to_vlm(
            args.output_root, "segmented_clips.csv", "segmentation_fix.json"
        )
        print(json.dumps(diagnosis, indent=2))
        return
    if args.rerender_cut_clips or args.fix_segmentation:
        assert_clip_set_not_frozen(args.output_root)
        if args.fix_segmentation and not args.rerender_cut_clips:
            diagnosis = diagnose_segmented_clips(
                args.clip_map,
                args.output_root / "clips_meta",
                summary_dir=args.output_root / "summary",
            )
            copy_summary_to_vlm(
                args.output_root, "segmented_clips.csv", "segmentation_fix.json"
            )
            print(json.dumps(diagnosis, indent=2))
            if diagnosis["action"] == "stop_and_reassess":
                raise SystemExit(
                    f"Segmented count {diagnosis['n_segmented']} > 30. Stop and reassess."
                )
        event_ids = currently_cut_event_ids(
            args.clip_map, args.output_root / "clips_meta"
        )
        rendered = rerender_probe_events(
            event_ids,
            events_root=args.events_root,
            tracker_output=args.tracker_output,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
            mask_root=args.mask_root,
            clip_map=args.clip_map,
            hashed_video_dirs=HASHED_VIDEO_DIRS,
        )
        print(json.dumps(rendered, indent=2))
        previous_path = args.output_root / "summary" / "segmentation_fix.json"
        previous = (
            json.loads(previous_path.read_text(encoding="utf-8"))
            if previous_path.is_file()
            else {}
        )
        report = finalize_long_clip_set(
            args.output_root, rendered, previous=previous
        )
        print(json.dumps(report, indent=2))
        return
    assert_clip_set_not_frozen(args.output_root)
    report = run_probe_clip_sample(
        events_root=args.events_root,
        tracker_output=args.tracker_output,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        seed=args.seed,
        workers=args.workers,
        sample_only=args.sample_only,
        mask_root=args.mask_root,
        keep_existing_manifest=args.keep_existing_manifest,
        top_up_to=args.top_up_to,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
