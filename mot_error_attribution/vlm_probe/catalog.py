from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import (
    DEFAULT_CLIP_MAP,
    DEFAULT_CLIPS_DIR,
    DEFAULT_CLIPS_META_DIR,
    DEFAULT_HASHED_DIR,
    SMOKE_CAUSES,
    SMOKE_N,
)


@dataclass
class ClipRecord:
    clip_id: str
    hashed_filename: str
    event_id: str
    source_filename: str
    cause: str
    sequence: str
    error_type: str
    video_path: Path | None = None
    n_encoded_frames: int | None = None
    highlight_before: int | None = None
    highlight_after: int | None = None
    clip_frames: list[int] = field(default_factory=list)
    failure_window: tuple[int, int] | None = None
    segments: list[list[int]] = field(default_factory=list)
    cut_between_segments: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def fn_resumes_same_id(self) -> bool | None:
        if self.error_type == "FRAG_NO_SWITCH":
            return True
        if self.error_type != "FN":
            return None
        if self.highlight_after is None:
            return False
        if self.highlight_before is None:
            return True
        return self.highlight_before == self.highlight_after

    @property
    def stage1_bucket(self) -> str:
        if self.error_type == "IDSW":
            return "IDSW"
        if self.error_type == "FP_ID_HIJACK":
            return "FP_ID_HIJACK"
        if self.error_type == "FRAG_NO_SWITCH":
            return "FN_resumes_same_id"
        if self.error_type == "FN":
            if self.fn_resumes_same_id:
                return "FN_resumes_same_id"
            return "FN_never_resumes"
        return self.error_type


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_video_path(
    row: dict[str, str],
    *,
    clips_dir: Path,
    hashed_dir: Path | None = None,
) -> Path | None:
    source = (row.get("source_filename") or "").strip()
    event_id = (row.get("event_id") or "").strip()
    hashed = (row.get("hashed_filename") or "").strip()
    candidates = []
    if source:
        candidates.append(clips_dir / source)
    if event_id:
        candidates.append(clips_dir / f"{event_id}.mp4")
    if hashed:
        candidates.append(clips_dir / hashed)
        if hashed_dir is not None:
            candidates.append(hashed_dir / hashed)
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0] if candidates else None


def _highlight_ids(meta: dict[str, Any]) -> tuple[int | None, int | None]:
    block = (
        meta.get("highlighted_track_ids")
        or meta.get("highlighted_track_ids")
        or {}
    )
    if not isinstance(block, dict):
        return None, None
    before = block.get("before", block.get("before"))
    after = block.get("after", block.get("after"))
    return _optional_int(before), _optional_int(after)


def _n_encoded_frames(meta: dict[str, Any]) -> int | None:
    for key in ("n_encoded_frames", "n_encoded_frames", "n_clip_frames"):
        parsed = _optional_int(meta.get(key))
        if parsed is not None:
            return parsed
    frames = meta.get("clip_frames")
    if isinstance(frames, list) and frames:
        return len(frames)
    return None


def load_clip_meta(meta_dir: Path, event_id: str) -> dict[str, Any]:
    path = meta_dir / f"{event_id}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _failure_window(meta: dict[str, Any]) -> tuple[int, int] | None:
    window = meta.get("failure_window")
    if isinstance(window, (list, tuple)) and len(window) == 2:
        start, end = _optional_int(window[0]), _optional_int(window[1])
        if start is not None and end is not None:
            return start, end
    return None


def _clip_frames(meta: dict[str, Any]) -> list[int]:
    frames = meta.get("clip_frames")
    if not isinstance(frames, list):
        return []
    out: list[int] = []
    for item in frames:
        parsed = _optional_int(item)
        if parsed is not None:
            out.append(parsed)
    return out


def _segments(meta: dict[str, Any]) -> list[list[int]]:
    raw = meta.get("segments")
    if not isinstance(raw, list):
        return []
    segments: list[list[int]] = []
    for block in raw:
        if not isinstance(block, list):
            continue
        frames = [parsed for parsed in (_optional_int(item) for item in block) if parsed is not None]
        if frames:
            segments.append(frames)
    return segments


def load_catalog(
    clip_map: Path | None = None,
    *,
    clips_dir: Path | None = None,
    meta_dir: Path | None = None,
    hashed_dir: Path | None = None,
) -> list[ClipRecord]:
    map_path = Path(clip_map or DEFAULT_CLIP_MAP)
    clips = Path(clips_dir or DEFAULT_CLIPS_DIR)
    meta = Path(meta_dir or DEFAULT_CLIPS_META_DIR)
    hashed = Path(hashed_dir or DEFAULT_HASHED_DIR)
    if not map_path.is_file():
        raise FileNotFoundError(f"Missing clip map: {map_path}")
    records: list[ClipRecord] = []
    with map_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            event_id = (row.get("event_id") or "").strip()
            sidecar = load_clip_meta(meta, event_id) if event_id else {}
            before, after = _highlight_ids(sidecar)
            records.append(
                ClipRecord(
                    clip_id=(row.get("clip_id") or "").strip(),
                    hashed_filename=(row.get("hashed_filename") or "").strip(),
                    event_id=event_id,
                    source_filename=(row.get("source_filename") or "").strip(),
                    cause=(row.get("cause") or "").strip(),
                    sequence=(row.get("sequence") or "").strip(),
                    error_type=(row.get("error_type") or "").strip(),
                    video_path=resolve_video_path(
                        row, clips_dir=clips, hashed_dir=hashed
                    ),
                    n_encoded_frames=_n_encoded_frames(sidecar),
                    highlight_before=before,
                    highlight_after=after,
                    clip_frames=_clip_frames(sidecar),
                    failure_window=_failure_window(sidecar),
                    segments=_segments(sidecar),
                    cut_between_segments=bool(sidecar.get("cut_between_segments")),
                )
            )
    records.sort(key=lambda item: item.clip_id)
    return records


def select_smoke_clips(
    records: Iterable[ClipRecord],
    *,
    n: int = SMOKE_N,
    causes: tuple[str, ...] = SMOKE_CAUSES,
) -> list[ClipRecord]:
    by_cause: dict[str, list[ClipRecord]] = defaultdict(list)
    for record in records:
        by_cause[record.cause].append(record)
    for pool in by_cause.values():
        pool.sort(key=lambda item: item.clip_id)
    missing = [cause for cause in causes if not by_cause.get(cause)]
    if missing:
        raise ValueError(f"Smoke causes missing from catalog: {missing}")
    per = max(1, n // len(causes))
    selected: list[ClipRecord] = []
    seen: set[str] = set()
    for cause in causes:
        for record in by_cause[cause][:per]:
            if record.clip_id not in seen:
                selected.append(record)
                seen.add(record.clip_id)
    if len(selected) < n:
        extras = [
            record
            for cause in causes
            for record in by_cause[cause]
            if record.clip_id not in seen
        ]
        for record in extras:
            selected.append(record)
            seen.add(record.clip_id)
            if len(selected) >= n:
                break
    selected.sort(key=lambda item: item.clip_id)
    return selected[:n]


def _take_sorted(pool: list[ClipRecord], n: int, seen: set[str]) -> list[ClipRecord]:
    picked: list[ClipRecord] = []
    for record in sorted(pool, key=lambda item: item.clip_id):
        if record.clip_id in seen:
            continue
        picked.append(record)
        seen.add(record.clip_id)
        if len(picked) >= n:
            break
    return picked


def select_perception_clips(records: Iterable[ClipRecord], n: int = 10) -> list[ClipRecord]:
    """V2: 10 clips spanning MOT20-01 (day) and MOT20-03 (night)."""
    items = list(records)
    day = [record for record in items if record.sequence == "MOT20-01"]
    night = [record for record in items if record.sequence == "MOT20-03"]
    per = max(1, n // 2)
    seen: set[str] = set()
    selected = _take_sorted(day, per, seen) + _take_sorted(night, n - per, seen)
    if len(selected) < n:
        selected.extend(_take_sorted(items, n - len(selected), seen))
    selected.sort(key=lambda item: item.clip_id)
    return selected[:n]


def can_extract_fn_stills(record: ClipRecord) -> bool:
    if record.error_type != "FN" or not record.clip_frames or record.failure_window is None:
        return False
    fs, fe = record.failure_window
    frames = record.clip_frames
    before = any(frame < fs for frame in frames)
    inside = any(fs <= frame <= fe for frame in frames)
    return before and inside


def select_fn_binary_clips(records: Iterable[ClipRecord], n: int = 50) -> list[ClipRecord]:
    """V3/W1: FN events with a frame before the gap and a frame inside it."""
    eligible = [record for record in records if can_extract_fn_stills(record)]
    day = [record for record in eligible if record.sequence == "MOT20-01"]
    night = [record for record in eligible if record.sequence == "MOT20-03"]
    per = max(1, n // 2)
    seen: set[str] = set()
    selected = _take_sorted(day, per, seen) + _take_sorted(night, n - per, seen)
    if len(selected) < n:
        selected.extend(_take_sorted(eligible, n - len(selected), seen))
    selected.sort(key=lambda item: item.clip_id)
    return selected[:n]


def select_centred_preview_clips(records: Iterable[ClipRecord], n: int = 20) -> list[ClipRecord]:
    """V4: 20 mixed clips for the event-centred Stage 1 preview."""
    items = list(records)
    buckets = {
        "FN": [record for record in items if record.error_type == "FN"],
        "IDSW": [record for record in items if record.error_type == "IDSW"],
        "FP_ID_HIJACK": [record for record in items if record.error_type == "FP_ID_HIJACK"],
        "FRAG_NO_SWITCH": [record for record in items if record.error_type == "FRAG_NO_SWITCH"],
    }
    quotas = {"FN": 8, "IDSW": 6, "FP_ID_HIJACK": 4, "FRAG_NO_SWITCH": 2}
    seen: set[str] = set()
    selected: list[ClipRecord] = []
    for error_type, quota in quotas.items():
        selected.extend(_take_sorted(buckets[error_type], quota, seen))
    if len(selected) < n:
        selected.extend(_take_sorted(items, n - len(selected), seen))
    selected.sort(key=lambda item: item.clip_id)
    return selected[:n]


def probe_n_frames(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            capture.release()
        if count > 0:
            return count
    except Exception:
        pass
    import shutil
    import subprocess

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    text = (result.stdout or "").strip()
    if text.isdigit():
        return int(text)
    return None
