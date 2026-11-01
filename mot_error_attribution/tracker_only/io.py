"""Load frozen regime-A clip assets without re-mining."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from mot_error_attribution.annotator_agreement import (
    parse_result_fields,
)
from mot_error_attribution.vlm_probe.summarize import write_csv, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STYLE_ROOT = REPO_ROOT / "vlm_clip_style_tests"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tracker-only-corpus"
DEFAULT_EXPORT = REPO_ROOT / "annotations" / "gt_v3.json"
DEFAULT_GT_V2 = REPO_ROOT / "annotations" / "human" / "verifier_forward.csv"

GHOST_CLIP_IDS = (
    "C002",
    "C010",
    "C025",
    "C059",
    "C063",
    "C090",
    "C105",
    "C107",
    "C174",
    "C260",
    "C262",
    "C277",
)
GHOST_CLIP_SET = frozenset(GHOST_CLIP_IDS)

ERROR_KIND_TO_PRIMARY = {
    "miss": "miss",
    "identity_change": "identity_change",
    "drift": "drift",
    "no_error_visible": "no_error",
    "no_error": "no_error",
}
NONCORE_KINDS = frozenset({"cannot_determine", "could_not_identify_target"})
STATUS_FROM_KIND = {
    "miss": "valid",
    "identity_change": "valid",
    "drift": "valid",
    "no_error_visible": "valid",
    "no_error": "valid",
    "cannot_determine": "ambiguous",
    "could_not_identify_target": "target_not_identifiable",
}
VISIBILITY_FROM_LS = {
    "clearly": "fully_visible",
    "partly": "partly_visible",
    "not_at_all": "not_visible",
    "cannot_tell": "uncertain",
}
CONDITION_FROM_CAUSE = {
    "hidden_by_object": "static_occlusion",
    "hidden_by_one_person": "single_person_occlusion",
    "hidden_by_standing_person": "single_person_occlusion",
    "hidden_by_crowd": "crowd_occlusion",
    "partly_blocked_by_person": "single_person_occlusion",
    "partly_blocked_by_static": "static_occlusion",
    "partly_outside_the_frame": "truncation",
    "clearly_visible_but_missed": "no_obvious_visual_trigger",
}
PLAN_EXPECTED_COUNTS = {
    "miss": 143,
    "identity_change": 64,
    "no_error": 50,
    "non_core": 24,
    "ghost_track": 12,
    "drift": 2,
}
LS_EXPECTED_COUNTS = {
    "miss": 143,
    "identity_change": 65,
    "no_error": 50,
    "non_core": 24,
    "drift": 2,
}


def clip_frames_digest(frames: Sequence[int]) -> str:
    payload = json.dumps([int(frame) for frame in frames], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def join_frames(frames: Sequence[int]) -> str:
    return ";".join(str(int(frame)) for frame in frames)


def parse_frames(value: Any) -> list[int]:
    if value in (None, "", []):
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item) for item in str(value).split(";") if str(item).strip()]


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_manifest(style_root: Path) -> dict[str, dict[str, str]]:
    path = Path(style_root) / "manifest.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["clip_id"]: row for row in csv.DictReader(handle)}


def list_clip_ids(style_root: Path) -> list[str]:
    mined = Path(style_root) / "mined_results"
    return sorted(path.stem for path in mined.glob("C*.json"))


def latest_annotation_fields(task: Mapping[str, Any]) -> dict[str, str]:
    chosen: Mapping[str, Any] | None = None
    for annotation in task.get("annotations") or ():
        if annotation.get("was_cancelled"):
            continue
        if chosen is None or str(
            annotation.get("updated_at") or annotation.get("created_at") or ""
        ) >= str(chosen.get("updated_at") or chosen.get("created_at") or ""):
            chosen = annotation
    if chosen is None:
        return {}
    return parse_result_fields(chosen.get("result"))


def load_human_labels(path: Path) -> dict[str, dict[str, Any]]:
    tasks = load_json(path)
    if not isinstance(tasks, list):
        raise ValueError(f"Expected a JSON list of Label Studio tasks in {path}")
    labels: dict[str, dict[str, Any]] = {}
    for task in tasks:
        clip_id = str((task.get("data") or {}).get("clip_id") or "").strip()
        if not clip_id:
            continue
        fields = latest_annotation_fields(task)
        labels[clip_id] = {
            "clip_id": clip_id,
            "task_id": task.get("id"),
            "error_kind": fields.get("error_kind", ""),
            "visibility": fields.get("visibility", ""),
            "cause_miss": fields.get("cause_miss", ""),
            "cause_idsw": fields.get("cause_idsw", ""),
            "cause_drift": fields.get("cause_drift", ""),
            "confidence": fields.get("confidence", ""),
            "free_text": fields.get("free_text", ""),
            "source_video": (task.get("data") or {}).get("video", ""),
        }
    return labels


def load_gt_v2_records(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not Path(path).is_file():
        return {}
    payload = load_json(path)
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"GT has no records list: {path}")
    return {str(row["clip_id"]): row for row in records if row.get("clip_id")}


def load_clip_bundle(style_root: Path, clip_id: str, manifest: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    root = Path(style_root)
    mined = load_json(root / "mined_results" / f"{clip_id}.json")
    annotation = load_json(root / "annotations" / f"{clip_id}.json")
    matching = list(annotation["target"]["matching_by_frame"])
    tracker_by_frame: dict[int, set[int]] = {}
    matched_gt_by_track_frame: dict[tuple[int, int], int | None] = {}
    for row in annotation.get("prediction_annotations") or ():
        frame = int(row["source_frame"])
        track_id = int(row["track_id"])
        tracker_by_frame.setdefault(frame, set()).add(track_id)
        matched = row.get("matched_gt_id")
        matched_gt_by_track_frame[(frame, track_id)] = (
            None if matched in (None, "") else int(matched)
        )
    clip_frames = [int(frame) for frame in annotation["clip_frames"]]
    return {
        "clip_id": clip_id,
        "event_id": mined.get("event_id") or annotation.get("event_id"),
        "sequence": mined["sequence"],
        "error_type": mined["error_type"],
        "focal_gt_id": int(mined["focal_gt_id"]),
        "frame_start": int(mined["frame_start"]),
        "frame_end": int(mined["frame_end"]),
        "clip_alignment": mined.get("clip_alignment") or {},
        "clip_frames": clip_frames,
        "matching_by_frame": matching,
        "tracker_by_frame": tracker_by_frame,
        "matched_gt_by_track_frame": matched_gt_by_track_frame,
        "annotation": annotation,
        "mined": mined,
        "regime_a_manifest_sha256": (manifest.get(clip_id) or {}).get("sha256", ""),
        "clip_frames_sha256": clip_frames_digest(clip_frames),
    }


def write_atomic_json(path: Path, payload: Any) -> None:
    write_json(path, payload)


def write_atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    write_csv(path, rows, fields)
