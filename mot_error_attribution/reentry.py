"""Reentry diagnosis and the absence-triggered rule (reentry-plan.md)."""

from __future__ import annotations

import csv
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import (
    DEFAULT_CLASSES,
    DEFAULT_THRESHOLDS,
    EVALUATION_SEQUENCES,
    PipelineConfig,
)
from .geometry import border_edges, near_frame_edge
from .models import BBox
from .io import _read_gt, _read_seqinfo
from .occluder_masks import load_events_jsonl, write_events_jsonl


REENTRY_MIN_ABSENT_FRAMES = int(DEFAULT_THRESHOLDS["REENTRY_MIN_ABSENT_FRAMES"])
EDGE_MARGIN_PX = float(DEFAULT_THRESHOLDS["EDGE_MARGIN_PX"])
TARGET_CLASSES = set(DEFAULT_CLASSES["TARGET_CLASSES"])

GTPresent = Mapping[int, set[int]]
BoxAt = Callable[[int, int], BBox | None]


def _as_present_set(value: Any) -> set[int]:
    if value is None:
        return set()
    return {int(item) for item in value}


def reentry_span_stats(
    event: Mapping[str, Any],
    gt_present: GTPresent,
) -> dict[str, Any]:
    start = int(event["frame_start"])
    end = int(event["frame_end"])
    focal = int(event["focal_gt_id"])
    frames = list(range(start, end + 1))
    present = [
        frame for frame in frames if focal in _as_present_set(gt_present.get(frame))
    ]
    absent = [frame for frame in frames if frame not in present]
    return {
        "span_frames": len(frames),
        "present_frames": present,
        "absent_frames": absent,
        "n_absent": len(absent),
        "n_present": len(present),
    }


def last_box_before_gap(
    event: Mapping[str, Any],
    present_frames: list[int],
    absent_frames: list[int],
    box_at: BoxAt,
    gt_present: GTPresent,
) -> tuple[BBox | None, int | None]:
    if not absent_frames:
        last_frame = max(present_frames) if present_frames else None
        if last_frame is None:
            return None, None
        return box_at(int(event["focal_gt_id"]), last_frame), last_frame

    before = [frame for frame in present_frames if frame < min(absent_frames)]
    if not before:
        focal = int(event["focal_gt_id"])
        for frame in range(int(event["frame_start"]) - 1, 0, -1):
            if focal in _as_present_set(gt_present.get(frame)):
                before = [frame]
                break
    if not before:
        return None, None
    last_frame = max(before)
    return box_at(int(event["focal_gt_id"]), last_frame), last_frame


def check_reentry(
    event: Mapping[str, Any],
    gt_present: GTPresent,
    box_at: BoxAt,
    width: int,
    height: int,
    *,
    min_absent: int = REENTRY_MIN_ABSENT_FRAMES,
    margin: float = EDGE_MARGIN_PX,
) -> tuple[str, dict[str, Any]] | None:
    """Absence-triggered reentry. Border proximity is only a sub-classifier."""
    if str(event.get("error_type")) != "IDSW":
        return None
    stats = reentry_span_stats(event, gt_present)
    absent = stats["absent_frames"]
    present = stats["present_frames"]
    if len(absent) < int(min_absent):
        return None
    if not present:
        return None
    last_box, last_frame = last_box_before_gap(
        event, present, absent, box_at, gt_present
    )
    if last_box is None or last_frame is None:
        return None
    extra = {
        "absent_frames": len(absent),
        "reentry_last_seen_frame": last_frame,
        "reentry_min_absent_frames": int(min_absent),
    }
    if near_frame_edge(last_box, width, height, margin):
        extra["exit_edge"] = ",".join(border_edges(last_box, width, height, margin))
        return ("reentry", extra)
    return ("gt_annotation_gap", extra)


def diagnostic_row(
    event: Mapping[str, Any],
    gt_present: GTPresent,
    box_at: BoxAt,
    width: int,
    height: int,
    *,
    margin: float = EDGE_MARGIN_PX,
) -> dict[str, Any]:
    stats = reentry_span_stats(event, gt_present)
    present = stats["present_frames"]
    last_frame = max(present) if present else None
    last_box = (
        box_at(int(event["focal_gt_id"]), last_frame) if last_frame is not None else None
    )
    near_edge = (
        near_frame_edge(last_box, width, height, margin) if last_box is not None else None
    )
    edges = (
        border_edges(last_box, width, height, margin) if last_box is not None else ()
    )
    absent = int(stats["n_absent"])
    return {
        "event_id": event["event_id"],
        "sequence": event["sequence"],
        "span_frames": stats["span_frames"],
        "absent_frames": absent,
        "last_seen_near_edge": "" if near_edge is None else str(bool(near_edge)).lower(),
        "which_edge": ",".join(edges),
        "trigger_was_border_only": str(bool(near_edge) and absent == 0).lower(),
        "error_type": event.get("error_type"),
        "focal_gt_id": event.get("focal_gt_id"),
        "frame_start": event.get("frame_start"),
        "frame_end": event.get("frame_end"),
    }


def load_sequence_gt(
    dataset_root: Path, sequence: str
) -> tuple[dict[int, set[int]], dict[tuple[int, int], BBox], int, int]:
    num_frames, width, height, _fps = _read_seqinfo(dataset_root, sequence)
    selected = set(range(1, num_frames + 1))
    by_frame, _full, _selected, _flags = _read_gt(dataset_root, sequence, selected)
    present: dict[int, set[int]] = {frame: set() for frame in selected}
    boxes: dict[tuple[int, int], BBox] = {}
    for frame, frame_boxes in by_frame.items():
        ids: set[int] = set()
        for box in frame_boxes:
            if box.flag == 0 or box.cls not in TARGET_CLASSES:
                continue
            ids.add(int(box.gt_id))
            boxes[(int(frame), int(box.gt_id))] = box.bbox
        present[int(frame)] = ids
    return present, boxes, width, height


def _box_lookup(boxes: dict[tuple[int, int], BBox]) -> BoxAt:
    def box_at(gt_id: int, frame: int) -> BBox | None:
        return boxes.get((int(frame), int(gt_id)))

    return box_at


def load_reentry_events(events_root: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for sequence in EVALUATION_SEQUENCES:
        matches = sorted(events_root.glob(f"{sequence}__*.jsonl"))
        if not matches:
            raise FileNotFoundError(f"No sidecar events for {sequence} in {events_root}")
        events.extend(
            event
            for event in load_events_jsonl(matches[0])
            if str(event.get("cause")) == "reentry"
        )
    return events


def diagnose_reentry_corpus(
    events_root: Path,
    dataset_root: Path,
) -> list[dict[str, Any]]:
    cache: dict[str, tuple[dict[int, set[int]], dict[tuple[int, int], BBox], int, int]] = {}
    rows: list[dict[str, Any]] = []
    for event in load_reentry_events(events_root):
        sequence = str(event["sequence"])
        if sequence not in cache:
            cache[sequence] = load_sequence_gt(dataset_root, sequence)
        present, boxes, width, height = cache[sequence]
        rows.append(
            diagnostic_row(event, present, _box_lookup(boxes), width, height)
        )
    return rows


def write_diagnostic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "event_id",
        "sequence",
        "span_frames",
        "absent_frames",
        "last_seen_near_edge",
        "which_edge",
        "trigger_was_border_only",
        "error_type",
        "focal_gt_id",
        "frame_start",
        "frame_end",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    absent_zero = sum(1 for row in rows if int(row["absent_frames"]) == 0)
    large_absent = sum(
        1 for row in rows if int(row["absent_frames"]) >= REENTRY_MIN_ABSENT_FRAMES
    )
    border_only = sum(1 for row in rows if row["trigger_was_border_only"] == "true")
    if n == 0:
        branch = "none"
    elif absent_zero >= (n + 1) // 2 or border_only >= (n + 1) // 2:
        branch = "R3"
    elif large_absent >= (n + 1) // 2:
        branch = "R4"
    else:
        branch = "R3"
    return {
        "n_reentry": n,
        "absent_zero": absent_zero,
        "absent_ge_min": large_absent,
        "trigger_was_border_only": border_only,
        "branch": branch,
        "reentry_min_absent_frames": REENTRY_MIN_ABSENT_FRAMES,
        "event_window": int(DEFAULT_THRESHOLDS["EVENT_WINDOW"]),
    }


def relabel_reentry_events(
    events: list[dict[str, Any]],
    present: GTPresent,
    box_at: BoxAt,
    width: int,
    height: int,
    *,
    min_absent: int = REENTRY_MIN_ABSENT_FRAMES,
    margin: float = EDGE_MARGIN_PX,
) -> list[dict[str, Any]]:
    """Reclassify current reentry events. Other events are unchanged."""
    updated: list[dict[str, Any]] = []
    for event in events:
        copy = deepcopy(event)
        if str(copy.get("cause")) != "reentry":
            updated.append(copy)
            continue
        evidence = dict(copy.get("evidence") or {})
        stats = reentry_span_stats(copy, present)
        evidence["absent_frames"] = stats["n_absent"]
        evidence["reentry_min_absent_frames"] = int(min_absent)
        decision = check_reentry(
            copy,
            present,
            box_at,
            width,
            height,
            min_absent=min_absent,
            margin=margin,
        )
        if decision is None:
            copy["cause"] = "unattributed_idsw"
            evidence["reentry_rejected"] = True
            evidence["residual_reason"] = "reentry_requires_gt_absence"
            evidence.pop("exit_edge", None)
        else:
            cause, extra = decision
            copy["cause"] = cause
            evidence.update(extra)
            evidence["reentry_rejected"] = False
            if cause != "reentry":
                evidence.pop("exit_edge", None)
        copy["evidence"] = evidence
        updated.append(copy)
    return updated


def assert_reentry_gates(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    min_absent: int = REENTRY_MIN_ABSENT_FRAMES,
    event_window: int = int(DEFAULT_THRESHOLDS["EVENT_WINDOW"]),
) -> dict[str, Any]:
    if len(before) != len(after):
        raise AssertionError(
            f"total events changed: {len(before)} -> {len(after)}"
        )
    before_ids = [event["event_id"] for event in before]
    after_ids = [event["event_id"] for event in after]
    if before_ids != after_ids:
        raise AssertionError("event identity order changed during reentry relabel")

    def _type_count(events: list[dict[str, Any]], error_type: str) -> int:
        return sum(1 for event in events if event.get("error_type") == error_type)

    if _type_count(before, "FN") != _type_count(after, "FN"):
        raise AssertionError("FN event count changed")
    if _type_count(before, "IDSW") != _type_count(after, "IDSW"):
        raise AssertionError("IDSW event count changed")

    if int(min_absent) == int(event_window):
        raise AssertionError(
            "REENTRY_MIN_ABSENT_FRAMES must not equal EVENT_WINDOW"
        )

    remaining = [event for event in after if event.get("cause") == "reentry"]
    for event in remaining:
        evidence = event.get("evidence") or {}
        absent = int(evidence.get("absent_frames", -1))
        if absent < int(min_absent):
            raise AssertionError(
                f"{event['event_id']} is reentry with absent_frames={absent}"
            )
        if not evidence.get("exit_edge"):
            raise AssertionError(
                f"{event['event_id']} is reentry without exit_edge"
            )
    return {
        "total_events": len(after),
        "reentry_before": sum(1 for event in before if event.get("cause") == "reentry"),
        "reentry_after": len(remaining),
        "gt_annotation_gap": sum(
            1 for event in after if event.get("cause") == "gt_annotation_gap"
        ),
        "fn_closure_holds": True,
        "idsw_closure_holds": True,
    }


def apply_reentry_postpass(
    events_root: Path,
    dataset_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_root = Path(output_root)
    events_out = output_root / "events"
    summary_dir = output_root / "summary"
    events_out.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    diagnostic_rows = diagnose_reentry_corpus(events_root, dataset_root)
    write_diagnostic_csv(summary_dir / "reentry_diagnostic.csv", diagnostic_rows)
    summary = diagnostic_summary(diagnostic_rows)

    cache: dict[str, tuple[dict[int, set[int]], dict[tuple[int, int], BBox], int, int]] = {}
    transitions: Counter[tuple[str, str]] = Counter()
    all_before: list[dict[str, Any]] = []
    all_after: list[dict[str, Any]] = []
    by_sequence_after: dict[str, list[dict[str, Any]]] = {}

    for sequence in EVALUATION_SEQUENCES:
        matches = sorted(Path(events_root).glob(f"{sequence}__*.jsonl"))
        if not matches:
            raise FileNotFoundError(f"No sidecar events for {sequence} in {events_root}")
        source = matches[0]
        before = load_events_jsonl(source)
        if sequence not in cache:
            cache[sequence] = load_sequence_gt(dataset_root, sequence)
        present, boxes, width, height = cache[sequence]
        if summary["branch"] == "R3":
            after = relabel_reentry_events(
                before, present, _box_lookup(boxes), width, height
            )
        else:
            after = deepcopy(before)
        gates = assert_reentry_gates(before, after)
        write_events_jsonl(events_out / source.name, after)
        all_before.extend(before)
        all_after.extend(after)
        by_sequence_after[sequence] = after
        for old, new in zip(before, after):
            if old.get("cause") != new.get("cause"):
                transitions[(str(old["cause"]), str(new["cause"]))] += 1

    overall_gates = assert_reentry_gates(all_before, all_after)
    cause_after = Counter(str(event["cause"]) for event in all_after)
    if cause_after.get("gt_annotation_gap", 0) == 0:
        gap_note = "gt_annotation_gap count is 0; category dropped"
    else:
        gap_note = (
            "gt_annotation_gap is a GT-quality observation; annotators see it as "
            "changed_for_no_visible_reason"
        )

    report = {
        "diagnostic": summary,
        "gates": overall_gates,
        "cause_after": dict(cause_after),
        "transitions": {
            f"{old}->{new}": count for (old, new), count in sorted(transitions.items())
        },
        "gt_annotation_gap_note": gap_note,
        "events_out": str(events_out),
        "diagnostic_csv": str(summary_dir / "reentry_diagnostic.csv"),
    }
    (summary_dir / "reentry_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def reentry_constants_ok(config: PipelineConfig | None = None) -> bool:
    thresholds = (config or PipelineConfig()).thresholds
    return int(thresholds["REENTRY_MIN_ABSENT_FRAMES"]) != int(
        thresholds["EVENT_WINDOW"]
    )
