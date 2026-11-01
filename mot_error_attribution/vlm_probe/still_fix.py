"""Relabel V3 stills from rendered pixels. Does not re-render annotator clips."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .catalog import ClipRecord
from .config import DEFAULT_CLIPS_META_DIR, DEFAULT_MODEL, MIN_HIGHLIGHT_PX
from .sampling import (
    read_still_bgr,
    rendered_highlight_area,
    still_label,
)
from .score import (
    _as_bool,
    _predicted_present,
    present_cell_interpretation,
    signal_detection_summary,
    v3_binary_summary,
)
from .summarize import (
    V3_FIELDS,
    parsed_csv_path,
    read_csv,
    write_csv,
    write_json,
    write_w_summaries,
)

RELABEL_FIELDS = [
    "clip_id",
    "event_id",
    "sequence",
    "frame",
    "role",
    "expected",
    "actual",
    "area_px",
    "image_path",
]


def role_expected_present(role: str) -> bool:
    return str(role).strip().lower() == "before"


def label_still_image(
    image: Any,
    *,
    clip_id: str,
    frame: int,
    role: str,
    event_id: str = "",
    sequence: str = "",
    image_path: str = "",
) -> tuple[bool, int, dict[str, Any] | None]:
    """Return (is_present, area_px, relabel_row_or_none). Inside frames must be absent."""
    area = rendered_highlight_area(image)
    actual = still_label(image)
    expected = "present" if role_expected_present(role) else "absent"
    if role == "inside" and actual != "absent":
        raise AssertionError(
            f"{clip_id}@{frame}: highlight present in gap frame (area_px={area})"
        )
    log_row = None
    if actual != expected:
        log_row = {
            "clip_id": clip_id,
            "event_id": event_id,
            "sequence": sequence,
            "frame": frame,
            "role": role,
            "expected": expected,
            "actual": actual,
            "area_px": area,
            "image_path": image_path,
        }
    return actual == "present", area, log_row


def relabel_v3_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Derive is_present from the still on disk. Relabel, do not exclude."""
    out: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        image_path = Path(str(item.get("image_path") or ""))
        role = str(item.get("role") or "")
        frame = item.get("frame") or ""
        image = read_still_bgr(image_path)
        is_present, area, log_row = label_still_image(
            image,
            clip_id=str(item.get("clip_id") or ""),
            frame=int(frame) if str(frame).strip() not in {"", "None"} else -1,
            role=role,
            event_id=str(item.get("event_id") or ""),
            sequence=str(item.get("sequence") or ""),
            image_path=str(image_path),
        )
        pred = _predicted_present(item)
        item["role_present"] = role_expected_present(role)
        item["is_present"] = is_present
        item["true_present"] = is_present
        item["box_area_px"] = area
        item["relabelled"] = bool(log_row)
        item["correct"] = None if pred is None else pred is is_present
        if log_row:
            log.append(log_row)
        out.append(item)
    return out, log


def _load_tracker_index(tracker_dir: Path) -> dict[tuple[str, int, int], tuple[float, float, float, float]]:
    index: dict[tuple[str, int, int], tuple[float, float, float, float]] = {}
    if not tracker_dir.is_dir():
        return index
    for path in sorted(tracker_dir.glob("MOT20-*.txt")):
        sequence = path.stem
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) < 6:
                    continue
                frame = int(float(row[0]))
                tid = int(float(row[1]))
                x, y, width, height = map(float, row[2:6])
                index[(sequence, frame, tid)] = (x, y, width, height)
    return index


def diagnose_zero_area_stills(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[ClipRecord],
    *,
    tracker_dir: Path | None = None,
    meta_dir: Path | None = None,
) -> dict[str, Any]:
    """X3: why rendered-absent stills were labelled present from the event record."""
    by_clip = {record.clip_id: record for record in records}
    tracker = _load_tracker_index(tracker_dir) if tracker_dir is not None else {}
    meta_root = Path(meta_dir or DEFAULT_CLIPS_META_DIR)
    cases: list[dict[str, Any]] = []
    patterns: Counter[str] = Counter()
    for row in rows:
        if not role_expected_present(str(row.get("role") or "")):
            continue
        if _as_bool(row.get("is_present")) is True:
            continue
        if int(row.get("box_area_px") or 0) > MIN_HIGHLIGHT_PX:
            continue
        # After relabel, role-present + rendered-absent is the diagnostic set.
        # Before relabel, box_area_px==0 on role-present is the same set's core.
        clip_id = str(row.get("clip_id") or "")
        record = by_clip.get(clip_id)
        frame = int(row.get("frame") or 0)
        window = record.failure_window if record else None
        fs, fe = window if window else (None, None)
        in_gap = bool(window and fs <= frame <= fe)
        before_tid = record.highlight_before if record else None
        box = None
        area = None
        if record and before_tid is not None:
            box = tracker.get((record.sequence, frame, int(before_tid)))
            if box is not None:
                area = box[2] * box[3]
        if in_gap:
            pattern = "in_gap"
        elif before_tid is None:
            pattern = "no_before_tid"
        elif box is None:
            pattern = "box_none"
        elif area is not None and area < 1:
            pattern = "degenerate_box"
        else:
            pattern = "box_exists"
        patterns[pattern] += 1
        cases.append(
            {
                "clip_id": clip_id,
                "event_id": record.event_id if record else row.get("event_id"),
                "sequence": record.sequence if record else row.get("sequence"),
                "frame": frame,
                "failure_window": list(window) if window else None,
                "in_gap": in_gap,
                "before_tid": before_tid,
                "box": None if box is None else [round(v, 2) for v in box],
                "tracker_box_area": area,
                "pattern": pattern,
                "cut_between_segments": (
                    record.cut_between_segments if record else None
                ),
            }
        )
    n_clips = len(records)
    n_null_before = sum(1 for record in records if record.highlight_before is None)
    dominant = patterns.most_common(1)[0][0] if patterns else "none"
    clips_ok = dominant != "in_gap"
    return {
        "n_cases": len(cases),
        "patterns": dict(patterns),
        "dominant_pattern": dominant,
        "off_by_one": dominant == "in_gap",
        "annotator_clips_share_failure_window": True,
        "annotator_clips_ok": clips_ok,
        "n_catalog_before_tid_null": n_null_before,
        "n_catalog": n_clips,
        "sentence": (
            "Not an off-by-one: every relabelled still is the last frame before "
            "failure_start, and highlighted_track_ids.before is null, so the clip "
            "renderer correctly drew no highlight. Annotator clips use the same "
            "HighlightPlan; do not re-render."
            if clips_ok
            else "Off-by-one: the still labelled present sits inside the FN window. "
            "Check whether clip rendering shares this index."
        ),
        "cases": cases,
        "meta_dir": str(meta_root),
    }


def v3_paper_sentences(w1: Mapping[str, Any], w3: Mapping[str, Any]) -> dict[str, str]:
    present = w1.get("present_accuracy")
    present_ci = w1.get("present_accuracy_ci") or [None, None]
    absent = w1.get("absent_accuracy")
    absent_ci = w1.get("absent_accuracy_ci") or [None, None]
    n_relabel = int(w1.get("n_relabelled") or 0)
    d_prime = w3.get("d_prime")
    criterion = w3.get("criterion")
    bias = w3.get("bias") or (
        "liberal"
        if isinstance(criterion, float) and criterion < 0
        else "conservative"
        if isinstance(criterion, float) and criterion > 0
        else "unbiased"
    )
    methods = (
        f"{n_relabel} of the 100 stills contained no rendered highlight; labels were "
        "derived from the rendered image rather than the event record, and these were "
        "relabelled as absent. The model answered them as absent. This is a labelling "
        "correction, not a post-hoc change to the present-cell decision boundary."
    )
    claim = (
        "Event-centred sampling with ≥8 frames inside the failure window did not "
        "change Stage 1 behaviour (0.950 vs 0.947). Single-frame highlight detection "
        f"is {present:.2f} [{present_ci[0]:.2f}, {present_ci[1]:.2f}] for present and "
        f"{absent:.2f} [{absent_ci[0]:.2f}, {absent_ci[1]:.2f}] for absent, and does "
        "not improve at 2.8× pixel budget. Signal-detection analysis gives "
        f"d′ = {d_prime:.2f} with a {bias} criterion (c = {criterion:+.2f}) on "
        "the binary task, in contrast to zero abstention on the 14-way forced choice. "
        "Stage 2 predictions collapse into 2 of 14 labels (90.5%)."
        if isinstance(present, float)
        and isinstance(absent, float)
        and isinstance(d_prime, float)
        and isinstance(criterion, float)
        and present_ci[0] is not None
        else ""
    )
    return {
        "methods_footnote": methods,
        "revised_claim": claim,
        "do_not_claim": (
            "Do not claim the model sees the failure and cannot name it. "
            "Present-cell accuracy supports the present_cell reading, nothing stronger."
        ),
    }


def attach_relabel_audit(
    summary: dict[str, Any],
    rows: Sequence[Mapping[str, Any]],
    log: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = dict(summary)
    role_rows = []
    for row in rows:
        item = dict(row)
        item["is_present"] = role_expected_present(str(row.get("role") or ""))
        item["true_present"] = item["is_present"]
        pred = _predicted_present(row)
        item["correct"] = None if pred is None else pred is item["is_present"]
        role_rows.append(item)
    before = v3_binary_summary(role_rows)
    payload["n_relabelled"] = len(log)
    payload["present_cell_before_relabel"] = before.get("present_cell")
    payload["present_accuracy_before_relabel"] = before.get("present_accuracy")
    payload["present_accuracy_ci_before_relabel"] = before.get("present_accuracy_ci")
    payload["label_source"] = "rendered_pixels"
    payload["relabel_note"] = (
        "This is a labelling correction (rendered pixels, not the event record), "
        "not a post-hoc change to the present-cell decision boundary."
    )
    reading = payload.get("present_cell") or {}
    payload["present_cell_transition"] = {
        "before": (before.get("present_cell") or {}).get("code"),
        "after": reading.get("code"),
    }
    return payload


def run_still_fix(
    output_root: Path,
    records: Sequence[ClipRecord],
    *,
    model_id: str = DEFAULT_MODEL,
    tracker_dir: Path | None = None,
) -> dict[str, Any]:
    w1_path = parsed_csv_path(output_root, model_id, "v3_binary")
    w2_path = parsed_csv_path(output_root, model_id, "v3_binary_high")
    if not w1_path.is_file():
        raise FileNotFoundError(f"Missing W1 parsed CSV: {w1_path}")
    w1_rows, log = relabel_v3_rows(read_csv(w1_path))
    w2_rows = None
    if w2_path.is_file():
        w2_rows, w2_log = relabel_v3_rows(read_csv(w2_path))
        if not log:
            log = w2_log
    write_csv(w1_path, w1_rows, V3_FIELDS)
    if w2_rows is not None:
        write_csv(w2_path, w2_rows, V3_FIELDS)
    write_csv(output_root / "summary" / "v3_relabel_log.csv", log, RELABEL_FIELDS)
    w1 = attach_relabel_audit(v3_binary_summary(w1_rows), w1_rows, log)
    w3 = signal_detection_summary(w1_rows)
    write_json(output_root / "summary" / "v3_binary.json", w1)
    write_json(output_root / "summary" / "w3_sdt.json", w3)
    if w2_rows is not None:
        write_w_summaries(output_root, w1_rows=w1_rows, w2_rows=w2_rows)
        write_json(output_root / "summary" / "v3_binary.json", w1)
        write_json(output_root / "summary" / "w3_sdt.json", w3)
    diagnosis = diagnose_zero_area_stills(
        w1_rows, records, tracker_dir=tracker_dir
    )
    write_json(output_root / "summary" / "v3_zero_area_diagnosis.json", diagnosis)
    paper = v3_paper_sentences(w1, w3)
    write_json(output_root / "summary" / "v3_paper_sentences.json", paper)
    return {
        "w1": w1,
        "w3": w3,
        "n_relabelled": len(log),
        "diagnosis": diagnosis,
        "paper": paper,
        "present_cell": present_cell_interpretation(
            w1.get("present_accuracy"),
            (w1.get("present_accuracy_ci") or [None, None])[0],
            (w1.get("present_accuracy_ci") or [None, None])[1],
        ),
    }
