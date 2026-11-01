"""Entry-annotation sliver analysis (E1–E4). Read-only; last diagnostic."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .annotator_agreement import (
    DEFAULT_CLIP_MAP,
    DEFAULT_CLIPS_META,
    DEFAULT_DATASET,
    DEFAULT_EXPORT,
    DEFAULT_OUT,
    DEFAULT_TRACKER,
    MATCH_IOU,
    load_clip_map,
    load_geometry,
    load_null_before,
    rater_ids_from_rows,
    rater_name,
)
from .config import DEFAULT_CLASSES, DEFAULT_THRESHOLDS
from .duration_crosstab import (
    DEFAULT_VALIDATED,
    DEFAULT_WITHHELD,
    classify_duration_profile,
    load_annotation_rows,
    load_durations,
    no_error_crosstab,
    rater_no_error_flags,
)
from .geometry import border_edges, iou
from .models import GTBox, SequenceData
from .probe_clips import SOURCE_FPS
from .vlm_probe.summarize import write_csv, write_json


TARGET_CLASSES = frozenset(DEFAULT_CLASSES["TARGET_CLASSES"])
EDGE_MARGIN_PX = float(DEFAULT_THRESHOLDS["EDGE_MARGIN_PX"])
MIN_FN_RUN_FRAMES = int(DEFAULT_THRESHOLDS["MIN_FN_RUN_FRAMES"])
OFFSET_ENTRY_MAX = 25
DETECTABLE_LOOKAHEAD = 150
DEFAULT_DETECTABLE_H_PX = 50.0


def box_height(box: tuple[float, float, float, float]) -> float:
    return max(0.0, float(box[3]) - float(box[1]))


def box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def is_target(box: GTBox) -> bool:
    return int(box.flag) != 0 and int(box.cls) in TARGET_CLASSES


def gt_boxes_for_id(sequence: SequenceData, gt_id: int) -> list[GTBox]:
    return [
        box
        for frame in sequence.selected_frames
        for box in sequence.gt_by_frame.get(frame, [])
        if int(box.gt_id) == int(gt_id) and is_target(box)
    ]


def box_on_frame(boxes: Sequence[GTBox], frame: int) -> GTBox | None:
    for box in boxes:
        if int(box.frame) == int(frame):
            return box
    return None


def frame_matched(sequence: SequenceData, gt_box: GTBox) -> bool:
    tracker = sequence.tracker_by_frame.get(int(gt_box.frame), [])
    if not tracker:
        return False
    return max(iou(gt_box.bbox, box.bbox) for box in tracker) >= MATCH_IOU


def first_matched_frame(sequence: SequenceData, boxes: Sequence[GTBox]) -> int | None:
    for box in sorted(boxes, key=lambda item: int(item.frame)):
        if frame_matched(sequence, box):
            return int(box.frame)
    return None


def matched_heights(sequence: SequenceData) -> list[float]:
    heights: list[float] = []
    for frame in sequence.selected_frames:
        gt_boxes = [box for box in sequence.gt_by_frame.get(frame, []) if is_target(box)]
        if not gt_boxes:
            continue
        tracker = sequence.tracker_by_frame.get(frame, [])
        if not tracker:
            continue
        for gt_box in gt_boxes:
            best = max(iou(gt_box.bbox, box.bbox) for box in tracker)
            if best >= MATCH_IOU:
                heights.append(box_height(gt_box.bbox))
    return heights


def calibrate_detectable_height(
    sequences: Mapping[str, SequenceData],
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for name, sequence in sorted(sequences.items()):
        heights = matched_heights(sequence)
        array = np.asarray(heights, dtype=float) if heights else np.asarray([], dtype=float)
        report[name] = {
            "sequence": name,
            "n_matched_gt_frames": len(heights),
            "width": sequence.width,
            "height": sequence.height,
            "p5": float(np.percentile(array, 5)) if len(array) else None,
            "median": float(np.percentile(array, 50)) if len(array) else None,
            "detectable_h_px": (
                float(np.percentile(array, 5)) if len(array) else DEFAULT_DETECTABLE_H_PX
            ),
        }
    return report


def first_detectable_frame(
    boxes: Sequence[GTBox], first: int, threshold: float
) -> int | None:
    by_frame = {int(box.frame): box for box in boxes}
    last = first + DETECTABLE_LOOKAHEAD
    for frame in range(first, last + 1):
        box = by_frame.get(frame)
        if box is not None and box_height(box.bbox) >= threshold:
            return frame
    return None


def describe(values: Sequence[float | int]) -> dict[str, Any]:
    numeric = [float(item) for item in values if item is not None and item != ""]
    if not numeric:
        return {"n": 0, "median": None, "p25": None, "p75": None, "min": None, "max": None, "mean": None}
    array = np.asarray(numeric, dtype=float)
    return {
        "n": len(array),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def assign_bucket(
    *,
    offset_from_first: int | None,
    at_border: bool,
    ever_detected: bool,
) -> str:
    if not ever_detected:
        return "never_acquired"
    if at_border and offset_from_first is not None and offset_from_first <= OFFSET_ENTRY_MAX:
        return "entry"
    if not at_border:
        return "mid_frame"
    return "late_border"


def typology_row(
    event: Mapping[str, Any],
    sequence: SequenceData,
    *,
    detectable_h_px: float,
) -> dict[str, Any]:
    focal = int(event["focal_gt_id"])
    boxes = gt_boxes_for_id(sequence, focal)
    first = min((int(box.frame) for box in boxes), default=None)
    last = max((int(box.frame) for box in boxes), default=None)
    start = int(event["frame_start"])
    end = int(event["frame_end"])
    first_box = box_on_frame(boxes, first) if first is not None else None
    fail_box = box_on_frame(boxes, start)
    edges = (
        border_edges(first_box.bbox, sequence.width, sequence.height, EDGE_MARGIN_PX)
        if first_box is not None
        else ()
    )
    first_det = first_matched_frame(sequence, boxes) if boxes else None
    ever_detected = first_det is not None
    offset = None if first is None else start - first
    detectable = (
        first_detectable_frame(boxes, first, detectable_h_px) if first is not None else None
    )
    sliver_geom = None if first is None or detectable is None else detectable - first
    sliver_real = None if first is None or first_det is None else first_det - first
    bucket = assign_bucket(
        offset_from_first=offset,
        at_border=bool(edges),
        ever_detected=ever_detected,
    )
    n_gt = len(boxes)
    n_matched = sum(1 for box in boxes if frame_matched(sequence, box))
    return {
        "clip_id": event.get("clip_id", ""),
        "event_id": event["event_id"],
        "sequence": event.get("sequence", sequence.name),
        "focal_gt_id": focal,
        "frame_start": start,
        "frame_end": end,
        "fn_run_len": end - start + 1,
        "gt_first_frame": first if first is not None else "",
        "gt_last_frame": last if last is not None else "",
        "offset_from_first": offset if offset is not None else "",
        "gt_h_at_first": box_height(first_box.bbox) if first_box is not None else "",
        "gt_area_at_first": box_area(first_box.bbox) if first_box is not None else "",
        "gt_h_at_fail_start": box_height(fail_box.bbox) if fail_box is not None else "",
        "at_border": bool(edges),
        "which_edge": ",".join(edges),
        "track_len": (last - first + 1) if first is not None and last is not None else "",
        "ever_detected": ever_detected,
        "identity_recall": (n_matched / n_gt) if n_gt else 0.0,
        "first_det_frame": first_det if first_det is not None else "",
        "detectable_frame": detectable if detectable is not None else "",
        "detectable_h_px": detectable_h_px,
        "sliver_frames_geom": sliver_geom if sliver_geom is not None else "",
        "sliver_frames_real": sliver_real if sliver_real is not None else "",
        "bucket": bucket,
        "offset_entry_max": OFFSET_ENTRY_MAX,
    }


def load_probe_events(
    clip_map: Sequence[Mapping[str, str]], withheld_dir: Path
) -> list[dict[str, Any]]:
    events = []
    root = Path(withheld_dir)
    for row in clip_map:
        event_id = row.get("event_id") or ""
        path = root / f"{event_id}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        events.append(
            {
                "clip_id": row["clip_id"],
                "event_id": event_id,
                "sequence": payload.get("sequence") or row.get("sequence"),
                "cause": row.get("cause") or payload.get("cause"),
                "error_type": row.get("error_type") or payload.get("error_type"),
                "focal_gt_id": payload["focal_gt_id"],
                "frame_start": payload["frame_start"],
                "frame_end": payload["frame_end"],
            }
        )
    return events


def collapse_hypotheses(classifications: Mapping[str, Mapping[str, Any]]) -> str:
    votes = [item.get("hypothesis") for item in classifications.values()]
    unique_votes = {vote for vote in votes if vote}
    if not unique_votes:
        return "unknown"
    if unique_votes == {"H-presentation"}:
        return "H-presentation"
    if unique_votes == {"H-invisible"}:
        return "H-invisible"
    if unique_votes == {"both"}:
        return "both"
    if "H-presentation" in unique_votes:
        return "mixed"
    if len(unique_votes) == 1:
        return next(iter(unique_votes))
    return "mixed"


def mechanism_a_profile(hypothesis: str) -> str:
    """Plan E3: high/flat supports two mechanisms; declining collapses A into B."""
    if hypothesis in {"H-invisible", "both"}:
        return "flat_high"
    if hypothesis == "H-presentation":
        return "declining"
    return str(hypothesis)


def e5_sentences(
    *,
    n_null_before: int,
    n_entry: int,
    n_mid_frame: int,
    sliver_real: Mapping[str, Any],
    h_first: Mapping[str, Any],
    share_above_min_fn: float | None,
    e3_hypothesis: str,
    e3_entry_hypothesis: str,
) -> dict[str, str]:
    median_n = sliver_real.get("median")
    n_label = f"{median_n:.0f}" if median_n is not None else "N"
    seconds = f"{median_n / SOURCE_FPS:.2f}" if median_n is not None else "N/25"
    h_label = f"{h_first.get('median'):.0f}" if h_first.get("median") is not None else "H"
    entry = (
        "MOT20 annotates pedestrians from the moment of entry, when as little as part of "
        "the head is visible. The detector cannot fire until the target is substantially "
        f"inside the frame — a median of {n_label} frames (≈{seconds} s) later, with median "
        f"GT box height at first annotation of {h_label} px. Every intervening frame is "
        f"scored as a false negative. On the {n_entry} of {n_null_before} clips that begin "
        "at the frame border, two annotators agreed at 79% (on the full 90-clip null-before "
        "set) that no failure was visible. Their judgement is correct: no observer watching "
        "tracker output would identify a missing box on a sliver of head as a tracking "
        f"failure. The remaining {n_mid_frame} clips are interior births (crowd occlusion "
        "exit), not camera-border entry."
    )
    brief = (
        'On the remaining clips, "no error visible" declined monotonically with failure '
        "duration, from 68%/58% at three frames to 4%/7% above one hundred. At the clip "
        "playback rate the shortest occupy under 0.5 s."
    )
    vlm = (
        'Human "no error visible" falls from ~65% to ~14% across duration bins; the model\'s '
        "`stays_on_person` rate remains 92-100% in every bin except the longest (74%). The "
        "model fails on failures lasting over ten seconds of playback that annotators "
        "identify reliably."
    )
    kappa = (
        "Rule-vs-human agreement of κ = 0.13-0.18 measures the distance between MOT20's "
        "annotation policy and what a human observer identifies as a tracking failure, "
        "not an error in either."
    )
    return {
        "entry_annotation": entry,
        "brief_failures": brief,
        "vlm": vlm,
        "kappa_reframe": kappa,
        "e3_hypothesis": e3_hypothesis,
        "e3_entry_hypothesis": e3_entry_hypothesis,
        "filter_note": (
            f"Share of entry FN runs at or above MIN_FN_RUN_FRAMES={MIN_FN_RUN_FRAMES}: "
            f"{share_above_min_fn}"
        ),
    }


def run_entry_analysis(
    *,
    out_dir: Path,
    validated_path: Path = DEFAULT_VALIDATED,
    export_path: Path = DEFAULT_EXPORT,
    clip_map_path: Path = DEFAULT_CLIP_MAP,
    clips_meta_dir: Path = DEFAULT_CLIPS_META,
    withheld_dir: Path = DEFAULT_WITHHELD,
    tracker_output: Path = DEFAULT_TRACKER,
    dataset_root: Path = DEFAULT_DATASET,
    sequences: Mapping[str, SequenceData] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    clip_map = load_clip_map(clip_map_path)
    null_before = load_null_before(clip_map, clips_meta_dir)
    events = [
        event
        for event in load_probe_events(clip_map, withheld_dir)
        if null_before.get(event["clip_id"])
    ]
    loaded = dict(sequences) if sequences is not None else load_geometry(
        tracker_output, dataset_root
    )
    calibration = calibrate_detectable_height(loaded)
    write_csv(
        out_dir / "detectable_height_calibration.csv",
        list(calibration.values()),
        (
            "sequence",
            "n_matched_gt_frames",
            "width",
            "height",
            "p5",
            "median",
            "detectable_h_px",
        ),
    )

    typology = []
    for event in events:
        sequence = loaded[str(event["sequence"])]
        threshold = float(
            calibration.get(str(event["sequence"]), {}).get(
                "detectable_h_px", DEFAULT_DETECTABLE_H_PX
            )
        )
        typology.append(typology_row(event, sequence, detectable_h_px=threshold))
    write_csv(
        summary_dir / "null_before_typology.csv",
        typology,
        (
            "clip_id",
            "event_id",
            "sequence",
            "focal_gt_id",
            "frame_start",
            "frame_end",
            "fn_run_len",
            "gt_first_frame",
            "gt_last_frame",
            "offset_from_first",
            "gt_h_at_first",
            "gt_area_at_first",
            "gt_h_at_fail_start",
            "at_border",
            "which_edge",
            "track_len",
            "ever_detected",
            "identity_recall",
            "first_det_frame",
            "detectable_frame",
            "detectable_h_px",
            "sliver_frames_geom",
            "sliver_frames_real",
            "bucket",
            "offset_entry_max",
        ),
    )
    bucket_counts = Counter(row["bucket"] for row in typology)
    entry_rows = [row for row in typology if row["bucket"] == "entry"]
    mid_rows = [row for row in typology if row["bucket"] == "mid_frame"]
    sliver_real = describe(
        [row["sliver_frames_real"] for row in entry_rows if row["sliver_frames_real"] != ""]
    )
    sliver_geom = describe(
        [row["sliver_frames_geom"] for row in entry_rows if row["sliver_frames_geom"] != ""]
    )
    h_first = describe(
        [row["gt_h_at_first"] for row in entry_rows if row["gt_h_at_first"] != ""]
    )
    run_lengths = [int(row["fn_run_len"]) for row in entry_rows]
    run_stats = describe(run_lengths)
    share_above = (
        float(np.mean(np.asarray(run_lengths) >= MIN_FN_RUN_FRAMES)) if run_lengths else None
    )
    mid_h_first = describe(
        [row["gt_h_at_first"] for row in mid_rows if row["gt_h_at_first"] != ""]
    )
    clip_bucket = {row["clip_id"]: row["bucket"] for row in typology}

    durations = load_durations(
        clip_map, clips_meta_dir=clips_meta_dir, withheld_dir=withheld_dir
    )
    annotation_rows = load_annotation_rows(
        validated_path=validated_path if Path(validated_path).is_file() else None,
        export_path=export_path,
    )
    rater_ids = rater_ids_from_rows(annotation_rows)
    null_clips = [event["clip_id"] for event in events]
    e3_rows = []
    e3_by_bucket: list[dict[str, Any]] = []
    e3_class: dict[str, Any] = {}
    e3_entry_class: dict[str, Any] = {}
    for rater_id in rater_ids:
        flags = rater_no_error_flags(annotation_rows, rater_id)
        members = [clip_id for clip_id in null_clips if clip_id in flags]
        table = no_error_crosstab(members, durations, flags)
        for row in table:
            row["rater_id"] = rater_id
            row["rater_name"] = rater_name(rater_id)
            row["bucket"] = "all_null_before"
        e3_rows.extend(table)
        e3_class[str(rater_id)] = classify_duration_profile(table)
        entry_members = [clip_id for clip_id in members if clip_bucket.get(clip_id) == "entry"]
        entry_table = no_error_crosstab(entry_members, durations, flags)
        e3_entry_class[str(rater_id)] = classify_duration_profile(entry_table)
        for bucket in ("entry", "mid_frame", "never_acquired", "late_border"):
            subset = [clip_id for clip_id in members if clip_bucket.get(clip_id) == bucket]
            split = no_error_crosstab(subset, durations, flags)
            for row in split:
                row["rater_id"] = rater_id
                row["rater_name"] = rater_name(rater_id)
                row["bucket"] = bucket
            e3_by_bucket.extend(split)
    crosstab_fields = (
        "rater_id",
        "rater_name",
        "bucket",
        "duration_bin",
        "n",
        "n_no_error_visible",
        "n_other",
        "rate_no_error_visible",
    )
    write_csv(
        out_dir / "entry_null_before_duration_crosstab.csv",
        e3_rows,
        crosstab_fields,
    )
    write_csv(
        out_dir / "entry_duration_crosstab_by_bucket.csv",
        e3_by_bucket,
        crosstab_fields,
    )
    e3_hypothesis = collapse_hypotheses(e3_class)
    e3_entry_hypothesis = collapse_hypotheses(e3_entry_class)
    sentences = e5_sentences(
        n_null_before=len(events),
        n_entry=len(entry_rows),
        n_mid_frame=len(mid_rows),
        sliver_real=sliver_real,
        h_first=h_first,
        share_above_min_fn=share_above,
        e3_hypothesis=e3_hypothesis,
        e3_entry_hypothesis=e3_entry_hypothesis,
    )
    report = {
        "n_null_before": len(events),
        "bucket_counts": dict(bucket_counts),
        "entry_dominates": bucket_counts.get("entry", 0) >= 0.5 * max(len(events), 1),
        "offset_entry_max": OFFSET_ENTRY_MAX,
        "edge_margin_px": EDGE_MARGIN_PX,
        "which_edge_entry": dict(Counter(row["which_edge"] for row in entry_rows)),
        "calibration": calibration,
        "sliver_frames_real": sliver_real,
        "sliver_frames_geom": sliver_geom,
        "gt_h_at_first": h_first,
        "mid_frame": {
            "n": len(mid_rows),
            "gt_h_at_first": mid_h_first,
            "n_at_sequence_start": sum(
                int(row["gt_first_frame"] or 0) == 1 for row in mid_rows
            ),
            "note": (
                "Interior GT births at full pedestrian height; not camera-border "
                "entry. Consistent with crowd occlusion exit."
            ),
        },
        "entry_fn_run_lengths": run_stats,
        "share_entry_fn_ge_min_fn": share_above,
        "min_fn_run_frames": MIN_FN_RUN_FRAMES,
        "e3_classification": e3_class,
        "e3_hypothesis": e3_hypothesis,
        "e3_profile": mechanism_a_profile(e3_hypothesis),
        "e3_entry_classification": e3_entry_class,
        "e3_entry_hypothesis": e3_entry_hypothesis,
        "e3_entry_profile": mechanism_a_profile(e3_entry_hypothesis),
        "e5": sentences,
    }
    write_json(out_dir / "entry_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entry-annotation sliver analysis (E1–E4)"
    )
    parser.add_argument("--validated", type=Path, default=DEFAULT_VALIDATED)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--clip-map", type=Path, default=DEFAULT_CLIP_MAP)
    parser.add_argument("--clips-meta", type=Path, default=DEFAULT_CLIPS_META)
    parser.add_argument("--withheld", type=Path, default=DEFAULT_WITHHELD)
    parser.add_argument("--tracker-output", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    report = run_entry_analysis(
        out_dir=args.out_dir,
        validated_path=args.validated,
        export_path=args.export,
        clip_map_path=args.clip_map,
        clips_meta_dir=args.clips_meta,
        withheld_dir=args.withheld,
        tracker_output=args.tracker_output,
        dataset_root=args.dataset_root,
    )
    print(
        json.dumps(
            {
                "n_null_before": report["n_null_before"],
                "bucket_counts": report["bucket_counts"],
                "entry_dominates": report["entry_dominates"],
                "sliver_frames_real_median": report["sliver_frames_real"].get("median"),
                "gt_h_at_first_median": report["gt_h_at_first"].get("median"),
                "share_entry_fn_ge_min_fn": report["share_entry_fn_ge_min_fn"],
                "e3_hypothesis": report["e3_hypothesis"],
                "e3_profile": report["e3_profile"],
                "e3_entry_hypothesis": report["e3_entry_hypothesis"],
                "e3_entry_profile": report["e3_entry_profile"],
            },
            indent=2,
        )
    )
    return report
