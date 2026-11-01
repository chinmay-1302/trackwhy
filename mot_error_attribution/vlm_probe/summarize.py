from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .score import (
    degeneracy_report,
    hallucinated_occluder_rows,
    map_rule_cause,
    parse_failure_rate,
    signal_detection_summary,
    stage1_table,
    stage2_confusion,
    three_way_bucket,
    uniform_vs_centred_stage1,
    uniform_vs_centred_stage2,
    v2_perception_summary,
    v3_binary_summary,
    w2_resolution_report,
    w4_followup,
)

STAGE1_FIELDS = [
    "clip_id",
    "event_id",
    "cause",
    "error_type",
    "stage1_bucket",
    "sampling",
    "n_frames_used",
    "n_frames_in_window",
    "n_frames_total",
    "elapsed_sec",
    "parse_status",
    "box_behaviour",
    "stage1_correct",
    "disappears_at",
    "person_still_visible_after",
    "person_description",
    "confidence",
    "video_path",
    "raw_path",
]

STAGE2_FIELDS = [
    "clip_id",
    "event_id",
    "true_cause",
    "error_type",
    "sampling",
    "n_frames_used",
    "n_frames_in_window",
    "n_frames_total",
    "elapsed_sec",
    "parse_status",
    "error_kind",
    "cause",
    "evidence",
    "occluder_visible_in_frame",
    "confidence",
    "hallucinated_occluder",
    "video_path",
    "raw_path",
]

V2_FIELDS = [
    "clip_id",
    "event_id",
    "sequence",
    "true_track_number",
    "track_number",
    "track_number_match",
    "position",
    "clothing",
    "found_highlighted_box",
    "parse_status",
    "elapsed_sec",
    "image_path",
    "raw_path",
]

V3_FIELDS = [
    "clip_id",
    "event_id",
    "sequence",
    "frame",
    "role",
    "is_present",
    "true_present",
    "predicted",
    "highlighted_box_present",
    "correct",
    "box_area_px",
    "max_pixels_used",
    "role_present",
    "relabelled",
    "parse_status",
    "elapsed_sec",
    "image_path",
    "raw_path",
]


def model_dir_name(model_id: str) -> str:
    return model_id.replace("/", "--").replace(" ", "_")


def raw_path(
    output_root: Path,
    model_id: str,
    stage: str,
    clip_id: str,
    sampling: str | None = None,
) -> Path:
    base = output_root / "raw" / model_dir_name(model_id) / stage
    if sampling:
        base = base / sampling
    return base / f"{clip_id}.txt"


def parsed_csv_path(
    output_root: Path,
    model_id: str,
    stage: str,
    sampling: str | None = None,
) -> Path:
    slug = model_dir_name(model_id)
    if sampling:
        return output_root / "parsed" / f"{slug}_{stage}_{sampling}.csv"
    return output_root / "parsed" / f"{slug}_{stage}.csv"


def write_raw(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _bool_cell(value: str) -> bool | None:
    text = (value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _prepare_stage2_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for row in rows:
        item = dict(row)
        item["true_cause"] = row.get("true_cause") or row.get("cause_true") or row.get("rule_cause")
        if "cause" in row and row.get("pred_cause"):
            item["cause"] = row.get("pred_cause")
        visible = row.get("occluder_visible_in_frame")
        if isinstance(visible, str):
            item["occluder_visible_in_frame"] = _bool_cell(visible)
        prepared.append(item)
    return prepared


def write_summaries(
    output_root: Path,
    *,
    stage1_rows: Sequence[Mapping[str, Any]] | None = None,
    stage2_rows: Sequence[Mapping[str, Any]] | None = None,
    human_rows: Sequence[Mapping[str, Any]] | None = None,
    human_could_not_rate: float | None = None,
) -> dict[str, Any]:
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    if stage1_rows is not None:
        table = stage1_table(stage1_rows, human_could_not_rate=human_could_not_rate)
        write_csv(
            summary_dir / "stage1_accuracy.csv",
            table,
            [
                "bucket",
                "n",
                "n_parsed",
                "n_correct",
                "accuracy",
                "n_no_highlighted_box_found",
                "no_highlighted_box_found_rate",
                "human_could_not_identify_target_rate",
            ],
        )
        report["stage1_parse"] = parse_failure_rate(stage1_rows)
        report["stage1_n"] = len(list(stage1_rows))
    if stage2_rows is not None:
        prepared = _prepare_stage2_rows(stage2_rows)
        confusion = stage2_confusion(prepared)
        write_csv(
            summary_dir / "stage2_confusion.csv",
            confusion,
            ["true_cause", "pred_cause", "n"],
        )
        hallucinations = hallucinated_occluder_rows(prepared)
        write_csv(
            summary_dir / "hallucinated_occluders.csv",
            hallucinations,
            [
                "clip_id",
                "event_id",
                "true_cause",
                "pred_cause",
                "occluder_visible_in_frame",
                "evidence",
                "confidence",
            ],
        )
        report["stage2_parse"] = parse_failure_rate(prepared)
        report["n_hallucinated_occluders"] = len(hallucinations)
        if human_rows:
            by_clip = {
                str(row.get("clip_id") or row.get("clip_id")): row for row in human_rows
            }
            three_way = []
            for row in prepared:
                clip_id = str(row.get("clip_id") or "")
                human = by_clip.get(clip_id, {})
                human_cause = human.get("cause") or human.get("adjudicated_cause")
                bucket = three_way_bucket(
                    rule_cause=str(row.get("true_cause") or ""),
                    human_cause=human_cause,
                    vlm_cause=row.get("cause") if row.get("parse_status") == "ok" else None,
                )
                three_way.append(
                    {
                        "clip_id": clip_id,
                        "rule_cause": row.get("true_cause"),
                        "human_cause": human_cause or "",
                        "vlm_cause": row.get("cause") if row.get("parse_status") == "ok" else "",
                        "bucket": bucket,
                    }
                )
            write_csv(
                summary_dir / "three_way_agreement.csv",
                three_way,
                ["clip_id", "rule_cause", "human_cause", "vlm_cause", "bucket"],
            )
            report["three_way"] = parse_failure_rate(
                [{"parse_status": "ok"} for _ in three_way]
            )
            report["three_way_n"] = len(three_way)
    write_json(summary_dir / "parse_report.json", report)
    return report


def write_v2_summaries(
    output_root: Path,
    *,
    v2_rows: Sequence[Mapping[str, Any]] | None = None,
    v3_rows: Sequence[Mapping[str, Any]] | None = None,
    uniform_stage1: Sequence[Mapping[str, Any]] | None = None,
    centred_stage1: Sequence[Mapping[str, Any]] | None = None,
    uniform_stage2: Sequence[Mapping[str, Any]] | None = None,
    centred_stage2: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    if v2_rows is not None:
        payload = v2_perception_summary(v2_rows)
        write_json(summary_dir / "v2_perception.json", payload)
        write_csv(
            summary_dir / "v2_perception.csv",
            v2_rows,
            [
                "clip_id",
                "sequence",
                "true_track_number",
                "track_number",
                "track_number_match",
                "found_highlighted_box",
                "position",
                "clothing",
                "parse_status",
            ],
        )
        report["v2"] = payload
    if v3_rows is not None:
        payload = v3_binary_summary(v3_rows)
        write_json(summary_dir / "v3_binary.json", payload)
        write_csv(
            summary_dir / "v3_binary.csv",
            v3_rows,
            [
                "clip_id",
                "sequence",
                "frame",
                "role",
                "is_present",
                "true_present",
                "predicted",
                "highlighted_box_present",
                "correct",
                "box_area_px",
                "max_pixels_used",
                "parse_status",
            ],
        )
        report["v3"] = payload
    if uniform_stage1 is not None and centred_stage1 is not None:
        table = uniform_vs_centred_stage1(uniform_stage1, centred_stage1)
        write_csv(
            summary_dir / "stage1_uniform_vs_centred.csv",
            table,
            [
                "condition",
                "n",
                "n_parsed",
                "n_correct",
                "accuracy",
                "n_stays_on_person",
                "stays_on_person_rate",
                "n_no_highlighted_box_found",
                "no_highlighted_box_found_rate",
            ],
        )
        report["stage1_uniform_vs_centred"] = table
    if uniform_stage2 is not None and centred_stage2 is not None:
        table = uniform_vs_centred_stage2(uniform_stage2, centred_stage2)
        write_csv(
            summary_dir / "stage2_uniform_vs_centred.csv",
            table,
            ["condition", "n", "n_correct_mapped", "accuracy_mapped"],
        )
        report["stage2_uniform_vs_centred"] = table
    if uniform_stage2 is not None:
        payload = degeneracy_report(uniform_stage2)
        write_json(summary_dir / "v1_degeneracy.json", payload)
        report["v1_degeneracy"] = payload
    report_path = summary_dir / "v2_report.json"
    existing: dict[str, Any] = {}
    if report_path.is_file():
        try:
            existing = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        if not isinstance(existing, dict):
            existing = {}
    existing.update(report)
    write_json(report_path, existing)
    return existing


def _csv_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower() if value is not None else ""
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no", ""}:
        return False if text else None
    return None


def _by_clip(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("clip_id") or ""): row for row in rows}


def _stage2_mapped_ok(row: Mapping[str, Any]) -> bool | None:
    if str(row.get("parse_status") or "") != "ok":
        return None
    pred = str(row.get("cause") or "")
    truth = str(row.get("true_cause") or "")
    if not pred or not truth:
        return None
    return map_rule_cause(truth) == pred


def write_frozen_17_report(
    output_root: Path,
    *,
    clip_ids: Sequence[str],
    original_stage1: Sequence[Mapping[str, Any]],
    frozen_stage1: Sequence[Mapping[str, Any]],
    original_stage2: Sequence[Mapping[str, Any]],
    frozen_stage2: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    old1 = _by_clip(original_stage1)
    new1 = _by_clip(frozen_stage1)
    old2 = _by_clip(original_stage2)
    new2 = _by_clip(frozen_stage2)
    pairs: list[dict[str, Any]] = []
    for clip_id in clip_ids:
        before1 = old1.get(clip_id) or {}
        after1 = new1.get(clip_id) or {}
        before2 = old2.get(clip_id) or {}
        after2 = new2.get(clip_id) or {}
        pairs.append(
            {
                "clip_id": clip_id,
                "stage1_bucket": after1.get("stage1_bucket") or before1.get("stage1_bucket") or "",
                "stage1_old": before1.get("box_behaviour") or "",
                "stage1_new": after1.get("box_behaviour") or "",
                "stage1_correct_old": _csv_bool(before1.get("stage1_correct")),
                "stage1_correct_new": _csv_bool(after1.get("stage1_correct")),
                "stage1_changed": (before1.get("box_behaviour") or "")
                != (after1.get("box_behaviour") or ""),
                "stage2_old": before2.get("cause") or "",
                "stage2_new": after2.get("cause") or "",
                "stage2_mapped_old": _stage2_mapped_ok(before2),
                "stage2_mapped_new": _stage2_mapped_ok(after2),
                "stage2_changed": (before2.get("cause") or "") != (after2.get("cause") or ""),
            }
        )
    n = len(pairs)
    n_s1_old = sum(1 for row in pairs if row["stage1_correct_old"])
    n_s1_new = sum(1 for row in pairs if row["stage1_correct_new"])
    n_s2_old = sum(1 for row in pairs if row["stage2_mapped_old"])
    n_s2_new = sum(1 for row in pairs if row["stage2_mapped_new"])
    report = {
        "n": n,
        "clip_ids": list(clip_ids),
        "note": (
            "Stage 1/2 re-run on the frozen re-rendered clips. "
            "Does not replace v1_degeneracy.json."
        ),
        "stage1_n_correct_old": n_s1_old,
        "stage1_n_correct_new": n_s1_new,
        "stage1_n_changed": sum(1 for row in pairs if row["stage1_changed"]),
        "stage2_n_mapped_old": n_s2_old,
        "stage2_n_mapped_new": n_s2_new,
        "stage2_n_changed": sum(1 for row in pairs if row["stage2_changed"]),
        "headline_moved": n_s1_new != n_s1_old or n_s2_new != n_s2_old,
        "pairs": pairs,
    }
    summary_dir = Path(output_root) / "summary"
    write_json(summary_dir / "v1_frozen_17.json", report)
    write_csv(
        summary_dir / "v1_frozen_17_pairs.csv",
        pairs,
        [
            "clip_id",
            "stage1_bucket",
            "stage1_old",
            "stage1_new",
            "stage1_correct_old",
            "stage1_correct_new",
            "stage1_changed",
            "stage2_old",
            "stage2_new",
            "stage2_mapped_old",
            "stage2_mapped_new",
            "stage2_changed",
        ],
    )
    return report


def write_w_summaries(
    output_root: Path,
    *,
    w1_rows: Sequence[Mapping[str, Any]] | None = None,
    w2_rows: Sequence[Mapping[str, Any]] | None = None,
    v2_low_rows: Sequence[Mapping[str, Any]] | None = None,
    v2_high_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    if w1_rows is not None:
        w1 = v3_binary_summary(w1_rows)
        write_json(summary_dir / "v3_binary.json", w1)
        write_json(summary_dir / "w3_sdt.json", signal_detection_summary(w1_rows))
        report["w1"] = w1
        report["w3"] = signal_detection_summary(w1_rows)
    if w2_rows is not None:
        write_json(summary_dir / "v3_binary_high.json", v3_binary_summary(w2_rows))
        report["w2_high"] = v3_binary_summary(w2_rows)
    if w1_rows is not None and w2_rows is not None:
        w2 = w2_resolution_report(w1_rows, w2_rows)
        write_json(summary_dir / "w2_resolution.json", w2)
        report["w2"] = w2
    if v2_low_rows is not None or v2_high_rows is not None:
        w4 = w4_followup(v2_low_rows, v2_high_rows)
        write_json(summary_dir / "w4_v2.json", w4)
        report["w4"] = w4
    return report
