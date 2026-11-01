"""Duration × no_error_visible cross-tab (D1–D5). Read-only analysis."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .annotator_agreement import (
    DEFAULT_CLIP_MAP,
    DEFAULT_CLIPS_META,
    DEFAULT_EXPORT,
    DEFAULT_OUT,
    complete_usable,
    load_clip_map,
    load_export,
    load_null_before,
    parse_export,
    rater_ids_from_rows,
    rater_name,
)
from .config import DEFAULT_THRESHOLDS
from .probe_clips import SOURCE_FPS
from .vlm_probe.config import CLIP_FPS
from .vlm_probe.summarize import write_csv, write_json


NO_ERROR = "no_error_visible"
DURATION_EDGES = (0, 3, 5, 10, 20, 50, 100, 10_000)
DURATION_LABELS = ("3", "4-5", "6-10", "11-20", "21-50", "51-100", ">100")
MIN_FN_RUN_FRAMES = int(DEFAULT_THRESHOLDS["MIN_FN_RUN_FRAMES"])
DEFAULT_VALIDATED = DEFAULT_OUT / "validated.csv"
DEFAULT_STAGE1 = Path("out/vlm_probe/parsed/Qwen--Qwen3-VL-8B-Instruct_stage1.csv")
DEFAULT_SEGMENTED = Path("out/probe_clips/summary/segmented_clips.csv")
DEFAULT_MANIFEST = Path("out/probe_clips/summary/clip_sample_manifest.csv")
DEFAULT_WITHHELD = Path("out/probe_clips/withheld")


def duration_bin(frames: int) -> str:
    """pandas.cut(right=True) equivalent: (0,3], (3,5], ..."""
    value = int(frames)
    for label, low, high in zip(DURATION_LABELS, DURATION_EDGES, DURATION_EDGES[1:]):
        if low < value <= high:
            return label
    return DURATION_LABELS[-1]


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def duration_from_meta(meta: Mapping[str, Any]) -> int | None:
    window = meta.get("failure_window")
    if isinstance(window, (list, tuple)) and len(window) >= 2:
        start, end = _optional_int(window[0]), _optional_int(window[1])
        if start is not None and end is not None:
            return end - start + 1
    start = _optional_int(meta.get("frame_start"))
    end = _optional_int(meta.get("frame_end"))
    if start is not None and end is not None:
        return end - start + 1
    return None


def load_durations(
    clip_map: Sequence[Mapping[str, str]],
    *,
    clips_meta_dir: Path,
    withheld_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, int]:
    by_event_manifest: dict[str, int] = {}
    if manifest_path and Path(manifest_path).is_file():
        with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                n_frames = _optional_int(row.get("n_frames"))
                if n_frames is not None:
                    by_event_manifest[row["event_id"]] = n_frames
    durations: dict[str, int] = {}
    meta_root = Path(clips_meta_dir)
    withheld_root = Path(withheld_dir) if withheld_dir else None
    for row in clip_map:
        clip_id = row["clip_id"]
        event_id = row.get("event_id") or ""
        duration = None
        meta_path = meta_root / f"{event_id}.json"
        if meta_path.is_file():
            duration = duration_from_meta(json.loads(meta_path.read_text(encoding="utf-8")))
        if duration is None and withheld_root is not None:
            withheld_path = withheld_root / f"{event_id}.json"
            if withheld_path.is_file():
                duration = duration_from_meta(
                    json.loads(withheld_path.read_text(encoding="utf-8"))
                )
        if duration is None:
            duration = by_event_manifest.get(event_id)
        if duration is not None:
            durations[clip_id] = duration
    return durations


def load_stage1_behaviour(path: Path) -> dict[str, str]:
    if not Path(path).is_file():
        return {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {
            row["clip_id"]: str(row.get("box_behaviour") or "")
            for row in csv.DictReader(handle)
        }


def load_segmented_clip_ids(path: Path) -> set[str]:
    if not Path(path).is_file():
        return set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {row["clip_id"] for row in csv.DictReader(handle)}


def load_annotation_rows(
    *,
    validated_path: Path | None,
    export_path: Path | None,
) -> list[dict[str, Any]]:
    if validated_path and Path(validated_path).is_file():
        rows = []
        with Path(validated_path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                copy = dict(row)
                copy["rater_id"] = int(row["rater_id"])
                copy["incomplete"] = str(row.get("incomplete") or "").lower() in {
                    "1",
                    "true",
                    "yes",
                }
                copy["void_excluded"] = str(row.get("void_excluded") or "").lower() in {
                    "1",
                    "true",
                    "yes",
                }
                rows.append(copy)
        return rows
    if export_path and Path(export_path).is_file():
        return parse_export(load_export(export_path))
    raise FileNotFoundError("Need --validated or --export")


def playback_arithmetic(durations: Sequence[int]) -> dict[str, Any]:
    values = [int(item) for item in durations]
    array = np.asarray(values, dtype=float)
    on_screen = array / float(CLIP_FPS)
    source_time = array / float(SOURCE_FPS)
    return {
        "clip_fps": float(CLIP_FPS),
        "source_fps": float(SOURCE_FPS),
        "min_fn_run_frames": MIN_FN_RUN_FRAMES,
        "n": len(values),
        "median_duration_frames": float(np.median(array)) if len(values) else None,
        "median_on_screen_seconds": float(np.median(on_screen)) if len(values) else None,
        "share_under_0_5s_on_screen": float(np.mean(on_screen < 0.5)) if len(values) else None,
        "median_source_seconds": float(np.median(source_time)) if len(values) else None,
        "note": (
            "On-screen seconds use clip fps (10). Source seconds use MOT20 fps (25). "
            "A 3-frame gap is 0.3s of playback and 0.12s of original time."
        ),
    }


def no_error_crosstab(
    clip_ids: Sequence[str],
    durations: Mapping[str, int],
    flags: Mapping[str, bool],
) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    for clip_id in clip_ids:
        if clip_id not in durations:
            continue
        label = duration_bin(durations[clip_id])
        counts[label] += 1
        if flags.get(clip_id):
            hits[label] += 1
    rows = []
    for label in DURATION_LABELS:
        n = counts[label]
        n_yes = hits[label]
        rows.append(
            {
                "duration_bin": label,
                "n": n,
                "n_no_error_visible": n_yes,
                "n_other": n - n_yes,
                "rate_no_error_visible": (n_yes / n) if n else "",
            }
        )
    return rows


def rater_no_error_flags(
    rows: Sequence[Mapping[str, Any]], rater_id: int
) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for row in rows:
        if int(row["rater_id"]) != int(rater_id) or not complete_usable(row):
            continue
        kind = str(row.get("error_kind") or row.get("human_cause") or "")
        flags[str(row["clip_id"])] = kind == NO_ERROR
    return flags


def classify_duration_profile(crosstab: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    usable = [
        row
        for row in crosstab
        if int(row["n"] or 0) >= 5 and row["rate_no_error_visible"] != ""
    ]
    if len(usable) < 3:
        return {"hypothesis": "insufficient", "reason": "too few bins with n>=5"}
    rates = [float(row["rate_no_error_visible"]) for row in usable]
    short = [
        float(row["rate_no_error_visible"])
        for row in crosstab
        if row["duration_bin"] in {"3", "4-5"} and int(row["n"] or 0)
    ]
    long = [
        float(row["rate_no_error_visible"])
        for row in crosstab
        if row["duration_bin"] in {"21-50", "51-100", ">100"} and int(row["n"] or 0) >= 5
    ]
    short_rate = float(np.mean(short)) if short else None
    long_rate = float(np.mean(long)) if long else None
    declines = sum(rates[i] >= rates[i + 1] - 0.05 for i in range(len(rates) - 1))
    monotone = declines >= max(1, len(rates) - 2)
    high_short = short_rate is not None and short_rate >= 0.5
    high_long = long_rate is not None and long_rate >= 0.5
    low_long = long_rate is not None and long_rate <= 0.35
    drop = (
        short_rate is not None
        and long_rate is not None
        and (short_rate - long_rate) >= 0.15
    )
    if high_short and high_long:
        hypothesis = "both"
        reason = "high no_error_visible at short and long extremes"
    elif drop or (monotone and low_long):
        hypothesis = "H-presentation"
        reason = "no_error_visible is higher on short failures than on long ones"
    elif (
        short_rate is not None
        and long_rate is not None
        and abs(short_rate - long_rate) < 0.15
    ):
        hypothesis = "H-invisible"
        reason = "no_error_visible is similar across short and long bins"
    else:
        hypothesis = "H-invisible"
        reason = "duration does not cleanly explain no_error_visible"
    return {
        "hypothesis": hypothesis,
        "reason": reason,
        "short_rate": short_rate,
        "long_rate": long_rate,
        "usable_bins": [row["duration_bin"] for row in usable],
    }


def d6_sentences(
    *,
    rates_by_rater: Mapping[int, float],
    has_before_class: Mapping[str, Any],
    threshold_frames: int | None,
    threshold_seconds: float | None,
) -> dict[str, str]:
    pct = "–".join(f"{100 * rates_by_rater[rater]:.0f}" for rater in sorted(rates_by_rater))
    n_label = str(threshold_frames) if threshold_frames is not None else "N"
    x_label = f"{threshold_seconds:.1f}" if threshold_seconds is not None else "X"
    presentation = (
        f'Annotators labelled {pct}% of clips "no error visible," concentrated on failures '
        f"shorter than {n_label} frames. At the clip's playback rate these occupy under "
        f"{x_label} seconds. A substantial share of what MOT metrics count as tracking "
        "failures is not perceptible to human observers watching the tracker's output."
    )
    invisible = (
        f'Annotators labelled {pct}% of clips "no error visible" regardless of failure '
        "duration. Agreement between them on this label was κ = 0.70, the highest of any "
        "category. A large fraction of counted tracking failures has no visible correlate "
        "in the tracker's output."
    )
    rule_human = (
        "Annotators agreed with each other at κ = 0.44 but with the rule-based attributor at "
        "κ = 0.13–0.18, driven primarily by the \"no error visible\" response. Agreement was "
        "moderate for pedestrian-occlusion causes (κ ≈ 0.36–0.49) and near zero for "
        "truncation, drift, and visible-miss."
    )
    chosen = has_before_class.get("hypothesis") or "unknown"
    if chosen == "H-presentation":
        recommended = presentation
    elif chosen == "both":
        recommended = presentation + " Long segmented clips are a second source."
    else:
        recommended = invisible
    return {
        "H-presentation": presentation,
        "H-invisible": invisible,
        "rule_human_kappa": rule_human,
        "recommended": recommended,
        "recommended_hypothesis": chosen,
    }


def classify_vlm_vs_human(profiles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def rates(field: str) -> list[float]:
        values = []
        for row in profiles:
            raw = row.get(field)
            if raw == "" or raw is None:
                continue
            if int(row.get("n") or 0) < 5:
                continue
            if row.get("duration_bin") == ">100":
                continue
            values.append(float(raw))
        return values

    vlm = rates("vlm_rate")
    human_fields = [key for key in (profiles[0] if profiles else {}) if key.startswith("rater_") and key.endswith("_rate")]
    vlm_span = (max(vlm) - min(vlm)) if len(vlm) >= 3 else None
    human_spans = []
    for field in human_fields:
        values = rates(field)
        if len(values) >= 3:
            human_spans.append(max(values) - min(values))
    vlm_flat = vlm_span is not None and vlm_span < 0.20
    humans_duration = bool(human_spans) and min(human_spans) >= 0.30
    if vlm_flat and humans_duration:
        reading = "vlm_flat_humans_duration"
        note = (
            "VLM stays_on_person is high in every duration bin; human no_error_visible "
            "falls as duration increases. The VLM null is its own; the human result is presentation."
        )
    elif vlm_flat is False and humans_duration:
        reading = "same_duration_profile"
        note = "Humans and the VLM both fail more on short clips."
    else:
        reading = "mixed"
        note = "Duration profiles do not line up cleanly across observers."
    return {
        "reading": reading,
        "note": note,
        "vlm_rate_span": vlm_span,
        "human_rate_spans": human_spans,
    }


def threshold_from_crosstab(crosstab: Sequence[Mapping[str, Any]]) -> tuple[int | None, float | None]:
    """First bin (from short) where no_error_visible drops below 0.5 with n>=5."""
    upper = {"3": 3, "4-5": 5, "6-10": 10, "11-20": 20, "21-50": 50, "51-100": 100}
    for row in crosstab:
        n = int(row["n"] or 0)
        rate = row["rate_no_error_visible"]
        if n < 5 or rate == "":
            continue
        if float(rate) < 0.5:
            frames = upper.get(str(row["duration_bin"]))
            if frames is None:
                return None, None
            return frames, frames / float(CLIP_FPS)
    return None, None


def three_way_rows(
    clip_ids: Sequence[str],
    durations: Mapping[str, int],
    vlm: Mapping[str, bool],
    rater_a: Mapping[str, bool],
    rater_b: Mapping[str, bool],
    rater_a_id: int,
    rater_b_id: int,
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, bool, bool, bool]] = Counter()
    for clip_id in clip_ids:
        if clip_id not in durations or clip_id not in vlm:
            continue
        if clip_id not in rater_a or clip_id not in rater_b:
            continue
        key = (
            duration_bin(durations[clip_id]),
            bool(vlm[clip_id]),
            bool(rater_a[clip_id]),
            bool(rater_b[clip_id]),
        )
        counts[key] += 1
    rows = []
    for label in DURATION_LABELS:
        for vlm_flag in (True, False):
            for a_flag in (True, False):
                for b_flag in (True, False):
                    n = counts[(label, vlm_flag, a_flag, b_flag)]
                    if n == 0:
                        continue
                    rows.append(
                        {
                            "duration_bin": label,
                            "vlm_stays_on_person": vlm_flag,
                            f"rater_{rater_a_id}_no_error_visible": a_flag,
                            f"rater_{rater_b_id}_no_error_visible": b_flag,
                            "n": n,
                        }
                    )
    return rows


def observer_profile_rows(
    clip_ids: Sequence[str],
    durations: Mapping[str, int],
    observers: Mapping[str, Mapping[str, bool]],
) -> list[dict[str, Any]]:
    rows = []
    for label in DURATION_LABELS:
        members = [
            clip_id
            for clip_id in clip_ids
            if clip_id in durations and duration_bin(durations[clip_id]) == label
        ]
        item: dict[str, Any] = {"duration_bin": label, "n": len(members)}
        for name, flags in observers.items():
            usable = [clip_id for clip_id in members if clip_id in flags]
            n_yes = sum(1 for clip_id in usable if flags[clip_id])
            item[f"{name}_n"] = len(usable)
            item[f"{name}_n_yes"] = n_yes
            item[f"{name}_rate"] = (n_yes / len(usable)) if usable else ""
        rows.append(item)
    return rows


def segmented_cause_rows(
    rows: Sequence[Mapping[str, Any]],
    segmented: set[str],
    rater_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    causes: Counter[str] = Counter()
    n = 0
    n_no_error = 0
    for row in rows:
        if int(row["rater_id"]) != int(rater_id) or not complete_usable(row):
            continue
        if str(row["clip_id"]) not in segmented:
            continue
        n += 1
        cause = str(row.get("human_cause") or row.get("error_kind") or "")
        causes[cause] += 1
        if cause == NO_ERROR or row.get("error_kind") == NO_ERROR:
            n_no_error += 1
    table = [
        {
            "rater_id": rater_id,
            "human_cause": cause,
            "n": count,
            "share": (count / n) if n else 0.0,
        }
        for cause, count in causes.most_common()
    ]
    summary = {
        "rater_id": rater_id,
        "n_segmented_labelled": n,
        "n_no_error_visible": n_no_error,
        "rate_no_error_visible": (n_no_error / n) if n else None,
    }
    return table, summary


def run_duration_crosstab(
    *,
    out_dir: Path,
    validated_path: Path = DEFAULT_VALIDATED,
    export_path: Path = DEFAULT_EXPORT,
    clip_map_path: Path = DEFAULT_CLIP_MAP,
    clips_meta_dir: Path = DEFAULT_CLIPS_META,
    withheld_dir: Path = DEFAULT_WITHHELD,
    manifest_path: Path = DEFAULT_MANIFEST,
    stage1_path: Path = DEFAULT_STAGE1,
    segmented_path: Path = DEFAULT_SEGMENTED,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_map = load_clip_map(clip_map_path)
    durations = load_durations(
        clip_map,
        clips_meta_dir=clips_meta_dir,
        withheld_dir=withheld_dir,
        manifest_path=manifest_path,
    )
    null_before = load_null_before(clip_map, clips_meta_dir)
    rows = load_annotation_rows(
        validated_path=validated_path if Path(validated_path).is_file() else None,
        export_path=export_path,
    )
    rater_ids = rater_ids_from_rows(rows)
    clip_ids = [row["clip_id"] for row in clip_map if row["clip_id"] in durations]

    d3 = playback_arithmetic([durations[clip_id] for clip_id in clip_ids])
    write_json(out_dir / "duration_arithmetic.json", d3)

    rates_by_rater: dict[int, float] = {}
    has_before_class: dict[str, Any] = {}
    all_crosstabs: list[dict[str, Any]] = []
    for rater_id in rater_ids:
        flags = rater_no_error_flags(rows, rater_id)
        labelled = [clip_id for clip_id in clip_ids if clip_id in flags]
        if labelled:
            rates_by_rater[rater_id] = sum(flags[clip_id] for clip_id in labelled) / len(labelled)
        subsets = {
            "all": labelled,
            "has_before": [clip_id for clip_id in labelled if not null_before.get(clip_id)],
            "null_before": [clip_id for clip_id in labelled if null_before.get(clip_id)],
        }
        for subset_name, members in subsets.items():
            table = no_error_crosstab(members, durations, flags)
            for row in table:
                row["rater_id"] = rater_id
                row["rater_name"] = rater_name(rater_id)
                row["subset"] = subset_name
            all_crosstabs.extend(table)
            if subset_name == "has_before":
                has_before_class[str(rater_id)] = classify_duration_profile(table)
                has_before_class[str(rater_id)]["n_has_before"] = len(members)
                has_before_class[str(rater_id)]["n_no_error_has_before"] = sum(
                    flags[clip_id] for clip_id in members
                )
    write_csv(
        out_dir / "duration_no_error_crosstab.csv",
        all_crosstabs,
        (
            "subset",
            "rater_id",
            "rater_name",
            "duration_bin",
            "n",
            "n_no_error_visible",
            "n_other",
            "rate_no_error_visible",
        ),
    )

    vlm_behaviour = load_stage1_behaviour(stage1_path)
    vlm_flags = {
        clip_id: behaviour == "stays_on_person"
        for clip_id, behaviour in vlm_behaviour.items()
    }
    observer_flags: dict[str, dict[str, bool]] = {"vlm": vlm_flags}
    for rater_id in rater_ids:
        observer_flags[f"rater_{rater_id}"] = rater_no_error_flags(rows, rater_id)
    profiles = observer_profile_rows(clip_ids, durations, observer_flags)
    profile_fields = ["duration_bin", "n"]
    for name in observer_flags:
        profile_fields.extend([f"{name}_n", f"{name}_n_yes", f"{name}_rate"])
    write_csv(out_dir / "duration_observer_profiles.csv", profiles, profile_fields)
    vlm_vs_human = classify_vlm_vs_human(profiles)

    three_way = []
    if len(rater_ids) >= 2:
        three_way = three_way_rows(
            clip_ids,
            durations,
            vlm_flags,
            observer_flags[f"rater_{rater_ids[0]}"],
            observer_flags[f"rater_{rater_ids[1]}"],
            rater_ids[0],
            rater_ids[1],
        )
        fields = [
            "duration_bin",
            "vlm_stays_on_person",
            f"rater_{rater_ids[0]}_no_error_visible",
            f"rater_{rater_ids[1]}_no_error_visible",
            "n",
        ]
        write_csv(out_dir / "duration_three_way.csv", three_way, fields)

    segmented = load_segmented_clip_ids(segmented_path)
    segmented_tables: list[dict[str, Any]] = []
    segmented_summaries: list[dict[str, Any]] = []
    for rater_id in rater_ids:
        table, summary = segmented_cause_rows(rows, segmented, rater_id)
        overall_flags = rater_no_error_flags(rows, rater_id)
        overall_rate = (
            sum(overall_flags.values()) / len(overall_flags) if overall_flags else None
        )
        summary["overall_rate_no_error_visible"] = overall_rate
        summary["elevated"] = bool(
            summary["rate_no_error_visible"] is not None
            and overall_rate is not None
            and summary["rate_no_error_visible"] >= overall_rate + 0.10
        )
        segmented_tables.extend(table)
        segmented_summaries.append(summary)
    write_csv(
        out_dir / "segmented_cause_distribution.csv",
        segmented_tables,
        ("rater_id", "human_cause", "n", "share"),
    )
    write_json(out_dir / "segmented_no_error_report.json", {"n_segmented": len(segmented), "raters": segmented_summaries})

    all_class: dict[str, Any] = {}
    for rater_id in rater_ids:
        all_table = [
            row
            for row in all_crosstabs
            if row["subset"] == "all" and row["rater_id"] == rater_id
        ]
        all_class[str(rater_id)] = classify_duration_profile(all_table)
    primary = next(
        (has_before_class[str(rater_id)] for rater_id in rater_ids if str(rater_id) in has_before_class),
        {},
    )
    votes = [item.get("hypothesis") for item in has_before_class.values()]
    if votes and votes.count("H-presentation") >= (len(votes) + 1) // 2:
        primary = {**primary, "hypothesis": "H-presentation"}
    all_tables = [
        row
        for row in all_crosstabs
        if row["subset"] == "all" and rater_ids and row["rater_id"] == rater_ids[0]
    ]
    threshold_frames, threshold_seconds = threshold_from_crosstab(all_tables)
    sentences = d6_sentences(
        rates_by_rater=rates_by_rater,
        has_before_class=primary,
        threshold_frames=threshold_frames,
        threshold_seconds=threshold_seconds,
    )
    vlm_yes = [clip_id for clip_id in clip_ids if vlm_flags.get(clip_id)]
    report = {
        "arithmetic": d3,
        "n_clips_with_duration": len(durations),
        "n_vlm_stays_on_person": sum(vlm_flags.get(clip_id, False) for clip_id in clip_ids),
        "n_vlm": sum(clip_id in vlm_flags for clip_id in clip_ids),
        "no_error_rate_by_rater": {
            str(rater_id): rates_by_rater.get(rater_id) for rater_id in rater_ids
        },
        "all_classification": all_class,
        "has_before_classification": has_before_class,
        "vlm_vs_human": vlm_vs_human,
        "threshold_frames": threshold_frames,
        "threshold_on_screen_seconds": threshold_seconds,
        "n_vlm_stays_on_person_named": len(vlm_yes),
        "d6": sentences,
        "segmented": segmented_summaries,
    }
    write_json(out_dir / "duration_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Duration × no_error_visible cross-tab (D1–D5)"
    )
    parser.add_argument("--validated", type=Path, default=DEFAULT_VALIDATED)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--clip-map", type=Path, default=DEFAULT_CLIP_MAP)
    parser.add_argument("--clips-meta", type=Path, default=DEFAULT_CLIPS_META)
    parser.add_argument("--withheld", type=Path, default=DEFAULT_WITHHELD)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage1", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--segmented", type=Path, default=DEFAULT_SEGMENTED)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    report = run_duration_crosstab(
        out_dir=args.out_dir,
        validated_path=args.validated,
        export_path=args.export,
        clip_map_path=args.clip_map,
        clips_meta_dir=args.clips_meta,
        withheld_dir=args.withheld,
        manifest_path=args.manifest,
        stage1_path=args.stage1,
        segmented_path=args.segmented,
    )
    first_class = next(iter((report.get("has_before_classification") or {}).values()), {})
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "median_duration_frames": report["arithmetic"]["median_duration_frames"],
                "median_on_screen_seconds": report["arithmetic"]["median_on_screen_seconds"],
                "share_under_0_5s_on_screen": report["arithmetic"]["share_under_0_5s_on_screen"],
                "n_vlm_stays_on_person": report["n_vlm_stays_on_person"],
                "hypothesis": first_class.get("hypothesis"),
                "recommended_hypothesis": report["d6"]["recommended_hypothesis"],
            },
            indent=2,
        )
    )
    return report
