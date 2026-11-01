"""Read-only annotator agreement analysis (Z1–Z6). Does not touch the corpus."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import DEFAULT_CLASSES, PipelineConfig
from .geometry import ioa, iou
from .human_verification import cohen_kappa
from .io import load_sequences
from .models import SequenceData
from .probe_clips import load_sidecar_events
from .vlm_probe.config import RULE_CAUSE_TO_VLM
from .vlm_probe.summarize import write_csv, write_json


KIND_IS_CAUSE = frozenset(
    {"no_error_visible", "could_not_identify_target", "cannot_determine"}
)
BRANCH_CAUSE_FIELD = {
    "miss": "cause_miss",
    "identity_change": "cause_idsw",
    "drift": "cause_drift",
}
HIDDEN_BY_CAUSES = frozenset(
    {
        "hidden_by_object",
        "hidden_by_one_person",
        "hidden_by_standing_person",
        "hidden_by_crowd",
    }
)
PARTLY_CAUSES = frozenset(
    {
        "partly_blocked_by_person",
        "partly_blocked_by_static",
        "partly_outside_the_frame",
    }
)
CROWD_MERGE = {
    "hidden_by_one_person": "hidden_by_people",
    "hidden_by_crowd": "hidden_by_people",
}
ERROR_TYPE_TO_KIND = {
    "FN": "miss",
    "FRAG_NO_SWITCH": "miss",
    "IDSW": "identity_change",
    "FP_ID_HIJACK": "drift",
}
IDENTITY_SWAP_RULE_CAUSES = frozenset(
    {
        "partial_occl_pedestrian",
        "full_occl_pedestrian",
        "full_occl_crowd_aggregate",
        "partial_occl_static",
        "full_occl_static",
    }
)
UNANSWERABLE_KINDS = frozenset({"could_not_identify_target", "no_error_visible"})
COMMENT_TAGS = (
    "ghost_highlight",
    "tree_occlusion",
    "pillar_occlusion",
    "localization_flicker",
    "out_of_frame",
    "duplicate_boxes",
    "crowd_occlusion",
    "third_id_not_highlighted",
    "target_invisible_no_valid_answer",
    "time_skip_confusion",
)
FUNNEL_RAW = 282
FUNNEL_CANOPY = 12
FUNNEL_TRUNCATION = 205
FUNNEL_FLICKER = 37
MATCH_IOU = 0.5
STATIONARY_VAR_THRESH = 25.0
IOA_OCCLUDER_THRESH = 0.5
GHOST_OCCLUDER_CLASSES = frozenset(
    DEFAULT_CLASSES["STATIC_PERSON"] + DEFAULT_CLASSES["OCCLUDER_CLASSES"]
)
RATER_NAMES = {
    3: "observer_a",
    8: "observer_b",
}
VALIDATED_FIELDS = (
    "clip_id",
    "rater_id",
    "rater_name",
    "annotation_id",
    "error_kind",
    "visibility",
    "cause_miss",
    "cause_idsw",
    "cause_drift",
    "hard_to_spot",
    "confidence",
    "human_cause",
    "incomplete",
    "incomplete_reason",
    "contradiction",
    "contradiction_reason",
    "void_excluded",
    "free_text",
)

DEFAULT_EXPORT = Path("annotations/human/observer_export.json")
DEFAULT_CLIP_MAP = Path("out/labelstudio_probe_clips/clip_map.csv")
DEFAULT_CLIPS_META = Path("out/probe_clips/clips_meta")
DEFAULT_EVENTS_ROOT = Path("out/min_fn3/events")
DEFAULT_TRACKER = Path("tracker_output/tracktrack")
DEFAULT_DATASET = Path("datasets/MOT20")
DEFAULT_VOID_CLIPS = Path("out/probe_clips/summary/void_a1_clip_ids.json")
DEFAULT_OUT = Path("out/annotator_agreement")


def _choice(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ";".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        if "choices" in value:
            return _choice(value.get("choices"))
        if "text" in value:
            return _choice(value.get("text"))
        return ""
    return str(value).strip()


def parse_result_fields(result: Sequence[Mapping[str, Any]] | None) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in result or ():
        name = str(item.get("from_name") or "").strip()
        if not name:
            continue
        fields[name] = _choice(item.get("value"))
    return fields


def human_cause(fields: Mapping[str, str]) -> str:
    kind = str(fields.get("error_kind") or "").strip()
    if kind in KIND_IS_CAUSE:
        return kind
    branch = BRANCH_CAUSE_FIELD.get(kind)
    if branch:
        return str(fields.get(branch) or "").strip()
    return ""


def incomplete_reason(fields: Mapping[str, str]) -> str:
    kind = str(fields.get("error_kind") or "").strip()
    if not kind:
        return "missing_error_kind"
    if not str(fields.get("confidence") or "").strip():
        return "missing_confidence"
    if kind in KIND_IS_CAUSE:
        return ""
    branch = BRANCH_CAUSE_FIELD.get(kind)
    if branch is None:
        return "unknown_error_kind"
    if not str(fields.get(branch) or "").strip():
        return f"missing_{branch}"
    return ""


def contradiction_reason(fields: Mapping[str, str]) -> str:
    if str(fields.get("error_kind") or "").strip() != "miss":
        return ""
    visibility = str(fields.get("visibility") or "").strip()
    cause = str(fields.get("cause_miss") or "").strip()
    if not visibility or not cause:
        return ""
    expected = {
        "not_at_all": HIDDEN_BY_CAUSES,
        "partly": PARTLY_CAUSES,
        "clearly": frozenset({"clearly_visible_but_missed"}),
        "cannot_tell": frozenset({"cannot_determine"}),
    }.get(visibility)
    if expected is None:
        return f"unknown_visibility:{visibility}"
    if cause in expected:
        return ""
    return f"visibility={visibility} vs cause_miss={cause}"


def rater_name(rater_id: int) -> str:
    return RATER_NAMES.get(rater_id, str(rater_id))


def _annotation_sort_key(annotation: Mapping[str, Any]) -> str:
    return str(
        annotation.get("updated_at")
        or annotation.get("created_at")
        or annotation.get("id")
        or ""
    )


def parse_export(
    tasks: Sequence[Mapping[str, Any]],
    *,
    void_clips: set[str] | None = None,
    void_rater_id: int | None = None,
) -> list[dict[str, Any]]:
    void = void_clips or set()
    rows: list[dict[str, Any]] = []
    for task in tasks:
        clip_id = str((task.get("data") or {}).get("clip_id") or "").strip()
        if not clip_id:
            continue
        by_rater: dict[int, Mapping[str, Any]] = {}
        for annotation in task.get("annotations") or ():
            if annotation.get("was_cancelled"):
                continue
            rater = annotation.get("completed_by")
            if rater is None:
                continue
            rater_id = int(rater)
            previous = by_rater.get(rater_id)
            if previous is None or _annotation_sort_key(annotation) >= _annotation_sort_key(
                previous
            ):
                by_rater[rater_id] = annotation
        for rater_id, annotation in sorted(by_rater.items()):
            fields = parse_result_fields(annotation.get("result"))
            missing = incomplete_reason(fields)
            contradiction = contradiction_reason(fields)
            excluded = bool(
                void_rater_id is not None
                and rater_id == int(void_rater_id)
                and clip_id in void
            )
            cause = "" if missing else human_cause(fields)
            rows.append(
                {
                    "clip_id": clip_id,
                    "rater_id": rater_id,
                    "rater_name": rater_name(rater_id),
                    "annotation_id": annotation.get("id", ""),
                    "error_kind": fields.get("error_kind", ""),
                    "visibility": fields.get("visibility", ""),
                    "cause_miss": fields.get("cause_miss", ""),
                    "cause_idsw": fields.get("cause_idsw", ""),
                    "cause_drift": fields.get("cause_drift", ""),
                    "hard_to_spot": fields.get("hard_to_spot", ""),
                    "confidence": fields.get("confidence", ""),
                    "human_cause": cause,
                    "incomplete": bool(missing),
                    "incomplete_reason": missing,
                    "contradiction": bool(contradiction),
                    "contradiction_reason": contradiction,
                    "void_excluded": excluded,
                    "free_text": fields.get("free_text", ""),
                }
            )
    return rows


def load_export(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of tasks in {path}")
    return payload


def load_void_clip_ids(path: Path | None) -> set[str]:
    if path is None or not Path(path).is_file():
        return set()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(item) for item in payload.get("clip_ids") or []}
    if isinstance(payload, list):
        return {str(item) for item in payload}
    return set()


def load_clip_map(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validation_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_rater: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_rater[int(row["rater_id"])].append(row)
    raters = []
    for rater_id in sorted(by_rater):
        items = by_rater[rater_id]
        n = len(items)
        incomplete = [row for row in items if row["incomplete"]]
        contradictions = [row for row in items if row["contradiction"]]
        identify = [
            row for row in items if row.get("error_kind") == "could_not_identify_target"
        ]
        raters.append(
            {
                "rater_id": rater_id,
                "rater_name": rater_name(rater_id),
                "n": n,
                "n_complete": n - len(incomplete),
                "n_incomplete": len(incomplete),
                "incomplete_clip_ids": [row["clip_id"] for row in incomplete],
                "n_contradiction": len(contradictions),
                "contradiction_rate": (len(contradictions) / n) if n else 0.0,
                "contradiction_clip_ids": [row["clip_id"] for row in contradictions],
                "n_could_not_identify_target": len(identify),
                "could_not_identify_target_rate": (len(identify) / n) if n else 0.0,
                "could_not_identify_target_clip_ids": [row["clip_id"] for row in identify],
            }
        )
    return {
        "n_rows": len(rows),
        "n_clips": len({row["clip_id"] for row in rows}),
        "n_raters": len(by_rater),
        "raters": raters,
    }


def write_validated(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(path, rows, VALIDATED_FIELDS)


def complete_usable(row: Mapping[str, Any]) -> bool:
    return (not row.get("incomplete")) and (not row.get("void_excluded")) and bool(
        row.get("human_cause")
    )


def ratings_by_clip(
    rows: Sequence[Mapping[str, Any]],
    *,
    rater_ids: Sequence[int] | None = None,
) -> dict[str, dict[int, Mapping[str, Any]]]:
    wanted = set(rater_ids) if rater_ids is not None else None
    by_clip: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if not complete_usable(row):
            continue
        rater_id = int(row["rater_id"])
        if wanted is not None and rater_id not in wanted:
            continue
        by_clip[str(row["clip_id"])][rater_id] = row
    return dict(by_clip)


def paired_clips(
    by_clip: Mapping[str, Mapping[int, Mapping[str, Any]]],
    rater_ids: Sequence[int],
) -> list[str]:
    needed = set(rater_ids)
    return sorted(
        clip_id for clip_id, raters in by_clip.items() if needed <= set(raters)
    )


def observed_all_agree(label_rows: Sequence[Sequence[str]]) -> float:
    if not label_rows:
        return 0.0
    return sum(len(set(row)) == 1 for row in label_rows) / len(label_rows)


def fleiss_kappa(ratings: Sequence[Sequence[str]]) -> float:
    if not ratings:
        raise ValueError("Fleiss' kappa requires at least one subject")
    n_raters = len(ratings[0])
    if n_raters < 2 or any(len(row) != n_raters for row in ratings):
        raise ValueError("Fleiss' kappa requires every subject to have the same rater count")
    n = len(ratings)
    categories = sorted({label for row in ratings for label in row})
    subject_p = []
    category_counts: Counter[str] = Counter()
    for row in ratings:
        counts = Counter(row)
        category_counts.update(row)
        subject_p.append(
            (sum(count * count for count in counts.values()) - n_raters)
            / (n_raters * (n_raters - 1))
        )
    p_bar = sum(subject_p) / n
    total = n * n_raters
    p_e = sum((category_counts[label] / total) ** 2 for label in categories)
    if p_e == 1.0:
        return 1.0 if p_bar == 1.0 else 0.0
    return (p_bar - p_e) / (1.0 - p_e)


def multi_rater_kappa(rater_labels: Sequence[Sequence[str]]) -> tuple[float, str]:
    if len(rater_labels) < 2:
        raise ValueError("Need at least two raters")
    n = len(rater_labels[0])
    if n == 0 or any(len(labels) != n for labels in rater_labels):
        raise ValueError("Raters must have equal-length non-empty label lists")
    if len(rater_labels) == 2:
        return cohen_kappa(rater_labels[0], rater_labels[1]), "cohen"
    subjects = [list(subject) for subject in zip(*rater_labels)]
    return fleiss_kappa(subjects), "fleiss"


def _merge_crowd(label: str) -> str:
    return CROWD_MERGE.get(label, label)


def kappa_table_row(
    *,
    cut: str,
    clip_ids: Sequence[str],
    rater_ids: Sequence[int],
    labels_for: Any,
    note: str = "",
) -> dict[str, Any]:
    rater_labels = [labels_for(rater_id, clip_ids) for rater_id in rater_ids]
    subjects = [list(subject) for subject in zip(*rater_labels)]
    kappa, method = multi_rater_kappa(rater_labels)
    return {
        "cut": cut,
        "n": len(clip_ids),
        "n_raters": len(rater_ids),
        "method": method,
        "observed_agreement": observed_all_agree(subjects),
        "kappa": kappa,
        "note": note,
    }


def human_human_kappa_rows(
    rows: Sequence[Mapping[str, Any]],
    rater_ids: Sequence[int],
) -> list[dict[str, Any]]:
    by_clip = ratings_by_clip(rows, rater_ids=rater_ids)
    clips = paired_clips(by_clip, rater_ids)
    if not clips:
        return []

    def labels(rater_id: int, clip_ids: Sequence[str], remap=lambda value: value):
        return [remap(by_clip[clip_id][rater_id]["human_cause"]) for clip_id in clip_ids]

    table = [
        kappa_table_row(
            cut="overall",
            clip_ids=clips,
            rater_ids=rater_ids,
            labels_for=labels,
        )
    ]
    causes = sorted({by_clip[clip_id][rater_id]["human_cause"] for clip_id in clips for rater_id in rater_ids})
    for cause in causes:
        table.append(
            kappa_table_row(
                cut=f"one_vs_rest:{cause}",
                clip_ids=clips,
                rater_ids=rater_ids,
                labels_for=lambda rater_id, clip_ids, cause=cause: [
                    str(by_clip[clip_id][rater_id]["human_cause"] == cause)
                    for clip_id in clip_ids
                ],
            )
        )
    high = [
        clip_id
        for clip_id in clips
        if all(by_clip[clip_id][rater_id]["confidence"] == "high" for rater_id in rater_ids)
    ]
    if high:
        table.append(
            kappa_table_row(
                cut="confidence_high",
                clip_ids=high,
                rater_ids=rater_ids,
                labels_for=labels,
                note="both/all raters confidence=high",
            )
        )
    no_cd = [
        clip_id
        for clip_id in clips
        if all(
            by_clip[clip_id][rater_id]["human_cause"] != "cannot_determine"
            for rater_id in rater_ids
        )
    ]
    if no_cd:
        table.append(
            kappa_table_row(
                cut="excluding_cannot_determine",
                clip_ids=no_cd,
                rater_ids=rater_ids,
                labels_for=labels,
            )
        )
    table.append(
        kappa_table_row(
            cut="crowd_merged",
            clip_ids=clips,
            rater_ids=rater_ids,
            labels_for=lambda rater_id, clip_ids: labels(rater_id, clip_ids, _merge_crowd),
            note="hidden_by_one_person and hidden_by_crowd -> hidden_by_people",
        )
    )
    return table


def map_rule_to_human_cause(rule_cause: str, error_type: str) -> str:
    kind = ERROR_TYPE_TO_KIND.get(str(error_type))
    cause = str(rule_cause or "").strip()
    if kind == "identity_change":
        if cause in IDENTITY_SWAP_RULE_CAUSES:
            return "swapped_during_crossing"
        if cause == "unattributed_idsw":
            return "changed_for_no_visible_reason"
        if cause == "reentry":
            return "left_and_returned"
    if kind == "drift":
        if cause == "track_drift_to_static":
            return "drifted_to_standing_person"
        if cause == "track_drift_to_moving":
            return "drifted_to_moving_person"
    if kind == "miss":
        mapped = RULE_CAUSE_TO_VLM.get(cause)
        if mapped:
            return mapped
    return RULE_CAUSE_TO_VLM.get(cause, cause)


def rule_human_outputs(
    rows: Sequence[Mapping[str, Any]],
    clip_map: Sequence[Mapping[str, str]],
    rater_ids: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    meta = {row["clip_id"]: row for row in clip_map}
    by_clip = ratings_by_clip(rows, rater_ids=rater_ids)
    kappa_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    distribution: list[dict[str, Any]] = []
    for rater_id in rater_ids:
        scored_rule: list[str] = []
        scored_human: list[str] = []
        unattr: Counter[str] = Counter()
        for clip_id, raters in sorted(by_clip.items()):
            if rater_id not in raters:
                continue
            info = meta.get(clip_id)
            if info is None:
                continue
            rule_cause = str(info.get("cause") or "")
            human = str(raters[rater_id]["human_cause"])
            mapped = map_rule_to_human_cause(rule_cause, str(info.get("error_type") or ""))
            pair_rows.append(
                {
                    "clip_id": clip_id,
                    "rater_id": rater_id,
                    "rater_name": rater_name(rater_id),
                    "event_id": info.get("event_id", ""),
                    "error_type": info.get("error_type", ""),
                    "rule_cause": rule_cause,
                    "mapped_rule_cause": mapped,
                    "human_cause": human,
                    "agree": mapped == human,
                    "unattributed_fn": rule_cause == "unattributed_fn",
                }
            )
            if rule_cause == "unattributed_fn":
                unattr[human] += 1
                continue
            scored_rule.append(mapped)
            scored_human.append(human)
        if scored_rule:
            kappa, method = multi_rater_kappa([scored_rule, scored_human])
            kappa_rows.append(
                {
                    "cut": f"rule_vs_rater_{rater_id}",
                    "n": len(scored_rule),
                    "n_raters": 2,
                    "method": method,
                    "observed_agreement": observed_all_agree(
                        list(zip(scored_rule, scored_human))
                    ),
                    "kappa": kappa,
                    "note": f"excludes unattributed_fn; rater {rater_id} ({rater_name(rater_id)})",
                }
            )
            causes = sorted(set(scored_rule) | set(scored_human))
            for cause in causes:
                left = [str(label == cause) for label in scored_rule]
                right = [str(label == cause) for label in scored_human]
                kappa_rows.append(
                    {
                        "cut": f"rule_vs_rater_{rater_id}:one_vs_rest:{cause}",
                        "n": len(scored_rule),
                        "n_raters": 2,
                        "method": "cohen",
                        "observed_agreement": observed_all_agree(list(zip(left, right))),
                        "kappa": cohen_kappa(left, right),
                        "note": "",
                    }
                )
        for cause, count in sorted(unattr.items()):
            distribution.append(
                {
                    "rater_id": rater_id,
                    "rater_name": rater_name(rater_id),
                    "human_cause": cause,
                    "n": count,
                    "share": count / sum(unattr.values()) if unattr else 0.0,
                }
            )
    confusion_counts: Counter[tuple[str, str]] = Counter()
    for row in pair_rows:
        if row["unattributed_fn"]:
            continue
        confusion_counts[(row["mapped_rule_cause"], row["human_cause"])] += 1
    labels = sorted({key[0] for key in confusion_counts} | {key[1] for key in confusion_counts})
    confusion = []
    for rule_label in labels:
        item = {"mapped_rule_cause": rule_label}
        total = 0
        for human_label in labels:
            count = confusion_counts[(rule_label, human_label)]
            item[human_label] = count
            total += count
        item["row_total"] = total
        confusion.append(item)
    return kappa_rows, pair_rows, confusion, distribution


def load_null_before(
    clip_map: Sequence[Mapping[str, str]], clips_meta_dir: Path
) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    root = Path(clips_meta_dir)
    for row in clip_map:
        clip_id = row["clip_id"]
        event_id = row.get("event_id") or ""
        path = root / f"{event_id}.json"
        if not path.is_file():
            flags[clip_id] = False
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        before = (meta.get("highlighted_track_ids") or {}).get("before")
        flags[clip_id] = before is None
    return flags


def _crosstab_rows(
    index_values: Sequence[Any],
    column_values: Sequence[Any],
) -> list[dict[str, Any]]:
    counts: Counter[tuple[Any, Any]] = Counter(zip(index_values, column_values))
    rows = sorted({str(value) for value in index_values})
    cols = sorted({str(value) for value in column_values})
    table = []
    for row in rows:
        row_total = sum(counts[(row, col)] for col in cols)
        item = {"index": row, "n": row_total}
        for col in cols:
            count = counts[(row, col)]
            item[col] = (count / row_total) if row_total else 0.0
            item[f"{col}_count"] = count
        table.append(item)
    return table


def null_before_outputs(
    rows: Sequence[Mapping[str, Any]],
    rater_ids: Sequence[int],
    null_before: Mapping[str, bool],
) -> dict[str, Any]:
    by_clip = ratings_by_clip(rows, rater_ids=rater_ids)
    clips = paired_clips(by_clip, rater_ids)
    kinds: list[str] = []
    causes: list[str] = []
    kind_index: list[str] = []
    cause_index: list[str] = []
    for clip_id in clips:
        flag = "null_before" if null_before.get(clip_id) else "has_before"
        for rater_id in rater_ids:
            kinds.append(str(by_clip[clip_id][rater_id]["error_kind"]))
            causes.append(str(by_clip[clip_id][rater_id]["human_cause"]))
            kind_index.append(flag)
            cause_index.append(flag)

    def unanswerable_rate(subset: Sequence[str]) -> float:
        if not subset:
            return 0.0
        hits = 0
        n = 0
        for clip_id in subset:
            for rater_id in rater_ids:
                n += 1
                if by_clip[clip_id][rater_id]["error_kind"] in UNANSWERABLE_KINDS:
                    hits += 1
        return hits / n if n else 0.0

    null_clips = [clip_id for clip_id in clips if null_before.get(clip_id)]
    other_clips = [clip_id for clip_id in clips if not null_before.get(clip_id)]
    agree_null = (
        sum(
            len({by_clip[clip_id][rater_id]["human_cause"] for rater_id in rater_ids}) == 1
            for clip_id in null_clips
        )
        / len(null_clips)
        if null_clips
        else 0.0
    )
    agree_other = (
        sum(
            len({by_clip[clip_id][rater_id]["human_cause"] for rater_id in rater_ids}) == 1
            for clip_id in other_clips
        )
        / len(other_clips)
        if other_clips
        else 0.0
    )
    rate_null = unanswerable_rate(null_clips)
    rate_other = unanswerable_rate(other_clips)
    elevated = rate_null >= max(rate_other * 1.5, rate_other + 0.10) and bool(null_clips)
    extra_kappa = []
    if other_clips:
        extra_kappa.append(
            kappa_table_row(
                cut="excluding_null_before",
                clip_ids=other_clips,
                rater_ids=rater_ids,
                labels_for=lambda rater_id, clip_ids: [
                    by_clip[clip_id][rater_id]["human_cause"] for clip_id in clip_ids
                ],
                note="primary if null-before unanswerable rate is elevated",
            )
        )
    if null_clips:
        extra_kappa.append(
            kappa_table_row(
                cut="null_before_only",
                clip_ids=null_clips,
                rater_ids=rater_ids,
                labels_for=lambda rater_id, clip_ids: [
                    by_clip[clip_id][rater_id]["human_cause"] for clip_id in clip_ids
                ],
            )
        )
    return {
        "n_paired": len(clips),
        "n_null_before": len(null_clips),
        "n_has_before": len(other_clips),
        "agreement_null_before": agree_null,
        "agreement_has_before": agree_other,
        "unanswerable_rate_null_before": rate_null,
        "unanswerable_rate_has_before": rate_other,
        "null_before_elevated": elevated,
        "error_kind_crosstab": _crosstab_rows(kind_index, kinds),
        "cause_crosstab": _crosstab_rows(cause_index, causes),
        "kappa_rows": extra_kappa,
    }


def max_iou_during_gap(event: Mapping[str, Any], sequence: SequenceData) -> float:
    focal = int(event["focal_gt_id"])
    best = 0.0
    start = int(event["frame_start"])
    end = int(event["frame_end"])
    for frame in range(start, end + 1):
        gt = [box for box in sequence.gt_by_frame.get(frame, []) if box.gt_id == focal]
        if not gt:
            continue
        target = gt[0].bbox
        frame_best = max(
            (iou(target, box.bbox) for box in sequence.tracker_by_frame.get(frame, [])),
            default=0.0,
        )
        if frame_best > best:
            best = frame_best
    return best


def classify_max_iou(value: float) -> str:
    if value <= 0.0:
        return "genuine_non_detection"
    if value < MATCH_IOU:
        return "localization_drift"
    return "iou_at_or_above_match"


def localization_rows(
    events: Sequence[Mapping[str, Any]],
    sequences: Mapping[str, SequenceData],
    causes: Sequence[str],
) -> list[dict[str, Any]]:
    wanted = set(causes)
    rows = []
    for event in events:
        if event.get("cause") not in wanted:
            continue
        sequence = sequences[str(event["sequence"])]
        max_iou = max_iou_during_gap(event, sequence)
        rows.append(
            {
                "event_id": event["event_id"],
                "sequence": event["sequence"],
                "cause": event["cause"],
                "error_type": event.get("error_type", ""),
                "frame_start": event["frame_start"],
                "frame_end": event["frame_end"],
                "focal_gt_id": event["focal_gt_id"],
                "max_iou_during_gap": max_iou,
                "class": classify_max_iou(max_iou),
            }
        )
    return rows


def funnel_rows(n_drift: int) -> list[dict[str, Any]]:
    remaining = FUNNEL_RAW - FUNNEL_CANOPY - FUNNEL_TRUNCATION - FUNNEL_FLICKER
    final = remaining - n_drift
    return [
        {"stage": "visible_miss (raw)", "delta_or_count": FUNNEL_RAW, "label": ""},
        {
            "stage": "unannotated canopy",
            "delta_or_count": -FUNNEL_CANOPY,
            "label": "visible_miss_unconfirmed",
        },
        {
            "stage": "frame truncation",
            "delta_or_count": -FUNNEL_TRUNCATION,
            "label": "visible_miss_truncated",
        },
        {
            "stage": "sub-3-frame flicker",
            "delta_or_count": -FUNNEL_FLICKER,
            "label": "excluded_micro_gap",
        },
        {
            "stage": "localization drift below IoU 0.5",
            "delta_or_count": -n_drift,
            "label": "localization_drift",
        },
        {"stage": "visible_miss (final)", "delta_or_count": final, "label": "headline"},
    ]


def _box_centre(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def ghost_track_row(event: Mapping[str, Any], sequence: SequenceData) -> dict[str, Any]:
    tid = event.get("prev_trk_id")
    centres: list[tuple[float, float]] = []
    ioa_best = 0.0
    n_boxes = 0
    if tid is not None and tid != "":
        track_id = int(tid)
        for frame, boxes in sequence.tracker_by_frame.items():
            track_boxes = [box for box in boxes if int(box.trk_id) == track_id]
            if not track_boxes:
                continue
            occluders = [
                box
                for box in sequence.gt_by_frame.get(frame, [])
                if int(box.cls) in GHOST_OCCLUDER_CLASSES
            ]
            for box in track_boxes:
                n_boxes += 1
                centres.append(_box_centre(box.bbox))
                for occluder in occluders:
                    ioa_best = max(ioa_best, ioa(box.bbox, occluder.bbox))
    centre_var = (
        float(np.var(np.asarray(centres), axis=0).sum()) if len(centres) >= 2 else ""
    )
    stationary = (
        centre_var != "" and float(centre_var) < STATIONARY_VAR_THRESH
    )
    on_occluder = ioa_best > IOA_OCCLUDER_THRESH
    return {
        "event_id": event["event_id"],
        "sequence": event.get("sequence", ""),
        "cause": event.get("cause", ""),
        "prev_trk_id": tid if tid is not None else "",
        "n_track_boxes": n_boxes,
        "centre_var": centre_var,
        "ioa_occluder": ioa_best,
        "prev_track_stationary": stationary,
        "prev_track_on_occluder": on_occluder,
        "likely_ghost": bool(stationary and on_occluder),
        "stationary_var_thresh": STATIONARY_VAR_THRESH,
        "ioa_occluder_thresh": IOA_OCCLUDER_THRESH,
    }


def ghost_track_rows(
    events: Sequence[Mapping[str, Any]], sequences: Mapping[str, SequenceData]
) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        if str(event.get("error_type") or "") != "IDSW":
            continue
        sequence = sequences[str(event["sequence"])]
        rows.append(ghost_track_row(event, sequence))
    return rows


def scaffold_comment_tags(
    path: Path, clip_ids: Sequence[str]
) -> list[dict[str, str]]:
    existing: dict[str, dict[str, str]] = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                existing[row["clip_id"]] = row
    rows = []
    for clip_id in clip_ids:
        prior = existing.get(clip_id, {})
        tag = str(prior.get("tag") or "").strip()
        note = str(prior.get("note_excerpt") or "").strip()
        has_comment = str(prior.get("has_comment") or "").strip().lower() in {
            "1",
            "true",
            "yes",
        } or bool(tag or note)
        rows.append(
            {
                "clip_id": clip_id,
                "has_comment": str(has_comment).lower(),
                "tag": tag,
                "note_excerpt": note,
            }
        )
    write_csv(path, rows, ("clip_id", "has_comment", "tag", "note_excerpt"))
    return rows


def comment_agreement_crosstab(
    comment_rows: Sequence[Mapping[str, str]],
    rows: Sequence[Mapping[str, Any]],
    rater_ids: Sequence[int],
) -> list[dict[str, Any]]:
    by_clip = ratings_by_clip(rows, rater_ids=rater_ids)
    clips = paired_clips(by_clip, rater_ids)
    tagged = {row["clip_id"]: row for row in comment_rows}
    if not any(row.get("has_comment") == "true" for row in comment_rows):
        return []
    index = []
    values = []
    for clip_id in clips:
        has_comment = str(tagged.get(clip_id, {}).get("has_comment") or "false")
        agree = (
            "agree"
            if len({by_clip[clip_id][rater_id]["human_cause"] for rater_id in rater_ids})
            == 1
            else "disagree"
        )
        index.append(has_comment)
        values.append(agree)
    return _crosstab_rows(index, values)


def load_geometry(
    tracker_output: Path, dataset_root: Path
) -> dict[str, SequenceData]:
    sequences, _audit = load_sequences(
        tracker_output, dataset_root, PipelineConfig(render_clips=False)
    )
    return sequences


def rater_ids_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return sorted({int(row["rater_id"]) for row in rows})


def run_validation(
    export_path: Path,
    out_csv: Path,
    *,
    void_clips_path: Path | None = None,
    void_rater_id: int | None = None,
    report_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks = load_export(export_path)
    void = load_void_clip_ids(void_clips_path)
    rows = parse_export(tasks, void_clips=void, void_rater_id=void_rater_id)
    write_validated(out_csv, rows)
    report = validation_report(rows)
    report["export"] = str(export_path)
    report["void_rater_id"] = void_rater_id
    report["n_void_clips"] = len(void)
    destination = report_path or out_csv.with_name("validation_report.json")
    write_json(destination, report)
    return rows, report


def run_agreement(
    *,
    export_path: Path,
    out_dir: Path,
    clip_map_path: Path = DEFAULT_CLIP_MAP,
    clips_meta_dir: Path = DEFAULT_CLIPS_META,
    events_root: Path = DEFAULT_EVENTS_ROOT,
    tracker_output: Path = DEFAULT_TRACKER,
    dataset_root: Path = DEFAULT_DATASET,
    void_clips_path: Path | None = None,
    void_rater_id: int | None = None,
    skip_geometry: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows, z1 = run_validation(
        export_path,
        out_dir / "validated.csv",
        void_clips_path=void_clips_path,
        void_rater_id=void_rater_id,
        report_path=out_dir / "validation_report.json",
    )
    rater_ids = rater_ids_from_rows(rows)
    clip_map = load_clip_map(clip_map_path) if clip_map_path.is_file() else []

    human_rows = human_human_kappa_rows(rows, rater_ids)
    write_csv(
        out_dir / "kappa_human_human.csv",
        human_rows,
        ("cut", "n", "n_raters", "method", "observed_agreement", "kappa", "note"),
    )

    rule_kappa, rule_pairs, confusion, unattr = rule_human_outputs(
        rows, clip_map, rater_ids
    )
    write_csv(
        out_dir / "kappa_rule_human.csv",
        rule_kappa,
        ("cut", "n", "n_raters", "method", "observed_agreement", "kappa", "note"),
    )
    write_csv(
        out_dir / "rule_human_pairs.csv",
        rule_pairs,
        (
            "clip_id",
            "rater_id",
            "rater_name",
            "event_id",
            "error_type",
            "rule_cause",
            "mapped_rule_cause",
            "human_cause",
            "agree",
            "unattributed_fn",
        ),
    )
    confusion_fields = ["mapped_rule_cause"]
    if confusion:
        confusion_fields.extend(
            [key for key in confusion[0] if key not in {"mapped_rule_cause", "row_total"}]
        )
        confusion_fields.append("row_total")
    write_csv(out_dir / "confusion_rule_human.csv", confusion, confusion_fields)
    write_csv(
        out_dir / "unattributed_fn_human_distribution.csv",
        unattr,
        ("rater_id", "rater_name", "human_cause", "n", "share"),
    )

    null_before = load_null_before(clip_map, clips_meta_dir) if clip_map else {}
    z3 = null_before_outputs(rows, rater_ids, null_before) if clip_map else {
        "kappa_rows": [],
        "error_kind_crosstab": [],
        "cause_crosstab": [],
    }
    if z3.get("kappa_rows"):
        human_rows.extend(z3["kappa_rows"])
        write_csv(
            out_dir / "kappa_human_human.csv",
            human_rows,
            ("cut", "n", "n_raters", "method", "observed_agreement", "kappa", "note"),
        )
    kind_crosstab = z3.get("error_kind_crosstab") or []
    cause_crosstab = z3.get("cause_crosstab") or []
    if kind_crosstab:
        fields = ["index", "n"] + [
            key for key in kind_crosstab[0] if key not in {"index", "n"}
        ]
        write_csv(out_dir / "null_before_error_kind_crosstab.csv", kind_crosstab, fields)
    if cause_crosstab:
        fields = ["index", "n"] + [
            key for key in cause_crosstab[0] if key not in {"index", "n"}
        ]
        write_csv(out_dir / "null_before_cause_crosstab.csv", cause_crosstab, fields)
    write_json(
        out_dir / "null_before_report.json",
        {key: value for key, value in z3.items() if key not in {"error_kind_crosstab", "cause_crosstab", "kappa_rows"}},
    )

    loc_rows: list[dict[str, Any]] = []
    ghost_rows: list[dict[str, Any]] = []
    funnel: list[dict[str, Any]] = []
    if not skip_geometry:
        sequences = load_geometry(tracker_output, dataset_root)
        events = load_sidecar_events(events_root)
        loc_rows = localization_rows(
            events, sequences, ("visible_miss", "unattributed_fn")
        )
        write_csv(
            out_dir / "localization_drift.csv",
            loc_rows,
            (
                "event_id",
                "sequence",
                "cause",
                "error_type",
                "frame_start",
                "frame_end",
                "focal_gt_id",
                "max_iou_during_gap",
                "class",
            ),
        )
        n_drift = sum(
            1
            for row in loc_rows
            if row["cause"] == "visible_miss" and row["class"] == "localization_drift"
        )
        funnel = funnel_rows(n_drift)
        write_csv(
            out_dir / "visible_miss_funnel.csv",
            funnel,
            ("stage", "delta_or_count", "label"),
        )
        ghost_rows = ghost_track_rows(events, sequences)
        write_csv(
            out_dir / "ghost_tracks.csv",
            ghost_rows,
            (
                "event_id",
                "sequence",
                "cause",
                "prev_trk_id",
                "n_track_boxes",
                "centre_var",
                "ioa_occluder",
                "prev_track_stationary",
                "prev_track_on_occluder",
                "likely_ghost",
                "stationary_var_thresh",
                "ioa_occluder_thresh",
            ),
        )

    clip_ids = [row["clip_id"] for row in clip_map] or sorted({row["clip_id"] for row in rows})
    comment_path = summary_dir / "comment_tags.csv"
    comment_rows = scaffold_comment_tags(comment_path, clip_ids)
    comment_xtab = comment_agreement_crosstab(comment_rows, rows, rater_ids)
    if comment_xtab:
        fields = ["index", "n"] + [
            key for key in comment_xtab[0] if key not in {"index", "n"}
        ]
        write_csv(out_dir / "comment_agreement_crosstab.csv", comment_xtab, fields)
    free_text_rows = [
        {
            "clip_id": row["clip_id"],
            "rater_id": row["rater_id"],
            "rater_name": row["rater_name"],
            "free_text": row["free_text"],
        }
        for row in rows
        if row.get("free_text")
    ]
    write_csv(
        out_dir / "annotator_free_text.csv",
        free_text_rows,
        ("clip_id", "rater_id", "rater_name", "free_text"),
    )

    summary = {
        "n_raters": len(rater_ids),
        "rater_ids": rater_ids,
        "validation": z1,
        "human_human": human_rows,
        "null_before_elevated": z3.get("null_before_elevated"),
        "n_null_before": z3.get("n_null_before"),
        "n_localization_visible_miss": sum(
            1 for row in loc_rows if row["cause"] == "visible_miss"
        ),
        "n_localization_drift_visible_miss": sum(
            1
            for row in loc_rows
            if row["cause"] == "visible_miss" and row["class"] == "localization_drift"
        ),
        "n_ghost_likely": sum(1 for row in ghost_rows if row["likely_ghost"]),
        "n_idsw_scored": len(ghost_rows),
        "comment_tags_allowed": list(COMMENT_TAGS),
        "funnel": funnel,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def build_validate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a Label Studio annotator export (Z1)"
    )
    parser.add_argument("export", type=Path, nargs="?", default=DEFAULT_EXPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT / "validated.csv")
    parser.add_argument("--void-clips", type=Path, default=None)
    parser.add_argument("--void-rater-id", type=int, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser


def build_agreement_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run annotator agreement analysis Z1–Z6"
    )
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--clip-map", type=Path, default=DEFAULT_CLIP_MAP)
    parser.add_argument("--clips-meta", type=Path, default=DEFAULT_CLIPS_META)
    parser.add_argument("--events-root", type=Path, default=DEFAULT_EVENTS_ROOT)
    parser.add_argument("--tracker-output", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--void-clips", type=Path, default=None)
    parser.add_argument("--void-rater-id", type=int, default=None)
    parser.add_argument(
        "--skip-geometry",
        action="store_true",
        help="Skip Z4/Z5 (no MOT/tracker load)",
    )
    return parser


def main_validate(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_validate_parser().parse_args(argv)
    _rows, report = run_validation(
        args.export,
        args.out,
        void_clips_path=args.void_clips,
        void_rater_id=args.void_rater_id,
        report_path=args.report,
    )
    print(json.dumps({key: report[key] for key in ("n_rows", "n_clips", "n_raters", "raters") if key in report}, indent=2, default=str))
    return report


def main_agreement(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_agreement_parser().parse_args(argv)
    summary = run_agreement(
        export_path=args.export,
        out_dir=args.out_dir,
        clip_map_path=args.clip_map,
        clips_meta_dir=args.clips_meta,
        events_root=args.events_root,
        tracker_output=args.tracker_output,
        dataset_root=args.dataset_root,
        void_clips_path=args.void_clips,
        void_rater_id=args.void_rater_id,
        skip_geometry=args.skip_geometry,
    )
    print(json.dumps(
        {
            "out_dir": str(args.out_dir),
            "n_raters": summary["n_raters"],
            "rater_ids": summary["rater_ids"],
            "n_ghost_likely": summary["n_ghost_likely"],
            "n_localization_drift_visible_miss": summary["n_localization_drift_visible_miss"],
            "null_before_elevated": summary["null_before_elevated"],
        },
        indent=2,
    ))
    return summary
