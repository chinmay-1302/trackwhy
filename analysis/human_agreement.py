"""Cohen's kappa and observer rates from the shipped human CSVs."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

RULE_CAUSE_TO_HUMAN = {
    "full_occl_static": "hidden_by_object",
    "full_occl_pedestrian": "hidden_by_one_person",
    "full_occl_crowd_aggregate": "hidden_by_crowd",
    "partial_occl_pedestrian": "partly_blocked_by_person",
    "partial_occl_static": "partly_blocked_by_static",
    "visible_miss": "clearly_visible_but_missed",
    "visible_miss_truncated": "partly_outside_the_frame",
    "visible_miss_unconfirmed": "clearly_visible_but_missed",
    "track_drift_to_static": "drifted_to_standing_person",
    "unattributed_fn": "cannot_determine",
    "unattributed_idsw": "cannot_determine",
}
IDENTITY_SWAP_RULE_CAUSES = {
    "partial_occl_pedestrian",
    "full_occl_pedestrian",
    "full_occl_crowd_aggregate",
    "partial_occl_static",
    "full_occl_static",
}
ERROR_TYPE_TO_KIND = {
    "FN": "miss",
    "FRAG_NO_SWITCH": "miss",
    "IDSW": "identity_change",
    "FP_ID_HIJACK": "drift",
}


def load_observer(path: Path) -> dict[str, dict[str, str]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["clip_id"]] = row
    return rows


def complete(row: dict[str, str]) -> bool:
    incomplete = str(row.get("incomplete", "")).lower() in {"true", "1"}
    return (not incomplete) and bool(row.get("human_cause"))


def paired_clip_ids(left: dict[str, dict], right: dict[str, dict]) -> list[str]:
    return sorted(
        clip_id
        for clip_id in left
        if clip_id in right and complete(left[clip_id]) and complete(right[clip_id])
    )


def cohen_kappa(left_labels: list[str], right_labels: list[str]) -> float:
    if len(left_labels) != len(right_labels) or not left_labels:
        raise ValueError("Cohen's kappa needs paired non-empty labels")
    n = len(left_labels)
    observed = sum(a == b for a, b in zip(left_labels, right_labels)) / n
    labels = set(left_labels) | set(right_labels)
    left_counts = Counter(left_labels)
    right_counts = Counter(right_labels)
    expected = sum(left_counts[label] * right_counts[label] for label in labels) / (
        n * n
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def no_error_rate(rows: dict[str, dict]) -> float:
    return sum(row["error_kind"] == "no_error_visible" for row in rows.values()) / len(
        rows
    )


def map_rule_to_human(rule_cause: str, error_type: str) -> str:
    kind = ERROR_TYPE_TO_KIND.get(error_type)
    cause = str(rule_cause or "").strip()
    if kind == "identity_change":
        if cause in IDENTITY_SWAP_RULE_CAUSES:
            return "swapped_during_crossing"
        if cause == "unattributed_idsw":
            return "changed_for_no_visible_reason"
        if cause == "reentry":
            return "left_and_returned"
    if kind == "drift" and cause == "track_drift_to_static":
        return "drifted_to_standing_person"
    return RULE_CAUSE_TO_HUMAN.get(cause, cause)


def observer_table(
    observer_a: dict[str, dict],
    observer_b: dict[str, dict],
) -> dict[str, float | int]:
    clips = paired_clip_ids(observer_a, observer_b)
    labels_a = [observer_a[clip]["human_cause"] for clip in clips]
    labels_b = [observer_b[clip]["human_cause"] for clip in clips]
    return {
        "n_paired": len(clips),
        "agreement": sum(a == b for a, b in zip(labels_a, labels_b)) / len(clips),
        "kappa": cohen_kappa(labels_a, labels_b),
        "observer_a_no_error_rate": no_error_rate(observer_a),
        "observer_b_no_error_rate": no_error_rate(observer_b),
        "no_error_visible_kappa": cohen_kappa(
            [label == "no_error_visible" for label in labels_a],
            [label == "no_error_visible" for label in labels_b],
        ),
    }


def rule_observer_kappa(
    observer: dict[str, dict],
    clip_causes: dict[str, dict],
    paired: list[str],
) -> dict[str, float | int]:
    rule_labels = []
    human_labels = []
    for clip_id in paired:
        info = clip_causes[clip_id]
        if clip_id not in observer or not complete(observer[clip_id]):
            continue
        if info["cause"] == "unattributed_fn":
            continue
        rule_labels.append(map_rule_to_human(info["cause"], info["error_type"]))
        human_labels.append(observer[clip_id]["human_cause"])
    return {
        "n": len(rule_labels),
        "agreement": sum(a == b for a, b in zip(rule_labels, human_labels))
        / len(rule_labels),
        "kappa": cohen_kappa(rule_labels, human_labels),
    }


def verifier_flips(forward: list[dict], informed: list[dict]) -> dict[str, int]:
    informed_map = {row["clip_id"]: row["primary_error"] for row in informed}
    flips = [
        row["clip_id"]
        for row in forward
        if row["primary_error"] == "no_error"
        and informed_map.get(row["clip_id"]) == "miss"
    ]
    return {
        "n_forward": len(forward),
        "n_no_error_forward": sum(row["primary_error"] == "no_error" for row in forward),
        "n_flips_to_miss": len(flips),
    }
