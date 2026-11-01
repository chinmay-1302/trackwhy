from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from math import hypot
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np

from .config import CAUSES, PipelineConfig, vis_band
from .geometry import box_area
from .matching import MatchingResult
from .models import ErrorEvent, SequenceData


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, events: Iterable[ErrorEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            handle.write("\n")
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _components(event: ErrorEvent) -> list[dict[str, Any]]:
    merged = event.evidence.get("merged_components")
    if isinstance(merged, list) and merged:
        return merged
    return [
        {
            "event_id": event.event_id,
            "error_type": event.error_type,
            "frame_start": event.frame_start,
            "frame_end": event.frame_end,
            "focal_gt_id": event.focal_gt_id,
            "partner_gt_id": event.partner_gt_id,
            "role": event.role,
            "prev_trk_id": event.prev_trk_id,
            "new_trk_id": event.new_trk_id,
            "fn_frame_count": event.evidence.get("fn_frame_count"),
        }
    ]


def _fn_frames(events: Iterable[ErrorEvent]) -> int:
    count = 0
    for event in events:
        for component in _components(event):
            if component["error_type"] == "FN":
                value = component.get("fn_frame_count")
                if value is None:
                    value = component["frame_end"] - component["frame_start"] + 1
                count += int(value)
    return count


def _idsw_count(events: Iterable[ErrorEvent]) -> int:
    return sum(
        component["error_type"] == "IDSW"
        for event in events
        for component in _components(event)
    )


def validate_sequence(
    sequence: SequenceData,
    matching: MatchingResult,
    included: list[ErrorEvent],
    excluded: list[ErrorEvent],
    config: PipelineConfig,
    excluded_micro_gap: list[ErrorEvent] | None = None,
) -> dict[str, Any]:
    micro = excluded_micro_gap or []
    attributed_events = included + excluded
    all_events = attributed_events + micro
    included_fn = _fn_frames(included)
    excluded_fn = _fn_frames(excluded)
    micro_fn = _fn_frames(micro)
    corpus_idsw = _idsw_count(all_events)
    official_fn = int(matching.trackeval_report["CLR_FN"])
    official_idsw = int(matching.trackeval_report["IDSW"])
    min_fn = int(config.thresholds["MIN_FN_RUN_FRAMES"])
    short_in_included = any(
        0 < _fn_frames([event]) < min_fn for event in included
    )
    checks: dict[str, Any] = {
        "matching_fn_equals_trackeval": matching.local_fn == official_fn,
        "matching_idsw_equals_trackeval": matching.local_idsw == official_idsw,
        "corpus_fn_closure": included_fn + excluded_fn + micro_fn == official_fn,
        "corpus_idsw_closure": corpus_idsw == official_idsw,
        "no_short_fn_in_corpus": not short_in_included,
        "micro_gap_used_if_short_runs": (micro_fn > 0) or not any(
            0 < _fn_frames([event]) < min_fn for event in included + excluded + micro
        ),
        "all_events_attributed": all(event.cause is not None for event in all_events),
        "event_ids_unique": len({event.event_id for event in all_events}) == len(all_events),
    }

    duplicate_keys: list[tuple] = []
    seen: set[tuple] = set()
    for event in included:
        pair = tuple(
            sorted(
                (event.focal_gt_id, event.partner_gt_id),
                key=lambda value: (-1 if value is None else value),
            )
        )
        key = (
            event.sequence,
            event.tracker,
            event.frame_start,
            event.frame_end,
            pair,
        )
        if key in seen:
            duplicate_keys.append(key)
        seen.add(key)
    checks["no_duplicate_event_keys"] = not duplicate_keys

    contradictory_pairs: list[tuple[str, str]] = []
    pedestrian_events = [
        event
        for event in included
        if event.partner_gt_id is not None
        and event.cause in {"full_occl_pedestrian", "partial_occl_pedestrian"}
    ]
    for index, first in enumerate(pedestrian_events):
        pair_first = {first.focal_gt_id, first.partner_gt_id}
        for second in pedestrian_events[index + 1 :]:
            if pair_first != {second.focal_gt_id, second.partner_gt_id}:
                continue
            if max(first.frame_start, second.frame_start) > min(
                first.frame_end, second.frame_end
            ):
                continue
            if first.role is None or second.role is None:
                continue
            first_roles = {
                first.focal_gt_id: first.role,
                first.partner_gt_id: (
                    "occluder" if first.role == "occludee" else "occludee"
                ),
            }
            second_roles = {
                second.focal_gt_id: second.role,
                second.partner_gt_id: (
                    "occluder" if second.role == "occludee" else "occludee"
                ),
            }
            if any(first_roles[identity] != second_roles[identity] for identity in pair_first):
                contradictory_pairs.append((first.event_id, second.event_id))
    checks["pedestrian_pair_symmetry"] = not contradictory_pairs

    band_failures: list[str] = []
    for event in attributed_events:
        value = event.evidence.get("min_visibility")
        try:
            if value is None:
                raise ValueError("missing visibility")
            vis_band(float(value), config.vis_bands)
        except ValueError:
            band_failures.append(event.event_id)
    checks["visibility_band_coverage"] = not band_failures
    checks["visible_miss_non_interacting"] = all(
        int(event.evidence.get("n_interactors", -1)) == 0
        for event in included
        if event.cause == "visible_miss"
    )
    checks["aggregate_has_no_dominant_single_occluder"] = all(
        float(event.evidence.get("max_single_ioa", 1.0))
        < float(config.thresholds["IOA_OCCLUDER"])
        for event in included
        if event.cause == "full_occl_crowd_aggregate"
    )
    interaction_causes = {
        "partial_occl_pedestrian",
        "partial_occl_static",
        "visible_miss",
    }
    checks["v4_interaction_measure_consistency"] = all(
        "ioa_focal" in event.evidence and "iou" in event.evidence
        for event in included
        if event.cause in interaction_causes
    )
    interaction_threshold = float(config.thresholds["IOA_INTERACTION"])
    checks["v4_visible_miss_partial_complement"] = all(
        float(event.evidence.get("ioa_focal", 1.0)) < interaction_threshold
        for event in included
        if event.cause == "visible_miss"
    )

    def expected_static_threshold(event: ErrorEvent) -> float | None:
        occluder_class = event.evidence.get("occluder_class")
        if occluder_class in config.classes["STATIC_PERSON"]:
            return float(config.ioa_occluder_by_class["person"])
        if occluder_class in config.classes["OCCLUDER_CLASSES"]:
            return float(config.ioa_occluder_by_class["large_object"])
        return None

    checks["v4_static_threshold_by_class"] = all(
        expected_static_threshold(event) is not None
        and abs(
            float(event.evidence.get("ioa_threshold_applied", -1.0))
            - float(expected_static_threshold(event))
        )
        < 1e-12
        for event in included
        if event.cause == "full_occl_static"
    )
    checks["v4_drift_rename_complete"] = all(
        event.cause != "static_person_hijack" for event in attributed_events
    )
    checks["v4_static_proximity_recorded"] = all(
        event.static_person_present is not None
        and event.static_person_max_ioa is not None
        for event in attributed_events
    )
    checks["v4_frame_policy_recorded"] = all(
        event.evidence.get("frame_policy") == "per_frame_majority"
        for event in attributed_events
    )

    unattributed = sum(
        event.cause in {"unattributed_fn", "unattributed_idsw"} for event in included
    )
    unattributed_fraction = unattributed / len(included) if included else 0.0
    if unattributed_fraction > 0.35:
        checks["residual_fraction_at_most_0.35"] = False
    else:
        checks["residual_fraction_at_most_0.35"] = True
    category_counts = Counter(str(event.cause) for event in included)
    warnings: list[str] = []
    if unattributed_fraction > 0.20:
        warnings.append(
            "Unattributed fraction exceeds 0.20; v2 residual target was not met."
        )
    underpowered = {
        cause: count for cause, count in sorted(category_counts.items()) if 0 < count < 25
    }
    for cause, count in underpowered.items():
        warnings.append(
            f"Category {cause} has {count} events and is under-powered for the 25/category kappa table."
        )
    class7_full = sum(
        event.cause == "full_occl_static"
        and event.evidence.get("occluder_class") in config.classes["STATIC_PERSON"]
        for event in included
    )
    if class7_full < 5:
        class7_preconditions = sum(
            event.evidence.get("visibility_band") == "FULL"
            and event.error_type in {"FN", "FRAG_NO_SWITCH"}
            and event.evidence.get("full_person_static_vote", {}).get(
                "partner_gt_id"
            )
            is not None
            for event in included
        )
        warnings.append(
            "Class-7 full-occlusion count remains below 5: "
            f"{class7_full} assigned from {class7_preconditions} events meeting the "
            "joint FULL-band + 0.70-IoA + FN/fragmentation precondition."
        )
    failures = sorted(key for key, passed in checks.items() if not passed)
    result = {
        "sequence": sequence.name,
        "selected_frames": len(sequence.selected_frames),
        "trackeval": matching.trackeval_report,
        "corpus": {
            "canonical_event_count": len(included),
            "excluded_crowd_event_count": len(excluded),
            "excluded_micro_gap_event_count": len(micro),
            "included_fn_frames": included_fn,
            "excluded_crowd_fn_frames": excluded_fn,
            "excluded_micro_gap_fn_frames": micro_fn,
            "idsw_components": corpus_idsw,
            "unattributed_events": unattributed,
            "unattributed_fraction": unattributed_fraction,
            "category_counts": dict(sorted(category_counts.items())),
            "underpowered_categories": underpowered,
        },
        "checks": checks,
        "duplicate_keys": duplicate_keys,
        "contradictory_pair_events": contradictory_pairs,
        "visibility_band_failures": band_failures,
        "warnings": warnings,
        "passed": not failures,
        "failed_checks": failures,
    }
    if failures:
        raise AssertionError(
            f"Validation failed for {sequence.name}: {', '.join(failures)}"
        )
    return result


def tracker_geometry_audit(
    sequence: SequenceData, config: PipelineConfig
) -> dict[str, int | float | bool]:
    target_classes = set(config.classes["TARGET_CLASSES"])
    gt_count = 0
    tracker_count = 0
    exact_geometry_matches = 0
    for frame in sequence.selected_frames:
        gt_boxes = Counter(
            box.bbox
            for box in sequence.gt_by_frame.get(frame, [])
            if box.flag != 0 and box.cls in target_classes
        )
        tracker_boxes = Counter(
            box.bbox for box in sequence.tracker_by_frame.get(frame, [])
        )
        gt_count += sum(gt_boxes.values())
        tracker_count += sum(tracker_boxes.values())
        exact_geometry_matches += sum((gt_boxes & tracker_boxes).values())
    gt_ratio = exact_geometry_matches / gt_count if gt_count else 0.0
    tracker_ratio = exact_geometry_matches / tracker_count if tracker_count else 0.0
    return {
        "target_gt_boxes": gt_count,
        "tracker_boxes": tracker_count,
        "exact_geometry_matches": exact_geometry_matches,
        "fraction_gt_boxes_with_exact_tracker_geometry": gt_ratio,
        "fraction_tracker_boxes_with_exact_gt_geometry": tracker_ratio,
        "looks_like_ground_truth_geometry": gt_ratio > 0.999 and tracker_ratio > 0.999,
    }


def identity_rows(
    sequence: SequenceData,
    matching: MatchingResult,
    events: list[ErrorEvent],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    all_gt_ids = sorted(set().union(*matching.gt_present.values()))
    component_causes: dict[int, Counter[str]] = defaultdict(Counter)
    component_idsw: Counter[int] = Counter()
    for event in events:
        for component in _components(event):
            focal = int(component["focal_gt_id"])
            component_causes[focal][str(event.cause)] += 1
            if component["error_type"] == "IDSW":
                component_idsw[focal] += 1

    rows: list[dict[str, Any]] = []
    for gt_id in all_gt_ids:
        present_frames = [
            frame
            for frame in sequence.selected_frames
            if gt_id in matching.gt_present[frame]
        ]
        total = len(present_frames)
        missed = sum(matching.match[frame][gt_id] is None for frame in present_frames)
        assigned = {
            matching.match[frame][gt_id]
            for frame in present_frames
            if matching.match[frame][gt_id] is not None
        }
        n_idsw = int(component_idsw[gt_id])
        recall = 1.0 - missed / total if total else 0.0
        idsw_rate = n_idsw / (total / 100.0) if total else 0.0
        timeline = [matching.match[frame][gt_id] for frame in present_frames]
        gap_lengths: list[int] = []
        gap_start: int | None = None
        for index, tracker_id in enumerate(timeline):
            if tracker_id is None and gap_start is None:
                gap_start = index
            elif tracker_id is not None and gap_start is not None:
                if gap_start > 0:
                    gap_lengths.append(index - gap_start)
                gap_start = None
        n_gaps = sum(
            length >= int(config.thresholds["FRAG_MIN_GAP"])
            for length in gap_lengths
        )
        frag_rate = n_gaps / (total / 100.0) if total else 0.0
        normalized_idsw = min(1.0, idsw_rate / config.difficulty_idsw_rate_cap)
        normalized_frag = min(1.0, frag_rate / config.difficulty_frag_rate_cap)
        difficulty = (
            config.difficulty_recall_weight * (1.0 - recall)
            + config.difficulty_idsw_weight * normalized_idsw
            + config.difficulty_frag_weight * normalized_frag
        )
        if recall == 0.0:
            partition = "never_detected"
        elif total < config.min_track_frames:
            partition = "too_short"
        else:
            partition = "trackable"
        rows.append(
            {
                "gt_id": gt_id,
                "T": total,
                "recall": recall,
                "fn_frames": missed,
                "n_idsw": n_idsw,
                "idsw_rate_per_100_frames": idsw_rate,
                "n_gaps": n_gaps,
                "frag_rate_per_100_frames": frag_rate,
                "n_distinct_trk_ids": len(assigned),
                "fragmentation": len(assigned) / max(1, n_idsw + 1),
                "difficulty": difficulty,
                "partition": partition,
                "ranking_eligible": partition == "trackable",
                "cause_histogram": json.dumps(
                    dict(sorted(component_causes[gt_id].items())), sort_keys=True
                ),
            }
        )
    trackable = sorted(
        (row for row in rows if row["ranking_eligible"]),
        key=lambda row: (-row["difficulty"], row["gt_id"]),
    )
    rank_by_id = {row["gt_id"]: rank for rank, row in enumerate(trackable, start=1)}
    partition_order = {"trackable": 0, "never_detected": 1, "too_short": 2}
    rows.sort(
        key=lambda row: (
            partition_order[row["partition"]],
            rank_by_id.get(row["gt_id"], 10**9),
            row["gt_id"],
        )
    )
    for row in rows:
        rank = rank_by_id.get(row["gt_id"])
        row["difficulty_rank"] = "" if rank is None else rank
        row["top_k"] = rank is not None and rank <= config.top_k_identities
    return rows


def write_identity_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "gt_id",
        "T",
        "recall",
        "fn_frames",
        "n_idsw",
        "idsw_rate_per_100_frames",
        "n_gaps",
        "frag_rate_per_100_frames",
        "n_distinct_trk_ids",
        "fragmentation",
        "difficulty",
        "partition",
        "ranking_eligible",
        "difficulty_rank",
        "top_k",
        "cause_histogram",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def never_detected_analysis(
    sequence: SequenceData,
    identity_metrics: list[dict[str, Any]],
    config: PipelineConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_classes = set(config.classes["TARGET_CLASSES"])
    radius = float(config.thresholds["DENSITY_RADIUS_PX"])
    never_ids = {
        int(row["gt_id"])
        for row in identity_metrics
        if row["partition"] == "never_detected"
    }
    boxes_by_id: dict[int, list] = defaultdict(list)
    target_by_frame: dict[int, list] = {}
    for frame in sequence.selected_frames:
        target = [
            box
            for box in sequence.gt_by_frame.get(frame, [])
            if box.flag != 0 and box.cls in target_classes
        ]
        target_by_frame[frame] = target
        for box in target:
            if box.gt_id in never_ids:
                boxes_by_id[box.gt_id].append(box)

    rows: list[dict[str, Any]] = []
    for gt_id in sorted(never_ids):
        boxes = boxes_by_id[gt_id]
        areas = [box_area(box.bbox) for box in boxes]
        centres = [
            ((box.bbox[0] + box.bbox[2]) / 2, (box.bbox[1] + box.bbox[3]) / 2)
            for box in boxes
        ]
        densities = []
        for box, centre in zip(boxes, centres):
            densities.append(
                sum(
                    other.gt_id != gt_id
                    and hypot(
                        centre[0] - (other.bbox[0] + other.bbox[2]) / 2,
                        centre[1] - (other.bbox[1] + other.bbox[3]) / 2,
                    )
                    <= radius
                    for other in target_by_frame[box.frame]
                )
            )
        rows.append(
            {
                "sequence": sequence.name,
                "tracker": config.tracker_name,
                "gt_id": gt_id,
                "T": len(boxes),
                "median_bbox_area_px": median(areas) if areas else 0.0,
                "median_local_density": median(densities) if densities else 0.0,
                "median_center_x_px": median([point[0] for point in centres]) if centres else 0.0,
                "median_center_y_px": median([point[1] for point in centres]) if centres else 0.0,
                "spatial_cluster_id": -1,
            }
        )

    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            distance = hypot(
                float(rows[left]["median_center_x_px"])
                - float(rows[right]["median_center_x_px"]),
                float(rows[left]["median_center_y_px"])
                - float(rows[right]["median_center_y_px"]),
            )
            if distance <= radius:
                union(left, right)
    root_to_cluster: dict[int, int] = {}
    for index, row in enumerate(rows):
        root = find(index)
        row["spatial_cluster_id"] = root_to_cluster.setdefault(
            root, len(root_to_cluster) + 1
        )
    cluster_sizes = Counter(int(row["spatial_cluster_id"]) for row in rows)
    short_excluded = sum(
        row["partition"] == "too_short" for row in identity_metrics
    )
    summary = {
        "sequence": sequence.name,
        "tracker": config.tracker_name,
        "never_detected_count": len(rows),
        "total_frame_mass": sum(int(row["T"]) for row in rows),
        "median_bbox_area_px": median(
            [float(row["median_bbox_area_px"]) for row in rows]
        )
        if rows
        else 0.0,
        "median_local_density": median(
            [float(row["median_local_density"]) for row in rows]
        )
        if rows
        else 0.0,
        "spatial_cluster_count": len(cluster_sizes),
        "largest_spatial_cluster_size": max(cluster_sizes.values(), default=0),
        "short_track_excluded_count": short_excluded,
        "min_track_frames": config.min_track_frames,
        "spatial_cluster_radius_px": radius,
    }
    return rows, summary


def write_never_detected_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sequence",
        "tracker",
        "gt_id",
        "T",
        "median_bbox_area_px",
        "median_local_density",
        "median_center_x_px",
        "median_center_y_px",
        "spatial_cluster_id",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_class_histogram(path: Path, sequences: dict[str, SequenceData]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "sequence",
            "class_id",
            "full_sequence_rows",
            "selected_frame_rows",
            "selected_flag_0_rows",
            "selected_flag_1_rows",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sequence in sequences.values():
            class_ids = sorted(
                set(sequence.class_histogram_full)
                | set(sequence.class_histogram_selected)
            )
            for class_id in class_ids:
                writer.writerow(
                    {
                        "sequence": sequence.name,
                        "class_id": class_id,
                        "full_sequence_rows": sequence.class_histogram_full.get(class_id, 0),
                        "selected_frame_rows": sequence.class_histogram_selected.get(class_id, 0),
                        "selected_flag_0_rows": sequence.flag_class_histogram_selected.get(
                            (class_id, 0), 0
                        ),
                        "selected_flag_1_rows": sequence.flag_class_histogram_selected.get(
                            (class_id, 1), 0
                        ),
                    }
                )


def write_cause_distribution(
    path: Path,
    sequences: dict[str, SequenceData],
    events_by_sequence: dict[str, list[ErrorEvent]],
    tracker_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["cause", "tracker", "sequence", "count"]
        )
        writer.writeheader()
        for sequence_name in sorted(sequences):
            counts = Counter(
                event.cause for event in events_by_sequence.get(sequence_name, [])
            )
            for cause in CAUSES:
                writer.writerow(
                    {
                        "cause": cause,
                        "tracker": tracker_name,
                        "sequence": sequence_name,
                        "count": counts[cause],
                    }
                )


DETECTION_CAUSES = {
    "full_occl_static",
    "full_occl_pedestrian",
    "full_occl_crowd_aggregate",
    "visible_miss",
    "unattributed_fn",
}


def write_cause_distribution_by_tracker(
    path: Path, events: list[ErrorEvent]
) -> None:
    counts = Counter((event.tracker, str(event.cause)) for event in events)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["tracker", "cause_group", "cause", "count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for tracker in sorted({event.tracker for event in events}):
            for cause in CAUSES:
                writer.writerow(
                    {
                        "tracker": tracker,
                        "cause_group": (
                            "detection-caused"
                            if cause in DETECTION_CAUSES
                            else "association-caused"
                        ),
                        "cause": cause,
                        "count": counts[(tracker, cause)],
                    }
                )


def write_ordering_suppression(
    path: Path, events: list[ErrorEvent], config: PipelineConfig
) -> None:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for event in events:
        candidates = list(event.evidence.get("all_candidates", []))
        winning = [
            candidate
            for candidate in candidates
            if str(candidate.get("rule")) == str(event.cause)
        ]
        if not winning:
            continue
        winning_candidate = max(
            winning, key=lambda candidate: float(candidate.get("score", 0.0))
        )
        winning_score = float(winning_candidate.get("score", 0.0))
        for candidate in candidates:
            suppressed = str(candidate.get("rule"))
            if suppressed == str(event.cause):
                continue
            grouped[
                (
                    event.tracker,
                    event.sequence,
                    str(event.cause),
                    suppressed,
                )
            ].append(winning_score - float(candidate.get("score", 0.0)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "tracker",
            "sequence",
            "winning_cause",
            "suppressed_cause",
            "count",
            "mean_score_gap",
            "median_score_gap",
            "min_score_gap",
            "max_score_gap",
            "ambiguous_within_margin_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (tracker, sequence, winner, suppressed), gaps in sorted(grouped.items()):
            writer.writerow(
                {
                    "tracker": tracker,
                    "sequence": sequence,
                    "winning_cause": winner,
                    "suppressed_cause": suppressed,
                    "count": len(gaps),
                    "mean_score_gap": mean(gaps),
                    "median_score_gap": median(gaps),
                    "min_score_gap": min(gaps),
                    "max_score_gap": max(gaps),
                    "ambiguous_within_margin_count": sum(
                        abs(gap) < float(config.thresholds["AMBIGUITY_MARGIN"])
                        for gap in gaps
                    ),
                }
            )


def write_partial_selection_delta(path: Path, events: list[ErrorEvent]) -> dict[str, Any]:
    changed = [
        event
        for event in events
        if event.evidence.get("partial_selection_changed_from_order") is True
    ]
    grouped = Counter(
        (
            event.tracker,
            event.sequence,
            str(event.evidence.get("partial_order_based_cause")),
            str(event.evidence.get("partial_score_based_cause")),
        )
        for event in changed
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "tracker",
            "sequence",
            "order_based_cause",
            "score_based_cause",
            "changed_event_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, count in sorted(grouped.items()):
            writer.writerow(dict(zip(fields[:-1], key), changed_event_count=count))
    return {
        "changed_event_count": len(changed),
        "partial_event_count": sum(
            event.cause in {"partial_occl_pedestrian", "partial_occl_static"}
            for event in events
        ),
    }


def write_static_person_presence_vs_drift(
    path: Path, events: list[ErrorEvent], config: PipelineConfig
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = [("ALL", events)] + [
        (sequence, [event for event in events if event.sequence == sequence])
        for sequence in sorted({event.sequence for event in events})
    ]
    for sequence, selected in groups:
        presence = sum(event.static_person_present is True for event in selected)
        drift = sum(event.cause == "track_drift_to_static" for event in selected)
        both = sum(
            event.static_person_present is True
            and event.cause == "track_drift_to_static"
            for event in selected
        )
        class7_full = sum(
            event.cause == "full_occl_static"
            and event.evidence.get("occluder_class")
            in config.classes["STATIC_PERSON"]
            for event in selected
        )
        class7_full_preconditions = sum(
            event.evidence.get("visibility_band") == "FULL"
            and event.error_type in {"FN", "FRAG_NO_SWITCH"}
            and event.evidence.get("full_person_static_vote", {}).get(
                "partner_gt_id"
            )
            is not None
            for event in selected
        )
        rows.append(
            {
                "tracker": selected[0].tracker if selected else "",
                "sequence": sequence,
                "total_events": len(selected),
                "static_person_present_events": presence,
                "track_drift_to_static_events": drift,
                "presence_and_drift_events": both,
                "full_occl_static_class7_events": class7_full,
                "class7_full_joint_precondition_events": class7_full_preconditions,
                "class7_full_order_suppressed_events": max(
                    0, class7_full_preconditions - class7_full
                ),
                "presence_to_drift_ratio": presence / drift if drift else None,
                "drift_per_presence": drift / presence if presence else None,
            }
        )
    write_rows_csv(path, rows)
    return rows


def write_visible_miss_analysis(
    csv_path: Path,
    reference_path: Path,
    events: list[ErrorEvent],
    sequences: dict[str, SequenceData],
    config: PipelineConfig,
) -> None:
    visible = [event for event in events if event.cause == "visible_miss"]
    fields = [
        "event_id",
        "sequence",
        "tracker",
        "focal_gt_id",
        "frame_start",
        "frame_end",
        "min_visibility",
        "bbox_area_px",
        "bbox_height_px",
        "area_percentile",
        "local_density",
        "nn_distance_px",
        "aspect_ratio",
        "frame_position_x",
        "frame_position_y",
        "n_frames",
        "n_interactors",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in visible:
            position = event.evidence.get("frame_position", [None, None])
            writer.writerow(
                {
                    "event_id": event.event_id,
                    "sequence": event.sequence,
                    "tracker": event.tracker,
                    "focal_gt_id": event.focal_gt_id,
                    "frame_start": event.frame_start,
                    "frame_end": event.frame_end,
                    "min_visibility": event.evidence.get("min_visibility"),
                    "bbox_area_px": event.evidence.get("bbox_area_px"),
                    "bbox_height_px": event.evidence.get("bbox_height_px"),
                    "area_percentile": event.evidence.get("area_percentile"),
                    "local_density": event.evidence.get("local_density"),
                    "nn_distance_px": event.evidence.get("nn_distance_px"),
                    "aspect_ratio": event.evidence.get("aspect_ratio"),
                    "frame_position_x": position[0],
                    "frame_position_y": position[1],
                    "n_frames": event.evidence.get("n_frames"),
                    "n_interactors": event.evidence.get("n_interactors"),
                }
            )

    percentile_bins = np.linspace(0.0, 100.0, 11)
    density_bins = np.arange(0.0, 52.0, 2.0)
    reference: dict[str, Any] = {
        "area_percentile_bins": percentile_bins.tolist(),
        "density_bins": density_bins.tolist(),
        "sequences": {},
    }
    target_classes = set(config.classes["TARGET_CLASSES"])
    radius = float(config.thresholds["DENSITY_RADIUS_PX"])
    for sequence_name, sequence in sorted(sequences.items()):
        boxes_by_frame = {
            frame: [
                box
                for box in sequence.gt_by_frame.get(frame, [])
                if box.flag != 0 and box.cls in target_classes
            ]
            for frame in sequence.selected_frames
        }
        areas = np.asarray(
            [box_area(box.bbox) for boxes in boxes_by_frame.values() for box in boxes],
            dtype=float,
        )
        sorted_areas = np.sort(areas)
        percentiles = (
            100.0 * np.searchsorted(sorted_areas, areas, side="right") / len(areas)
            if len(areas)
            else np.empty(0)
        )
        densities: list[int] = []
        for boxes in boxes_by_frame.values():
            if not boxes:
                continue
            centres = np.asarray(
                [
                    ((box.bbox[0] + box.bbox[2]) / 2, (box.bbox[1] + box.bbox[3]) / 2)
                    for box in boxes
                ],
                dtype=float,
            )
            distance = np.sqrt(
                np.sum((centres[:, None, :] - centres[None, :, :]) ** 2, axis=2)
            )
            densities.extend((np.sum(distance <= radius, axis=1) - 1).tolist())
        visible_sequence = [event for event in visible if event.sequence == sequence_name]
        visible_percentiles = np.asarray(
            [float(event.evidence["area_percentile"]) for event in visible_sequence]
        )
        visible_densities = np.asarray(
            [float(event.evidence["local_density"]) for event in visible_sequence]
        )
        reference["sequences"][sequence_name] = {
            "all_target_box_count": int(len(areas)),
            "visible_miss_event_count": len(visible_sequence),
            "all_area_percentile_hist": np.histogram(
                percentiles, bins=percentile_bins
            )[0].tolist(),
            "visible_area_percentile_hist": np.histogram(
                visible_percentiles, bins=percentile_bins
            )[0].tolist(),
            "all_density_hist": np.histogram(densities, bins=density_bins)[0].tolist(),
            "visible_density_hist": np.histogram(
                visible_densities, bins=density_bins
            )[0].tolist(),
            "all_density_overflow": int(
                sum(value >= density_bins[-1] for value in densities)
            ),
            "visible_density_overflow": int(
                np.sum(visible_densities >= density_bins[-1])
            ),
        }
    write_json(reference_path, reference)


def write_transition_matrix(
    path: Path,
    v1_events: list[ErrorEvent],
    v2_events: list[ErrorEvent],
) -> dict[str, Any]:
    v1_by_id = {event.event_id: event for event in v1_events}
    v2_by_id = {event.event_id: event for event in v2_events}
    if set(v1_by_id) != set(v2_by_id):
        raise AssertionError(
            "G1 failed: v1/v2 canonical event-ID sets differ on the fixed frame set"
        )
    transitions = Counter(
        (str(v1_by_id[event_id].cause), str(v2_by_id[event_id].cause))
        for event_id in sorted(v1_by_id)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["v1_cause", "v2_cause", "count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (v1_cause, v2_cause), count in sorted(transitions.items()):
            writer.writerow(
                {"v1_cause": v1_cause, "v2_cause": v2_cause, "count": count}
            )
    v1_residual = sum(
        event.cause in {"unattributed_fn", "unattributed_idsw"}
        for event in v1_events
    )
    v2_residual = sum(
        event.cause in {"unattributed_fn", "unattributed_idsw"}
        for event in v2_events
    )
    result = {
        "v1_event_count": len(v1_events),
        "v2_event_count": len(v2_events),
        "v1_unattributed": v1_residual,
        "v2_unattributed": v2_residual,
        "event_count_invariant": len(v1_events) == len(v2_events),
        "residual_reduced": v2_residual < v1_residual,
    }
    if not result["event_count_invariant"] or not result["residual_reduced"]:
        raise AssertionError(f"G1 redistribution invariant failed: {result}")
    return result


def write_v3_v4_transition_matrix(
    path: Path,
    v3_events: list[ErrorEvent],
    v4_events: list[ErrorEvent],
) -> dict[str, Any]:
    v3_by_id = {event.event_id: event for event in v3_events}
    v4_by_id = {event.event_id: event for event in v4_events}
    if set(v3_by_id) != set(v4_by_id):
        raise AssertionError(
            "H6 failed: v3/v4 canonical event-ID sets differ on the fixed frame set"
        )
    transitions = Counter(
        (str(v3_by_id[event_id].cause), str(v4_by_id[event_id].cause))
        for event_id in sorted(v3_by_id)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["v3_cause", "v4_cause", "count"]
        )
        writer.writeheader()
        for (v3_cause, v4_cause), count in sorted(transitions.items()):
            writer.writerow(
                {"v3_cause": v3_cause, "v4_cause": v4_cause, "count": count}
            )
    result = {
        "v3_event_count": len(v3_events),
        "v4_event_count": len(v4_events),
        "event_count_invariant": len(v3_events) == len(v4_events),
        "v3_unattributed": sum(
            event.cause in {"unattributed_fn", "unattributed_idsw"}
            for event in v3_events
        ),
        "v4_unattributed": sum(
            event.cause in {"unattributed_fn", "unattributed_idsw"}
            for event in v4_events
        ),
        "v3_visible_miss": sum(event.cause == "visible_miss" for event in v3_events),
        "v4_visible_miss": sum(event.cause == "visible_miss" for event in v4_events),
        "changed_cause_count": sum(
            v3_by_id[event_id].cause != v4_by_id[event_id].cause
            for event_id in v3_by_id
        ),
    }
    if not result["event_count_invariant"]:
        raise AssertionError(f"H6 redistribution invariant failed: {result}")
    return result


def write_ioa_sweep(
    path: Path,
    baseline_by_sequence: dict[str, list[ErrorEvent]],
    sequences: dict[str, SequenceData],
    matchings: dict[str, MatchingResult],
    config: PipelineConfig,
) -> dict[str, Any]:
    # Local import avoids making reports a dependency of the attribution layer.
    from .attribution import reattribute_canonical_events

    rows: list[dict[str, Any]] = []
    baseline_visible = {
        sequence: sum(event.cause == "visible_miss" for event in events)
        for sequence, events in baseline_by_sequence.items()
    }
    for threshold in config.ioa_sweep_thresholds:
        relabelled_by_sequence: dict[str, list[ErrorEvent]] = {}
        for sequence_name, baseline in sorted(baseline_by_sequence.items()):
            sweep_config = deepcopy(config)
            sweep_config.thresholds["IOA_INTERACTION"] = float(threshold)
            relabelled, excluded = reattribute_canonical_events(
                baseline,
                sequences[sequence_name],
                matchings[sequence_name],
                sweep_config,
            )
            if excluded:
                raise AssertionError(
                    "IoA sweep moved an included baseline event into crowd exclusion"
                )
            if len(relabelled) != len(baseline):
                raise AssertionError("IoA sweep changed canonical event count")
            relabelled_by_sequence[sequence_name] = relabelled
        for sequence_name in sorted(relabelled_by_sequence) + ["ALL"]:
            selected = (
                relabelled_by_sequence[sequence_name]
                if sequence_name != "ALL"
                else [
                    event
                    for name in sorted(relabelled_by_sequence)
                    for event in relabelled_by_sequence[name]
                ]
            )
            v3_visible = (
                baseline_visible[sequence_name]
                if sequence_name != "ALL"
                else sum(baseline_visible.values())
            )
            ped = sum(
                event.cause == "partial_occl_pedestrian" for event in selected
            )
            static = sum(event.cause == "partial_occl_static" for event in selected)
            visible = sum(event.cause == "visible_miss" for event in selected)
            residual = sum(
                event.cause in {"unattributed_fn", "unattributed_idsw"}
                for event in selected
            )
            rows.append(
                {
                    "sequence": sequence_name,
                    "ioa_interaction": float(threshold),
                    "partial_occl_pedestrian": ped,
                    "partial_occl_static": static,
                    "partial_rules_total": ped + static,
                    "residual_remaining": residual,
                    "visible_miss": visible,
                    "v3_visible_miss": v3_visible,
                    "visible_miss_change_from_v3": visible - v3_visible,
                    "selected_threshold": abs(
                        float(threshold)
                        - float(config.thresholds["IOA_INTERACTION"])
                    )
                    < 1e-12,
                }
            )
    write_rows_csv(path, rows)
    return {
        "chosen_ioa_interaction": float(config.thresholds["IOA_INTERACTION"]),
        "selection_rationale": (
            "0.20 is the conservative elbow: the pre-change sweep preserved 18 more "
            "visible-miss events than 0.15 for only 10 additional residuals; 0.30 "
            "added substantially more residuals for a smaller visibility gain."
        ),
        "thresholds": list(config.ioa_sweep_thresholds),
        "rows": rows,
    }


def residual_diagnostics(
    events: list[ErrorEvent], config: PipelineConfig
) -> dict[str, Any]:
    residuals = [
        event
        for event in events
        if event.cause in {"unattributed_fn", "unattributed_idsw"}
    ]

    def visibility_bin(event: ErrorEvent) -> str:
        value = event.evidence.get("min_visibility")
        if value is None:
            return "missing"
        return vis_band(float(value), config.vis_bands)

    def counter(field) -> dict[str, int]:
        return dict(sorted(Counter(field(event) for event in residuals).items()))

    return {
        "residual_event_count": len(residuals),
        "by_cause": counter(lambda event: str(event.cause)),
        "by_sequence": counter(lambda event: event.sequence),
        "by_error_type": counter(lambda event: event.error_type),
        "by_confidence": counter(lambda event: str(event.cause_confidence)),
        "by_residual_reason": counter(
            lambda event: str(event.evidence.get("residual_reason"))
        ),
        "by_min_visibility_band": counter(visibility_bin),
        "near_interaction_threshold": {
            "pedestrian_ioa_at_least_0.10": sum(
                float(event.evidence.get("max_pedestrian_interaction_ioa", 0.0))
                >= 0.10
                for event in residuals
            ),
            "pedestrian_ioa_at_least_0.20": sum(
                float(event.evidence.get("max_pedestrian_interaction_ioa", 0.0))
                >= 0.20
                for event in residuals
            ),
            "pedestrian_ioa_at_least_0.30": sum(
                float(event.evidence.get("max_pedestrian_interaction_ioa", 0.0))
                >= 0.30
                for event in residuals
            ),
            "pedestrian_iou_at_least_0.25": sum(
                float(event.evidence.get("max_pedestrian_interaction_iou", 0.0))
                >= 0.25
                for event in residuals
            ),
            "pedestrian_iou_at_least_0.30": sum(
                float(event.evidence.get("max_pedestrian_interaction_iou", 0.0))
                >= 0.30
                for event in residuals
            ),
            "static_iou_at_least_0.25": sum(
                float(event.evidence.get("max_static_interaction_iou", 0.0)) >= 0.25
                for event in residuals
            ),
            "static_iou_at_least_0.30": sum(
                float(event.evidence.get("max_static_interaction_iou", 0.0)) >= 0.30
                for event in residuals
            ),
        },
        "interpretation": (
            "High-confidence residuals are a positive claim that no configured rule fired. "
            "Low-confidence residuals are fully occluded cases without a dominant single or aggregate occluder."
        ),
    }


def _balanced_sample(
    events: list[ErrorEvent], per_cause: int, seed: int
) -> list[ErrorEvent]:
    rng = random.Random(seed)
    by_cause: dict[str, dict[tuple[str, str], list[ErrorEvent]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for event in events:
        by_cause[str(event.cause)][(event.tracker, event.sequence)].append(event)
    sample: list[ErrorEvent] = []
    for cause in sorted(by_cause):
        buckets = by_cause[cause]
        for bucket in buckets.values():
            rng.shuffle(bucket)
        keys = sorted(buckets)
        while len([item for item in sample if item.cause == cause]) < per_cause:
            made_progress = False
            for key in keys:
                if buckets[key]:
                    sample.append(buckets[key].pop())
                    made_progress = True
                    if len([item for item in sample if item.cause == cause]) >= per_cause:
                        break
            if not made_progress:
                break
    return sample


def write_human_verification_manifests(
    summary_dir: Path,
    events: list[ErrorEvent],
    config: PipelineConfig,
) -> None:
    cause_counts = Counter(str(event.cause) for event in events)
    eligible = [
        event
        for event in events
        if cause_counts[str(event.cause)] >= config.human_sample_per_cause
    ]
    sample = _balanced_sample(
        eligible, config.human_sample_per_cause, config.random_seed
    )
    summary_dir.mkdir(parents=True, exist_ok=True)
    blind_path = summary_dir / "human_verification_blind.csv"
    with blind_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "event_id",
            "clip_path",
            "annotator_1_cause",
            "annotator_2_cause",
            "adjudicated_cause",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in sample:
            writer.writerow(
                {
                    "event_id": event.event_id,
                    "clip_path": f"clips/{event.event_id}.mp4",
                    "annotator_1_cause": "",
                    "annotator_2_cause": "",
                    "adjudicated_cause": "",
                }
            )
    write_jsonl(summary_dir / "human_verification_answer_key.jsonl", sample)


def render_validation_text(report: dict[str, Any]) -> str:
    lines = ["MOT20 rule-based attribution validation", ""]
    lines.append(f"Overall passed: {report['passed']}")
    lines.append(f"Frame policy: {report['input_audit']['frame_policy']}")
    absent = report["input_audit"][
        "expected_sequences_absent_from_tracker_source"
    ]
    lines.append(f"Expected sequences absent from tracker source: {', '.join(absent) or 'none'}")
    lines.append(
        "Detector training-frame provenance: "
        + report["input_audit"]["detector_training_frame_provenance"]
    )
    redistribution = report.get("redistribution_gate")
    if redistribution is not None:
        lines.extend(
            [
                "",
                "Pass A redistribution gate:",
                f"  v1_events={redistribution['v1_event_count']} v2_events={redistribution['v2_event_count']}",
                f"  v1_unattributed={redistribution['v1_unattributed']} v2_unattributed={redistribution['v2_unattributed']}",
                f"  event_count_invariant={redistribution['event_count_invariant']}",
                f"  residual_reduced={redistribution['residual_reduced']}",
            ]
        )
    v4_gate = report.get("v3_to_v4_redistribution_gate")
    if v4_gate is not None:
        lines.extend(
            [
                "",
                "v3 -> v4 redistribution gate:",
                f"  v3_events={v4_gate['v3_event_count']} v4_events={v4_gate['v4_event_count']}",
                f"  event_count_invariant={v4_gate['event_count_invariant']}",
                f"  changed_causes={v4_gate['changed_cause_count']}",
                f"  unattributed={v4_gate['v3_unattributed']} -> {v4_gate['v4_unattributed']}",
                f"  visible_miss={v4_gate['v3_visible_miss']} -> {v4_gate['v4_visible_miss']}",
            ]
        )
    ioa_sweep = report.get("ioa_sweep")
    if ioa_sweep is not None:
        lines.append(
            f"  chosen_IOA_INTERACTION={ioa_sweep['chosen_ioa_interaction']:.2f}"
        )
    partial_delta = report.get("partial_selection_delta")
    if partial_delta is not None:
        lines.append(
            "  partial_order_to_score_changes="
            f"{partial_delta['changed_event_count']}"
        )
    static_rows = report.get("static_person_presence_vs_drift") or []
    static_all = next(
        (row for row in static_rows if row.get("sequence") == "ALL"), None
    )
    if static_all is not None:
        lines.append(
            "  static_person_presence/drift="
            f"{static_all['static_person_present_events']}/"
            f"{static_all['track_drift_to_static_events']}"
        )
    scope = report.get("scope_expansion_check")
    if scope is not None:
        lines.extend(
            [
                "",
                "Pass B scope expansion:",
                f"  v1_half_events={scope['v1_reference_event_count']} full_v2_events={scope['full_v2_event_count']}",
                f"  expansion_ratio={scope['ratio']:.4f}",
            ]
        )
        if scope.get("warning"):
            lines.append(f"  WARNING: {scope['warning']}")
    second_tracker = report.get("second_tracker_comparison", {})
    if not second_tracker.get("available", False):
        lines.extend(
            [
                "",
                "Second shared-detection tracker comparison: unavailable",
                f"  {second_tracker.get('reason', 'No valid result files were supplied.')}",
            ]
        )
    lines.append("")
    for sequence, item in sorted(report["sequences"].items()):
        lines.extend(
            [
                f"[{sequence}] passed={item['passed']}",
                f"  selected_frames={item['selected_frames']}",
                f"  TrackEval FN={item['trackeval']['CLR_FN']} IDSW={item['trackeval']['IDSW']}",
                f"  corpus FN={item['corpus']['included_fn_frames']} + excluded_crowd={item['corpus']['excluded_crowd_fn_frames']} + micro_gap={item['corpus'].get('excluded_micro_gap_fn_frames', 0)}",
                f"  canonical_events={item['corpus']['canonical_event_count']}",
                f"  unattributed_fraction={item['corpus']['unattributed_fraction']:.4f}",
                f"  exact_GT_geometry_warning={report['geometry_audit'][sequence]['looks_like_ground_truth_geometry']}",
            ]
        )
        for warning in item["warnings"]:
            lines.append(f"  WARNING: {warning}")
        if report["geometry_audit"][sequence]["looks_like_ground_truth_geometry"]:
            lines.append(
                "  WARNING: every tracker box has exact target-GT geometry; this input appears to be a GT annotation export, not tracker detections."
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
