from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, replace
from math import hypot
from statistics import mean, median
from typing import Any, Callable, Optional

from .config import PipelineConfig, vis_band
from .reentry import check_reentry
from .events import event_id_for
from .geometry import (
    box_area,
    intersection_box,
    ioa,
    iou,
    near_frame_edge,
    rectangle_union_area,
)
from .matching import MatchingResult
from .models import ErrorEvent, GTBox, SequenceData


def _boxes_for_identity(
    sequence: SequenceData,
    gt_id: int,
    frames: set[int],
    target_classes: set[int] | None = None,
) -> list[GTBox]:
    boxes = [
        box
        for frame in sorted(frames)
        for box in sequence.gt_by_frame.get(frame, [])
        if box.gt_id == gt_id
    ]
    if target_classes is not None:
        boxes = [box for box in boxes if box.cls in target_classes and box.flag != 0]
    return boxes


def _best_overlap(
    sequence: SequenceData,
    focal_by_frame: dict[int, GTBox],
    frames: set[int],
    classes: set[int],
    overlap_fn: Callable,
    *,
    exclude_id: Optional[int] = None,
) -> tuple[Optional[GTBox], float, Optional[int], int]:
    best: Optional[GTBox] = None
    best_value = 0.0
    best_frame: Optional[int] = None
    interacting_ids: set[int] = set()
    for frame in sorted(frames):
        focal = focal_by_frame.get(frame)
        if focal is None:
            continue
        for candidate in sequence.gt_by_frame.get(frame, []):
            if candidate.cls not in classes or candidate.gt_id == exclude_id:
                continue
            value = float(overlap_fn(focal.bbox, candidate.bbox))
            if value > 0:
                interacting_ids.add(candidate.gt_id)
            if value > best_value:
                best = candidate
                best_value = value
                best_frame = frame
    return best, best_value, best_frame, len(interacting_ids)


def _role_assignment(
    sequence: SequenceData,
    focal_id: int,
    partner_id: int,
    window_frames: set[int],
    config: PipelineConfig,
) -> tuple[Optional[str], str, dict[str, float | int | None]]:
    target_classes = set(config.classes["TARGET_CLASSES"])
    focal_boxes = _boxes_for_identity(
        sequence, focal_id, window_frames, target_classes
    )
    partner_boxes = _boxes_for_identity(
        sequence, partner_id, window_frames, target_classes
    )
    focal_mean = mean(box.visibility for box in focal_boxes) if focal_boxes else None
    partner_mean = mean(box.visibility for box in partner_boxes) if partner_boxes else None
    threshold = float(config.thresholds["VIS_DROP_DELTA"])
    evidence: dict[str, float | int | None] = {
        "focal_mean_visibility_window": focal_mean,
        "partner_mean_visibility_window": partner_mean,
        "visibility_drop_threshold": threshold,
        "focal_visibility_observations": len(focal_boxes),
        "partner_visibility_observations": len(partner_boxes),
    }
    if focal_mean is None or partner_mean is None:
        evidence["visibility_mean_delta_partner_minus_focal"] = None
        return None, "low", evidence
    delta = partner_mean - focal_mean
    evidence["visibility_mean_delta_partner_minus_focal"] = delta
    if delta >= threshold:
        return "occludee", "high", evidence
    if -delta >= threshold:
        return "occluder", "high", evidence
    return None, "low", evidence


def _max_absent_run(
    event: ErrorEvent, sequence: SequenceData, matching: MatchingResult
) -> int:
    run = 0
    maximum = 0
    for frame in sequence.selected_frames:
        if frame <= event.frame_start or frame >= event.frame_end:
            continue
        if event.focal_gt_id not in matching.gt_present[frame]:
            run += 1
            maximum = max(maximum, run)
        else:
            run = 0
    return maximum


def _edge_evidence(
    event: ErrorEvent,
    focal_by_frame: dict[int, GTBox],
    sequence: SequenceData,
    margin: float,
) -> tuple[bool, Optional[float]]:
    candidate = focal_by_frame.get(event.frame_start)
    if candidate is None:
        prior_frames = [frame for frame in focal_by_frame if frame <= event.frame_start]
        candidate = focal_by_frame[max(prior_frames)] if prior_frames else None
    if candidate is None:
        return False, None
    box = candidate.bbox
    edge_distance = min(
        box[0], box[1], sequence.width - box[2], sequence.height - box[3]
    )
    return near_frame_edge(box, sequence.width, sequence.height, margin), edge_distance


def _analysis_context(
    sequence: SequenceData, config: PipelineConfig
) -> dict[str, Any]:
    target_classes = set(config.classes["TARGET_CLASSES"])
    target_by_frame = {
        frame: [
            box
            for box in sequence.gt_by_frame.get(frame, [])
            if box.flag != 0 and box.cls in target_classes
        ]
        for frame in sequence.selected_frames
    }
    all_areas = sorted(
        box_area(box.bbox)
        for boxes in target_by_frame.values()
        for box in boxes
    )
    return {"target_by_frame": target_by_frame, "all_target_areas": all_areas}


@dataclass
class RuleVote:
    partner: GTBox | None
    ioa_focal: float
    iou: float
    vote_fraction: float
    n_frames_voted: int
    supporting_frames: list[int]
    per_frame_votes: list[dict[str, Any]]
    max_ioa_focal: float
    iou_at_max_ioa: float
    n_candidate_ids: int


def _per_frame_vote(
    sequence: SequenceData,
    focal_by_frame: dict[int, GTBox],
    classes: set[int],
    threshold: float,
    *,
    exclude_id: int,
) -> RuleVote:
    """Evaluate one geometric rule independently per frame, then majority-vote partner."""
    votes: list[tuple[GTBox, float, float]] = []
    vote_records: list[dict[str, Any]] = []
    max_ioa = 0.0
    iou_at_max = 0.0
    candidate_ids: set[int] = set()
    for frame, focal in sorted(focal_by_frame.items()):
        frame_candidates: list[tuple[GTBox, float, float]] = []
        for candidate in sequence.gt_by_frame.get(frame, []):
            if candidate.gt_id == exclude_id or candidate.cls not in classes:
                continue
            focal_ioa = float(ioa(focal.bbox, candidate.bbox))
            pair_iou = float(iou(focal.bbox, candidate.bbox))
            if focal_ioa > max_ioa or (
                focal_ioa == max_ioa and pair_iou > iou_at_max
            ):
                max_ioa = focal_ioa
                iou_at_max = pair_iou
            if focal_ioa >= threshold:
                frame_candidates.append((candidate, focal_ioa, pair_iou))
                candidate_ids.add(candidate.gt_id)
        if not frame_candidates:
            vote_records.append(
                {
                    "frame": frame,
                    "partner_gt_id": None,
                    "occluder_class": None,
                    "ioa_focal": 0.0,
                    "iou": 0.0,
                }
            )
            continue
        chosen = max(
            frame_candidates,
            key=lambda item: (item[1], item[2], -item[0].gt_id),
        )
        votes.append(chosen)
        vote_records.append(
            {
                "frame": frame,
                "partner_gt_id": chosen[0].gt_id,
                "occluder_class": chosen[0].cls,
                "ioa_focal": chosen[1],
                "iou": chosen[2],
            }
        )
    total_frames = len(focal_by_frame)
    if not votes:
        return RuleVote(
            partner=None,
            ioa_focal=0.0,
            iou=0.0,
            vote_fraction=0.0,
            n_frames_voted=total_frames,
            supporting_frames=[],
            per_frame_votes=vote_records,
            max_ioa_focal=max_ioa,
            iou_at_max_ioa=iou_at_max,
            n_candidate_ids=len(candidate_ids),
        )
    by_partner: dict[tuple[int, int], list[tuple[GTBox, float, float]]] = defaultdict(list)
    for vote in votes:
        by_partner[(vote[0].gt_id, vote[0].cls)].append(vote)
    winner_key, winner_votes = max(
        by_partner.items(),
        key=lambda item: (
            len(item[1]),
            median(vote[1] for vote in item[1]),
            median(vote[2] for vote in item[1]),
            -item[0][0],
        ),
    )
    supporting_frames = [
        int(record["frame"])
        for record in vote_records
        if record["partner_gt_id"] == winner_key[0]
        and record["occluder_class"] == winner_key[1]
    ]
    return RuleVote(
        partner=winner_votes[0][0],
        ioa_focal=float(median(vote[1] for vote in winner_votes)),
        iou=float(median(vote[2] for vote in winner_votes)),
        vote_fraction=len(winner_votes) / total_frames if total_frames else 0.0,
        n_frames_voted=total_frames,
        supporting_frames=supporting_frames,
        per_frame_votes=vote_records,
        max_ioa_focal=max_ioa,
        iou_at_max_ioa=iou_at_max,
        n_candidate_ids=len(candidate_ids),
    )


def _vote_evidence(vote: RuleVote, threshold: float) -> dict[str, Any]:
    return {
        "partner_gt_id": vote.partner.gt_id if vote.partner else None,
        "occluder_class": vote.partner.cls if vote.partner else None,
        "ioa_focal": vote.ioa_focal,
        "iou": vote.iou,
        "max_ioa_focal": vote.max_ioa_focal,
        "iou_at_max_ioa": vote.iou_at_max_ioa,
        "ioa_threshold": threshold,
        "vote_fraction": vote.vote_fraction,
        "frame_policy": "per_frame_majority",
        "n_frames_voted": vote.n_frames_voted,
        "supporting_frames": vote.supporting_frames,
        "n_candidate_ids": vote.n_candidate_ids,
        "per_frame_votes": vote.per_frame_votes,
    }


def _aggregate_occlusion(
    sequence: SequenceData,
    focal_by_frame: dict[int, GTBox],
    frames: set[int],
    config: PipelineConfig,
) -> dict[str, Any]:
    allowed_classes = (
        set(config.classes["TARGET_CLASSES"])
        | set(config.classes["STATIC_PERSON"])
        | set(config.classes["OCCLUDER_CLASSES"])
    )
    minimum = int(config.thresholds["AGG_MIN_OCCLUDERS"])
    best = {
        "coverage": 0.0,
        "frame": None,
        "n_occluders": 0,
        "occluder_ids": [],
        "max_single_ioa": 0.0,
        "vote_fraction": 0.0,
        "n_frames_voted": len(focal_by_frame),
        "supporting_frames": [],
        "per_frame_votes": [],
    }
    qualifying: list[dict[str, Any]] = []
    per_frame: list[dict[str, Any]] = []
    coverage_threshold = float(config.thresholds["AGG_COVERAGE_THRESH"])
    dominant_threshold = float(config.thresholds["IOA_OCCLUDER"])
    for frame in sorted(frames):
        focal = focal_by_frame.get(frame)
        if focal is None:
            continue
        others: list[GTBox] = []
        intersections = []
        for candidate in sequence.gt_by_frame.get(frame, []):
            if (
                candidate.gt_id == focal.gt_id
                or candidate.cls not in allowed_classes
                or iou(focal.bbox, candidate.bbox) <= 0
            ):
                continue
            intersection = intersection_box(focal.bbox, candidate.bbox)
            if intersection is not None:
                others.append(candidate)
                intersections.append(intersection)
        focal_area = box_area(focal.bbox)
        coverage = (
            rectangle_union_area(intersections) / focal_area if focal_area > 0 else 0.0
        )
        max_single = max(
            (ioa(focal.bbox, box.bbox) for box in others), default=0.0
        )
        record = {
            "frame": frame,
            "coverage": coverage,
            "n_occluders": len(others),
            "occluder_ids": sorted({box.gt_id for box in others}),
            "max_single_ioa": max_single,
            "qualified": (
                len(others) >= minimum
                and coverage >= coverage_threshold
                and max_single < dominant_threshold
            ),
        }
        per_frame.append(record)
        if record["qualified"]:
            qualifying.append(record)
        if coverage > float(best["coverage"]):
            best = {
                "coverage": coverage,
                "frame": frame,
                "n_occluders": len(others),
                "occluder_ids": sorted({box.gt_id for box in others}),
                "max_single_ioa": max_single,
            }
    if qualifying:
        representative = max(
            qualifying,
            key=lambda item: (item["coverage"], item["n_occluders"], -item["frame"]),
        )
        best.update(representative)
        best["coverage"] = float(median(item["coverage"] for item in qualifying))
        best["max_single_ioa"] = max(
            float(item["max_single_ioa"]) for item in qualifying
        )
    best["vote_fraction"] = (
        len(qualifying) / len(focal_by_frame) if focal_by_frame else 0.0
    )
    best["n_frames_voted"] = len(focal_by_frame)
    best["supporting_frames"] = [int(item["frame"]) for item in qualifying]
    best["per_frame_votes"] = per_frame
    best["qualified"] = bool(qualifying)
    return best


def _visible_miss_covariates(
    focal_boxes: list[GTBox],
    sequence: SequenceData,
    context: dict[str, Any],
    config: PipelineConfig,
) -> dict[str, Any]:
    areas = [box_area(box.bbox) for box in focal_boxes]
    heights = [max(0.0, box.bbox[3] - box.bbox[1]) for box in focal_boxes]
    aspects = [
        (box.bbox[2] - box.bbox[0]) / max(1e-12, box.bbox[3] - box.bbox[1])
        for box in focal_boxes
    ]
    centres = [
        ((box.bbox[0] + box.bbox[2]) / 2, (box.bbox[1] + box.bbox[3]) / 2)
        for box in focal_boxes
    ]
    median_area = median(areas) if areas else 0.0
    all_areas: list[float] = context["all_target_areas"]
    percentile = (
        100.0 * bisect_right(all_areas, median_area) / len(all_areas)
        if all_areas
        else 0.0
    )
    density_radius = float(config.thresholds["DENSITY_RADIUS_PX"])
    densities: list[int] = []
    nearest: list[float] = []
    for box, centre in zip(focal_boxes, centres):
        distances = [
            hypot(
                centre[0] - (other.bbox[0] + other.bbox[2]) / 2,
                centre[1] - (other.bbox[1] + other.bbox[3]) / 2,
            )
            for other in context["target_by_frame"].get(box.frame, [])
            if other.gt_id != box.gt_id
        ]
        densities.append(sum(distance <= density_radius for distance in distances))
        if distances:
            nearest.append(min(distances))
    return {
        "bbox_area_px": median_area,
        "bbox_height_px": median(heights) if heights else 0.0,
        "area_percentile": percentile,
        "local_density": mean(densities) if densities else 0.0,
        "nn_distance_px": median(nearest) if nearest else None,
        "aspect_ratio": median(aspects) if aspects else 0.0,
        "frame_position": [
            median([centre[0] / sequence.width for centre in centres]) if centres else 0.0,
            median([centre[1] / sequence.height for centre in centres]) if centres else 0.0,
        ],
        "n_frames": len(focal_boxes),
        "density_radius_px": density_radius,
    }


def attribute_event(
    event: ErrorEvent,
    sequence: SequenceData,
    matching: MatchingResult,
    config: PipelineConfig,
    context: dict[str, Any] | None = None,
) -> ErrorEvent:
    context = _analysis_context(sequence, config) if context is None else context
    thresholds = config.thresholds
    selected = set(sequence.selected_frames)
    event_frames = {
        frame
        for frame in selected
        if event.frame_start <= frame <= event.frame_end
    }
    window_radius = int(thresholds["EVENT_WINDOW"])
    window_frames = {
        frame
        for frame in selected
        if event.frame_start - window_radius <= frame <= event.frame_end + window_radius
    }
    target_classes = set(config.classes["TARGET_CLASSES"])
    focal_boxes = _boxes_for_identity(
        sequence, event.focal_gt_id, event_frames, target_classes
    )
    focal_by_frame = {box.frame: box for box in focal_boxes}
    focal_window_boxes = _boxes_for_identity(
        sequence, event.focal_gt_id, window_frames, target_classes
    )
    focal_window_by_frame = {box.frame: box for box in focal_window_boxes}
    visibility_values = [box.visibility for box in focal_boxes]
    visibility_source = "event"
    if not visibility_values:
        visibility_values = [box.visibility for box in focal_window_boxes]
        visibility_source = "window"
    minimum_visibility = min(visibility_values) if visibility_values else None
    band = vis_band(minimum_visibility, config.vis_bands) if minimum_visibility is not None else None

    crowd, crowd_ioa, crowd_frame, _ = _best_overlap(
        sequence,
        focal_by_frame,
        event_frames,
        set(config.classes["CROWD"]),
        ioa,
        exclude_id=event.focal_gt_id,
    )
    full_threshold = float(thresholds["IOA_OCCLUDER"])
    interaction_threshold = float(thresholds["IOA_INTERACTION"])
    large_static_threshold = float(
        config.ioa_occluder_by_class["large_object"]
    )
    person_static_threshold = float(config.ioa_occluder_by_class["person"])
    large_static_vote = _per_frame_vote(
        sequence,
        focal_by_frame,
        set(config.classes["OCCLUDER_CLASSES"]),
        large_static_threshold,
        exclude_id=event.focal_gt_id,
    )
    person_static_vote = _per_frame_vote(
        sequence,
        focal_by_frame,
        set(config.classes["STATIC_PERSON"]),
        person_static_threshold,
        exclude_id=event.focal_gt_id,
    )
    full_pedestrian_vote = _per_frame_vote(
        sequence,
        focal_by_frame,
        target_classes,
        full_threshold,
        exclude_id=event.focal_gt_id,
    )
    partial_pedestrian_vote = _per_frame_vote(
        sequence,
        focal_by_frame,
        target_classes,
        interaction_threshold,
        exclude_id=event.focal_gt_id,
    )
    partial_static_classes = set(config.classes["STATIC_PERSON"]) | set(
        config.classes["OCCLUDER_CLASSES"]
    )
    partial_static_vote = _per_frame_vote(
        sequence,
        focal_by_frame,
        partial_static_classes,
        interaction_threshold,
        exclude_id=event.focal_gt_id,
    )
    static_presence_vote = _per_frame_vote(
        sequence,
        focal_window_by_frame,
        set(config.classes["STATIC_PERSON"]),
        interaction_threshold,
        exclude_id=event.focal_gt_id,
    )
    static_person_present = (
        static_presence_vote.max_ioa_focal >= interaction_threshold
    )
    aggregate = _aggregate_occlusion(
        sequence, focal_by_frame, event_frames, config
    )
    margin = float(thresholds["EDGE_MARGIN_PX"])
    near_edge, edge_distance = _edge_evidence(
        event, focal_window_by_frame, sequence, margin
    )
    absent_run = _max_absent_run(event, sequence, matching)
    min_absent = int(thresholds["REENTRY_MIN_ABSENT_FRAMES"])

    def _box_at(gt_id: int, frame: int):
        for box in sequence.gt_by_frame.get(frame, []):
            if box.gt_id == gt_id:
                return box.bbox
        return None

    reentry_decision = check_reentry(
        {
            "error_type": event.error_type,
            "frame_start": event.frame_start,
            "frame_end": event.frame_end,
            "focal_gt_id": event.focal_gt_id,
        },
        matching.gt_present,
        _box_at,
        sequence.width,
        sequence.height,
        min_absent=min_absent,
        margin=margin,
    )
    interaction_ids = {
        int(record["partner_gt_id"])
        for vote in (partial_pedestrian_vote, partial_static_vote)
        for record in vote.per_frame_votes
        if record["partner_gt_id"] is not None
    }
    event_max_interaction_ioa = max(
        partial_pedestrian_vote.max_ioa_focal,
        partial_static_vote.max_ioa_focal,
    )
    event_iou_at_max_ioa = (
        partial_pedestrian_vote.iou_at_max_ioa
        if partial_pedestrian_vote.max_ioa_focal
        >= partial_static_vote.max_ioa_focal
        else partial_static_vote.iou_at_max_ioa
    )
    evidence = dict(event.evidence)
    evidence.update(
        {
            "frame_policy": "per_frame_majority",
            "n_frames_voted": len(focal_by_frame),
            "event_focal_visibility_observations": len(visibility_values),
            "min_visibility": minimum_visibility,
            "visibility_source": visibility_source,
            "visibility_band": band,
            "visibility_bands": config.vis_bands,
            "max_crowd_ioa": crowd_ioa,
            "max_crowd_ioa_frame": crowd_frame,
            "crowd_gt_id": crowd.gt_id if crowd else None,
            "crowd_overlap_threshold": float(thresholds["CROWD_OVERLAP"]),
            "full_large_static_vote": _vote_evidence(
                large_static_vote, large_static_threshold
            ),
            "full_person_static_vote": _vote_evidence(
                person_static_vote, person_static_threshold
            ),
            "full_pedestrian_vote": _vote_evidence(
                full_pedestrian_vote, full_threshold
            ),
            "partial_pedestrian_vote": _vote_evidence(
                partial_pedestrian_vote, interaction_threshold
            ),
            "partial_static_vote": _vote_evidence(
                partial_static_vote, interaction_threshold
            ),
            "static_person_window_vote": _vote_evidence(
                static_presence_vote, interaction_threshold
            ),
            "max_pedestrian_interaction_ioa": partial_pedestrian_vote.max_ioa_focal,
            "max_pedestrian_interaction_iou": partial_pedestrian_vote.iou_at_max_ioa,
            "max_static_interaction_ioa": partial_static_vote.max_ioa_focal,
            "max_static_interaction_iou": partial_static_vote.iou_at_max_ioa,
            "interaction_ioa_threshold": interaction_threshold,
            "event_window": window_radius,
            "event_interactor_ids": sorted(interaction_ids),
            "n_interactors": len(interaction_ids),
            "event_max_interaction_ioa": event_max_interaction_ioa,
            "event_iou_at_max_interaction_ioa": event_iou_at_max_ioa,
            "aggregate_coverage": aggregate["coverage"],
            "aggregate_coverage_frame": aggregate["frame"],
            "aggregate_n_occluders": aggregate["n_occluders"],
            "aggregate_occluder_ids": aggregate["occluder_ids"],
            "aggregate_max_single_ioa": aggregate["max_single_ioa"],
            "aggregate_vote_fraction": aggregate["vote_fraction"],
            "aggregate_n_frames_voted": aggregate["n_frames_voted"],
            "aggregate_per_frame_votes": aggregate["per_frame_votes"],
            "aggregate_coverage_threshold": float(thresholds["AGG_COVERAGE_THRESH"]),
            "aggregate_min_occluders": int(thresholds["AGG_MIN_OCCLUDERS"]),
            "edge_margin_px": margin,
            "focal_edge_distance_px": edge_distance,
            "focal_before_gap_near_edge": near_edge,
            "max_gt_absent_run_between_matches": absent_run,
            "reentry_absence_threshold": min_absent,
            "reentry_min_absent_frames": min_absent,
            "static_person_present": static_person_present,
            "static_person_max_ioa": static_presence_vote.max_ioa_focal,
        }
    )
    attributed = replace(
        event,
        evidence=evidence,
        static_person_present=static_person_present,
        static_person_max_ioa=static_presence_vote.max_ioa_focal,
    )
    is_fn = event.error_type in {"FN", "FRAG_NO_SWITCH"}
    candidates: list[dict[str, Any]] = []

    def add_vote_candidate(rule: str, vote: RuleVote, threshold: float) -> None:
        if vote.partner is None:
            return
        candidates.append(
            {
                "rule": rule,
                "score": vote.ioa_focal,
                "partner_gt_id": vote.partner.gt_id,
                "evidence": _vote_evidence(vote, threshold),
            }
        )

    if band == "FULL" and is_fn:
        add_vote_candidate(
            "full_occl_static", large_static_vote, large_static_threshold
        )
        add_vote_candidate(
            "full_occl_static", person_static_vote, person_static_threshold
        )
    if band == "FULL":
        add_vote_candidate(
            "full_occl_pedestrian", full_pedestrian_vote, full_threshold
        )
    aggregate_fires = band == "FULL" and bool(aggregate["qualified"])
    if aggregate_fires:
        candidates.append(
            {
                "rule": "full_occl_crowd_aggregate",
                "score": float(aggregate["coverage"]),
                "partner_gt_id": None,
                "evidence": {
                    "coverage": aggregate["coverage"],
                    "n_occluders": aggregate["n_occluders"],
                    "occluder_ids": aggregate["occluder_ids"],
                    "max_single_ioa": aggregate["max_single_ioa"],
                    "vote_fraction": aggregate["vote_fraction"],
                    "frame_policy": "per_frame_majority",
                    "n_frames_voted": aggregate["n_frames_voted"],
                },
            }
        )
    partial_options: list[tuple[str, RuleVote]] = []
    if band in {"PARTIAL", "CLEAR"} and event.error_type in {"FN", "IDSW"}:
        if partial_pedestrian_vote.partner is not None:
            partial_options.append(
                ("partial_occl_pedestrian", partial_pedestrian_vote)
            )
            add_vote_candidate(
                "partial_occl_pedestrian",
                partial_pedestrian_vote,
                interaction_threshold,
            )
        if partial_static_vote.partner is not None:
            partial_options.append(("partial_occl_static", partial_static_vote))
            add_vote_candidate(
                "partial_occl_static",
                partial_static_vote,
                interaction_threshold,
            )
    order_based_cause = next(
        (
            cause
            for cause in ("partial_occl_pedestrian", "partial_occl_static")
            if any(option[0] == cause for option in partial_options)
        ),
        None,
    )
    ranked_partial = sorted(
        partial_options,
        key=lambda option: (
            -option[1].ioa_focal,
            option[0] != "partial_occl_pedestrian",
            option[1].partner.gt_id if option[1].partner else 10**9,
        ),
    )
    score_based_cause = ranked_partial[0][0] if ranked_partial else None
    score_gap = (
        ranked_partial[0][1].ioa_focal - ranked_partial[1][1].ioa_focal
        if len(ranked_partial) > 1
        else None
    )
    evidence.update(
        {
            "partial_selection_policy": "score_based_ioa_focal",
            "partial_order_based_cause": order_based_cause,
            "partial_score_based_cause": score_based_cause,
            "partial_selection_changed_from_order": (
                order_based_cause is not None
                and score_based_cause is not None
                and order_based_cause != score_based_cause
            ),
            "partial_score_gap": score_gap,
            "partial_ambiguity_margin": float(thresholds["AMBIGUITY_MARGIN"]),
        }
    )
    if reentry_decision is not None:
        reentry_cause, reentry_extra = reentry_decision
        candidates.append(
            {
                "rule": reentry_cause,
                "score": 1.0,
                "partner_gt_id": None,
                "evidence": {
                    "frame_policy": "per_frame_majority",
                    **reentry_extra,
                },
            }
        )
    no_interaction = event_max_interaction_ioa < interaction_threshold
    visible_miss_fires = band == "CLEAR" and is_fn and no_interaction
    if visible_miss_fires:
        candidates.append(
            {
                "rule": "visible_miss",
                "score": float(minimum_visibility or 0.0),
                "partner_gt_id": None,
                "evidence": {
                    "ioa_focal": event_max_interaction_ioa,
                    "iou": event_iou_at_max_ioa,
                    "ioa_threshold": interaction_threshold,
                    "n_interactors": 0,
                    "frame_policy": "per_frame_majority",
                    "n_frames_voted": len(focal_by_frame),
                    "vote_fraction": 1.0,
                },
            }
        )
    if event.error_type == "FP_ID_HIJACK":
        candidates.append(
            {
                "rule": "track_drift_to_static",
                "score": float(event.evidence.get("max_static_iou", 1.0)),
                "partner_gt_id": event.partner_gt_id,
                "evidence": {
                    "tracker_geometry_exception": True,
                    "frame_policy": "per_frame_majority",
                },
            }
        )
    candidates.sort(key=lambda item: (-float(item["score"]), item["rule"]))
    evidence["all_candidates"] = candidates

    def apply_vote(vote: RuleVote, threshold: float) -> None:
        evidence.update(_vote_evidence(vote, threshold))

    def vote_confidence(base: str, vote_fraction: float) -> str:
        return (
            "low"
            if vote_fraction < float(thresholds["FRAME_VOTE_CONFIDENCE"])
            else base
        )

    if crowd_ioa >= float(thresholds["CROWD_OVERLAP"]):
        attributed.cause = "EXCLUDED_CROWD"
        attributed.cause_confidence = "high"
        attributed.partner_gt_id = crowd.gt_id if crowd else None
        evidence["vote_fraction"] = 1.0
        return attributed

    if event.error_type == "FP_ID_HIJACK":
        attributed.cause = "track_drift_to_static"
        attributed.cause_confidence = "high"
        evidence["tracker_geometry_exception"] = True
        evidence["vote_fraction"] = 1.0
        return attributed

    if band == "FULL":
        static_choice: tuple[RuleVote, float] | None = None
        if is_fn and large_static_vote.partner is not None:
            static_choice = (large_static_vote, large_static_threshold)
        elif is_fn and person_static_vote.partner is not None:
            static_choice = (person_static_vote, person_static_threshold)
        if static_choice is not None:
            vote, applied_threshold = static_choice
            apply_vote(vote, applied_threshold)
            attributed.cause = "full_occl_static"
            attributed.cause_confidence = vote_confidence(
                "high", vote.vote_fraction
            )
            attributed.partner_gt_id = vote.partner.gt_id if vote.partner else None
            attributed.role = "occludee"
            evidence["occluder_class"] = vote.partner.cls if vote.partner else None
            evidence["ioa_threshold_applied"] = applied_threshold
            return attributed
        if full_pedestrian_vote.partner is not None:
            role, confidence, role_evidence = _role_assignment(
                sequence,
                event.focal_gt_id,
                full_pedestrian_vote.partner.gt_id,
                window_frames,
                config,
            )
            apply_vote(full_pedestrian_vote, full_threshold)
            evidence.update(role_evidence)
            attributed.cause = "full_occl_pedestrian"
            attributed.cause_confidence = vote_confidence(
                confidence, full_pedestrian_vote.vote_fraction
            )
            attributed.partner_gt_id = full_pedestrian_vote.partner.gt_id
            attributed.role = role
            return attributed
        if aggregate_fires:
            attributed.cause = "full_occl_crowd_aggregate"
            attributed.cause_confidence = vote_confidence(
                "high", float(aggregate["vote_fraction"])
            )
            attributed.partner_gt_id = None
            attributed.role = "occludee"
            evidence.update(
                {
                    "coverage": aggregate["coverage"],
                    "n_occluders": aggregate["n_occluders"],
                    "occluder_ids": aggregate["occluder_ids"],
                    "max_single_ioa": aggregate["max_single_ioa"],
                    "vote_fraction": aggregate["vote_fraction"],
                }
            )
            return attributed
        attributed.cause = (
            "unattributed_fn"
            if event.error_type in {"FN", "FRAG_NO_SWITCH"}
            else "unattributed_idsw"
        )
        attributed.cause_confidence = "low"
        evidence["residual_reason"] = "no_dominant_full_occluder"
        evidence["vote_fraction"] = 0.0
        return attributed

    if ranked_partial:
        cause, vote = ranked_partial[0]
        runner = ranked_partial[1] if len(ranked_partial) > 1 else None
        ambiguous = (
            runner is not None
            and score_gap is not None
            and score_gap < float(thresholds["AMBIGUITY_MARGIN"])
        )
        apply_vote(vote, interaction_threshold)
        evidence["partial_winner_score"] = vote.ioa_focal
        evidence["ambiguous_alternatives"] = (
            [
                {
                    "cause": runner[0],
                    "partner_gt_id": (
                        runner[1].partner.gt_id if runner[1].partner else None
                    ),
                    "ioa_focal": runner[1].ioa_focal,
                    "iou": runner[1].iou,
                }
            ]
            if ambiguous and runner is not None
            else []
        )
        attributed.cause = cause
        attributed.partner_gt_id = vote.partner.gt_id if vote.partner else None
        if cause == "partial_occl_pedestrian" and vote.partner is not None:
            role, role_confidence, role_evidence = _role_assignment(
                sequence,
                event.focal_gt_id,
                vote.partner.gt_id,
                window_frames,
                config,
            )
            attributed.role = role
            evidence.update(role_evidence)
            base_confidence = role_confidence
        else:
            attributed.role = "occludee"
            evidence["occluder_class"] = vote.partner.cls if vote.partner else None
            base_confidence = "medium"
        attributed.cause_confidence = (
            "low"
            if ambiguous
            else vote_confidence(base_confidence, vote.vote_fraction)
        )
        return attributed

    if reentry_decision is not None:
        reentry_cause, reentry_extra = reentry_decision
        attributed.cause = reentry_cause
        attributed.cause_confidence = "high"
        evidence.update(reentry_extra)
        evidence["vote_fraction"] = 1.0
        return attributed

    if visible_miss_fires:
        attributed.cause = "visible_miss"
        attributed.cause_confidence = "high"
        evidence.update(
            {
                "ioa_focal": event_max_interaction_ioa,
                "iou": event_iou_at_max_ioa,
                "ioa_threshold_applied": interaction_threshold,
                "n_interactors": 0,
                "vote_fraction": 1.0,
            }
        )
        evidence.update(
            _visible_miss_covariates(focal_boxes, sequence, context, config)
        )
        return attributed

    if event.error_type in {"FN", "FRAG_NO_SWITCH"}:
        attributed.cause = "unattributed_fn"
    else:
        attributed.cause = "unattributed_idsw"
    attributed.cause_confidence = "high"
    evidence["residual_reason"] = "no_attribution_rule_fired"
    evidence["vote_fraction"] = 0.0
    return attributed


def _component_record(event: ErrorEvent) -> dict:
    return {
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


def _component_records(event: ErrorEvent) -> list[dict]:
    existing = event.evidence.get("merged_components")
    if isinstance(existing, list) and existing:
        return [dict(component) for component in existing]
    return [_component_record(event)]


def _merge_cluster(cluster: list[ErrorEvent]) -> ErrorEvent:
    if len(cluster) == 1:
        return cluster[0]
    cause_priority = {
        "track_drift_to_static": 0,
        "full_occl_static": 1,
        "full_occl_pedestrian": 2,
        "full_occl_crowd_aggregate": 3,
        "partial_occl_pedestrian": 4,
        "partial_occl_static": 5,
        "reentry": 6,
        "gt_annotation_gap": 7,
        "visible_miss": 8,
        "unattributed_fn": 9,
        "unattributed_idsw": 10,
    }
    canonical = min(
        cluster,
        key=lambda item: (
            item.error_type != "FP_ID_HIJACK",
            item.role != "occludee",
            cause_priority.get(str(item.cause), 99),
            item.frame_start,
            item.focal_gt_id,
            item.error_type,
        ),
    )
    merged = replace(
        canonical,
        frame_start=min(event.frame_start for event in cluster),
        frame_end=max(event.frame_end for event in cluster),
        evidence=dict(canonical.evidence),
    )
    components = [
        component for event in cluster for component in _component_records(event)
    ]
    merged.evidence["merged_components"] = components
    merged.evidence["merged_component_count"] = len(components)
    merged.event_id = event_id_for(
        merged.sequence,
        merged.tracker,
        merged.error_type,
        merged.frame_start,
        merged.frame_end,
        merged.focal_gt_id,
        merged.partner_gt_id,
    )
    return merged


def merge_attributed_events(
    events: list[ErrorEvent], config: PipelineConfig
) -> list[ErrorEvent]:
    """Collapse overlapping symmetric pedestrian interactions into one record."""
    # FN and FRAG_NO_SWITCH deliberately arise from the same missed run. Keep the
    # raw components for metric closure, but do not store the same physical event
    # twice in the corpus.
    exact_groups: dict[tuple, list[ErrorEvent]] = defaultdict(list)
    for event in events:
        pair = tuple(
            sorted(
                (event.focal_gt_id, event.partner_gt_id),
                key=lambda value: (-1 if value is None else value),
            )
        )
        exact_groups[
            (
                event.sequence,
                event.tracker,
                event.frame_start,
                event.frame_end,
                pair,
            )
        ].append(event)
    events = [_merge_cluster(group) for group in exact_groups.values()]

    groups: dict[tuple, list[ErrorEvent]] = defaultdict(list)
    passthrough: list[ErrorEvent] = []
    for event in events:
        if event.partner_gt_id is None:
            passthrough.append(event)
            continue
        pair = tuple(sorted((event.focal_gt_id, event.partner_gt_id)))
        groups[(event.sequence, event.tracker, pair)].append(event)

    window = int(config.thresholds["EVENT_WINDOW"])
    merged_events: list[ErrorEvent] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda item: (item.frame_start, item.frame_end))
        cluster = [ordered[0]]
        expanded_end = ordered[0].frame_end + window
        for event in ordered[1:]:
            if event.frame_start - window <= expanded_end:
                cluster.append(event)
                expanded_end = max(expanded_end, event.frame_end + window)
            else:
                merged_events.append(_merge_cluster(cluster))
                cluster = [event]
                expanded_end = event.frame_end + window
        merged_events.append(_merge_cluster(cluster))
    result = passthrough + merged_events
    return sorted(
        result,
        key=lambda item: (
            item.sequence,
            item.frame_start,
            item.frame_end,
            item.focal_gt_id,
            item.error_type,
        ),
    )


def attribute_events(
    events: list[ErrorEvent],
    sequence: SequenceData,
    matching: MatchingResult,
    config: PipelineConfig,
) -> tuple[list[ErrorEvent], list[ErrorEvent]]:
    context = _analysis_context(sequence, config)
    attributed = [
        attribute_event(event, sequence, matching, config, context) for event in events
    ]
    excluded = [event for event in attributed if event.cause == "EXCLUDED_CROWD"]
    included = [event for event in attributed if event.cause != "EXCLUDED_CROWD"]
    return merge_attributed_events(included, config), excluded


def _fixed_event_key(event: ErrorEvent) -> tuple:
    pair = tuple(
        sorted(
            (event.focal_gt_id, event.partner_gt_id),
            key=lambda value: -1 if value is None else value,
        )
    )
    return (
        event.sequence,
        event.tracker,
        event.frame_start,
        event.frame_end,
        pair,
    )


def _resolve_fixed_corpus_partner_collisions(
    events: list[ErrorEvent],
) -> list[ErrorEvent]:
    """Keep H6 event identity/count fixed while retaining unique canonical keys.

    A widened interaction rule can turn two previously partnerless, per-identity
    events into the same symmetric pair. Merging would violate H6. We retain the
    geometrically selected partner in evidence on both records, but only the more
    specific canonical record stores it in ``partner_gt_id``.
    """
    groups: dict[tuple, list[ErrorEvent]] = defaultdict(list)
    for event in events:
        groups[_fixed_event_key(event)].append(event)
    resolved: list[ErrorEvent] = []
    cause_priority = {
        "track_drift_to_static": 0,
        "full_occl_static": 1,
        "full_occl_pedestrian": 2,
        "full_occl_crowd_aggregate": 3,
        "partial_occl_pedestrian": 4,
        "partial_occl_static": 5,
        "reentry": 6,
        "gt_annotation_gap": 7,
        "visible_miss": 8,
        "unattributed_fn": 9,
        "unattributed_idsw": 10,
    }
    for group in groups.values():
        if len(group) == 1:
            resolved.extend(group)
            continue
        keeper = min(
            group,
            key=lambda event: (
                cause_priority.get(str(event.cause), 99),
                event.role != "occludee",
                -float(event.evidence.get("vote_fraction", 0.0)),
                event.event_id,
            ),
        )
        collision_ids = sorted(event.event_id for event in group)
        for event in group:
            event.evidence["canonical_collision_event_ids"] = collision_ids
            if event is keeper:
                event.evidence["canonical_partner_retained_for_fixed_event_invariant"] = True
                resolved.append(event)
                continue
            event.evidence["geometric_partner_gt_id"] = event.partner_gt_id
            event.evidence["geometric_role"] = event.role
            event.evidence["canonical_partner_suppressed_for_fixed_event_invariant"] = True
            event.partner_gt_id = None
            event.role = None
            resolved.append(event)
    duplicate_keys = len({_fixed_event_key(event) for event in resolved}) != len(resolved)
    if duplicate_keys:
        raise AssertionError(
            "Fixed-corpus partner-collision resolution still produced duplicate keys"
        )
    return resolved


_STRUCTURAL_EVIDENCE_KEYS = {
    "fn_frame_count",
    "fn_frames",
    "gap_frame_count",
    "gap_frames",
    "frag_min_gap",
    "previous_match_frame",
    "new_match_frame",
    "intervening_frame_count",
    "max_static_iou",
    "hijack_iou_threshold",
    "match_iou_threshold",
    "hijack_frame_count",
    "max_prior_association_gap",
    "prior_association_frame",
    "merged_components",
    "merged_component_count",
}


def reattribute_canonical_events(
    baseline_events: list[ErrorEvent],
    sequence: SequenceData,
    matching: MatchingResult,
    config: PipelineConfig,
) -> tuple[list[ErrorEvent], list[ErrorEvent]]:
    """Relabel a fixed canonical corpus without changing event identity.

    Pass A uses this path because the v1 merger depended on attributed partners.
    Locking the v1 component groups makes G1 a true redistribution test rather
    than a comparison contaminated by changed merge behavior.
    """
    context = _analysis_context(sequence, config)
    attributed: list[ErrorEvent] = []
    for baseline in baseline_events:
        structural = {
            key: value
            for key, value in baseline.evidence.items()
            if key in _STRUCTURAL_EVIDENCE_KEYS
        }
        structural["baseline_cause"] = baseline.cause
        structural["baseline_partner_gt_id"] = baseline.partner_gt_id
        partner = (
            baseline.partner_gt_id
            if baseline.error_type == "FP_ID_HIJACK"
            else None
        )
        clean = replace(
            baseline,
            partner_gt_id=partner,
            role=None,
            cause=None,
            cause_confidence=None,
            evidence=structural,
        )
        relabelled = attribute_event(clean, sequence, matching, config, context)
        relabelled.event_id = baseline.event_id
        attributed.append(relabelled)
    attributed = _resolve_fixed_corpus_partner_collisions(attributed)
    excluded = [event for event in attributed if event.cause == "EXCLUDED_CROWD"]
    included = [event for event in attributed if event.cause != "EXCLUDED_CROWD"]
    return included, excluded
