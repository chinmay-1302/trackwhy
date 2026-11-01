from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]  # TrackEval on NumPy>=1.24

from trackeval.datasets import MotChallenge2DBox
from trackeval.metrics import CLEAR
from trackeval.metrics.clear import linear_sum_assignment

from .config import PipelineConfig
from .models import SequenceData, TrkBox


@dataclass
class MatchingResult:
    match: dict[int, dict[int, Optional[int]]]
    reverse: dict[int, dict[int, Optional[int]]]
    gt_present: dict[int, set[int]]
    tracker_by_frame: dict[int, list[TrkBox]]
    trackeval_report: dict[str, int | float]
    local_fn: int
    local_idsw: int


def _as_array(rows: list[tuple[float, float, float, float]]) -> np.ndarray:
    if not rows:
        return np.empty((0, 4), dtype=float)
    return np.asarray(rows, dtype=float)


def _build_raw_trackeval_data(
    sequence: SequenceData, config: PipelineConfig
) -> dict[str, Any]:
    frames = sequence.selected_frames
    raw: dict[str, Any] = {
        "seq": sequence.name,
        "num_timesteps": len(frames),
        "gt_ids": [],
        "gt_dets": [],
        "gt_classes": [],
        "gt_extras": [],
        "tracker_ids": [],
        "tracker_dets": [],
        "tracker_classes": [],
        "tracker_confidences": [],
        "similarity_scores": [],
    }
    for frame in frames:
        gt_boxes = sequence.gt_by_frame.get(frame, [])
        tracker_boxes = sequence.tracker_by_frame.get(frame, [])
        gt_dets = _as_array([box.bbox for box in gt_boxes])
        trk_dets = _as_array([box.bbox for box in tracker_boxes])
        raw["gt_ids"].append(np.asarray([box.gt_id for box in gt_boxes], dtype=int))
        raw["gt_dets"].append(gt_dets)
        raw["gt_classes"].append(np.asarray([box.cls for box in gt_boxes], dtype=int))
        raw["gt_extras"].append(
            {"zero_marked": np.asarray([box.flag for box in gt_boxes], dtype=int)}
        )
        raw["tracker_ids"].append(
            np.asarray([box.trk_id for box in tracker_boxes], dtype=int)
        )
        raw["tracker_dets"].append(trk_dets)
        raw["tracker_classes"].append(np.ones(len(tracker_boxes), dtype=int))
        # TrackEval requires this field but CLEAR matching does not use it. Neutral
        # constants ensure detector confidence is neither consumed nor propagated.
        raw["tracker_confidences"].append(np.ones(len(tracker_boxes), dtype=float))
        raw["similarity_scores"].append(
            MotChallenge2DBox._calculate_box_ious(
                gt_dets, trk_dets, box_format="x0y0x1y1"
            )
        )
    return raw


def _trackeval_preprocess(raw: dict[str, Any]) -> dict[str, Any]:
    """Invoke TrackEval's own MOT20 preprocessing without filesystem adapters."""
    dataset = MotChallenge2DBox.__new__(MotChallenge2DBox)
    dataset.benchmark = "MOT20"
    dataset.do_preproc = True
    dataset.class_name_to_class_id = {
        "pedestrian": 1,
        "person_on_vehicle": 2,
        "car": 3,
        "bicycle": 4,
        "motorbike": 5,
        "non_mot_vehicle": 6,
        "static_person": 7,
        "distractor": 8,
        "occluder": 9,
        "occluder_on_ground": 10,
        "occluder_full": 11,
        "reflection": 12,
        "crowd": 13,
    }
    dataset.valid_class_numbers = list(range(1, 14))
    return dataset.get_preprocessed_seq_data(raw, "pedestrian")


def _recover_tracker_id_map(
    raw: dict[str, Any], processed: dict[str, Any]
) -> dict[int, int]:
    """Map TrackEval's contiguous IDs back to tracker IDs after preprocessing."""
    observed_pairs: dict[int, int] = {}
    survivor_original_ids: set[int] = set()
    for raw_ids, raw_dets, relabeled_ids, kept_dets in zip(
        raw["tracker_ids"],
        raw["tracker_dets"],
        processed["tracker_ids"],
        processed["tracker_dets"],
    ):
        cursor = 0
        for relabeled_id, kept_det in zip(relabeled_ids.tolist(), kept_dets):
            while cursor < len(raw_dets) and not np.array_equal(raw_dets[cursor], kept_det):
                cursor += 1
            if cursor >= len(raw_dets):
                raise AssertionError("TrackEval preprocessing unexpectedly reordered tracker boxes")
            original_id = int(raw_ids[cursor])
            previous = observed_pairs.setdefault(int(relabeled_id), original_id)
            if previous != original_id:
                raise AssertionError(
                    "Could not recover a stable original tracker-ID mapping after TrackEval preprocessing"
                )
            survivor_original_ids.add(original_id)
            cursor += 1
    expected = {index: value for index, value in enumerate(sorted(survivor_original_ids))}
    if observed_pairs != expected:
        raise AssertionError(
            f"Recovered tracker-ID map differs from TrackEval relabeling: {observed_pairs} != {expected}"
        )
    return expected


def build_matching(sequence: SequenceData, config: PipelineConfig) -> MatchingResult:
    """Build the exact CLEAR assignment timeline and verify it against TrackEval.

    TrackEval's MOT20 preprocessing, box IoU implementation, Hungarian solver, and
    CLEAR continuity rule are used. The independently exposed timeline is required
    for attribution and is reconciled to TrackEval's official aggregate result.
    """
    raw = _build_raw_trackeval_data(sequence, config)
    processed = _trackeval_preprocess(raw)
    threshold = float(config.thresholds["MATCH_IOU"])
    official = CLEAR({"THRESHOLD": threshold, "PRINT_CONFIG": False}).eval_sequence(
        processed
    )

    target_classes = set(config.classes["TARGET_CLASSES"])
    target_ids = sorted(
        {
            box.gt_id
            for frame in sequence.selected_frames
            for box in sequence.gt_by_frame.get(frame, [])
            if box.flag != 0 and box.cls in target_classes
        }
    )
    gt_inverse = {index: gt_id for index, gt_id in enumerate(target_ids)}
    if len(gt_inverse) != processed["num_gt_ids"]:
        raise AssertionError("GT relabeling differs from TrackEval preprocessing")
    tracker_inverse = _recover_tracker_id_map(raw, processed)

    match: dict[int, dict[int, Optional[int]]] = {}
    reverse: dict[int, dict[int, Optional[int]]] = {}
    gt_present: dict[int, set[int]] = {}
    previous_tracker_id = np.nan * np.zeros(processed["num_gt_ids"])
    previous_timestep_tracker_id = np.nan * np.zeros(processed["num_gt_ids"])
    local_fn = 0
    local_idsw = 0
    eps = np.finfo(float).eps

    for index, frame in enumerate(sequence.selected_frames):
        gt_ids_t = processed["gt_ids"][index]
        tracker_ids_t = processed["tracker_ids"][index]
        original_gt_ids = [gt_inverse[int(item)] for item in gt_ids_t]
        match[frame] = {gt_id: None for gt_id in original_gt_ids}
        reverse[frame] = {
            box.trk_id: None for box in sequence.tracker_by_frame.get(frame, [])
        }
        gt_present[frame] = set(original_gt_ids)

        if len(gt_ids_t) == 0:
            continue
        if len(tracker_ids_t) == 0:
            local_fn += len(gt_ids_t)
            continue

        similarity = processed["similarity_scores"][index]
        score_matrix = (
            tracker_ids_t[np.newaxis, :]
            == previous_timestep_tracker_id[gt_ids_t[:, np.newaxis]]
        )
        score_matrix = 1000 * score_matrix + similarity
        score_matrix[similarity < threshold - eps] = 0
        rows, columns = linear_sum_assignment(-score_matrix)
        keep = score_matrix[rows, columns] > eps
        rows = rows[keep]
        columns = columns[keep]

        matched_gt_rel = gt_ids_t[rows]
        matched_trk_rel = tracker_ids_t[columns]
        for gt_rel, trk_rel in zip(matched_gt_rel.tolist(), matched_trk_rel.tolist()):
            gt_id = gt_inverse[int(gt_rel)]
            trk_id = tracker_inverse[int(trk_rel)]
            match[frame][gt_id] = trk_id
            reverse[frame][trk_id] = gt_id

        previous_matches = previous_tracker_id[matched_gt_rel]
        local_idsw += int(
            np.sum(
                (~np.isnan(previous_matches))
                & (matched_trk_rel != previous_matches)
            )
        )
        previous_tracker_id[matched_gt_rel] = matched_trk_rel
        previous_timestep_tracker_id[:] = np.nan
        previous_timestep_tracker_id[matched_gt_rel] = matched_trk_rel
        local_fn += len(gt_ids_t) - len(matched_gt_rel)

    official_fn = int(official["CLR_FN"])
    official_idsw = int(official["IDSW"])
    if local_fn != official_fn or local_idsw != official_idsw:
        raise AssertionError(
            f"TrackEval reconciliation failed for {sequence.name}: "
            f"local FN/IDSW={local_fn}/{local_idsw}, "
            f"TrackEval={official_fn}/{official_idsw}"
        )

    report_fields = (
        "CLR_TP",
        "CLR_FN",
        "CLR_FP",
        "IDSW",
        "Frag",
        "MOTA",
        "MOTP",
        "CLR_Re",
        "CLR_Pr",
    )
    report: dict[str, int | float] = {}
    for field in report_fields:
        value = official[field]
        report[field] = int(value) if field in {"CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "Frag"} else float(value)
    return MatchingResult(
        match=match,
        reverse=reverse,
        gt_present=gt_present,
        tracker_by_frame=sequence.tracker_by_frame,
        trackeval_report=report,
        local_fn=local_fn,
        local_idsw=local_idsw,
    )

