from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable, Optional

from .config import PipelineConfig
from .geometry import iou
from .matching import MatchingResult
from .models import ErrorEvent, SequenceData, TrkBox


def event_id_for(
    sequence: str,
    tracker: str,
    error_type: str,
    frame_start: int,
    frame_end: int,
    focal_gt_id: int,
    partner_gt_id: Optional[int] = None,
) -> str:
    source = (
        f"{sequence}|{tracker}|{error_type}|{frame_start}|{frame_end}|"
        f"{focal_gt_id}|{partner_gt_id}"
    )
    suffix = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    return f"{sequence}__{tracker}__{error_type}__{suffix}"


def _new_event(
    sequence: str,
    tracker: str,
    error_type: str,
    frame_start: int,
    frame_end: int,
    focal_gt_id: int,
    *,
    partner_gt_id: Optional[int] = None,
    prev_trk_id: Optional[int] = None,
    new_trk_id: Optional[int] = None,
    evidence: Optional[dict] = None,
) -> ErrorEvent:
    return ErrorEvent(
        event_id=event_id_for(
            sequence,
            tracker,
            error_type,
            frame_start,
            frame_end,
            focal_gt_id,
            partner_gt_id,
        ),
        sequence=sequence,
        tracker=tracker,
        error_type=error_type,
        frame_start=frame_start,
        frame_end=frame_end,
        focal_gt_id=focal_gt_id,
        partner_gt_id=partner_gt_id,
        evidence={} if evidence is None else evidence,
        prev_trk_id=prev_trk_id,
        new_trk_id=new_trk_id,
    )


def _none_runs(timeline: list[tuple[int, Optional[int]]]) -> Iterable[tuple[int, int]]:
    start: Optional[int] = None
    for index, (_, tracker_id) in enumerate(timeline):
        if tracker_id is None and start is None:
            start = index
        elif tracker_id is not None and start is not None:
            yield start, index - 1
            start = None
    if start is not None:
        yield start, len(timeline) - 1


def _extract_identity_events(
    sequence: SequenceData, matching: MatchingResult, config: PipelineConfig
) -> list[ErrorEvent]:
    events: list[ErrorEvent] = []
    all_gt_ids = sorted(set().union(*matching.gt_present.values()))
    frag_min_gap = int(config.thresholds["FRAG_MIN_GAP"])
    min_fn_run = int(config.thresholds["MIN_FN_RUN_FRAMES"])
    for gt_id in all_gt_ids:
        timeline = [
            (frame, matching.match[frame][gt_id])
            for frame in sequence.selected_frames
            if gt_id in matching.gt_present[frame]
        ]
        for start_index, end_index in _none_runs(timeline):
            missed_frames = [frame for frame, _ in timeline[start_index : end_index + 1]]
            fn_event = _new_event(
                sequence.name,
                config.tracker_name,
                "FN",
                missed_frames[0],
                missed_frames[-1],
                gt_id,
                evidence={
                    "fn_frame_count": len(missed_frames),
                    "fn_frames": missed_frames,
                    "min_fn_run_frames": min_fn_run,
                },
            )
            if len(missed_frames) < min_fn_run:
                fn_event.cause = "excluded_micro_gap"
                events.append(fn_event)
                continue
            events.append(fn_event)
            gap_length = end_index - start_index + 1
            before = timeline[start_index - 1][1] if start_index > 0 else None
            after = timeline[end_index + 1][1] if end_index + 1 < len(timeline) else None
            if gap_length >= frag_min_gap and before is not None and before == after:
                events.append(
                    _new_event(
                        sequence.name,
                        config.tracker_name,
                        "FRAG_NO_SWITCH",
                        missed_frames[0],
                        missed_frames[-1],
                        gt_id,
                        prev_trk_id=before,
                        new_trk_id=after,
                        evidence={
                            "gap_frame_count": gap_length,
                            "gap_frames": missed_frames,
                            "frag_min_gap": frag_min_gap,
                        },
                    )
                )

        non_none = [(frame, trk_id) for frame, trk_id in timeline if trk_id is not None]
        for (frame_a, id_a), (frame_b, id_b) in zip(non_none, non_none[1:]):
            if id_a == id_b:
                continue
            events.append(
                _new_event(
                    sequence.name,
                    config.tracker_name,
                    "IDSW",
                    frame_a,
                    frame_b,
                    gt_id,
                    prev_trk_id=id_a,
                    new_trk_id=id_b,
                    evidence={
                        "previous_match_frame": frame_a,
                        "new_match_frame": frame_b,
                        "intervening_frame_count": max(0, frame_b - frame_a - 1),
                    },
                )
            )
    return events


def _best_static_overlap(
    tracker_box: TrkBox,
    sequence: SequenceData,
    static_classes: set[int],
) -> tuple[Optional[int], float]:
    best_id: Optional[int] = None
    best_iou = 0.0
    for candidate in sequence.gt_by_frame.get(tracker_box.frame, []):
        if candidate.cls not in static_classes:
            continue
        overlap = iou(tracker_box.bbox, candidate.bbox)
        if overlap > best_iou:
            best_iou = overlap
            best_id = candidate.gt_id
    return best_id, best_iou


def _extract_static_hijacks(
    sequence: SequenceData, matching: MatchingResult, config: PipelineConfig
) -> list[ErrorEvent]:
    threshold = float(config.thresholds["HIJACK_IOU"])
    max_prior_gap = int(config.thresholds["EVENT_WINDOW"]) * 2
    static_classes = set(config.classes["STATIC_PERSON"])
    boxes_by_tracker: dict[int, dict[int, TrkBox]] = defaultdict(dict)
    for frame, boxes in sequence.tracker_by_frame.items():
        for box in boxes:
            boxes_by_tracker[box.trk_id][frame] = box

    events: list[ErrorEvent] = []
    for tracker_id, frame_boxes in sorted(boxes_by_tracker.items()):
        prior_gt: Optional[int] = None
        prior_gt_frame: Optional[int] = None
        run: Optional[dict] = None

        def close_run() -> None:
            nonlocal run
            if run is None:
                return
            events.append(
                _new_event(
                    sequence.name,
                    config.tracker_name,
                    "FP_ID_HIJACK",
                    run["frame_start"],
                    run["frame_end"],
                    run["focal_gt_id"],
                    partner_gt_id=run["partner_gt_id"],
                    prev_trk_id=tracker_id,
                    evidence={
                        "max_static_iou": run["max_static_iou"],
                        "hijack_iou_threshold": threshold,
                        "max_prior_association_gap": max_prior_gap,
                        "prior_association_frame": run["prior_association_frame"],
                        "hijack_frame_count": run["frame_count"],
                    },
                )
            )
            run = None

        for frame in sequence.selected_frames:
            tracker_box = frame_boxes.get(frame)
            if tracker_box is None:
                close_run()
                continue
            matched_gt = matching.reverse[frame].get(tracker_id)
            if matched_gt is not None:
                close_run()
                prior_gt = matched_gt
                prior_gt_frame = frame
                continue
            static_id, overlap = _best_static_overlap(
                tracker_box, sequence, static_classes
            )
            prior_is_recent = (
                prior_gt is not None
                and prior_gt_frame is not None
                and frame - prior_gt_frame <= max_prior_gap
            )
            if not prior_is_recent or static_id is None or overlap < threshold:
                close_run()
                continue
            if (
                run is not None
                and run["frame_end"] == frame - 1
                and run["partner_gt_id"] == static_id
                and run["focal_gt_id"] == prior_gt
            ):
                run["frame_end"] = frame
                run["frame_count"] += 1
                run["max_static_iou"] = max(run["max_static_iou"], overlap)
            else:
                close_run()
                run = {
                    "frame_start": frame,
                    "frame_end": frame,
                    "frame_count": 1,
                    "focal_gt_id": prior_gt,
                    "partner_gt_id": static_id,
                    "max_static_iou": overlap,
                    "prior_association_frame": prior_gt_frame,
                }
        close_run()
    return events


def dedupe_exact(events: list[ErrorEvent]) -> list[ErrorEvent]:
    unique: dict[tuple, ErrorEvent] = {}
    for event in events:
        key = (
            event.sequence,
            event.tracker,
            event.error_type,
            event.frame_start,
            event.frame_end,
            event.focal_gt_id,
            event.partner_gt_id,
            event.prev_trk_id,
            event.new_trk_id,
        )
        unique.setdefault(key, event)
    return sorted(
        unique.values(),
        key=lambda item: (
            item.frame_start,
            item.frame_end,
            item.focal_gt_id,
            item.error_type,
        ),
    )


def fn_component_length(event: ErrorEvent) -> int:
    merged = event.evidence.get("merged_components")
    components = merged if isinstance(merged, list) and merged else None
    if components is None:
        if event.error_type != "FN":
            return 0
        value = event.evidence.get("fn_frame_count")
        if value is None:
            return event.frame_end - event.frame_start + 1
        return int(value)
    total = 0
    for component in components:
        if component.get("error_type") != "FN":
            continue
        value = component.get("fn_frame_count")
        if value is None:
            value = int(component["frame_end"]) - int(component["frame_start"]) + 1
        total += int(value)
    return total


def split_micro_gap_events(
    events: list[ErrorEvent], config: PipelineConfig
) -> tuple[list[ErrorEvent], list[ErrorEvent]]:
    min_fn = int(config.thresholds["MIN_FN_RUN_FRAMES"])
    kept: list[ErrorEvent] = []
    micro: list[ErrorEvent] = []
    for event in events:
        if event.cause == "excluded_micro_gap":
            micro.append(event)
            continue
        length = fn_component_length(event)
        if length and length < min_fn:
            event.cause = "excluded_micro_gap"
            micro.append(event)
            continue
        kept.append(event)
    return kept, micro


def extract_events(
    sequence: SequenceData, matching: MatchingResult, config: PipelineConfig
) -> list[ErrorEvent]:
    events = _extract_identity_events(sequence, matching, config)
    events.extend(_extract_static_hijacks(sequence, matching, config))
    return dedupe_exact(events)
