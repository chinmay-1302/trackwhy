"""P1: pin the highlighted tracker ID from frozen TrackEval matching."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .io import join_frames


FLAG_NO_MATCH = "no_match_in_window"
FLAG_LATE_FIRST_MATCH = "late_first_match"

DERIVATION_FIELDS = (
    "clip_id",
    "event_id",
    "sequence",
    "error_type",
    "focal_gt_id",
    "frame_start",
    "frame_end",
    "n_clip_frames",
    "highlight_track_id",
    "first_match_frame",
    "first_match_clip_frame",
    "matched_frames",
    "pinned_track_lifetime",
    "pinned_track_match_frames",
    "n_matched_frames",
    "n_pinned_track_lifetime",
    "n_pinned_track_match_frames",
    "n_distinct_matched_tracks",
    "matched_track_ids",
    "flags",
    "exclude_from_core",
    "clip_frames_sha256",
    "regime_a_manifest_sha256",
)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def pin_highlight_track(
    matching_by_frame: Sequence[Mapping[str, Any]],
    *,
    clip_frames: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Return the first in-window TrackEval match to the focal GT identity."""
    matched_frames: list[int] = []
    distinct: list[int] = []
    highlight_track_id: int | None = None
    first_match_frame: int | None = None
    first_match_clip_frame: int | None = None
    rows = list(matching_by_frame)
    n_window = len(clip_frames) if clip_frames is not None else len(rows)
    late_cutoff = (2 * n_window) // 3
    for index, row in enumerate(rows):
        track_id = _optional_int(row.get("matched_prediction_track_id"))
        if track_id is None:
            continue
        source_frame = int(row["source_frame"])
        matched_frames.append(source_frame)
        if track_id not in distinct:
            distinct.append(track_id)
        if highlight_track_id is None:
            highlight_track_id = track_id
            first_match_frame = source_frame
            first_match_clip_frame = int(row.get("clip_frame") or index + 1)
    flags: list[str] = []
    if highlight_track_id is None:
        flags.append(FLAG_NO_MATCH)
    elif first_match_clip_frame is not None and first_match_clip_frame - 1 >= late_cutoff:
        flags.append(FLAG_LATE_FIRST_MATCH)
    return {
        "highlight_track_id": highlight_track_id,
        "first_match_frame": first_match_frame,
        "first_match_clip_frame": first_match_clip_frame,
        "matched_frames": matched_frames,
        "matched_track_ids": distinct,
        "n_distinct_matched_tracks": len(distinct),
        "flags": flags,
    }


def pinned_lifetime(
    highlight_track_id: int | None,
    clip_frames: Sequence[int],
    tracker_by_frame: Mapping[int, Sequence[int]],
) -> list[int]:
    if highlight_track_id is None:
        return []
    wanted = int(highlight_track_id)
    return [
        int(frame)
        for frame in clip_frames
        if wanted in set(map(int, tracker_by_frame.get(int(frame), ())))
    ]


def pinned_match_frames(
    highlight_track_id: int | None,
    matching_by_frame: Sequence[Mapping[str, Any]],
) -> list[int]:
    if highlight_track_id is None:
        return []
    wanted = int(highlight_track_id)
    return [
        int(row["source_frame"])
        for row in matching_by_frame
        if _optional_int(row.get("matched_prediction_track_id")) == wanted
    ]


def highlight_id_for_frame(
    highlight_track_id: int | None,
    frame: int,
    tracker_by_frame: Mapping[int, Sequence[int]],
) -> int | None:
    if highlight_track_id is None:
        return None
    wanted = int(highlight_track_id)
    if wanted in set(map(int, tracker_by_frame.get(int(frame), ()))):
        return wanted
    return None


def derive_clip(bundle: Mapping[str, Any]) -> dict[str, Any]:
    pin = pin_highlight_track(
        bundle["matching_by_frame"],
        clip_frames=bundle["clip_frames"],
    )
    lifetime = pinned_lifetime(
        pin["highlight_track_id"],
        bundle["clip_frames"],
        bundle["tracker_by_frame"],
    )
    match_frames = pinned_match_frames(
        pin["highlight_track_id"],
        bundle["matching_by_frame"],
    )
    flags = list(pin["flags"])
    return {
        "clip_id": bundle["clip_id"],
        "event_id": bundle["event_id"],
        "sequence": bundle["sequence"],
        "error_type": bundle["error_type"],
        "focal_gt_id": int(bundle["focal_gt_id"]),
        "frame_start": int(bundle["frame_start"]),
        "frame_end": int(bundle["frame_end"]),
        "n_clip_frames": len(bundle["clip_frames"]),
        "clip_frames": list(bundle["clip_frames"]),
        "highlight_track_id": pin["highlight_track_id"],
        "first_match_frame": pin["first_match_frame"],
        "first_match_clip_frame": pin["first_match_clip_frame"],
        "matched_frames": pin["matched_frames"],
        "pinned_track_lifetime": lifetime,
        "pinned_track_match_frames": match_frames,
        "n_matched_frames": len(pin["matched_frames"]),
        "n_pinned_track_lifetime": len(lifetime),
        "n_pinned_track_match_frames": len(match_frames),
        "n_distinct_matched_tracks": pin["n_distinct_matched_tracks"],
        "matched_track_ids": list(pin["matched_track_ids"]),
        "flags": flags,
        "exclude_from_core": FLAG_NO_MATCH in flags,
        "clip_frames_sha256": bundle["clip_frames_sha256"],
        "regime_a_manifest_sha256": bundle.get("regime_a_manifest_sha256", ""),
    }


def derivation_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **{field: row.get(field, "") for field in DERIVATION_FIELDS},
        "highlight_track_id": (
            "" if row.get("highlight_track_id") is None else row["highlight_track_id"]
        ),
        "first_match_frame": (
            "" if row.get("first_match_frame") is None else row["first_match_frame"]
        ),
        "first_match_clip_frame": (
            "" if row.get("first_match_clip_frame") is None else row["first_match_clip_frame"]
        ),
        "matched_frames": join_frames(row.get("matched_frames") or ()),
        "pinned_track_lifetime": join_frames(row.get("pinned_track_lifetime") or ()),
        "pinned_track_match_frames": join_frames(
            row.get("pinned_track_match_frames") or ()
        ),
        "matched_track_ids": join_frames(row.get("matched_track_ids") or ()),
        "flags": ";".join(row.get("flags") or ()),
        "exclude_from_core": bool(row.get("exclude_from_core")),
    }
