"""P2: automatic derivation verification. Do not auto-correct."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .io import GHOST_CLIP_SET

ID_PATTERN = re.compile(r"\bID\s*:?\s*(\d+)\b", re.IGNORECASE)
NEAR_FRAMES = 5
MIN_RENDERABLE_FRAMES = 5
V1_SINGLE_ID_TARGET = 72
V1_AGREEMENT_TARGET = 68
V1_AGREEMENT_RATE = 0.95

VERIFICATION_FIELDS = (
    "clip_id",
    "error_type",
    "highlight_track_id",
    "named_track_ids",
    "n_named_track_ids",
    "v1_single_id",
    "v1_agreement",
    "v1_multi_contains_pin",
    "v1_flag",
    "v2a_ok",
    "v2a_flag",
    "v2b_expected",
    "v2b_ok",
    "v2b_flag",
    "v3_lifetime",
    "v3_ok",
    "v3_flag",
    "unresolved_v1",
    "unresolved_v2",
    "exclude_from_core",
    "unscored_reason",
    "is_ghost_source",
)

RULE_HUMAN_CONSISTENCY_FIELDS = (
    "clip_id",
    "error_type",
    "primary_error",
    "rule_prediction_met",
    "agreement_class",
)

RULE_IMPLIED_PRIMARY = {
    "FN": "miss",
    "IDSW": "identity_change",
    "FRAG_NO_SWITCH": "identity_change",
    "FP_ID_HIJACK": "drift",
}

V2A_NULL_PIN = "v2a_null_pin"
V2A_NOT_RENDERABLE = "v2a_pinned_track_not_renderable"
V2A_NO_MATCH_FRAMES = "v2a_no_match_frames"
V2A_FIRST_MATCH_OUT_OF_WINDOW = "v2a_first_match_out_of_window"


def named_track_ids(free_text: str) -> list[int]:
    seen: list[int] = []
    for match in ID_PATTERN.finditer(free_text or ""):
        value = int(match.group(1))
        if value not in seen:
            seen.append(value)
    return seen


def v1_fields(
    *,
    clip_id: str,
    highlight_track_id: int | None,
    free_text: str,
    matched_track_ids: Sequence[int] = (),
) -> dict[str, Any]:
    named = named_track_ids(free_text)
    single = len(named) == 1
    ghost = clip_id in GHOST_CLIP_SET
    agreement: bool | None = None
    multi_contains: bool | None = None
    flag = ""
    matched = {int(item) for item in matched_track_ids}
    if highlight_track_id is not None:
        matched.add(int(highlight_track_id))
    if ghost:
        flag = ""
    elif single:
        named_id = named[0]
        pin_match = highlight_track_id is not None and named_id == int(highlight_track_id)
        in_matched = named_id in matched
        # Captions describe HighlightPlan IDs. Those are in-window TrackEval matches
        # to the same GT, but not always the first-match pin. Count either as
        # agreement so a window-boundary pin choice is not treated as a wrong GT.
        agreement = pin_match or in_matched
        if not agreement:
            flag = "v1_single_id_mismatch"
        elif not pin_match:
            flag = "v1_named_other_matched_track"
    elif len(named) >= 2:
        multi_contains = highlight_track_id is not None and int(highlight_track_id) in named
        if not multi_contains:
            flag = "v1_multi_id_missing_pin"
    return {
        "named_track_ids": named,
        "n_named_track_ids": len(named),
        "v1_single_id": single,
        "v1_agreement": agreement,
        "v1_multi_contains_pin": multi_contains,
        "v1_flag": flag,
        "unresolved_v1": bool(flag == "v1_single_id_mismatch"),
    }


def _in_window(frames: Sequence[int], start: int, end: int) -> list[int]:
    lo, hi = int(start), int(end)
    return [int(frame) for frame in frames if lo <= int(frame) <= hi]


def v2_idsw_ok(
    *,
    pinned_track_match_frames: Sequence[int],
    frame_end: int,
    near: int = NEAR_FRAMES,
) -> bool:
    """IDSW: the pinned track has stopped matching by the end of the switch.

    Stopping earlier than ``frame_end`` is consistent with a first-in-window pin
    that is replaced before the switch completes. Continuing to match *after*
    ``frame_end + near`` contradicts identity switch.
    """
    if not pinned_track_match_frames:
        return False
    last_match = max(int(frame) for frame in pinned_track_match_frames)
    return last_match <= int(frame_end) + int(near)


def v2_fn_ok(
    *,
    pinned_track_match_frames: Sequence[int],
    frame_start: int,
    frame_end: int,
    clip_frames: Sequence[int],
) -> bool:
    event_frames = _in_window(clip_frames, frame_start, frame_end)
    if not event_frames:
        return False
    matched = set(map(int, pinned_track_match_frames))
    return any(frame not in matched for frame in event_frames)


def v2_drift_ok(
    *,
    pinned_track_lifetime: Sequence[int],
    pinned_track_match_frames: Sequence[int],
    frame_end: int,
    near: int = NEAR_FRAMES,
    clip_frames: Sequence[int] = (),
) -> bool:
    if clip_frames and not any(int(frame) > int(frame_end) for frame in clip_frames):
        return True
    exists_after = any(int(frame) > int(frame_end) for frame in pinned_track_lifetime)
    matches_after = any(
        int(frame) > int(frame_end) + int(near) for frame in pinned_track_match_frames
    )
    return exists_after and not matches_after


def v2_frag_ok(
    *,
    pinned_track_lifetime: Sequence[int],
    clip_frames: Sequence[int],
    highlight_track_id: int | None,
    tracker_by_frame: Mapping[int, Sequence[int]] = (),
) -> bool:
    """FRAG_NO_SWITCH: pinned track dies in-window and its ID is not reused."""
    if highlight_track_id is None or not pinned_track_lifetime or not clip_frames:
        return False
    last_life = max(int(frame) for frame in pinned_track_lifetime)
    last_clip = max(int(frame) for frame in clip_frames)
    if last_life >= last_clip:
        return False
    wanted = int(highlight_track_id)
    by_frame = tracker_by_frame or {}
    return not any(
        int(frame) > last_life and wanted in set(map(int, by_frame.get(int(frame), ())))
        for frame in clip_frames
    )


def v2_hijack_ok(
    *,
    pinned_track_lifetime: Sequence[int],
    pinned_track_match_frames: Sequence[int],
    matched_frames: Sequence[int],
    frame_end: int,
    clip_frames: Sequence[int] = (),
    near: int = NEAR_FRAMES,
) -> bool:
    """FP_ID_HIJACK: pin persists unmatched, or another track takes the GT."""
    drift = v2_drift_ok(
        pinned_track_lifetime=pinned_track_lifetime,
        pinned_track_match_frames=pinned_track_match_frames,
        frame_end=frame_end,
        near=near,
        clip_frames=clip_frames,
    )
    last_pin = max(map(int, pinned_track_match_frames), default=None)
    last_any = max(map(int, matched_frames), default=None)
    foreign = (
        last_pin is not None and last_any is not None and last_any > last_pin
    )
    return drift or foreign


def v2a_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Derivation soundness. Independent of rule and human labels.

    A short in-window lifetime is reported for the eye check (original V3 was a
    flag, not a gate). It must not unscore: late-first-match clips can have a
    pin that only appears for a few frames, and those stay in the scored set.
    """
    gating: list[str] = []
    reported: list[str] = []
    highlight = row.get("highlight_track_id")
    if highlight in (None, ""):
        gating.append(V2A_NULL_PIN)
    lifetime = int(row.get("n_pinned_track_lifetime") or 0)
    if lifetime < MIN_RENDERABLE_FRAMES:
        reported.append(V2A_NOT_RENDERABLE)
    n_match = int(row.get("n_pinned_track_match_frames") or 0)
    if n_match < 1:
        gating.append(V2A_NO_MATCH_FRAMES)
    clip_frames = [int(frame) for frame in row.get("clip_frames") or ()]
    first_source = row.get("first_match_frame")
    first_clip = row.get("first_match_clip_frame")
    in_window = False
    if first_source not in (None, "") and clip_frames:
        in_window = int(first_source) in set(clip_frames)
    if not in_window and first_clip not in (None, "") and clip_frames:
        in_window = 1 <= int(first_clip) <= len(clip_frames)
    if highlight not in (None, "") and not in_window:
        gating.append(V2A_FIRST_MATCH_OUT_OF_WINDOW)
    reported = gating + reported
    ok = not gating
    return {
        "v2a_ok": ok,
        "v2a_flag": ";".join(reported),
        "unresolved_v2": not ok,
    }


def v2b_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Rule-vs-pin behaviour. Diagnostic only; never gates."""
    error_type = str(row.get("error_type") or "")
    start = int(row["frame_start"])
    end = int(row["frame_end"])
    match_frames = list(row.get("pinned_track_match_frames") or ())
    lifetime = list(row.get("pinned_track_lifetime") or ())
    clip_frames = list(row.get("clip_frames") or ())
    matched_frames = list(row.get("matched_frames") or ())
    expected = ""
    ok: bool | None = None
    if error_type == "FN":
        expected = "match_gap_overlaps_event_window"
        ok = v2_fn_ok(
            pinned_track_match_frames=match_frames,
            frame_start=start,
            frame_end=end,
            clip_frames=clip_frames,
        )
    elif error_type == "IDSW":
        expected = "pinned_stops_matching_near_frame_end"
        ok = v2_idsw_ok(pinned_track_match_frames=match_frames, frame_end=end)
    elif error_type == "FRAG_NO_SWITCH":
        expected = "pinned_track_terminates_id_not_reused"
        ok = v2_frag_ok(
            pinned_track_lifetime=lifetime,
            clip_frames=clip_frames,
            highlight_track_id=row.get("highlight_track_id"),
            tracker_by_frame=row.get("tracker_by_frame") or {},
        )
    elif error_type == "FP_ID_HIJACK":
        expected = "exists_unmatched_or_foreign_takeover"
        ok = v2_hijack_ok(
            pinned_track_lifetime=lifetime,
            pinned_track_match_frames=match_frames,
            matched_frames=matched_frames,
            frame_end=end,
            clip_frames=clip_frames,
        )
    flag = ""
    if expected and ok is False:
        flag = f"v2b_{error_type.lower()}_not_met"
    return {
        "v2b_expected": expected,
        "v2b_ok": ok,
        "v2b_flag": flag,
    }


def agreement_class(
    *,
    error_type: str,
    primary_error: str | None,
    rule_prediction_met: bool | None,
) -> str:
    implied = RULE_IMPLIED_PRIMARY.get(str(error_type) or "")
    primary = None if primary_error in (None, "") else str(primary_error)
    if primary == implied:
        return "rule_and_human_agree"
    if primary == "no_error":
        return "rule_fires_human_sees_nothing"
    return "rule_and_human_differ_in_kind"


def rule_human_row(
    *,
    clip_id: str,
    error_type: str,
    primary_error: str | None,
    rule_prediction_met: bool | None,
) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "error_type": error_type,
        "primary_error": "" if primary_error in (None, "") else primary_error,
        "rule_prediction_met": rule_prediction_met,
        "agreement_class": agreement_class(
            error_type=error_type,
            primary_error=primary_error,
            rule_prediction_met=rule_prediction_met,
        ),
    }


def v2_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Back-compat wrapper: V2b diagnostic fields under the old names."""
    v2b = v2b_fields(row)
    return {
        "v2_expected": v2b["v2b_expected"],
        "v2_ok": v2b["v2b_ok"],
        "v2_flag": v2b["v2b_flag"],
        "unresolved_v2": False,
    }


def v3_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    lifetime = len(row.get("pinned_track_lifetime") or ())
    ok = lifetime >= MIN_RENDERABLE_FRAMES or bool(row.get("exclude_from_core"))
    flag = "" if ok else V2A_NOT_RENDERABLE
    return {
        "v3_lifetime": lifetime,
        "v3_ok": ok,
        "v3_flag": flag,
    }


def _unscored_reason(
    *,
    row: Mapping[str, Any],
    v1_unresolved: bool,
    v2a_flag: str,
) -> str:
    reasons: list[str] = []
    if "no_match_in_window" in (row.get("flags") or ()):
        reasons.append("no_match_in_window")
    if v1_unresolved:
        reasons.append("v1_single_id_mismatch")
    if v2a_flag:
        reasons.extend(item for item in str(v2a_flag).split(";") if item)
    return ";".join(dict.fromkeys(reasons))


def verify_clip(
    row: Mapping[str, Any],
    *,
    free_text: str,
) -> dict[str, Any]:
    v1 = v1_fields(
        clip_id=str(row["clip_id"]),
        highlight_track_id=row.get("highlight_track_id"),
        free_text=free_text,
        matched_track_ids=row.get("matched_track_ids") or (),
    )
    v2a = v2a_fields(row)
    v2b = v2b_fields(row)
    v3 = v3_fields(row)
    ghost = str(row["clip_id"]) in GHOST_CLIP_SET
    v1_unresolved = v1["unresolved_v1"] and not ghost
    exclude = bool(row.get("exclude_from_core")) or v1_unresolved or not v2a["v2a_ok"]
    return {
        "clip_id": row["clip_id"],
        "error_type": row.get("error_type", ""),
        "highlight_track_id": row.get("highlight_track_id"),
        "named_track_ids": ";".join(str(item) for item in v1["named_track_ids"]),
        "n_named_track_ids": v1["n_named_track_ids"],
        "v1_single_id": v1["v1_single_id"],
        "v1_agreement": v1["v1_agreement"],
        "v1_multi_contains_pin": v1["v1_multi_contains_pin"],
        "v1_flag": v1["v1_flag"],
        "v2a_ok": v2a["v2a_ok"],
        "v2a_flag": v2a["v2a_flag"],
        "v2b_expected": v2b["v2b_expected"],
        "v2b_ok": v2b["v2b_ok"],
        "v2b_flag": v2b["v2b_flag"],
        "v3_lifetime": v3["v3_lifetime"],
        "v3_ok": v3["v3_ok"],
        "v3_flag": v3["v3_flag"],
        "unresolved_v1": v1_unresolved,
        "unresolved_v2": v2a["unresolved_v2"],
        "exclude_from_core": exclude,
        "unscored_reason": _unscored_reason(
            row=row, v1_unresolved=v1_unresolved, v2a_flag=v2a["v2a_flag"]
        )
        if exclude
        else "",
        "is_ghost_source": ghost,
    }


def v1_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evaluated = [
        row
        for row in rows
        if row.get("v1_single_id") and not row.get("is_ghost_source")
    ]
    agreements = [row for row in evaluated if row.get("v1_agreement")]
    n_eval = len(evaluated)
    n_agree = len(agreements)
    rate = (n_agree / n_eval) if n_eval else 0.0
    passed = n_eval == 0 or (
        n_agree >= V1_AGREEMENT_TARGET
        if n_eval >= V1_SINGLE_ID_TARGET
        else rate >= V1_AGREEMENT_RATE
    )
    if n_eval >= V1_SINGLE_ID_TARGET:
        passed = n_agree >= V1_AGREEMENT_TARGET
    return {
        "n_single_id_evaluated": n_eval,
        "n_single_id_agree": n_agree,
        "agreement_rate": rate,
        "target_agree": V1_AGREEMENT_TARGET,
        "target_n": V1_SINGLE_ID_TARGET,
        "passed": passed,
    }
