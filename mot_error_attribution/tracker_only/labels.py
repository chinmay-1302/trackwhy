"""P3 Option 1: inherit primary_error; subtypes stay regime-A semantics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .io import (
    CONDITION_FROM_CAUSE,
    ERROR_KIND_TO_PRIMARY,
    LS_EXPECTED_COUNTS,
    NONCORE_KINDS,
    PLAN_EXPECTED_COUNTS,
    STATUS_FROM_KIND,
    VISIBILITY_FROM_LS,
)

# Manual-review overlays win over GT-v2.1 for these clips.
MANUAL_REVIEW_SUBTYPES: dict[str, str | None] = {
    "C068": "duplicate_track",
    "C072": "duplicate_track",
    "C153": "duplicate_track",
    "C155": "identity_change_unspecified",
    "C158": "fragmentation",
    "C160": "fragmentation",
    "C169": "fragmentation",
    "C174": None,
    "C177": "fragmentation",
    "C182": "spontaneous_id_reassignment",
    "C190": None,
    "C202": "fragmentation",
    "C205": "identity_change_unspecified",
    "C211": "identity_change_unspecified",
    "C214": "identity_change_unspecified",
    "C222": "fragmentation",
    "C254": "identity_transfer",
}


def identity_subtype_for(
    clip_id: str,
    *,
    primary_error: str | None,
    gt_v2: Mapping[str, Any] | None,
) -> str | None:
    if primary_error != "identity_change":
        return None
    if clip_id in MANUAL_REVIEW_SUBTYPES:
        return MANUAL_REVIEW_SUBTYPES[clip_id]
    if gt_v2:
        value = gt_v2.get("identity_change_subtype")
        if value not in (None, ""):
            return str(value)
    return None


def drift_subtype_for(
    primary_error: str | None,
    gt_v2: Mapping[str, Any] | None,
) -> str | None:
    if primary_error != "drift":
        return None
    if gt_v2:
        value = gt_v2.get("drift_subtype")
        if value not in (None, ""):
            return str(value)
    return "to_standing_person"


def label_clip(
    *,
    clip_id: str,
    human: Mapping[str, Any],
    gt_v2: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    error_kind = str(human.get("error_kind") or "").strip()
    status = STATUS_FROM_KIND.get(error_kind, "ambiguous" if error_kind else "target_not_identifiable")
    if error_kind in NONCORE_KINDS:
        primary = None
        status = STATUS_FROM_KIND[error_kind]
    else:
        primary = ERROR_KIND_TO_PRIMARY.get(error_kind)
        if primary is None and error_kind:
            primary = None
            status = "ambiguous"
    visibility = None
    conditions: list[str] = []
    if primary == "miss":
        visibility = VISIBILITY_FROM_LS.get(str(human.get("visibility") or "").strip())
        cause = str(human.get("cause_miss") or "").strip()
        mapped = CONDITION_FROM_CAUSE.get(cause)
        if mapped:
            conditions = [mapped]
    subtype = identity_subtype_for(clip_id, primary_error=primary, gt_v2=gt_v2)
    return {
        "clip_id": clip_id,
        "annotation_status": status,
        "primary_error": primary,
        "identity_change_subtype": subtype,
        "identity_subtype_semantics": "regime_a_semantics",
        "drift_subtype": drift_subtype_for(primary, gt_v2),
        "target_visibility": visibility,
        "visual_conditions": conditions,
        "annotation_confidence": human.get("confidence") or None,
        "source_error_kind": error_kind,
        "free_text": human.get("free_text") or "",
    }


def class_count_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    primary = Counter(
        "non_core" if row.get("primary_error") in (None, "") else str(row["primary_error"])
        for row in rows
    )
    observed = {
        "miss": int(primary.get("miss", 0)),
        "identity_change": int(primary.get("identity_change", 0)),
        "no_error": int(primary.get("no_error", 0)),
        "non_core": int(primary.get("non_core", 0)),
        "drift": int(primary.get("drift", 0)),
        "ghost_track": int(primary.get("ghost_track", 0)),
        "n_source": len(rows),
    }
    return {
        "observed": observed,
        "label_studio_expected": dict(LS_EXPECTED_COUNTS),
        "derivation_plan_expected": dict(PLAN_EXPECTED_COUNTS),
        "delta_vs_label_studio": {
            key: observed.get(key, 0) - LS_EXPECTED_COUNTS.get(key, 0)
            for key in LS_EXPECTED_COUNTS
        },
        "delta_vs_plan": {
            key: observed.get(key, 0) - PLAN_EXPECTED_COUNTS.get(key, 0)
            for key in PLAN_EXPECTED_COUNTS
        },
    }
