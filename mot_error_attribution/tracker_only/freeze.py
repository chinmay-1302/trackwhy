"""Assemble and freeze the regime-B ground-truth JSON."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import GHOST_CLIP_SET, write_atomic_json
from .prompts import TRACKER_ONLY_PROMPT, prompt_sha256

SCORED_PRIMARY_ERRORS = frozenset({"miss", "identity_change", "no_error", "ghost_track"})
EMISSION_ONLY_ERRORS = frozenset({"ghost_track", "drift", "duplicate_track"})


def _count_primary(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(
        "non_core" if row.get("primary_error") in (None, "") else str(row["primary_error"])
        for row in records
    )
    return dict(sorted(counts.items()))


def build_record(
    *,
    clip_id: str,
    source_clip_id: str,
    derivation: Mapping[str, Any],
    labels: Mapping[str, Any],
    highlight_track_id: int | None,
    role: str,
) -> dict[str, Any]:
    primary = labels.get("primary_error") if role != "ghost" else "ghost_track"
    status = "valid" if role == "ghost" else labels.get("annotation_status")
    subtype = None if role == "ghost" else labels.get("identity_change_subtype")
    record = {
        "schema_version": "gt_tracker_only_v1",
        "clip_id": clip_id,
        "source_clip_id": source_clip_id,
        "record_role": role,
        "regime": "tracker_only",
        "event_id": derivation.get("event_id"),
        "sequence": derivation.get("sequence"),
        "error_type": derivation.get("error_type"),
        "focal_gt_id": derivation.get("focal_gt_id"),
        "highlight_track_id": highlight_track_id,
        "first_match_frame": derivation.get("first_match_frame"),
        "first_match_clip_frame": derivation.get("first_match_clip_frame"),
        "n_matched_frames": derivation.get("n_matched_frames"),
        "n_pinned_track_lifetime": derivation.get("n_pinned_track_lifetime"),
        "n_pinned_track_match_frames": derivation.get("n_pinned_track_match_frames"),
        "n_distinct_matched_tracks": derivation.get("n_distinct_matched_tracks"),
        "derivation_flags": list(derivation.get("flags") or []),
        "annotation_status": status,
        "primary_error": primary,
        "identity_change_subtype": subtype,
        "identity_subtype_semantics": "regime_a_semantics",
        "drift_subtype": None if role == "ghost" else labels.get("drift_subtype"),
        "target_visibility": None if role == "ghost" else labels.get("target_visibility"),
        "visual_conditions": [] if role == "ghost" else list(labels.get("visual_conditions") or []),
        "annotation_confidence": labels.get("annotation_confidence"),
        "clip_frames": list(derivation.get("clip_frames") or []),
        "clip_frames_sha256": derivation.get("clip_frames_sha256"),
        "regime_a_manifest_sha256": derivation.get("regime_a_manifest_sha256"),
        "scored_primary": False,
        "unscored_reason": "",
    }
    non_core = primary in (None, "") or status != "valid"
    no_match = "no_match_in_window" in (derivation.get("flags") or [])
    excluded = bool(derivation.get("exclude_from_core")) or no_match
    reasons: list[str] = []
    if non_core:
        reasons.append("non_core")
    extra = str(derivation.get("unscored_reason") or "")
    if extra:
        reasons.extend(item for item in extra.split(";") if item)
    elif excluded and not non_core:
        reasons.append("excluded_from_core")
    # Drift (and other emission-only classes) stay in the corpus but are not
    # scored_primary. They still need a typed reason so V2a does not leave them blank.
    if primary in EMISSION_ONLY_ERRORS - SCORED_PRIMARY_ERRORS and status == "valid":
        reasons.append("emission_count_and_precision_only")
    record["unscored_reason"] = ";".join(dict.fromkeys(reasons))
    record["scored_primary"] = (
        primary in SCORED_PRIMARY_ERRORS
        and status == "valid"
        and not excluded
        and not no_match
    )
    if record["scored_primary"]:
        record["unscored_reason"] = ""
    return record


def assemble_records(
    *,
    derivations: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    ghost_validations: Sequence[Mapping[str, Any]],
    drop_ghost_class: bool,
) -> list[dict[str, Any]]:
    ghosts = {str(row["clip_id"]): row for row in ghost_validations}
    records: list[dict[str, Any]] = []
    for derivation in derivations:
        clip_id = str(derivation["clip_id"])
        label = labels[clip_id]
        if clip_id in GHOST_CLIP_SET:
            ghost = ghosts.get(clip_id, {})
            records.append(
                build_record(
                    clip_id=f"{clip_id}-T",
                    source_clip_id=clip_id,
                    derivation=derivation,
                    labels=label,
                    highlight_track_id=derivation.get("highlight_track_id"),
                    role="target",
                )
            )
            if not drop_ghost_class:
                ghost_id = ghost.get("ghost_track_id")
                records.append(
                    build_record(
                        clip_id=f"{clip_id}-G",
                        source_clip_id=clip_id,
                        derivation=derivation,
                        labels=label,
                        highlight_track_id=(
                            None if ghost_id in (None, "") else int(ghost_id)
                        ),
                        role="ghost",
                    )
                )
        else:
            records.append(
                build_record(
                    clip_id=clip_id,
                    source_clip_id=clip_id,
                    derivation=derivation,
                    labels=label,
                    highlight_track_id=derivation.get("highlight_track_id"),
                    role="target",
                )
            )
    records.sort(key=lambda row: str(row["clip_id"]))
    return records


def payload_hash(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_clip_id(clip_id: str) -> str:
    text = str(clip_id)
    if text.endswith("-T") or text.endswith("-G"):
        return text.rsplit("-", 1)[0]
    return text


def blocking_unresolved(
    *,
    records: Sequence[Mapping[str, Any]],
    unresolved_v1: Sequence[str],
    unresolved_v2: Sequence[str],
) -> list[str]:
    """Unresolved flags only block freeze when they remain on a scored core clip."""
    scored_sources = {
        str(row.get("source_clip_id") or _source_clip_id(str(row["clip_id"])))
        for row in records
        if row.get("scored_primary")
    }
    blocking: list[str] = []
    for clip_id in list(unresolved_v1) + list(unresolved_v2):
        source = _source_clip_id(str(clip_id))
        if source in scored_sources and clip_id not in blocking:
            blocking.append(str(clip_id))
    return blocking


def build_candidate(
    *,
    records: Sequence[Mapping[str, Any]],
    class_counts: Mapping[str, Any],
    v1_gate: Mapping[str, Any],
    ghost_decision: Mapping[str, Any],
    unresolved_v1: Sequence[str],
    unresolved_v2: Sequence[str],
    prompt_text: str = TRACKER_ONLY_PROMPT,
    preview_pending: bool = True,
) -> dict[str, Any]:
    body = {
        "schema_version": "gt_tracker_only_v1",
        "regime": "tracker_only",
        "n_source": len({row["source_clip_id"] for row in records}),
        "n_records": len(records),
        "primary_error_counts": _count_primary(records),
        "class_count_report": dict(class_counts),
        "v1_gate": dict(v1_gate),
        "ghost_decision": dict(ghost_decision),
        "unresolved_v1": list(unresolved_v1),
        "unresolved_v2": list(unresolved_v2),
        "blocking_unresolved": blocking_unresolved(
            records=records,
            unresolved_v1=unresolved_v1,
            unresolved_v2=unresolved_v2,
        ),
        "prompt_sha256": prompt_sha256(prompt_text),
        "preview_pending_eye_check": bool(preview_pending),
        "correction": "v2a_v2b",
        "v2b_is_diagnostic": True,
        "scoring": {
            "primary_error_scored": ["miss", "identity_change", "no_error"],
            "emission_count_and_precision_only": [
                "ghost_track",
                "drift",
                "duplicate_track",
            ],
            "identity_change_subtype": "descriptive_regime_a_semantics_not_scored",
        },
        "records": list(records),
    }
    body["manifest_hash"] = payload_hash(
        {
            "clip_frames": {
                row["clip_id"]: row["clip_frames_sha256"] for row in records
            },
            "highlight_track_id": {
                row["clip_id"]: row["highlight_track_id"] for row in records
            },
            "prompt_sha256": body["prompt_sha256"],
        }
    )
    return body


def write_prompt(path: Path, text: str = TRACKER_ONLY_PROMPT) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return prompt_sha256(text)


def assert_v2_correction_invariants(
    *,
    records: Sequence[Mapping[str, Any]],
    v1_gate: Mapping[str, Any],
    previous_frozen: Mapping[str, Any] | None,
    consistencies: Sequence[Mapping[str, Any]],
) -> None:
    n_source = len({row["source_clip_id"] for row in records})
    if n_source != 284 or len(records) != 284:
        raise AssertionError(
            f"Expected 284 source / 284 records, got {n_source}/{len(records)}"
        )
    if int(v1_gate.get("n_single_id_agree") or 0) != 70:
        raise AssertionError(
            "V1 gate must stay 70/72 after the V2 correction, got "
            f"{v1_gate.get('n_single_id_agree')}"
        )
    if previous_frozen:
        prev = {
            str(row["clip_id"]): row
            for row in previous_frozen.get("records") or ()
        }
        for row in records:
            old = prev.get(str(row["clip_id"])) or prev.get(
                str(row["source_clip_id"])
            )
            if old is None:
                continue
            if row["clip_frames_sha256"] != old["clip_frames_sha256"]:
                raise AssertionError(
                    f"{row['clip_id']} clip_frames_sha256 changed"
                )
            if row["regime_a_manifest_sha256"] != old["regime_a_manifest_sha256"]:
                raise AssertionError(
                    f"{row['clip_id']} regime_a_manifest_sha256 changed"
                )
    for row in records:
        if row.get("scored_primary"):
            continue
        reason = str(row.get("unscored_reason") or "").strip()
        if not reason:
            raise AssertionError(f"{row['clip_id']} is unscored without a typed reason")
        if reason.startswith("v2b_") or ";v2b_" in f";{reason}":
            raise AssertionError(
                f"{row['clip_id']} is unscored solely/also for a V2b flag: {reason}"
            )
    v2b_by_clip = {str(row["clip_id"]): row for row in consistencies}
    for row in records:
        source = str(row.get("source_clip_id") or row["clip_id"])
        diagnostic = v2b_by_clip.get(source)
        if diagnostic is None:
            continue
        if not row.get("scored_primary"):
            reason = str(row.get("unscored_reason") or "")
            if reason in {
                f"v2b_{str(row.get('error_type') or '').lower()}_not_met",
                str(diagnostic.get("v2b_flag") or ""),
            }:
                raise AssertionError(
                    f"{row['clip_id']} excluded only because V2b failed"
                )


def freeze_candidate(
    candidate_path: Path,
    frozen_path: Path,
    *,
    preview_ok: bool = False,
    keep_preview_pending: bool = False,
) -> dict[str, Any]:
    if frozen_path.is_file():
        raise FileExistsError(
            f"{frozen_path} already exists; freeze is immutable. Write a new version."
        )
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    if not payload.get("v1_gate", {}).get("passed", False):
        raise ValueError("Cannot freeze: V1 agreement gate failed")
    blocking = list(payload.get("blocking_unresolved") or [])
    if not blocking:
        blocking = blocking_unresolved(
            records=payload.get("records") or [],
            unresolved_v1=payload.get("unresolved_v1") or [],
            unresolved_v2=payload.get("unresolved_v2") or [],
        )
    if blocking:
        raise ValueError(
            "Cannot freeze while unresolved V1/V2 flags remain on scored core clips: "
            f"{blocking}"
        )
    preview_dir = candidate_path.parent / "preview" / "manifest.json"
    if not preview_ok and not preview_dir.is_file():
        raise ValueError(
            "Cannot freeze: preview manifest missing. Render preview or pass preview_ok."
        )
    payload["frozen"] = True
    payload["preview_pending_eye_check"] = True if keep_preview_pending else not preview_ok
    payload["unreleased"] = True if keep_preview_pending else not preview_ok
    write_atomic_json(frozen_path, payload)
    return payload
