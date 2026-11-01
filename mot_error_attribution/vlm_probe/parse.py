from __future__ import annotations

import json
import re
from typing import Any

VALID_CAUSES = frozenset(
    {
        "hidden_by_object",
        "hidden_by_one_person",
        "hidden_by_standing_person",
        "hidden_by_crowd",
        "partly_blocked_by_person",
        "partly_blocked_by_static",
        "partly_outside_the_frame",
        "clearly_visible_but_missed",
        "left_and_returned",
        "swapped_during_crossing",
        "changed_for_no_visible_reason",
        "drifted_to_standing_person",
        "drifted_to_moving_person",
        "cannot_determine",
    }
)

BOX_BEHAVIOURS = frozenset(
    {
        "stays_on_person",
        "disappears",
        "disappears_and_returns",
        "moves_to_different_person",
        "no_highlighted_box_found",
    }
)
DISAPPEARS_AT = frozenset({"early", "middle", "late", "not_applicable"})
PERSON_STILL_VISIBLE = frozenset({"yes", "no", "partly", "not_applicable"})
CONFIDENCE = frozenset({"high", "medium", "low"})
ERROR_KINDS = frozenset(
    {"miss", "identity_change", "drift", "no_error", "could_not_find"}
)

_FENCE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)


def extract_json(text: str) -> tuple[dict[str, Any] | None, str]:
    stripped = _FENCE.sub("", (text or "").strip()).strip()
    match = re.search(r"\{.*\}", stripped, re.S)
    if not match:
        return None, "no_json"
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, "bad_json"
    if not isinstance(parsed, dict):
        return None, "bad_json"
    return parsed, "ok"


def parse(text: str) -> tuple[dict[str, Any] | None, str]:
    """P3 parser: pull JSON, then require a valid Stage 2 cause."""
    parsed, status = extract_json(text)
    if parsed is None:
        return None, status
    cause = parsed.get("cause")
    if cause not in VALID_CAUSES:
        return parsed, f"invalid_cause:{cause}"
    return parsed, "ok"


def _norm(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = _norm(value).lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def parse_stage1(text: str) -> tuple[dict[str, Any] | None, str]:
    parsed, status = extract_json(text)
    if parsed is None:
        return None, status
    behaviour = _norm(parsed.get("box_behaviour"))
    if behaviour not in BOX_BEHAVIOURS:
        return parsed, f"invalid_box_behaviour:{behaviour or None}"
    parsed["box_behaviour"] = behaviour
    disappears_at = _norm(parsed.get("disappears_at")) or "not_applicable"
    if disappears_at not in DISAPPEARS_AT:
        return parsed, f"invalid_disappears_at:{disappears_at}"
    parsed["disappears_at"] = disappears_at
    visible = _norm(parsed.get("person_still_visible_after")) or "not_applicable"
    if visible not in PERSON_STILL_VISIBLE:
        return parsed, f"invalid_person_still_visible_after:{visible}"
    parsed["person_still_visible_after"] = visible
    confidence = _norm(parsed.get("confidence")) or "low"
    if confidence not in CONFIDENCE:
        return parsed, f"invalid_confidence:{confidence}"
    parsed["confidence"] = confidence
    parsed["person_description"] = _norm(parsed.get("person_description"))
    return parsed, "ok"


def parse_stage2(text: str) -> tuple[dict[str, Any] | None, str]:
    parsed, status = parse(text)
    if parsed is None:
        return None, status
    if status != "ok":
        return parsed, status
    kind = _norm(parsed.get("error_kind"))
    if kind not in ERROR_KINDS:
        return parsed, f"invalid_error_kind:{kind or None}"
    parsed["error_kind"] = kind
    parsed["cause"] = _norm(parsed.get("cause"))
    parsed["evidence"] = _norm(parsed.get("evidence"))
    occluder = _as_bool(parsed.get("occluder_visible_in_frame"))
    if occluder is None:
        return parsed, "invalid_occluder_visible_in_frame"
    parsed["occluder_visible_in_frame"] = occluder
    confidence = _norm(parsed.get("confidence")) or "low"
    if confidence not in CONFIDENCE:
        return parsed, f"invalid_confidence:{confidence}"
    parsed["confidence"] = confidence
    return parsed, "ok"


def parse_frame_count(text: str) -> tuple[dict[str, Any] | None, str]:
    parsed, status = extract_json(text)
    if parsed is None:
        return None, status
    value = parsed.get("n_moments", parsed.get("n_frames", parsed.get("n_moments")))
    try:
        parsed["n_moments"] = int(value)
    except (TypeError, ValueError):
        return parsed, f"invalid_n_moments:{value}"
    return parsed, "ok"


def parse_v2_perception(text: str) -> tuple[dict[str, Any] | None, str]:
    parsed, status = extract_json(text)
    if parsed is None:
        return None, status
    found = _as_bool(parsed.get("found_highlighted_box"))
    if found is None:
        return parsed, "invalid_found_highlighted_box"
    parsed["found_highlighted_box"] = found
    parsed["track_number"] = _norm(parsed.get("track_number"))
    parsed["position"] = _norm(parsed.get("position"))
    parsed["clothing"] = _norm(parsed.get("clothing"))
    return parsed, "ok"


def parse_v3_binary(text: str) -> tuple[dict[str, Any] | None, str]:
    parsed, status = extract_json(text)
    if parsed is None:
        return None, status
    present = _as_bool(parsed.get("highlighted_box_present"))
    if present is None:
        return parsed, "invalid_highlighted_box_present"
    parsed["highlighted_box_present"] = present
    return parsed, "ok"
