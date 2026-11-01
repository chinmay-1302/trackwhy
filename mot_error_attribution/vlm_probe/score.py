from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .catalog import ClipRecord
from .config import (
    HIDDEN_BY_CAUSES,
    HUMAN_COULD_NOT_IDENTIFY,
    MAX_PIXELS,
    MAX_PIXELS_HIGH,
    PARTLY_BLOCKED_CAUSES,
    RULE_CAUSE_TO_VLM,
    VISIBLE_MISS_TRUTHS,
)

STAGE1_EXPECTED: dict[str, frozenset[str]] = {
    "FN_never_resumes": frozenset({"disappears"}),
    "FN_resumes_same_id": frozenset({"disappears_and_returns"}),
    "FN_never_resumes": frozenset({"disappears"}),
    "FN_resumes_same_id": frozenset({"disappears_and_returns"}),
    "IDSW": frozenset({"disappears_and_returns", "moves_to_different_person"}),
    "FP_ID_HIJACK": frozenset({"moves_to_different_person"}),
}


def expected_box_behaviours(record: ClipRecord) -> frozenset[str]:
    return STAGE1_EXPECTED.get(record.stage1_bucket, frozenset())


def stage1_correct(record: ClipRecord, box_behaviour: str | None) -> bool | None:
    expected = expected_box_behaviours(record)
    if not expected or not box_behaviour:
        return None
    if box_behaviour == "no_highlighted_box_found":
        return False
    return box_behaviour in expected


def is_hallucinated_occluder(true_cause: str, pred_cause: str | None) -> bool:
    return true_cause in VISIBLE_MISS_TRUTHS and (pred_cause or "") in HIDDEN_BY_CAUSES


def occluder_claim_inconsistent(pred_cause: str | None, occluder_visible: bool | None) -> bool:
    if pred_cause in HIDDEN_BY_CAUSES or pred_cause in PARTLY_BLOCKED_CAUSES:
        return occluder_visible is not True
    return False


def map_rule_cause(cause: str) -> str:
    return RULE_CAUSE_TO_VLM.get(cause, cause)


def three_way_bucket(
    *,
    rule_cause: str,
    human_cause: str | None,
    vlm_cause: str | None,
) -> str:
    rule = map_rule_cause(rule_cause)
    human = (human_cause or "").strip() or None
    vlm = (vlm_cause or "").strip() or None
    if human is None or vlm is None:
        return "incomplete"
    if human == rule == vlm:
        return "all_agree"
    if human != rule and vlm == rule:
        return "vlm_eq_rules_ne_human"
    if human != rule:
        return "human_ne_rules"
    if vlm == rule and vlm != human:
        return "vlm_eq_rules_ne_human"
    if vlm != rule and vlm != human:
        return "vlm_ne_both"
    if vlm == human and vlm != rule:
        return "vlm_eq_human_ne_rules"
    return "other"


def stage1_table(
    rows: Iterable[Mapping[str, Any]],
    *,
    human_could_not_rate: float | None = None,
) -> list[dict[str, Any]]:
    total = list(rows)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in total:
        groups[str(row.get("stage1_bucket") or row.get("error_type") or "unknown")].append(
            row
        )
    table: list[dict[str, Any]] = []
    ordered = sorted(groups)
    for bucket in ordered:
        items = groups[bucket]
        n = len(items)
        n_ok = sum(1 for item in items if item.get("stage1_correct") is True)
        n_no_box = sum(
            1
            for item in items
            if item.get("box_behaviour") == "no_highlighted_box_found"
        )
        n_parsed = sum(1 for item in items if item.get("parse_status") == "ok")
        table.append(
            {
                "bucket": bucket,
                "n": n,
                "n_parsed": n_parsed,
                "n_correct": n_ok,
                "accuracy": (n_ok / n_parsed) if n_parsed else "",
                "n_no_highlighted_box_found": n_no_box,
                "no_highlighted_box_found_rate": (n_no_box / n) if n else "",
            }
        )
    n_all = len(total)
    n_no_box_all = sum(
        1 for item in total if item.get("box_behaviour") == "no_highlighted_box_found"
    )
    n_parsed_all = sum(1 for item in total if item.get("parse_status") == "ok")
    n_ok_all = sum(1 for item in total if item.get("stage1_correct") is True)
    table.append(
        {
            "bucket": "ALL",
            "n": n_all,
            "n_parsed": n_parsed_all,
            "n_correct": n_ok_all,
            "accuracy": (n_ok_all / n_parsed_all) if n_parsed_all else "",
            "n_no_highlighted_box_found": n_no_box_all,
            "no_highlighted_box_found_rate": (n_no_box_all / n_all) if n_all else "",
            "human_could_not_identify_target_rate": (
                human_could_not_rate if human_could_not_rate is not None else ""
            ),
        }
    )
    return table


def stage2_confusion(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if row.get("parse_status") != "ok":
            continue
        truth = str(row.get("true_cause") or "")
        pred = str(row.get("cause") or "")
        counts[(truth, pred)] += 1
    out = [
        {"true_cause": truth, "pred_cause": pred, "n": n}
        for (truth, pred), n in sorted(counts.items())
    ]
    return out


def hallucinated_occluder_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("parse_status") != "ok":
            continue
        if not is_hallucinated_occluder(
            str(row.get("true_cause") or ""), row.get("cause")
        ):
            continue
        out.append(
            {
                "clip_id": row.get("clip_id"),
                "event_id": row.get("event_id"),
                "true_cause": row.get("true_cause"),
                "pred_cause": row.get("cause"),
                "occluder_visible_in_frame": row.get("occluder_visible_in_frame"),
                "evidence": row.get("evidence"),
                "confidence": row.get("confidence"),
            }
        )
    return out


def parse_failure_rate(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    n = len(items)
    n_ok = sum(1 for item in items if item.get("parse_status") == "ok")
    by_status = Counter(str(item.get("parse_status") or "") for item in items)
    return {
        "n": n,
        "n_ok": n_ok,
        "n_failed": n - n_ok,
        "failure_rate": ((n - n_ok) / n) if n else 0.0,
        "by_status": dict(by_status),
    }


def human_could_not_rate(human_rows: Iterable[Mapping[str, Any]]) -> float | None:
    items = list(human_rows)
    if not items:
        return None
    n = 0
    hits = 0
    for row in items:
        kind = str(
            row.get("error_kind")
            or row.get("human_error_kind")
            or row.get("annotator_error_kind")
            or ""
        )
        if not kind and not row.get("cause"):
            continue
        n += 1
        if kind == HUMAN_COULD_NOT_IDENTIFY or kind == "could_not_find":
            hits += 1
    if not n:
        return None
    return hits / n


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _as_correct(value: Any) -> bool:
    if value is True:
        return True
    return str(value).strip().lower() in {"true", "1", "yes"}


def wilson_ci(
    k: int, n: int, *, z: float = 1.959963984540054
) -> tuple[float | None, float | None]:
    """Wilson score interval. Implemented locally so we do not need statsmodels."""
    if n <= 0:
        return None, None
    k = max(0, min(int(k), int(n)))
    n = int(n)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def ci_excludes(lo: float | None, hi: float | None, value: float) -> bool | None:
    if lo is None or hi is None:
        return None
    return hi < value or lo > value


def clip_rate(k: int, n: int) -> float | None:
    """Proportion clipped to (0.5/n, 1 - 0.5/n) so z-scores stay finite."""
    if n <= 0:
        return None
    p = k / n
    floor = 0.5 / n
    return min(max(p, floor), 1.0 - floor)


def _positive_int(value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _is_present_row(row: Mapping[str, Any]) -> bool | None:
    if row.get("is_present") not in (None, ""):
        return _as_bool(row.get("is_present"))
    return _as_bool(row.get("true_present"))


def _predicted_present(row: Mapping[str, Any]) -> bool | None:
    if row.get("predicted") not in (None, ""):
        return _as_bool(row.get("predicted"))
    return _as_bool(row.get("highlighted_box_present"))


def present_cell_interpretation(
    present_accuracy: float | None,
    ci_lo: float | None,
    ci_hi: float | None,
) -> dict[str, Any]:
    """Pre-committed reading of the W1 present cell."""
    if present_accuracy is None:
        return {
            "code": "not_run",
            "sentence": "W1 has not run.",
            "w2_mandatory": True,
        }
    if present_accuracy >= 0.85:
        return {
            "code": "perception_adequate",
            "sentence": (
                "Perception adequate. Strongest framing of the null available."
            ),
            "w2_mandatory": False,
        }
    excludes = ci_excludes(ci_lo, ci_hi, 0.5)
    if excludes and ci_lo is not None and ci_lo > 0.5:
        return {
            "code": "perception_real_but_poor",
            "sentence": (
                "Perception is real but poor. Null claim stands, stated conservatively."
            ),
            "w2_mandatory": False,
        }
    return {
        "code": "cannot_detect_highlight",
        "sentence": (
            "Cannot detect the highlight. Perception is the confound; "
            "W2 becomes mandatory."
        ),
        "w2_mandatory": True,
    }


def _proportion_block(n_correct: int, n: int) -> dict[str, Any]:
    lo, hi = wilson_ci(n_correct, n)
    accuracy = (n_correct / n) if n else None
    return {
        "n": n,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "ci_lo": lo,
        "ci_hi": hi,
        "ci": [lo, hi] if lo is not None and hi is not None else None,
    }


def stays_on_person_rate(rows: Iterable[Mapping[str, Any]]) -> float | None:
    parsed = [row for row in rows if row.get("parse_status") == "ok"]
    if not parsed:
        return None
    hits = sum(1 for row in parsed if row.get("box_behaviour") == "stays_on_person")
    return hits / len(parsed)


def track_number_match(pred: str | None, expected: int | None) -> bool | None:
    if expected is None:
        return None
    text = (pred or "").strip().lower()
    if not text or text == "unreadable":
        return False
    digits = "".join(char for char in text if char.isdigit())
    if not digits:
        return False
    return int(digits) == int(expected)


def v3_binary_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    parsed = [row for row in items if row.get("parse_status") == "ok"]
    n_correct = sum(1 for row in parsed if _as_correct(row.get("correct")))
    present = [row for row in parsed if _is_present_row(row) is True]
    absent = [row for row in parsed if _is_present_row(row) is False]
    present_ok = sum(1 for row in present if _as_correct(row.get("correct")))
    absent_ok = sum(1 for row in absent if _as_correct(row.get("correct")))
    n = len(parsed)
    overall = _proportion_block(n_correct, n)
    present_block = _proportion_block(present_ok, len(present))
    absent_block = _proportion_block(absent_ok, len(absent))
    visible_present = [
        row
        for row in present
        if _positive_int(row.get("box_area_px"))
    ]
    visible_ok = sum(1 for row in visible_present if _as_correct(row.get("correct")))
    visible_block = _proportion_block(visible_ok, len(visible_present))
    reading = present_cell_interpretation(
        present_block["accuracy"], present_block["ci_lo"], present_block["ci_hi"]
    )
    by_sequence: dict[str, Any] = {}
    sequences = sorted({str(row.get("sequence") or "") for row in parsed if row.get("sequence")})
    for sequence in sequences:
        group = [row for row in parsed if str(row.get("sequence") or "") == sequence]
        g_present = [row for row in group if _is_present_row(row) is True]
        g_absent = [row for row in group if _is_present_row(row) is False]
        g_ok = sum(1 for row in group if _as_correct(row.get("correct")))
        g_present_ok = sum(1 for row in g_present if _as_correct(row.get("correct")))
        g_absent_ok = sum(1 for row in g_absent if _as_correct(row.get("correct")))
        by_sequence[sequence] = {
            **_proportion_block(g_ok, len(group)),
            "n_present": len(g_present),
            "present_accuracy": (g_present_ok / len(g_present)) if g_present else None,
            "present_ci": list(wilson_ci(g_present_ok, len(g_present))),
            "n_absent": len(g_absent),
            "absent_accuracy": (g_absent_ok / len(g_absent)) if g_absent else None,
            "absent_ci": list(wilson_ci(g_absent_ok, len(g_absent))),
        }
    return {
        "n": len(items),
        "n_parsed": n,
        "n_correct": n_correct,
        "accuracy": overall["accuracy"],
        "accuracy_ci": overall["ci"],
        "n_present": len(present),
        "present_accuracy": present_block["accuracy"],
        "present_accuracy_ci": present_block["ci"],
        "n_absent": len(absent),
        "absent_accuracy": absent_block["accuracy"],
        "absent_accuracy_ci": absent_block["ci"],
        "n_present_no_box": len(present) - len(visible_present),
        "n_present_with_box": len(visible_present),
        "present_with_box_accuracy": visible_block["accuracy"],
        "present_with_box_ci": visible_block["ci"],
        "present_with_box_cell": present_cell_interpretation(
            visible_block["accuracy"], visible_block["ci_lo"], visible_block["ci_hi"]
        ),
        "by_sequence": by_sequence,
        "present_cell": reading,
        "at_chance": bool(n and abs((n_correct / n) - 0.5) < 0.15),
        "near_perfect": bool(n and (n_correct / n) >= 0.9),
    }


def signal_detection_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = [row for row in rows if row.get("parse_status") == "ok"]
    present = [row for row in items if _is_present_row(row) is True]
    absent = [row for row in items if _is_present_row(row) is False]
    n_yes = sum(1 for row in items if _predicted_present(row) is True)
    n_hits = sum(1 for row in present if _predicted_present(row) is True)
    n_fa = sum(1 for row in absent if _predicted_present(row) is True)
    n = len(items)
    hit_rate = clip_rate(n_hits, len(present))
    fa_rate = clip_rate(n_fa, len(absent))
    d_prime = None
    criterion = None
    if hit_rate is not None and fa_rate is not None:
        from scipy.stats import norm

        z_hit = float(norm.ppf(hit_rate))
        z_fa = float(norm.ppf(fa_rate))
        d_prime = z_hit - z_fa
        criterion = -0.5 * (z_hit + z_fa)
    return {
        "n": n,
        "n_predicted_present": n_yes,
        "say_yes_rate": (n_yes / n) if n else None,
        "n_present": len(present),
        "n_hits": n_hits,
        "hit_rate": (n_hits / len(present)) if present else None,
        "hit_rate_clipped": hit_rate,
        "n_absent": len(absent),
        "n_false_alarms": n_fa,
        "false_alarm_rate": (n_fa / len(absent)) if absent else None,
        "false_alarm_rate_clipped": fa_rate,
        "d_prime": d_prime,
        "criterion": criterion,
        "bias": (
            "liberal"
            if criterion is not None and criterion < 0
            else "conservative"
            if criterion is not None and criterion > 0
            else "unbiased"
            if criterion == 0
            else None
        ),
    }


RESOLUTION_JUMP = 0.10


def w2_resolution_report(
    low_rows: Iterable[Mapping[str, Any]],
    high_rows: Iterable[Mapping[str, Any]],
    *,
    jump: float = RESOLUTION_JUMP,
) -> dict[str, Any]:
    low_items = list(low_rows)
    high_items = list(high_rows)
    low = v3_binary_summary(low_items)
    high = v3_binary_summary(high_items)
    low_present = low.get("present_accuracy")
    high_present = high.get("present_accuracy")
    delta = None
    if isinstance(low_present, (int, float)) and isinstance(high_present, (int, float)):
        delta = float(high_present) - float(low_present)
    if delta is None:
        code = "incomplete"
        sentence = "W2 has not run on the same stills as W1."
    elif delta >= jump:
        code = "resolution_was_a_contributor"
        sentence = (
            "Present-accuracy jumped at the higher pixel budget. Resolution was a "
            "real contributor. Video-stage results were obtained at the lower budget."
        )
    elif delta <= -jump:
        code = "resolution_hurt"
        sentence = (
            "Present-accuracy fell at the higher pixel budget. Report the paired "
            "numbers; do not treat resolution as the explanation for the v1 null."
        )
    elif isinstance(high_present, (int, float)) and high_present >= 0.85:
        code = "no_movement_perception_adequate"
        sentence = (
            "No movement at the higher pixel budget. Present-cell accuracy is already "
            "adequate at the lower budget; resolution is not a contributor."
        )
    else:
        code = "no_movement_perception_closed"
        sentence = (
            "No movement. The model cannot reliably see the highlight regardless of "
            "resolution, and the null is about the marking scheme rather than pixel budget."
        )
    low_pixels = _first_int(low_items, "max_pixels_used") or MAX_PIXELS
    high_pixels = _first_int(high_items, "max_pixels_used") or MAX_PIXELS_HIGH
    return {
        "low": {
            "present_accuracy": low_present,
            "present_accuracy_ci": low.get("present_accuracy_ci"),
            "accuracy": low.get("accuracy"),
            "n_present": low.get("n_present"),
            "max_pixels": low_pixels,
        },
        "high": {
            "present_accuracy": high_present,
            "present_accuracy_ci": high.get("present_accuracy_ci"),
            "accuracy": high.get("accuracy"),
            "n_present": high.get("n_present"),
            "max_pixels": high_pixels,
        },
        "present_accuracy_delta": delta,
        "jump_threshold": jump,
        "code": code,
        "sentence": sentence,
    }


def _first_int(rows: Iterable[Mapping[str, Any]], key: str) -> int | None:
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


W4_SENTENCE = (
    "The model localizes the highlighted box but reads its printed ID correctly "
    "only half the time."
)


def w4_followup(
    low_rows: Iterable[Mapping[str, Any]] | None = None,
    high_rows: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"sentence": W4_SENTENCE}
    if low_rows is not None:
        payload["low"] = v2_perception_summary(low_rows)
    if high_rows is not None:
        payload["high"] = v2_perception_summary(high_rows)
        low_acc = (payload.get("low") or {}).get("track_number_accuracy")
        high_acc = payload["high"].get("track_number_accuracy")
        if isinstance(low_acc, (int, float)) and isinstance(high_acc, (int, float)):
            payload["track_number_accuracy_delta"] = float(high_acc) - float(low_acc)
    return payload


def v2_perception_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    parsed = [row for row in items if row.get("parse_status") == "ok"]
    n = len(parsed)
    found = sum(1 for row in parsed if _as_bool(row.get("found_highlighted_box")) is True)
    number_ok = sum(1 for row in parsed if _as_correct(row.get("track_number_match")))
    return {
        "n": len(items),
        "n_parsed": n,
        "n_found": found,
        "found_rate": (found / n) if n else None,
        "n_track_number_match": number_ok,
        "track_number_accuracy": (number_ok / n) if n else None,
    }


def condition_stage1_row(rows: Iterable[Mapping[str, Any]], condition: str) -> dict[str, Any]:
    items = list(rows)
    parsed = [row for row in items if row.get("parse_status") == "ok"]
    n_parsed = len(parsed)
    n_ok = sum(1 for row in items if _as_correct(row.get("stage1_correct")))
    n_stays = sum(1 for row in parsed if row.get("box_behaviour") == "stays_on_person")
    n_no_box = sum(1 for row in items if row.get("box_behaviour") == "no_highlighted_box_found")
    return {
        "condition": condition,
        "n": len(items),
        "n_parsed": n_parsed,
        "n_correct": n_ok,
        "accuracy": (n_ok / n_parsed) if n_parsed else None,
        "n_stays_on_person": n_stays,
        "stays_on_person_rate": (n_stays / n_parsed) if n_parsed else None,
        "n_no_highlighted_box_found": n_no_box,
        "no_highlighted_box_found_rate": (n_no_box / len(items)) if items else None,
    }


def uniform_vs_centred_stage1(
    uniform_rows: Iterable[Mapping[str, Any]],
    centred_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        condition_stage1_row(uniform_rows, "uniform"),
        condition_stage1_row(centred_rows, "centred"),
    ]


def mapped_stage2_correct(true_cause: str, pred_cause: str | None) -> bool | None:
    if not true_cause or not pred_cause:
        return None
    return pred_cause == map_rule_cause(true_cause)


def condition_stage2_row(rows: Iterable[Mapping[str, Any]], condition: str) -> dict[str, Any]:
    items = [row for row in rows if row.get("parse_status") == "ok"]
    n = len(items)
    n_ok = sum(
        1
        for row in items
        if mapped_stage2_correct(str(row.get("true_cause") or ""), row.get("cause")) is True
    )
    return {
        "condition": condition,
        "n": n,
        "n_correct_mapped": n_ok,
        "accuracy_mapped": (n_ok / n) if n else None,
    }


def uniform_vs_centred_stage2(
    uniform_rows: Iterable[Mapping[str, Any]],
    centred_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        condition_stage2_row(uniform_rows, "uniform"),
        condition_stage2_row(centred_rows, "centred"),
    ]


def full_rerun_warranted(
    uniform_stays: float | None,
    centred_stays: float | None,
    *,
    drop: float = 0.10,
) -> bool | None:
    if uniform_stays is None or centred_stays is None:
        return None
    return (uniform_stays - centred_stays) >= drop


V3_REQUIRED = (
    "n",
    "n_parsed",
    "n_correct",
    "accuracy",
    "n_present",
    "present_accuracy",
    "n_absent",
    "absent_accuracy",
)
V2_REQUIRED = (
    "n",
    "n_parsed",
    "n_found",
    "found_rate",
    "n_track_number_match",
    "track_number_accuracy",
)
V4_REQUIRED = (
    "centred_stays_on_person_rate",
    "uniform_stays_on_person_rate",
    "full_rerun_warranted",
)


def assert_complete(
    payload: Mapping[str, Any] | None,
    required: Sequence[str],
    name: str,
) -> None:
    data = dict(payload or {})
    missing = [
        key for key in required if data.get(key) is None or data.get(key) == ""
    ]
    if data.get("n_parsed") == 0:
        raise ValueError(f"{name} has n_parsed=0")
    if missing:
        raise ValueError(f"{name} incomplete: {missing}")


def build_v4_decision(
    uniform_stays: float | None,
    centred_stays: float | None,
    *,
    drop: float = 0.10,
    v3_accuracy: float | None = None,
) -> dict[str, Any]:
    if centred_stays is None:
        return {
            "uniform_stays_on_person_rate": uniform_stays,
            "centred_stays_on_person_rate": None,
            "drop_threshold": drop,
            "full_rerun_warranted": None,
            "reason": "V4 has not run",
            "note": "V4 has not run. No re-run recommendation.",
        }
    if uniform_stays is None:
        return {
            "uniform_stays_on_person_rate": None,
            "centred_stays_on_person_rate": centred_stays,
            "drop_threshold": drop,
            "full_rerun_warranted": None,
            "reason": "v1 uniform Stage 1 is missing",
            "note": "v1 uniform Stage 1 is missing. No re-run recommendation.",
        }
    warranted = full_rerun_warranted(uniform_stays, centred_stays, drop=drop)
    if warranted:
        note = "Re-run both stages on all 284 with --sampling centred"
    elif v3_accuracy is not None:
        note = (
            "Stop. The null is clean if V3 showed the model can see the highlight."
        )
    else:
        note = "Centred stays_on_person did not drop enough to warrant a full re-run."
    return {
        "uniform_stays_on_person_rate": uniform_stays,
        "centred_stays_on_person_rate": centred_stays,
        "drop_threshold": drop,
        "full_rerun_warranted": warranted,
        "reason": None,
        "note": note,
    }


def degeneracy_report(stage2_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    parsed = [row for row in stage2_rows if row.get("parse_status") == "ok"]
    preds = [str(row.get("cause") or "") for row in parsed]
    counts = Counter(preds)
    n = len(preds)
    top2 = counts.most_common(2)
    top2_share = (sum(count for _, count in top2) / n) if n else 0.0
    nulls = (
        "cannot_determine",
        "clearly_visible_but_missed",
        "changed_for_no_visible_reason",
    )
    null_counts = {label: counts.get(label, 0) for label in nulls}
    drift_truth = [
        row for row in parsed if row.get("true_cause") == "track_drift_to_static"
    ]
    drift_pred = [
        row for row in parsed if row.get("cause") == "drifted_to_standing_person"
    ]
    tp = sum(1 for row in drift_truth if row.get("cause") == "drifted_to_standing_person")
    recall = (tp / len(drift_truth)) if drift_truth else None
    precision = (tp / len(drift_pred)) if drift_pred else None
    return {
        "n_parsed": n,
        "n_labels_used": len([label for label, count in counts.items() if count]),
        "label_counts": dict(counts),
        "top2": [{"cause": label, "n": count} for label, count in top2],
        "top2_share": top2_share,
        "null_answer_counts": null_counts,
        "null_answers_used": sum(null_counts.values()),
        "drift_recall": recall,
        "drift_precision": precision,
        "drift_n_truth": len(drift_truth),
        "drift_n_pred": len(drift_pred),
    }
