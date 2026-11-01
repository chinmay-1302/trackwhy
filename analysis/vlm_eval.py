"""Score shipped model outputs against the released ground-truth files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from analysis.human_agreement import cohen_kappa

PRIMARY_CLASSES = ("miss", "identity_change", "no_error")
PREDICTION_FAILURE = "__prediction_failure__"

REGIME_A_STYLES = (
    "02_inside_top_left",
    "09_target_only_inside_top_left",
    "18_target_only_inside_center_with_target_trajectory",
)
REGIME_B_STYLES = (
    "02_inside_top_left_tracker_only",
    "02_inside_top_left_tracker_only_base_prompt",
)
MODELS = (
    "Qwen3.8-27B",
    "InternVL3.5-8B",
    "Qwen3-VL-8B",
    "LLaVA-OneVision-2-8B",
)
STYLE_LABELS = {
    "02_inside_top_left": "02",
    "09_target_only_inside_top_left": "09",
    "18_target_only_inside_center_with_target_trajectory": "18",
    "02_inside_top_left_tracker_only": "tracker-only",
    "02_inside_top_left_tracker_only_base_prompt": "tracker-only, base",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def record_prediction_id(record: dict) -> str:
    return str(record.get("source_clip_id") or record["clip_id"])


def prediction_primary(path: Path) -> str:
    if not path.is_file():
        return PREDICTION_FAILURE
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return PREDICTION_FAILURE
    if not isinstance(value, dict):
        return PREDICTION_FAILURE
    primary = value.get("primary_error")
    if primary in PRIMARY_CLASSES:
        return str(primary)
    return PREDICTION_FAILURE


def regime_a_eligible(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        evaluation = record.get("evaluation") or {}
        if record.get("primary_error") not in PRIMARY_CLASSES:
            continue
        if not evaluation.get("include_in_core_error_eval"):
            continue
        rows.append(record)
    return rows


def regime_b_eligible(records: list[dict]) -> list[dict]:
    return [
        record
        for record in records
        if record.get("scored_primary") and record.get("primary_error") in PRIMARY_CLASSES
    ]


def score_condition(
    records: list[dict],
    prediction_dir: Path,
) -> dict:
    truth = [str(record["primary_error"]) for record in records]
    pred = [
        prediction_primary(prediction_dir / f"{record_prediction_id(record)}.json")
        for record in records
    ]
    kappa = cohen_kappa(truth, pred)
    recalls = {}
    for label in PRIMARY_CLASSES:
        support = sum(value == label for value in truth)
        hits = sum(t == label and p == label for t, p in zip(truth, pred))
        recalls[label] = hits / support if support else 0.0
        recalls[f"{label}_support"] = support
        recalls[f"{label}_hits"] = hits
    coverage = len(set(pred) & set(PRIMARY_CLASSES))
    return {
        "n": len(records),
        "kappa": kappa,
        "coverage": coverage,
        "prediction_counts": dict(Counter(pred)),
        **recalls,
    }


def evaluate_all(model_root: Path, gt_v3: dict, gt_tracker: dict) -> list[dict]:
    a_records = regime_a_eligible(gt_v3["records"])
    b_records = regime_b_eligible(gt_tracker["records"])
    rows = []
    for model in MODELS:
        for style in REGIME_A_STYLES:
            rows.append(
                {
                    "model": model,
                    "condition": STYLE_LABELS[style],
                    "style": style,
                    **score_condition(a_records, model_root / model / style),
                }
            )
        for style in REGIME_B_STYLES:
            rows.append(
                {
                    "model": model,
                    "condition": STYLE_LABELS[style],
                    "style": style,
                    **score_condition(b_records, model_root / model / style),
                }
            )
    return rows
