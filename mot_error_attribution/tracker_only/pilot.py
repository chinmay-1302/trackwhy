"""Regime-B human detection pilot: 30-clip blind existence check."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from statistics import NormalDist, median
from typing import Any, Mapping, Sequence

from mot_error_attribution.vlm_probe.summarize import write_csv

from .eyecheck import (
    encode_overlay_clip,
    iter_regime_b_frames,
    overlay_leaks_primary_error,
    source_clip_id,
)
from .io import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STYLE_ROOT,
    load_clip_bundle,
    load_json,
    load_manifest,
    write_atomic_json,
)


PILOT_SEED = 20260905
N_MISS = 10
N_IDENTITY = 5
N_NEGATIVE = 15
N_POSITIVE = N_MISS + N_IDENTITY
MIN_MOT20_03_POSITIVE = 6
MANDATORY_EXCLUDE = ("C198", "C044", "C070")
RESPONSE_FIELDS = ("clip_id", "answer", "seconds", "note")
SELECTION_FIELDS = ("clip_id", "true_class", "sequence", "subtype")
KEY_FIELDS = ("file_id", "clip_id", "true_class", "sequence", "subtype")
OBSERVERS = ("o1", "o2")
BRIEFING = (
    "The highlighted box is a track produced by an automatic tracker. "
    "Tell us whether it correctly follows one single person for the whole clip."
)
PAPER_DRAFT = """\
As a pilot, we ran the binary detection task past two human observers on the same 30
clips (15 positive, 15 negative): one naive to the corpus, one the corpus author.
{counts}
Both observers marked a substantial fraction of true failures as correct.
We report this as an existence check that the task is non-trivial for human
observers under tracker-only rendering, not as an estimate of human performance —
one observer is the corpus author, and the naive arm is a single participant.
"""


def eye_check_excluded_ids(eye_root: Path) -> set[str]:
    excluded = {str(item) for item in MANDATORY_EXCLUDE}
    selection_path = eye_root / "selection.json"
    if selection_path.is_file():
        payload = load_json(selection_path)
        for key in ("e1", "e2", "e3"):
            for row in payload.get(key) or ():
                excluded.add(str(row["clip_id"]))
        return excluded
    for folder in ("E1_fragmentation", "E2_negatives", "E3_late_match"):
        answers = eye_root / folder / "ANSWERS.csv"
        if not answers.is_file():
            continue
        with answers.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                excluded.add(str(row.get("clip_id") or ""))
    key_path = eye_root / "E2_negatives" / "KEY.csv"
    if key_path.is_file():
        with key_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                excluded.add(str(row.get("clip_id") or ""))
    return {item for item in excluded if item}


def eligible_scored(records: Sequence[Mapping[str, Any]], excluded: set[str]) -> list[dict[str, Any]]:
    out = []
    for row in records:
        if not row.get("scored_primary"):
            continue
        primary = str(row.get("primary_error") or "")
        if primary not in {"miss", "identity_change", "no_error"}:
            continue
        clip_id = source_clip_id(str(row["clip_id"]))
        if clip_id in excluded:
            continue
        out.append(
            {
                "clip_id": clip_id,
                "true_class": primary,
                "sequence": str(row.get("sequence") or ""),
                "subtype": str(row.get("identity_change_subtype") or ""),
                "highlight_track_id": row.get("highlight_track_id"),
            }
        )
    return out


def select_pilot_clips(
    records: Sequence[Mapping[str, Any]],
    excluded: set[str],
    *,
    seed: int = PILOT_SEED,
    n_miss: int = N_MISS,
    n_identity: int = N_IDENTITY,
    n_negative: int = N_NEGATIVE,
    min_mot20_03: int = MIN_MOT20_03_POSITIVE,
) -> list[dict[str, Any]]:
    pool = eligible_scored(records, excluded)
    miss = [row for row in pool if row["true_class"] == "miss"]
    ident = [row for row in pool if row["true_class"] == "identity_change"]
    neg = [row for row in pool if row["true_class"] == "no_error"]
    if len(miss) < n_miss or len(ident) < n_identity or len(neg) < n_negative:
        raise ValueError(
            f"Pilot pool too small: miss={len(miss)} ident={len(ident)} neg={len(neg)}"
        )
    rng = random.Random(int(seed))
    miss_03 = [row for row in miss if row["sequence"] == "MOT20-03"]
    ident_03 = [row for row in ident if row["sequence"] == "MOT20-03"]
    n_id_03 = min(2, n_identity, len(ident_03))
    n_miss_03 = min(n_miss, len(miss_03), max(0, min_mot20_03 - n_id_03))
    if n_miss_03 + n_id_03 < min_mot20_03:
        n_id_03 = min(n_identity, len(ident_03), min_mot20_03 - n_miss_03)
    if n_miss_03 + n_id_03 < min_mot20_03:
        raise ValueError("Not enough MOT20-03 positives for the pilot strata")
    chosen_miss_03 = rng.sample(miss_03, n_miss_03)
    chosen_id_03 = rng.sample(ident_03, n_id_03)
    used = {row["clip_id"] for row in chosen_miss_03 + chosen_id_03}
    rest_miss = [row for row in miss if row["clip_id"] not in used]
    rest_id = [row for row in ident if row["clip_id"] not in used]
    chosen_miss = chosen_miss_03 + rng.sample(rest_miss, n_miss - n_miss_03)
    chosen_id = chosen_id_03 + rng.sample(rest_id, n_identity - n_id_03)
    chosen_neg = rng.sample(neg, n_negative)
    selected = chosen_miss + chosen_id + chosen_neg
    n_03 = sum(
        1
        for row in selected
        if row["true_class"] != "no_error" and row["sequence"] == "MOT20-03"
    )
    if n_03 < min_mot20_03:
        raise ValueError(f"MOT20-03 positives={n_03}, need {min_mot20_03}")
    order = list(range(len(selected)))
    rng.shuffle(order)
    blinded = []
    for index, source_index in enumerate(order, start=1):
        row = dict(selected[source_index])
        row["file_id"] = f"P{index:02d}"
        row["index"] = index
        row["n"] = len(selected)
        blinded.append(row)
    return blinded


def pilot_overlay_lines(item: Mapping[str, Any]) -> list[str]:
    label = str(item.get("file_id") or item["clip_id"])
    return [
        f"[PILOT]  {label}",
        "The highlighted box is one tracker's track.",
        "WATCH FOR: anything wrong with how this track follows a person.",
        "ANSWER:    Does this track correctly follow a single person",
        "           throughout the clip?    YES / NO",
    ]


def e2_decoy_prior() -> dict[str, Any]:
    """H9: author-blind E2 decoy result, already collected."""
    hidden = [
        {
            "clip_id": "C246",
            "true_class": "miss",
            "sequence": "MOT20-03",
            "error_type": "FN",
            "mined_cause": "visible_miss_truncated",
            "target_visibility": "partly_visible",
            "visual_conditions": ["truncation"],
            "cause_miss": "partly_outside_the_frame",
        },
        {
            "clip_id": "C085",
            "true_class": "miss",
            "sequence": "MOT20-03",
            "error_type": "FN",
            "mined_cause": "visible_miss_truncated",
            "target_visibility": "partly_visible",
            "visual_conditions": ["truncation"],
            "cause_miss": "partly_outside_the_frame",
        },
    ]
    return {
        "observer": "author",
        "source": "E2 eye-check, labels unseen",
        "n_true_positive": 5,
        "n_true_positive_marked_clean": 2,
        "n_true_negative": 15,
        "n_true_negative_marked_broken": 1,
        "true_positives_marked_clean": hidden,
        "true_negative_marked_broken": ["C198"],
        "same_hidden_failure_type": True,
        "hidden_failure_type": (
            "partly-visible truncated miss (visible_miss_truncated / "
            "partly_outside_the_frame) on MOT20-03"
        ),
        "note": (
            "Both author-missed decoys are the same failure: a partly-visible "
            "person leaving the frame. Tracker-only rendering hides truncation "
            "misses of this kind."
        ),
    }


def wilson_interval(k: int, n: int, z: float = 1.96) -> dict[str, float]:
    if n <= 0:
        return {"k": 0, "n": 0, "rate": 0.0, "low": 0.0, "high": 0.0}
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    margin = (
        z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    )
    return {
        "k": k,
        "n": n,
        "rate": p,
        "low": max(0.0, center - margin),
        "high": min(1.0, center + margin),
    }


def loglinear_rate(k: int, n: int) -> float:
    return (k + 0.5) / (n + 1.0)


def dprime_and_c(hits: int, n_pos: int, false_alarms: int, n_neg: int) -> dict[str, float]:
    h = loglinear_rate(hits, n_pos)
    f = loglinear_rate(false_alarms, n_neg)
    normal = NormalDist()
    z_h = normal.inv_cdf(h)
    z_f = normal.inv_cdf(f)
    return {
        "dprime": z_h - z_f,
        "criterion_c": -0.5 * (z_h + z_f),
        "h_corrected": h,
        "f_corrected": f,
    }


def cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Cohen's kappa needs paired non-empty judgements")
    n = len(left)
    agree = sum(1 for a, b in zip(left, right) if a == b) / n
    values = sorted(set(left) | set(right))
    pe = 0.0
    for value in values:
        pe += (sum(1 for item in left if item == value) / n) * (
            sum(1 for item in right if item == value) / n
        )
    if pe >= 1.0:
        return 1.0
    return (agree - pe) / (1.0 - pe)


def _norm_answer(value: str) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    aliases = {
        "yes": "YES",
        "y": "YES",
        "fine": "YES",
        "clean": "YES",
        "correct": "YES",
        "no": "NO",
        "n": "NO",
        "miss": "NO",
        "broken": "NO",
        "wrong": "NO",
    }
    if text not in aliases:
        raise ValueError(f"Unrecognised pilot answer {value!r}; use YES or NO")
    return aliases[text]


def score_observer(
    answers: Sequence[Mapping[str, Any]],
    key_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    key = {str(row["file_id"]): row for row in key_rows}
    hits = 0
    false_alarms = 0
    seconds: list[float] = []
    rows = []
    for answer in answers:
        file_id = str(answer.get("clip_id") or "")
        meta = key[file_id]
        judgement = _norm_answer(str(answer.get("answer") or ""))
        positive = str(meta["true_class"]) in {"miss", "identity_change"}
        if judgement == "NO" and positive:
            hits += 1
        if judgement == "NO" and not positive:
            false_alarms += 1
        raw_seconds = answer.get("seconds")
        if str(raw_seconds or "").strip() != "":
            seconds.append(float(raw_seconds))
        rows.append(
            {
                "file_id": file_id,
                "clip_id": meta["clip_id"],
                "true_class": meta["true_class"],
                "answer": judgement,
                "seconds": raw_seconds,
                "note": answer.get("note", ""),
            }
        )
    n_pos = N_POSITIVE
    n_neg = N_NEGATIVE
    detect = dprime_and_c(hits, n_pos, false_alarms, n_neg)
    return {
        "n": len(rows),
        "hits": wilson_interval(hits, n_pos),
        "false_alarms": wilson_interval(false_alarms, n_neg),
        "dprime": detect["dprime"],
        "criterion_c": detect["criterion_c"],
        "median_seconds": median(seconds) if seconds else None,
        "n_timed": len(seconds),
        "rows": rows,
    }


def both_clean_positives(
    observer_scores: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    by_file: dict[str, dict[str, str]] = {}
    for observer, payload in observer_scores.items():
        for row in payload.get("rows") or ():
            file_id = str(row["file_id"])
            by_file.setdefault(
                file_id,
                {
                    "file_id": file_id,
                    "clip_id": str(row["clip_id"]),
                    "true_class": str(row["true_class"]),
                },
            )
            by_file[file_id][observer] = str(row["answer"])
    hidden = []
    for row in by_file.values():
        if row["true_class"] not in {"miss", "identity_change"}:
            continue
        answers = [row.get(name) for name in observer_scores]
        if answers and all(item == "YES" for item in answers):
            hidden.append(
                {
                    "clip_id": row["clip_id"],
                    "file_id": row["file_id"],
                    "true_class": row["true_class"],
                }
            )
    return hidden


def write_blank_responses(path: Path, items: Sequence[Mapping[str, Any]]) -> None:
    write_csv(
        path,
        [
            {"clip_id": row["file_id"], "answer": "", "seconds": "", "note": ""}
            for row in items
        ],
        RESPONSE_FIELDS,
    )


def render_pilot(
    *,
    frozen_path: Path,
    output_root: Path,
    style_root: Path = DEFAULT_STYLE_ROOT,
    dataset_root: Path | None = None,
    seed: int = PILOT_SEED,
) -> dict[str, Any]:
    from .eyecheck import dataset_root_from_summary

    payload = load_json(frozen_path)
    excluded = eye_check_excluded_ids(output_root / "preview" / "eyecheck")
    items = select_pilot_clips(payload.get("records") or [], excluded, seed=seed)
    for item in items:
        if overlay_leaks_primary_error(pilot_overlay_lines(item)):
            raise ValueError("Pilot overlay leaked a class label")
    manifest = load_manifest(style_root)
    data_root = Path(dataset_root) if dataset_root else dataset_root_from_summary(style_root)
    video_dir = output_root / "preview" / "pilot"
    closed_dir = output_root / "pilot"
    closed_dir.mkdir(parents=True, exist_ok=True)
    bundles = {
        item["clip_id"]: load_clip_bundle(style_root, item["clip_id"], manifest)
        for item in items
    }
    print(f"Rendering {len(items)} pilot clips to {video_dir}", flush=True)
    rendered = []
    for item in items:
        clip_id = str(item["clip_id"])
        bundle = bundles[clip_id]
        path = video_dir / f"{item['file_id']}.mp4"
        print(f"  [{item['index']}/{item['n']}] {item['file_id']}", flush=True)
        info = encode_overlay_clip(
            frames=iter_regime_b_frames(
                bundle=bundle,
                dataset_root=data_root,
                highlight_track_id=item.get("highlight_track_id"),
            ),
            top_lines=pilot_overlay_lines(item),
            bottom_text=f"clip {item['index']} of {item['n']}",
            output_path=path,
            n_frames=len(bundle["annotation"]["clip_frames"]),
            source_width=int(bundle["annotation"]["video"]["width"]),
            source_height=int(bundle["annotation"]["video"]["height"]),
        )
        rendered.append({**dict(item), **info})
    write_csv(
        closed_dir / "pilot_clip_selection.csv",
        rendered,
        SELECTION_FIELDS,
    )
    write_csv(output_root / "pilot_clip_selection.csv", rendered, SELECTION_FIELDS)
    write_csv(closed_dir / "PILOT_KEY.csv", rendered, KEY_FIELDS)
    for observer in OBSERVERS:
        write_blank_responses(video_dir / f"pilot_responses_{observer}.csv", rendered)
    (video_dir / "BRIEFING.txt").write_text(BRIEFING + "\n", encoding="utf-8")
    (closed_dir / "BRIEFING.txt").write_text(BRIEFING + "\n", encoding="utf-8")
    prior = e2_decoy_prior()
    write_atomic_json(closed_dir / "e2_decoy_prior.json", prior)
    write_atomic_json(
        closed_dir / "selection.json",
        {
            "schema_version": "pilot_selection_v1",
            "seed": int(seed),
            "n": len(rendered),
            "excluded": sorted(excluded),
            "items": rendered,
            "note": "Closed until both observers finish. Do not open PILOT_KEY.csv first.",
        },
    )
    (closed_dir / "paper_subsection.md").write_text(
        PAPER_DRAFT.format(counts="[Counts pending both observers.]") + "\n",
        encoding="utf-8",
    )
    report = {
        "n": len(rendered),
        "n_miss": sum(1 for row in rendered if row["true_class"] == "miss"),
        "n_identity_change": sum(
            1 for row in rendered if row["true_class"] == "identity_change"
        ),
        "n_no_error": sum(1 for row in rendered if row["true_class"] == "no_error"),
        "n_mot20_03_positive": sum(
            1
            for row in rendered
            if row["true_class"] != "no_error" and row["sequence"] == "MOT20-03"
        ),
        "video_dir": str(video_dir),
        "closed_dir": str(closed_dir),
        "e2_decoy_prior": prior,
    }
    write_atomic_json(closed_dir / "render_report.json", report)
    return report


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def score_pilot(
    *,
    output_root: Path,
    video_dir: Path | None = None,
) -> dict[str, Any]:
    closed_dir = output_root / "pilot"
    videos = video_dir or output_root / "preview" / "pilot"
    key = _read_csv(closed_dir / "PILOT_KEY.csv")
    observer_scores = {}
    paired: dict[str, dict[str, str]] = {}
    for observer in OBSERVERS:
        path = videos / f"pilot_responses_{observer}.csv"
        rows = _read_csv(path)
        missing = [
            row.get("clip_id")
            for row in rows
            if not str(row.get("answer") or "").strip()
        ]
        if missing:
            raise ValueError(f"{path.name} has blank answers: {missing}")
        observer_scores[observer] = score_observer(rows, key)
        for row in observer_scores[observer]["rows"]:
            paired.setdefault(str(row["file_id"]), {})[observer] = row["answer"]
    left = [paired[file_id]["o1"] for file_id in sorted(paired)]
    right = [paired[file_id]["o2"] for file_id in sorted(paired)]
    hidden = both_clean_positives(observer_scores)
    prior = e2_decoy_prior()
    results = {
        "schema_version": "pilot_results_v1",
        "observers": observer_scores,
        "cohens_kappa": cohen_kappa(left, right),
        "kappa_n": len(left),
        "kappa_note": "Weak at n=30; report as such.",
        "both_marked_clean_true_positives": hidden,
        "e2_decoy_prior": prior,
        "framing": {
            "if_both_dprime_above_0": (
                "The rendering is readable and the task is well-posed. "
                "Any VLM null is about the model, not the stimulus."
            ),
            "if_both_dprime_near_0": (
                "The task is hard for humans too under tracker-only rendering. "
                "Report this; do not bury it."
            ),
            "if_o1_below_o2": (
                "Expected: quantifies how much corpus familiarity is worth."
            ),
            "if_o1_above_o2": "Report it and do not explain it away.",
        },
    }
    write_atomic_json(closed_dir / "pilot_results.json", results)
    o1 = observer_scores["o1"]
    o2 = observer_scores["o2"]
    counts = (
        f"Naive observer: hits {o1['hits']['k']}/{o1['hits']['n']}, "
        f"false alarms {o1['false_alarms']['k']}/{o1['false_alarms']['n']}, "
        f"d′={o1['dprime']:.2f}. "
        f"Author: hits {o2['hits']['k']}/{o2['hits']['n']}, "
        f"false alarms {o2['false_alarms']['k']}/{o2['false_alarms']['n']}, "
        f"d′={o2['dprime']:.2f}. "
        f"Cohen's κ={results['cohens_kappa']:.2f} (n=30). "
        f"Both marked {len(hidden)} true failures as clean."
    )
    (closed_dir / "paper_subsection.md").write_text(
        PAPER_DRAFT.format(counts=counts) + "\n",
        encoding="utf-8",
    )
    return results


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Render or score the regime-B human detection pilot"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--style-root", type=Path, default=DEFAULT_STYLE_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--frozen",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "gt_tracker_only_frozen_released.json",
    )
    parser.add_argument("--seed", type=int, default=PILOT_SEED)
    parser.add_argument("--score", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root).resolve()
    if args.score:
        results = score_pilot(output_root=output_root)
        print(
            json.dumps(
                {
                    "scored": True,
                    "kappa": results["cohens_kappa"],
                    "hidden_failures": [
                        row["clip_id"]
                        for row in results["both_marked_clean_true_positives"]
                    ],
                },
                indent=2,
            )
        )
        return 0
    report = render_pilot(
        frozen_path=Path(args.frozen),
        output_root=output_root,
        style_root=Path(args.style_root),
        dataset_root=args.dataset_root,
        seed=int(args.seed),
    )
    print(json.dumps({k: v for k, v in report.items() if k != "e2_decoy_prior"}, indent=2))
    return 0
