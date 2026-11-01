"""Sidecar: drop FN/FRAG runs shorter than MIN_FN_RUN_FRAMES. Corpus frozen upstream."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import DEFAULT_THRESHOLDS, EVALUATION_SEQUENCES, PipelineConfig
from .fn_run_length import (
    fn_frame_total,
    fn_run_length,
    load_excluded_crowd,
    load_trackeval_fn,
)
from .occluder_masks import load_events_jsonl, write_events_jsonl
from .probe_clips import load_sidecar_events


MIN_FN_RUN_FRAMES = int(DEFAULT_THRESHOLDS["MIN_FN_RUN_FRAMES"])
FRAG_MIN_GAP = int(DEFAULT_THRESHOLDS["FRAG_MIN_GAP"])


def is_short_fn(event: dict[str, Any], min_fn: int = MIN_FN_RUN_FRAMES) -> bool:
    length = fn_run_length(event)
    return bool(length) and length < min_fn


def is_short_frag(event: dict[str, Any], min_fn: int = MIN_FN_RUN_FRAMES) -> bool:
    if event.get("error_type") != "FRAG_NO_SWITCH":
        return False
    span = int(event["frame_end"]) - int(event["frame_start"]) + 1
    return span < min_fn


def partition_events(
    events: list[dict[str, Any]],
    *,
    min_fn: int = MIN_FN_RUN_FRAMES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    micro: list[dict[str, Any]] = []
    for event in events:
        copy = deepcopy(event)
        if is_short_fn(copy, min_fn) or is_short_frag(copy, min_fn):
            copy["cause"] = "excluded_micro_gap"
            evidence = dict(copy.get("evidence") or {})
            evidence["min_fn_run_frames"] = min_fn
            copy["evidence"] = evidence
            micro.append(copy)
        else:
            kept.append(copy)
    return kept, micro


def apply_min_fn3_postpass(
    events_root: Path,
    excluded_root: Path,
    validation_report: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_root = Path(output_root)
    events_out = output_root / "events"
    micro_out = output_root / "excluded_micro_gap"
    summary_dir = output_root / "summary"
    for directory in (events_out, micro_out, summary_dir):
        directory.mkdir(parents=True, exist_ok=True)

    included = load_sidecar_events(events_root)
    crowd = load_excluded_crowd(excluded_root)
    clr_fn = load_trackeval_fn(validation_report)
    if int(DEFAULT_THRESHOLDS["FRAG_MIN_GAP"]) < MIN_FN_RUN_FRAMES:
        raise AssertionError("FRAG_MIN_GAP must be >= MIN_FN_RUN_FRAMES")
    if MIN_FN_RUN_FRAMES != FRAG_MIN_GAP:
        raise AssertionError("FRAG_MIN_GAP and MIN_FN_RUN_FRAMES must both be 3")

    kept, micro = partition_events(included)
    if len(kept) + len(micro) != len(included):
        raise AssertionError("event identity was not conserved")

    by_seq_kept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_seq_micro: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in kept:
        by_seq_kept[str(event["sequence"])].append(event)
    for event in micro:
        by_seq_micro[str(event["sequence"])].append(event)

    closure_rows = []
    all_ok = True
    for sequence in EVALUATION_SEQUENCES:
        attributed = fn_frame_total(by_seq_kept[sequence])
        crowd_fn = fn_frame_total(
            [event for event in crowd if event.get("sequence") == sequence]
        )
        micro_fn = fn_frame_total(by_seq_micro[sequence])
        official = int(clr_fn[sequence])
        ok = attributed + crowd_fn + micro_fn == official
        all_ok = all_ok and ok
        if micro_fn == 0 and any(
            is_short_fn(event) for event in included if event.get("sequence") == sequence
        ):
            ok = False
            all_ok = False
        closure_rows.append(
            {
                "sequence": sequence,
                "attributed_fn_frames": attributed,
                "excluded_crowd_fn_frames": crowd_fn,
                "excluded_micro_gap_fn_frames": micro_fn,
                "trackeval_clr_fn": official,
                "closed": ok,
            }
        )
        matches = sorted(Path(events_root).glob(f"{sequence}__*.jsonl"))
        name = matches[0].name if matches else f"{sequence}__tracktrack.jsonl"
        write_events_jsonl(events_out / name, by_seq_kept[sequence])
        write_events_jsonl(micro_out / name, by_seq_micro[sequence])

    if not all_ok:
        raise AssertionError(f"A3 three-way closure failed: {closure_rows}")
    if sum(row["excluded_micro_gap_fn_frames"] for row in closure_rows) == 0:
        raise AssertionError("micro-gap bucket is empty; filter is not wired")

    before = Counter(str(event["cause"]) for event in included)
    after = Counter(str(event["cause"]) for event in kept)
    report = {
        "min_fn_run_frames": MIN_FN_RUN_FRAMES,
        "frag_min_gap": FRAG_MIN_GAP,
        "n_before": len(included),
        "n_after": len(kept),
        "n_micro_gap_events": len(micro),
        "n_short_frag": sum(1 for event in micro if event.get("error_type") == "FRAG_NO_SWITCH"),
        "cause_before": dict(before),
        "cause_after": dict(after),
        "visible_miss_before": before.get("visible_miss", 0),
        "visible_miss_after": after.get("visible_miss", 0),
        "closure": {
            "passed": True,
            "rows": closure_rows,
            "attributed_fn_frames": sum(row["attributed_fn_frames"] for row in closure_rows),
            "excluded_crowd_fn_frames": sum(
                row["excluded_crowd_fn_frames"] for row in closure_rows
            ),
            "excluded_micro_gap_fn_frames": sum(
                row["excluded_micro_gap_fn_frames"] for row in closure_rows
            ),
            "trackeval_clr_fn": sum(row["trackeval_clr_fn"] for row in closure_rows),
        },
        "underpowered_visible_miss": after.get("visible_miss", 0) < 30,
        "events_out": str(events_out),
        "micro_gap_out": str(micro_out),
    }
    (summary_dir / "min_fn3_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move FN/FRAG runs shorter than 3 frames to excluded_micro_gap"
    )
    parser.add_argument(
        "--events-root", type=Path, default=Path("out/reentry_fix/events")
    )
    parser.add_argument(
        "--excluded-root",
        type=Path,
        default=Path("out/tracktrack_v4_full/excluded_crowd"),
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=Path("out/tracktrack_v4_full/summary/validation_report.json"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("out/min_fn3"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    PipelineConfig().validate()
    report = apply_min_fn3_postpass(
        events_root=args.events_root,
        excluded_root=args.excluded_root,
        validation_report=args.validation_report,
        output_root=args.output_root,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
