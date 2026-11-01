#!/usr/bin/env python3
"""Mine the TrackEval-anchored event pool and optionally check the release CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mot_error_attribution.config import PipelineConfig
from mot_error_attribution.min_fn3 import apply_min_fn3_postpass
from mot_error_attribution.pipeline import run_pipeline
from mot_error_attribution.reentry import apply_reentry_postpass


EXPECTED_FN = 34724
EXPECTED_IDSW = 253
EXPECTED_EVENTS = 1164


def events_from_jsonl(events_root: Path) -> list[dict]:
    rows = []
    for path in sorted(events_root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_events_csv(events: list[dict], destination: Path) -> None:
    identity = [
        "event_id",
        "sequence",
        "tracker",
        "error_type",
        "cause",
        "cause_confidence",
        "role",
        "frame_start",
        "frame_end",
        "focal_gt_id",
        "partner_gt_id",
        "prev_trk_id",
        "new_trk_id",
        "static_person_present",
        "static_person_max_ioa",
    ]
    evidence_keys = sorted(
        {key for event in events for key in (event.get("evidence") or {})}
    )
    fieldnames = identity + [f"evidence.{key}" for key in evidence_keys]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in sorted(events, key=lambda item: item["event_id"]):
            row = {
                key: "" if event.get(key) is None else event.get(key) for key in identity
            }
            evidence = event.get("evidence") or {}
            for key in evidence_keys:
                value = evidence.get(key)
                if isinstance(value, (dict, list)):
                    row[f"evidence.{key}"] = json.dumps(
                        value, separators=(",", ":"), ensure_ascii=True
                    )
                elif value is None:
                    row[f"evidence.{key}"] = ""
                else:
                    row[f"evidence.{key}"] = value
            writer.writerow(row)


def closure_from_report(report: dict) -> tuple[int, int]:
    fn = 0
    idsw = 0
    for sequence in report.get("sequences", {}).values():
        trackeval = sequence.get("trackeval") or sequence.get("trackeval_report") or {}
        if "CLR_FN" in trackeval:
            fn += int(trackeval["CLR_FN"])
            idsw += int(trackeval["IDSW"])
            continue
        official = sequence.get("official") or {}
        if "CLR_FN" in official:
            fn += int(official["CLR_FN"])
            idsw += int(official["IDSW"])
    return fn, idsw


def compare_release(produced: Path, released: Path) -> None:
    left = list(csv.DictReader(produced.open(encoding="utf-8")))
    right = list(csv.DictReader(released.open(encoding="utf-8")))
    assert len(left) == len(right) == EXPECTED_EVENTS, (
        len(left),
        len(right),
        EXPECTED_EVENTS,
    )
    left_sorted = sorted(left, key=lambda row: row["event_id"])
    right_sorted = sorted(right, key=lambda row: row["event_id"])
    left_ids = [row["event_id"] for row in left_sorted]
    right_ids = [row["event_id"] for row in right_sorted]
    assert left_ids == right_ids, "event_id set does not match the released pool"
    left_causes = [row["cause"] for row in left_sorted]
    right_causes = [row["cause"] for row in right_sorted]
    assert left_causes == right_causes, "cause labels do not match the released pool"
    print("event pool reproduces")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine TrackEval-anchored MOT20 failure events"
    )
    parser.add_argument("--tracker-output", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", default="out/mine_events")
    parser.add_argument("--out", default="annotations/events.reproduced.csv")
    parser.add_argument("--config")
    parser.add_argument("--tracker-name", default="tracktrack")
    parser.add_argument("--v3-corpus-root")
    parser.add_argument(
        "--occluder-export",
        help="Label-style polygon export for the unannotated-structure postpass",
    )
    parser.add_argument(
        "--check-release",
        action="store_true",
        help="Assert event_id and cause match annotations/events.csv",
    )
    parser.add_argument(
        "--released-events",
        default=str(ROOT / "annotations" / "events.csv"),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
    config.tracker_name = args.tracker_name
    config.render_clips = False
    output_root = Path(args.output_root)
    report = run_pipeline(
        tracker_output=Path(args.tracker_output),
        dataset_root=Path(args.dataset_root),
        output_root=output_root / "v4",
        config=config,
        v3_corpus_root=args.v3_corpus_root,
    )
    events_root = output_root / "v4" / "events"
    if args.occluder_export:
        from mot_error_attribution.occluder_masks import run_occluder_mask_postpass

        run_occluder_mask_postpass(
            export_path=Path(args.occluder_export),
            dataset_root=Path(args.dataset_root),
            events_root=events_root,
            output_root=output_root / "occluder",
        )
        events_root = output_root / "occluder" / "events"
        apply_reentry_postpass(
            events_root, Path(args.dataset_root), output_root / "reentry"
        )
        events_root = output_root / "reentry" / "events"
        apply_min_fn3_postpass(events_root, output_root / "min_fn3")
        events_root = output_root / "min_fn3" / "events"

    fn, idsw = closure_from_report(report)
    if fn and idsw:
        assert fn == EXPECTED_FN, f"closure broken: {fn} vs {EXPECTED_FN}"
        assert idsw == EXPECTED_IDSW, f"closure broken: {idsw} vs {EXPECTED_IDSW}"
        print(f"TrackEval closure: CLR_FN={fn} IDSW={idsw}")
    else:
        print("pipeline report did not expose CLR_FN totals; per-sequence asserts already ran")

    events = events_from_jsonl(events_root)
    write_events_csv(events, Path(args.out))
    print(f"wrote {len(events)} events to {args.out}")
    if args.check_release:
        compare_release(Path(args.out), Path(args.released_events))


if __name__ == "__main__":
    main()
