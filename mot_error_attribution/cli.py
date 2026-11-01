from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PipelineConfig
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rule-based, model-free MOT20 tracker error attribution"
    )
    parser.add_argument(
        "--tracker-output",
        "--tracker-json",
        dest="tracker_output",
        default="tracker_output/tracktrack",
        help="MOT .txt file/directory or COCO JSON; its records define authorized frames",
    )
    parser.add_argument("--dataset-root", default="datasets/MOT20")
    parser.add_argument("--output-root", default="out")
    parser.add_argument("--config", help="Optional JSON configuration overrides")
    parser.add_argument("--tracker-name", help="Override tracker label in output events")
    parser.add_argument(
        "--no-clips", action="store_true", help="Skip MP4 and sidecar rendering"
    )
    parser.add_argument(
        "--legacy-half-validation",
        action="store_true",
        help=(
            "Pass A only: reproduce the original validation-half frame set for "
            "the v1-to-v2 redistribution invariant"
        ),
    )
    parser.add_argument(
        "--human-annotations",
        help="Completed blind annotation CSV to score against the withheld key",
    )
    parser.add_argument(
        "--v1-corpus-root",
        help=(
            "v1 output root: required to lock canonical events in Pass A; in "
            "full-sequence Pass B it is used only for the scope-expansion check"
        ),
    )
    parser.add_argument(
        "--v3-corpus-root",
        help=(
            "v3 output root used to lock canonical event IDs/counts for the "
            "v4 redistribution invariant and v3-to-v4 transition matrix"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig.from_json(args.config)
    if args.tracker_name:
        config.tracker_name = args.tracker_name
    if args.no_clips:
        config.render_clips = False
    if args.legacy_half_validation:
        config.frame_scope = "legacy_half"
    report = run_pipeline(
        tracker_output=Path(args.tracker_output),
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        config=config,
        human_annotations=args.human_annotations,
        v1_corpus_root=args.v1_corpus_root,
        v3_corpus_root=args.v3_corpus_root,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "sequences": sorted(report["sequences"]),
                "total_canonical_events": report["total_canonical_events"],
                "total_excluded_crowd_events": report[
                    "total_excluded_crowd_events"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
