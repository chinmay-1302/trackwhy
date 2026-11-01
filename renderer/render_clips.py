#!/usr/bin/env python3
"""Render TrackWhy clips in either released regime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mot_error_attribution.clips import render_event_clip
from mot_error_attribution.config import PipelineConfig
from mot_error_attribution.io import load_sequences
from mot_error_attribution.matching import build_matching
from mot_error_attribution.models import ErrorEvent
from mot_error_attribution.probe_clips import render_event_clips

from renderer.config import PLAYBACK_FPS, SOURCE_FPS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render TrackWhy failure clips")
    parser.add_argument("--tracker-output", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--events-root", required=True, help="JSONL event directory")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--regime",
        choices=("supervision_free", "gt_informed"),
        default="supervision_free",
    )
    parser.add_argument("--event-id", help="Render a single event")
    return parser


def load_events(events_root: Path) -> list[dict]:
    rows = []
    for path in sorted(Path(events_root).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = PipelineConfig(render_clips=True)
    sequences, _coverage = load_sequences(
        Path(args.tracker_output), Path(args.dataset_root), config
    )
    events = load_events(Path(args.events_root))
    if args.event_id:
        events = [event for event in events if event["event_id"] == args.event_id]
        if not events:
            raise SystemExit(f"unknown event_id {args.event_id}")
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"source {SOURCE_FPS} fps, playback {PLAYBACK_FPS} fps, "
        f"regime={args.regime}, n={len(events)}"
    )
    if args.regime == "supervision_free":
        clips = output / "clips"
        audit = output / "audit"
        meta = output / "clips_meta"
        withheld = output / "withheld"
        for directory in (clips, audit, meta, withheld):
            directory.mkdir(parents=True, exist_ok=True)
        for event in events:
            render_event_clips(
                event,
                sequences[str(event["sequence"])],
                clips,
                audit,
                meta,
                withheld,
                None,
            )
        return
    for event in events:
        sequence = sequences[str(event["sequence"])]
        matching = build_matching(sequence, config)
        render_event_clip(
            ErrorEvent(**{key: event[key] for key in ErrorEvent.__dataclass_fields__ if key in event}),
            sequence,
            matching,
            output / "clips_gt_informed",
            config,
        )


if __name__ == "__main__":
    main()
