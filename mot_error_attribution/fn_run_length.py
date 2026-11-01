"""FN run-length audit. Read-only on the frozen corpus; A4 adds a sidecar field."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from .config import DEFAULT_THRESHOLDS, EVALUATION_SEQUENCES
from .occluder_masks import load_events_jsonl, write_events_jsonl
from .probe_clips import load_sidecar_events

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mot-error-attribution-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Documented v3 D1 default. Not present in the attributor config.
DOCUMENTED_MIN_FN_RUN_FRAMES = 3
FRAG_MIN_GAP = int(DEFAULT_THRESHOLDS["FRAG_MIN_GAP"])
BLIP_FRACTION = 0.10

VISIBLE_MISS_CAUSES = {
    "visible_miss",
    "visible_miss_truncated",
    "visible_miss_unconfirmed",
}


def _components(event: dict[str, Any]) -> list[dict[str, Any]]:
    merged = (event.get("evidence") or {}).get("merged_components")
    if isinstance(merged, list) and merged:
        return [dict(component) for component in merged]
    return [
        {
            "event_id": event.get("event_id"),
            "error_type": event.get("error_type"),
            "frame_start": event.get("frame_start"),
            "frame_end": event.get("frame_end"),
            "fn_frame_count": (event.get("evidence") or {}).get("fn_frame_count"),
        }
    ]


def fn_run_length(event: dict[str, Any]) -> int:
    """TrackEval FN frames in this record. FRAG_NO_SWITCH is the same miss."""
    total = 0
    for component in _components(event):
        if component.get("error_type") != "FN":
            continue
        value = component.get("fn_frame_count")
        if value is None:
            value = int(component["frame_end"]) - int(component["frame_start"]) + 1
        total += int(value)
    return total


def fn_frame_total(events: Iterable[dict[str, Any]]) -> int:
    return sum(fn_run_length(event) for event in events)


def load_identity_table(identities_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    table: dict[tuple[str, int], dict[str, Any]] = {}
    for sequence in EVALUATION_SEQUENCES:
        matches = sorted(identities_root.glob(f"{sequence}__*.csv"))
        matches = [path for path in matches if "never_detected" not in path.name]
        if not matches:
            raise FileNotFoundError(f"No identity CSV for {sequence} in {identities_root}")
        with matches[0].open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (sequence, int(row["gt_id"]))
                table[key] = {
                    "track_length": int(float(row["T"])),
                    "identity_recall": float(row["recall"]),
                    "fn_frames": int(float(row["fn_frames"])),
                    "partition": row.get("partition"),
                }
    return table


def load_trackeval_fn(validation_report: Path) -> dict[str, int]:
    report = json.loads(validation_report.read_text(encoding="utf-8"))
    sequences = report.get("sequences") or report
    return {
        sequence: int(payload["trackeval"]["CLR_FN"])
        for sequence, payload in sequences.items()
        if isinstance(payload, dict) and "trackeval" in payload
    }


def load_excluded_crowd(excluded_root: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not excluded_root.is_dir():
        return events
    for path in sorted(excluded_root.glob("*.jsonl")):
        events.extend(load_events_jsonl(path))
    return events


def attributor_min_fn_run_frames() -> int | None:
    if "MIN_FN_RUN_FRAMES" in DEFAULT_THRESHOLDS:
        return int(DEFAULT_THRESHOLDS["MIN_FN_RUN_FRAMES"])
    return None


def closure_check(
    included: list[dict[str, Any]],
    excluded_crowd: list[dict[str, Any]],
    clr_fn: dict[str, int],
) -> dict[str, Any]:
    by_seq_included: dict[str, int] = defaultdict(int)
    by_seq_crowd: dict[str, int] = defaultdict(int)
    for event in included:
        by_seq_included[str(event["sequence"])] += fn_run_length(event)
    for event in excluded_crowd:
        by_seq_crowd[str(event["sequence"])] += fn_run_length(event)

    rows: list[dict[str, Any]] = []
    all_ok = True
    for sequence in EVALUATION_SEQUENCES:
        attributed = int(by_seq_included.get(sequence, 0))
        crowd = int(by_seq_crowd.get(sequence, 0))
        micro = 0
        official = int(clr_fn[sequence])
        ok = attributed + crowd + micro == official
        all_ok = all_ok and ok
        rows.append(
            {
                "sequence": sequence,
                "attributed_fn_frames": attributed,
                "excluded_crowd_fn_frames": crowd,
                "excluded_micro_gap_fn_frames": micro,
                "trackeval_clr_fn": official,
                "closed": ok,
            }
        )
    return {
        "passed": all_ok,
        "filter_wired": False,
        "rows": rows,
        "attributed_fn_frames": sum(row["attributed_fn_frames"] for row in rows),
        "excluded_crowd_fn_frames": sum(row["excluded_crowd_fn_frames"] for row in rows),
        "excluded_micro_gap_fn_frames": 0,
        "trackeval_clr_fn": sum(row["trackeval_clr_fn"] for row in rows),
    }


def run_lengths(events: Iterable[dict[str, Any]]) -> list[int]:
    return [fn_run_length(event) for event in events if fn_run_length(event) > 0]


def histogram_summary(lengths: list[int], threshold: int) -> dict[str, Any]:
    hist = Counter(lengths)
    n_below = sum(count for length, count in hist.items() if length < threshold)
    frames_below = sum(length * count for length, count in hist.items() if length < threshold)
    ones_twos = hist.get(1, 0) + hist.get(2, 0)
    knee = ones_twos > 0 and hist.get(1, 0) + hist.get(2, 0) >= max(
        hist.get(3, 0), hist.get(4, 0), 1
    )
    return {
        "n_runs": len(lengths),
        "histogram": {str(length): hist[length] for length in sorted(hist)},
        "n_below_documented_threshold": n_below,
        "frames_below_documented_threshold": frames_below,
        "n_len_1": hist.get(1, 0),
        "n_len_2": hist.get(2, 0),
        "visible_knee_at_1_2": bool(knee and n_below > 0),
        "smooth_decay_no_knee": not bool(knee and n_below > hist.get(3, 0)),
    }


def write_histogram_plot(
    path: Path,
    lengths: list[int],
    *,
    documented: int,
    actual: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hist = Counter(lengths)
    xs = sorted(hist)
    ys = [hist[x] for x in xs]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(xs, ys, width=0.8, color="#4e79a7", align="center")
    ax.set_xscale("log")
    ax.set_xlabel("FN run length (frames)")
    ax.set_ylabel("Count")
    ax.set_title("FN run-length histogram")
    ax.axvline(
        documented,
        color="#d62728",
        linestyle="--",
        linewidth=1.4,
        label=f"documented MIN_FN_RUN_FRAMES={documented}",
    )
    if actual is not None:
        ax.axvline(
            actual,
            color="#2ca02c",
            linestyle=":",
            linewidth=1.4,
            label=f"wired MIN_FN_RUN_FRAMES={actual}",
        )
    else:
        ax.text(
            0.02,
            0.96,
            "wired MIN_FN_RUN_FRAMES: absent (no filter)",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="#444444",
        )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def hygiene_check(
    events: list[dict[str, Any]],
    sample_ids: set[str],
    *,
    min_fn: int,
    frag_min: int,
) -> dict[str, Any]:
    frag = [event for event in events if event.get("error_type") == "FRAG_NO_SWITCH"]
    short = []
    for event in frag:
        length = int(event["frame_end"]) - int(event["frame_start"]) + 1
        if length < min_fn:
            short.append(
                {
                    "event_id": event["event_id"],
                    "length": length,
                    "in_sample": event["event_id"] in sample_ids,
                }
            )
    return {
        "frag_min_gap": frag_min,
        "min_fn_run_frames_documented": min_fn,
        "min_fn_run_frames_wired": attributor_min_fn_run_frames(),
        "constants_agree": frag_min >= min_fn,
        "n_frag": len(frag),
        "n_frag_below_min_fn": len(short),
        "n_frag_below_min_fn_in_sample": sum(1 for row in short if row["in_sample"]),
        "short_frag_event_ids": [row["event_id"] for row in short],
        "zero_risk_to_fix": len(short) == 0
        or all(not row["in_sample"] for row in short),
    }


def add_run_fraction_covariates(
    events: list[dict[str, Any]],
    identities: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for event in events:
        copy = deepcopy(event)
        if copy.get("error_type") not in {"FN", "FRAG_NO_SWITCH"}:
            updated.append(copy)
            continue
        key = (str(copy["sequence"]), int(copy["focal_gt_id"]))
        identity = identities.get(key)
        if identity is None:
            raise KeyError(f"No identity row for {key}")
        length = int(copy["frame_end"]) - int(copy["frame_start"]) + 1
        track = int(identity["track_length"])
        evidence = dict(copy.get("evidence") or {})
        evidence["fn_run_fraction"] = (length / track) if track else None
        evidence["track_length"] = track
        evidence["identity_recall"] = float(identity["identity_recall"])
        copy["evidence"] = evidence
        updated.append(copy)
    return updated


def covariate_splits(
    events: list[dict[str, Any]],
    *,
    blip: float = BLIP_FRACTION,
) -> dict[str, Any]:
    def rows_for(causes: set[str]) -> list[dict[str, Any]]:
        selected = []
        for event in events:
            if event.get("cause") not in causes:
                continue
            if event.get("error_type") not in {"FN", "FRAG_NO_SWITCH"}:
                continue
            evidence = event.get("evidence") or {}
            if evidence.get("fn_run_fraction") is None:
                continue
            selected.append(event)
        return selected

    def split(selected: list[dict[str, Any]]) -> dict[str, Any]:
        low = [
            event
            for event in selected
            if float((event.get("evidence") or {})["fn_run_fraction"]) < blip
        ]
        return {
            "n": len(selected),
            "n_blip": len(low),
            "n_substantial": len(selected) - len(low),
            "blip_fraction": (len(low) / len(selected)) if selected else None,
            "mean_fn_run_fraction": (
                sum(
                    float((event.get("evidence") or {})["fn_run_fraction"])
                    for event in selected
                )
                / len(selected)
                if selected
                else None
            ),
        }

    residual = rows_for({"unattributed_fn"})
    visible = rows_for({"visible_miss"})
    visible_family = rows_for(VISIBLE_MISS_CAUSES)
    near_total = [
        event
        for event in events
        if event.get("error_type") in {"FN", "FRAG_NO_SWITCH"}
        and float((event.get("evidence") or {}).get("fn_run_fraction") or 0) >= 0.9
    ]
    return {
        "unattributed_fn": split(residual),
        "visible_miss": split(visible),
        "visible_miss_family": split(visible_family),
        "near_total_fn_runs": len(near_total),
        "near_total_mean_identity_recall": (
            sum(
                float((event.get("evidence") or {})["identity_recall"])
                for event in near_total
            )
            / len(near_total)
            if near_total
            else None
        ),
    }


def limitation_sentences(report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    hist = report["histogram"]
    hygiene = report["hygiene"]
    closure = report["closure"]
    splits = report["covariates"]
    wired = hygiene["min_fn_run_frames_wired"]
    if not closure["passed"]:
        lines.append(
            "STOP: FN three-way closure failed; TrackEval CLR_FN is not recovered "
            "from attributed + crowd + micro-gap frames."
        )
        return lines
    if wired is None:
        lines.append(
            "The v3 D1 micro-gap filter (MIN_FN_RUN_FRAMES, default 3, "
            "excluded_micro_gap) is not wired into the attributor. Every unmatched "
            f"GT frame is an FN event, including {hist['n_len_1']} 1-frame and "
            f"{hist['n_len_2']} 2-frame runs "
            f"({hist['frames_below_documented_threshold']} frames)."
        )
    if hist.get("smooth_decay_no_knee") or not hist.get("visible_knee_at_1_2"):
        lines.append(
            "The FN run-length histogram has no clear knee at 1–2 frames, so a "
            "threshold of 3 would be arbitrary. We report the figure and leave "
            "the corpus unfiltered."
        )
    else:
        lines.append(
            "The FN run-length histogram shows a 1–2 frame spike. The documented "
            "threshold of 3 sits at that knee, but the filter is not applied."
        )
    if not hygiene["constants_agree"]:
        affected = hygiene["n_frag_below_min_fn"]
        in_sample = hygiene["n_frag_below_min_fn_in_sample"]
        lines.append(
            f"FRAG_MIN_GAP={hygiene['frag_min_gap']} is below documented "
            f"MIN_FN_RUN_FRAMES={hygiene['min_fn_run_frames_documented']}. "
            f"{affected} FRAG_NO_SWITCH events are shorter than that threshold "
            f"({in_sample} in the 284-clip sample). The corpus is left unchanged."
        )
    vis = splits["visible_miss"]
    if vis["n"] and vis["blip_fraction"] is not None:
        lines.append(
            f"Of {vis['n']} visible_miss events, {vis['n_blip']} "
            f"({vis['blip_fraction']:.0%}) have fn_run_fraction < {BLIP_FRACTION} "
            "(blips on longer tracks). This is a wording caveat, not a recount."
        )
    residual = splits["unattributed_fn"]
    if residual["n"] and residual["blip_fraction"] is not None:
        lines.append(
            f"Of {residual['n']} unattributed_fn events, {residual['n_blip']} "
            f"({residual['blip_fraction']:.0%}) have fn_run_fraction < {BLIP_FRACTION}."
        )
    return lines


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_audit(
    events_root: Path,
    identities_root: Path,
    excluded_root: Path,
    validation_report: Path,
    output_root: Path,
    sample_manifest: Path | None = None,
) -> dict[str, Any]:
    included = load_sidecar_events(events_root)
    excluded = load_excluded_crowd(excluded_root)
    identities = load_identity_table(identities_root)
    clr_fn = load_trackeval_fn(validation_report)
    sample_ids: set[str] = set()
    if sample_manifest is not None and sample_manifest.is_file():
        with sample_manifest.open(newline="", encoding="utf-8") as handle:
            sample_ids = {row["event_id"] for row in csv.DictReader(handle)}

    closure = closure_check(included, excluded, clr_fn)
    if not closure["passed"]:
        raise AssertionError(f"A3 closure failed: {closure}")

    fn_pool = [
        event
        for event in included + excluded
        if fn_run_length(event) > 0
    ]
    # One length per FN component-run. FRAG_NO_SWITCH is the same miss.
    lengths = run_lengths(fn_pool)
    hist = histogram_summary(lengths, DOCUMENTED_MIN_FN_RUN_FRAMES)
    hygiene = hygiene_check(
        included + excluded,
        sample_ids,
        min_fn=DOCUMENTED_MIN_FN_RUN_FRAMES,
        frag_min=FRAG_MIN_GAP,
    )

    annotated = add_run_fraction_covariates(included, identities)
    splits = covariate_splits(annotated)

    summary_dir = output_root / "summary"
    events_out = output_root / "events"
    summary_dir.mkdir(parents=True, exist_ok=True)
    events_out.mkdir(parents=True, exist_ok=True)
    write_histogram_plot(
        summary_dir / "fn_run_length_histogram.png",
        lengths,
        documented=DOCUMENTED_MIN_FN_RUN_FRAMES,
        actual=attributor_min_fn_run_frames(),
    )
    by_seq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in annotated:
        by_seq[str(event["sequence"])].append(event)
    for sequence in EVALUATION_SEQUENCES:
        matches = sorted(Path(events_root).glob(f"{sequence}__*.jsonl"))
        name = matches[0].name if matches else f"{sequence}__tracktrack.jsonl"
        write_events_jsonl(events_out / name, by_seq[sequence])

    covariate_rows = []
    for event in annotated:
        if event.get("error_type") not in {"FN", "FRAG_NO_SWITCH"}:
            continue
        evidence = event.get("evidence") or {}
        covariate_rows.append(
            {
                "event_id": event["event_id"],
                "sequence": event["sequence"],
                "cause": event.get("cause"),
                "error_type": event.get("error_type"),
                "run_length": int(event["frame_end"]) - int(event["frame_start"]) + 1,
                "track_length": evidence.get("track_length"),
                "fn_run_fraction": evidence.get("fn_run_fraction"),
                "identity_recall": evidence.get("identity_recall"),
                "in_sample": str(event["event_id"] in sample_ids).lower(),
            }
        )
    write_rows_csv(summary_dir / "fn_run_fraction.csv", covariate_rows)
    write_rows_csv(summary_dir / "fn_run_length_histogram.csv", [
        {"run_length": int(length), "count": count}
        for length, count in sorted(
            ((int(key), value) for key, value in hist["histogram"].items()),
            key=lambda item: item[0],
        )
    ])
    write_rows_csv(summary_dir / "fn_closure.csv", closure["rows"])

    report = {
        "constants": {
            "FRAG_MIN_GAP": FRAG_MIN_GAP,
            "MIN_FN_RUN_FRAMES_documented": DOCUMENTED_MIN_FN_RUN_FRAMES,
            "MIN_FN_RUN_FRAMES_wired": attributor_min_fn_run_frames(),
            "probe_clip_MIN_FN_RUN_FRAMES": 3,
        },
        "closure": closure,
        "histogram": hist,
        "hygiene": hygiene,
        "covariates": splits,
    }
    report["limitations"] = limitation_sentences(report)
    (summary_dir / "fn_run_length_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (summary_dir / "limitations.txt").write_text(
        "\n".join(report["limitations"]) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit FN run lengths and add fn_run_fraction as a sidecar covariate"
    )
    parser.add_argument(
        "--events-root", type=Path, default=Path("out/reentry_fix/events")
    )
    parser.add_argument(
        "--identities-root",
        type=Path,
        default=Path("out/tracktrack_v4_full/identities"),
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
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=Path("out/probe_clips/summary/clip_sample_manifest.csv"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("out/fn_run_length")
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_audit(
        events_root=args.events_root,
        identities_root=args.identities_root,
        excluded_root=args.excluded_root,
        validation_report=args.validation_report,
        output_root=args.output_root,
        sample_manifest=args.sample_manifest,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
