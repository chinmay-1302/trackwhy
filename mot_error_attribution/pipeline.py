from __future__ import annotations

from pathlib import Path
from typing import Any
import json
from concurrent.futures import ThreadPoolExecutor

from .attribution import attribute_events, reattribute_canonical_events
from .clips import render_class_audit, render_event_clip
from .config import PipelineConfig
from .events import extract_events, split_micro_gap_events
from .human_verification import (
    migrate_annotation_vocabulary,
    score_human_annotations,
)
from .io import load_sequences
from .matching import build_matching
from .models import ErrorEvent
from .plots import generate_summary_plots
from .reports import (
    identity_rows,
    never_detected_analysis,
    residual_diagnostics,
    render_validation_text,
    tracker_geometry_audit,
    validate_sequence,
    write_cause_distribution,
    write_cause_distribution_by_tracker,
    write_class_histogram,
    write_human_verification_manifests,
    write_identity_csv,
    write_ioa_sweep,
    write_json,
    write_jsonl,
    write_never_detected_csv,
    write_ordering_suppression,
    write_partial_selection_delta,
    write_rows_csv,
    write_static_person_presence_vs_drift,
    write_transition_matrix,
    write_v3_v4_transition_matrix,
    write_visible_miss_analysis,
)


def _load_event_file(path: Path) -> list[ErrorEvent]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing baseline corpus file required for validation: {path}"
        )
    return [
        ErrorEvent(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_pipeline(
    tracker_output: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    config: PipelineConfig | None = None,
    *,
    human_annotations: str | Path | None = None,
    v1_corpus_root: str | Path | None = None,
    v3_corpus_root: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or PipelineConfig()
    cfg.validate()
    output = Path(output_root)
    summary_dir = output / "summary"
    for directory in (
        output / "events",
        output / "identities",
        output / "excluded_crowd",
        output / "excluded_micro_gap",
        output / "clips",
        summary_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    sequences, input_audit = load_sequences(tracker_output, dataset_root, cfg)
    if v1_corpus_root is not None and v3_corpus_root is not None:
        raise ValueError("Use either --v1-corpus-root or --v3-corpus-root, not both")
    baseline_source = v3_corpus_root if v3_corpus_root is not None else v1_corpus_root
    baseline_root = Path(baseline_source) if baseline_source is not None else None
    baseline_included_by_sequence: dict[str, list[ErrorEvent]] = {}
    baseline_excluded_by_sequence: dict[str, list[ErrorEvent]] = {}
    if baseline_root is not None:
        for sequence_name in sequences:
            stem = f"{sequence_name}__{cfg.tracker_name}.jsonl"
            baseline_included_by_sequence[sequence_name] = _load_event_file(
                baseline_root / "events" / stem
            )
            baseline_excluded_by_sequence[sequence_name] = _load_event_file(
                baseline_root / "excluded_crowd" / stem
            )
    if cfg.frame_scope == "legacy_half" and baseline_root is None:
        raise ValueError(
            "Pass A requires --v1-corpus-root so canonical events stay fixed for G1"
        )
    write_json(summary_dir / "resolved_config.json", cfg.as_dict())
    write_json(summary_dir / "input_audit.json", input_audit)
    write_class_histogram(summary_dir / "class_histogram.csv", sequences)
    class_audit_files = [
        str(path)
        for sequence in sequences.values()
        for path in render_class_audit(sequence, summary_dir / "class_audit")
    ]
    input_audit["class_audit_files"] = class_audit_files
    write_json(summary_dir / "input_audit.json", input_audit)

    events_by_sequence: dict[str, list[ErrorEvent]] = {}
    excluded_by_sequence: dict[str, list[ErrorEvent]] = {}
    micro_gap_by_sequence: dict[str, list[ErrorEvent]] = {}
    sequence_reports: dict[str, Any] = {}
    geometry_audits: dict[str, Any] = {}
    never_detected_summaries: list[dict[str, Any]] = []
    matching_by_sequence: dict[str, Any] = {}

    for sequence_name, sequence in sorted(sequences.items()):
        matching = build_matching(sequence, cfg)
        matching_by_sequence[sequence_name] = matching
        raw_events = extract_events(sequence, matching, cfg)
        raw_events, micro_gap = split_micro_gap_events(raw_events, cfg)
        if cfg.frame_scope == "legacy_half" or v3_corpus_root is not None:
            baseline_events = (
                baseline_included_by_sequence[sequence_name]
                + baseline_excluded_by_sequence[sequence_name]
            )
            events, excluded = reattribute_canonical_events(
                baseline_events, sequence, matching, cfg
            )
        else:
            events, excluded = attribute_events(raw_events, sequence, matching, cfg)
        report = validate_sequence(
            sequence, matching, events, excluded, cfg, excluded_micro_gap=micro_gap
        )
        rows = identity_rows(sequence, matching, events + excluded, cfg)
        never_rows, never_summary = never_detected_analysis(sequence, rows, cfg)

        stem = f"{sequence_name}__{cfg.tracker_name}"
        write_jsonl(output / "events" / f"{stem}.jsonl", events)
        write_jsonl(output / "excluded_crowd" / f"{stem}.jsonl", excluded)
        write_jsonl(output / "excluded_micro_gap" / f"{stem}.jsonl", micro_gap)
        write_identity_csv(output / "identities" / f"{stem}.csv", rows)
        write_never_detected_csv(
            output / "identities" / f"{stem}__never_detected.csv", never_rows
        )
        never_detected_summaries.append(never_summary)
        events_by_sequence[sequence_name] = events
        excluded_by_sequence[sequence_name] = excluded
        micro_gap_by_sequence[sequence_name] = micro_gap
        sequence_reports[sequence_name] = report
        geometry_audits[sequence_name] = tracker_geometry_audit(sequence, cfg)

    all_events = [
        event
        for sequence_name in sorted(events_by_sequence)
        for event in events_by_sequence[sequence_name]
    ]
    write_cause_distribution(
        summary_dir / "cause_distribution.csv",
        sequences,
        events_by_sequence,
        cfg.tracker_name,
    )
    write_cause_distribution_by_tracker(
        summary_dir / "cause_distribution_by_tracker.csv", all_events
    )
    write_ordering_suppression(
        summary_dir / "ordering_suppression.csv", all_events, cfg
    )
    partial_delta = write_partial_selection_delta(
        summary_dir / "partial_selection_delta.csv", all_events
    )
    static_presence_rows = write_static_person_presence_vs_drift(
        summary_dir / "static_person_presence_vs_drift.csv", all_events, cfg
    )
    write_visible_miss_analysis(
        summary_dir / "visible_miss_analysis.csv",
        summary_dir / "visible_miss_reference.json",
        all_events,
        sequences,
        cfg,
    )
    write_rows_csv(
        summary_dir / "never_detected_summary.csv", never_detected_summaries
    )
    write_human_verification_manifests(summary_dir, all_events, cfg)
    write_json(
        summary_dir / "residual_diagnostics.json",
        residual_diagnostics(all_events, cfg),
    )
    redistribution_gate = None
    v3_v4_gate = None
    ioa_sweep = None
    scope_expansion_check = None
    if cfg.frame_scope == "legacy_half":
        baseline_all = [
            event
            for sequence_name in sorted(baseline_included_by_sequence)
            for event in baseline_included_by_sequence[sequence_name]
        ]
        redistribution_gate = write_transition_matrix(
            summary_dir / "v1_to_v2_transition_matrix.csv",
            baseline_all,
            all_events,
        )
        write_json(
            summary_dir / "v1_to_v2_redistribution_gate.json",
            redistribution_gate,
        )
    elif v3_corpus_root is not None:
        baseline_all = [
            event
            for sequence_name in sorted(baseline_included_by_sequence)
            for event in baseline_included_by_sequence[sequence_name]
        ]
        v3_v4_gate = write_v3_v4_transition_matrix(
            summary_dir / "v3_to_v4_transition_matrix.csv",
            baseline_all,
            all_events,
        )
        write_json(summary_dir / "v3_to_v4_redistribution_gate.json", v3_v4_gate)
        ioa_sweep = write_ioa_sweep(
            summary_dir / "ioa_sweep.csv",
            baseline_included_by_sequence,
            sequences,
            matching_by_sequence,
            cfg,
        )
        write_json(summary_dir / "ioa_sweep_selection.json", ioa_sweep)
    elif baseline_root is not None:
        baseline_count = sum(map(len, baseline_included_by_sequence.values()))
        ratio = len(all_events) / baseline_count if baseline_count else None
        scope_expansion_check = {
            "v1_reference_event_count": baseline_count,
            "full_v2_event_count": len(all_events),
            "ratio": ratio,
            "warning": (
                "Full corpus contains fewer than 1.5x the v1 half-split events."
                if ratio is not None and ratio < 1.5
                else None
            ),
        }
    validation_report = {
        "passed": all(item["passed"] for item in sequence_reports.values()),
        "input_audit": input_audit,
        "geometry_audit": geometry_audits,
        "sequences": sequence_reports,
        "total_canonical_events": len(all_events),
        "total_excluded_crowd_events": sum(map(len, excluded_by_sequence.values())),
        "total_excluded_micro_gap_events": sum(
            map(len, micro_gap_by_sequence.values())
        ),
        "redistribution_gate": redistribution_gate,
        "v3_to_v4_redistribution_gate": v3_v4_gate,
        "ioa_sweep": ioa_sweep,
        "partial_selection_delta": partial_delta,
        "static_person_presence_vs_drift": static_presence_rows,
        "scope_expansion_check": scope_expansion_check,
        "second_tracker_comparison": {
            "available": False,
            "reason": (
                "No ByteTrack, OC-SORT, or BoT-SORT MOT result files sharing the "
                "cached YOLOX-X detections were present under tracker_output."
            ),
        },
    }
    write_json(summary_dir / "validation_report.json", validation_report)
    text_report = render_validation_text(validation_report)
    with (summary_dir / "validation_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(text_report)
    if cfg.render_clips:
        for sequence_name, sequence in sorted(sequences.items()):
            matching = matching_by_sequence[sequence_name]
            with ThreadPoolExecutor(max_workers=cfg.clip_workers) as executor:
                futures = [
                    executor.submit(
                        render_event_clip,
                        event,
                        sequence,
                        matching,
                        output / "clips",
                        cfg,
                    )
                    for event in events_by_sequence[sequence_name]
                ]
                for future in futures:
                    future.result()

    generate_summary_plots(output)

    if human_annotations is not None:
        migrated_annotations = summary_dir / "human_annotations_v4_migrated.csv"
        migrate_annotation_vocabulary(Path(human_annotations), migrated_annotations)
        score_human_annotations(
            migrated_annotations,
            summary_dir / "human_verification_answer_key.jsonl",
            summary_dir / "human_verification_agreement.csv",
        )
    return validation_report
