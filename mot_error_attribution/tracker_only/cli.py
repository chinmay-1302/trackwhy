"""CLI: derive, verify, label, extract ghosts, preview, freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .derive import DERIVATION_FIELDS, derivation_csv_row, derive_clip
from .freeze import (
    assemble_records,
    assert_v2_correction_invariants,
    build_candidate,
    freeze_candidate,
    write_prompt,
)
from .ghosts import (
    GHOST_DISAGREEMENT_FIELDS,
    GHOST_VALIDATION_FIELDS,
    extract_ghosts,
    ghost_class_decision,
    ghost_gt_disagreement,
    validate_ghost,
)
from .io import (
    DEFAULT_EXPORT,
    DEFAULT_GT_V2,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STYLE_ROOT,
    GHOST_CLIP_IDS,
    list_clip_ids,
    load_clip_bundle,
    load_gt_v2_records,
    load_human_labels,
    load_json,
    load_manifest,
    write_atomic_csv,
    write_atomic_json,
)
from .labels import class_count_report, label_clip
from .prompts import regime_b_prompt
from .preview import PREVIEW_STYLE, render_preview_clip, select_preview_clip_ids
from .verify import (
    RULE_HUMAN_CONSISTENCY_FIELDS,
    VERIFICATION_FIELDS,
    rule_human_row,
    v1_gate,
    verify_clip,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive the regime-B tracker-only corpus from frozen regime-A assets"
    )
    parser.add_argument("--style-root", type=Path, default=DEFAULT_STYLE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--gt-v3", type=Path, default=None)
    parser.add_argument("--gt-v2", type=Path, default=DEFAULT_GT_V2)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-preview", action="store_true")
    parser.add_argument("--preview-limit", type=int, default=0)
    parser.add_argument("--preview-ok", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--clip-id", action="append", default=[])
    return parser


def _load_labels(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if args.gt_v3 and Path(args.gt_v3).is_file():
        records = load_gt_v2_records(args.gt_v3)
        human = load_human_labels(args.export)
        return records, human
    return load_gt_v2_records(args.gt_v2), load_human_labels(args.export)


def run(args: argparse.Namespace) -> dict[str, Any]:
    style_root = Path(args.style_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(style_root)
    clip_ids = args.clip_id or list_clip_ids(style_root)
    gt_v2, human = _load_labels(args)
    use_v3 = bool(args.gt_v3 and Path(args.gt_v3).is_file())

    derivations: list[dict[str, Any]] = []
    bundles: dict[str, dict[str, Any]] = {}
    for clip_id in clip_ids:
        bundle = load_clip_bundle(style_root, clip_id, manifest)
        bundles[clip_id] = bundle
        derivations.append(derive_clip(bundle))
    write_atomic_json(output_root / "derivation.json", derivations)
    write_atomic_csv(
        output_root / "derivation.csv",
        [derivation_csv_row(row) for row in derivations],
        DERIVATION_FIELDS,
    )

    verifications = [
        verify_clip(
            {**row, "tracker_by_frame": bundles[row["clip_id"]]["tracker_by_frame"]},
            free_text=(human.get(row["clip_id"]) or {}).get("free_text", ""),
        )
        for row in derivations
    ]
    write_atomic_csv(
        output_root / "derivation_verification.csv",
        verifications,
        VERIFICATION_FIELDS,
    )
    gate = v1_gate(verifications)
    write_atomic_json(output_root / "v1_gate.json", gate)
    unresolved_v1 = [row["clip_id"] for row in verifications if row.get("unresolved_v1")]
    unresolved_v2 = [row["clip_id"] for row in verifications if row.get("unresolved_v2")]
    verify_by_id = {row["clip_id"]: row for row in verifications}
    for row in derivations:
        verified = verify_by_id.get(row["clip_id"]) or {}
        if verified.get("exclude_from_core"):
            row["exclude_from_core"] = True
        if verified.get("unscored_reason"):
            row["unscored_reason"] = verified["unscored_reason"]

    labels: dict[str, dict[str, Any]] = {}
    for clip_id in clip_ids:
        if use_v3 and clip_id in gt_v2:
            record = gt_v2[clip_id]
            labels[clip_id] = {
                "clip_id": clip_id,
                "annotation_status": record.get("annotation_status"),
                "primary_error": record.get("primary_error"),
                "identity_change_subtype": record.get("identity_change_subtype"),
                "identity_subtype_semantics": "regime_a_semantics",
                "drift_subtype": record.get("drift_subtype"),
                "target_visibility": record.get("target_visibility"),
                "visual_conditions": list(record.get("visual_conditions") or []),
                "annotation_confidence": record.get("annotation_confidence"),
                "source_error_kind": "",
                "free_text": (human.get(clip_id) or {}).get("free_text", ""),
            }
        else:
            labels[clip_id] = label_clip(
                clip_id=clip_id,
                human=human.get(clip_id) or {},
                gt_v2=gt_v2.get(clip_id),
            )
    source_count_report = class_count_report(list(labels.values()))
    write_atomic_json(output_root / "class_count_report.json", source_count_report)
    print("Class counts vs Label Studio / derivation-plan table:")
    print(json.dumps(source_count_report, indent=2, default=str))
    consistencies = [
        rule_human_row(
            clip_id=row["clip_id"],
            error_type=str(row.get("error_type") or ""),
            primary_error=(labels.get(row["clip_id"]) or {}).get("primary_error"),
            rule_prediction_met=verify_by_id[row["clip_id"]].get("v2b_ok"),
        )
        for row in derivations
    ]
    write_atomic_csv(
        output_root / "rule_human_consistency.csv",
        consistencies,
        RULE_HUMAN_CONSISTENCY_FIELDS,
    )

    captions = {
        clip_id: (human.get(clip_id) or {}).get("free_text", "")
        for clip_id in GHOST_CLIP_IDS
        if clip_id in set(clip_ids)
    }
    cache_path = output_root / "ghost_extraction.json"
    cached = {}
    if cache_path.is_file():
        cached_payload = load_json(cache_path)
        if isinstance(cached_payload, list):
            cached = {str(row["clip_id"]): row for row in cached_payload}
    ghost_rows = extract_ghosts(
        captions, skip_llm=bool(args.skip_llm), cached=cached
    )
    write_atomic_json(cache_path, ghost_rows)
    ghost_validations = [
        validate_ghost(
            row,
            derivation=next(item for item in derivations if item["clip_id"] == row["clip_id"]),
            bundle=bundles[row["clip_id"]],
        )
        for row in ghost_rows
        if row["clip_id"] in bundles
    ]
    write_atomic_csv(
        output_root / "ghost_extraction_validation.csv",
        ghost_validations,
        GHOST_VALIDATION_FIELDS,
    )
    ghost_decision = ghost_class_decision(ghost_validations)
    write_atomic_json(output_root / "ghost_decision.json", ghost_decision)
    disagreements = [
        row
        for row in (
            ghost_gt_disagreement(item, bundles[item["clip_id"]])
            for item in ghost_rows
            if item["clip_id"] in bundles
        )
        if row is not None
    ]
    write_atomic_csv(
        output_root / "gt_ghost_disagreement.csv",
        disagreements,
        GHOST_DISAGREEMENT_FIELDS,
    )

    prompt_text = regime_b_prompt(
        include_ghost=not bool(ghost_decision["drop_ghost_class"])
    )
    records = assemble_records(
        derivations=derivations,
        labels=labels,
        ghost_validations=ghost_validations,
        drop_ghost_class=bool(ghost_decision["drop_ghost_class"]),
    )
    candidate = build_candidate(
        records=records,
        class_counts=source_count_report,
        v1_gate=gate,
        ghost_decision=ghost_decision,
        unresolved_v1=unresolved_v1,
        unresolved_v2=unresolved_v2,
        prompt_text=prompt_text,
        preview_pending=True,
    )
    write_prompt(output_root / "prompt_regime_b.txt", prompt_text)
    candidate_path = output_root / "gt_tracker_only_candidate.json"
    write_atomic_json(candidate_path, candidate)
    previous_frozen_path = output_root / "gt_tracker_only_frozen.json"
    previous_frozen = (
        load_json(previous_frozen_path) if previous_frozen_path.is_file() else None
    )
    assert_v2_correction_invariants(
        records=records,
        v1_gate=gate,
        previous_frozen=previous_frozen,
        consistencies=consistencies,
    )

    preview_rows: list[dict[str, Any]] = []
    if not args.skip_preview:
        summary = load_json(style_root / "dataset_summary.json")
        dataset_root = Path(
            args.dataset_root or summary["source"]["dataset_root"]
        )
        flagged = sorted(
            set(unresolved_v1)
            | set(unresolved_v2)
            | {
                row["clip_id"]
                for row in verifications
                if row.get("v3_flag") or row.get("v1_flag") or row.get("v2a_flag")
            }
            | {
                row["clip_id"]
                for row in derivations
                if "late_first_match" in (row.get("flags") or ())
            }
            | {
                row["clip_id"]
                for row in ghost_validations
                if row.get("needs_review") or row.get("ghost_not_renderable")
            }
        )
        preview_ids = select_preview_clip_ids(records, flagged)
        if args.preview_limit:
            preview_ids = preview_ids[: int(args.preview_limit)]
        highlight_by_source = {
            row["source_clip_id"]: row["highlight_track_id"]
            for row in records
            if str(row["clip_id"]).endswith("-T") or "-" not in str(row["clip_id"])[1:]
        }
        ghost_highlight = {
            row["source_clip_id"]: row["highlight_track_id"]
            for row in records
            if str(row["clip_id"]).endswith("-G")
        }
        preview_dir = output_root / "preview" / PREVIEW_STYLE
        print(f"Rendering {len(preview_ids)} preview clips to {preview_dir}")
        for index, clip_id in enumerate(preview_ids, start=1):
            print(f"  [{index}/{len(preview_ids)}] {clip_id}", flush=True)
            annotation = bundles[clip_id]["annotation"]
            preview_rows.append(
                render_preview_clip(
                    annotation=annotation,
                    dataset_root=dataset_root,
                    highlight_track_id=highlight_by_source.get(clip_id),
                    output_path=preview_dir / f"{clip_id}.mp4",
                )
            )
            if clip_id in ghost_highlight and ghost_highlight[clip_id] is not None:
                preview_rows.append(
                    render_preview_clip(
                        annotation=annotation,
                        dataset_root=dataset_root,
                        highlight_track_id=ghost_highlight[clip_id],
                        output_path=preview_dir / f"{clip_id}-G.mp4",
                    )
                )
        write_atomic_json(output_root / "preview" / "manifest.json", preview_rows)

    frozen = None
    if args.freeze:
        frozen = freeze_candidate(
            candidate_path,
            output_root / "gt_tracker_only_frozen_v2.json",
            preview_ok=bool(args.preview_ok),
            keep_preview_pending=True,
        )

    report = {
        "n_source": len(derivations),
        "n_records": len(records),
        "v1_gate": gate,
        "unresolved_v1": unresolved_v1,
        "unresolved_v2": unresolved_v2,
        "class_count_report": source_count_report,
        "ghost_decision": ghost_decision,
        "n_preview": len(preview_rows),
        "frozen": bool(frozen),
        "candidate": str(candidate_path),
    }
    write_atomic_json(output_root / "run_report.json", report)
    if not gate["passed"]:
        raise SystemExit(
            "V1 agreement gate failed: "
            f"{gate['n_single_id_agree']}/{gate['n_single_id_evaluated']} "
            f"(need ≥{gate['target_agree']} of {gate['target_n']} or 95%). "
            "Derivation CSVs were written; freeze is blocked."
        )
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
