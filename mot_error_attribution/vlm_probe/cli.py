from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .backends import GenerateResult, build_backend
from .catalog import ClipRecord, load_catalog, probe_n_frames, select_smoke_clips
from .config import (
    CENTRED_FRAMES,
    DEFAULT_CLIP_MAP,
    DEFAULT_CLIPS_DIR,
    DEFAULT_CLIPS_META_DIR,
    DEFAULT_HASHED_DIR,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_ROOT,
    FRAMES,
    MAX_PIXELS,
    MAX_PIXELS_HIGH,
    SAMPLING_CENTRED,
    SAMPLING_FROZEN,
    SMOKE_N,
    V2_N,
    V3_N_EVENTS,
    V4_N,
)
from .parse import parse_frame_count, parse_stage1, parse_stage2
from .prompts import FRAME_COUNT_PROMPT, STAGE1_PROMPT, STAGE2_PROMPT
from .score import (
    human_could_not_rate,
    is_hallucinated_occluder,
    stage1_correct,
)
from .summarize import (
    STAGE1_FIELDS,
    STAGE2_FIELDS,
    parsed_csv_path,
    raw_path,
    read_csv,
    write_csv,
    write_json,
    write_raw,
    write_summaries,
    write_v2_summaries,
    write_w_summaries,
    write_frozen_17_report,
)
from .v2 import (
    compare_conditions,
    print_v2_report,
    print_v3_report,
    print_v4_decision,
    print_w2_report,
    print_w3_report,
    print_w4_report,
    run_centred_stage,
    run_v2,
    run_v3,
    run_v4_preview,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (_repo_root() / candidate).resolve()


def _must_resolve(path: Path | str) -> Path:
    resolved = _resolve(path)
    assert resolved is not None
    return resolved


def _pixels(args: argparse.Namespace, default: int = MAX_PIXELS) -> int:
    value = getattr(args, "max_pixels", None)
    return int(value) if value else default


def _select_clips(
    records: list[ClipRecord],
    *,
    smoke: bool,
    clip_ids: Sequence[str],
    limit: int | None,
) -> list[ClipRecord]:
    if smoke:
        selected = select_smoke_clips(records)
    elif clip_ids:
        wanted = {item.strip() for item in clip_ids if item.strip()}
        selected = [record for record in records if record.clip_id in wanted]
        missing = wanted - {record.clip_id for record in selected}
        if missing:
            raise ValueError(f"Unknown clip_id(s): {sorted(missing)}")
    else:
        selected = list(records)
    if limit is not None:
        selected = selected[:limit]
    return selected


def _prompt_for(stage: str) -> str:
    if stage == "frame_count":
        return FRAME_COUNT_PROMPT
    if stage == "stage1":
        return STAGE1_PROMPT
    if stage == "stage2":
        return STAGE2_PROMPT
    raise ValueError(f"Unknown stage: {stage}")


def _parse_stage(stage: str, text: str) -> tuple[dict[str, Any] | None, str]:
    if stage == "frame_count":
        return parse_frame_count(text)
    if stage == "stage1":
        return parse_stage1(text)
    if stage == "stage2":
        return parse_stage2(text)
    raise ValueError(f"Unknown stage: {stage}")


def _row_for(
    record: ClipRecord,
    stage: str,
    result: GenerateResult,
    parsed: dict[str, Any] | None,
    status: str,
    raw: Path,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "clip_id": record.clip_id,
        "event_id": record.event_id,
        "cause": record.cause,
        "true_cause": record.cause,
        "error_type": record.error_type,
        "stage1_bucket": record.stage1_bucket,
        "n_frames_used": result.n_frames_used,
        "elapsed_sec": round(result.elapsed_sec, 3),
        "parse_status": status,
        "video_path": str(record.video_path) if record.video_path else "",
        "raw_path": str(raw),
    }
    fields = parsed or {}
    if stage == "stage1":
        behaviour = fields.get("box_behaviour")
        row.update(
            {
                "box_behaviour": behaviour or "",
                "stage1_correct": stage1_correct(record, behaviour),
                "disappears_at": fields.get("disappears_at", ""),
                "person_still_visible_after": fields.get("person_still_visible_after", ""),
                "person_description": fields.get("person_description", ""),
                "confidence": fields.get("confidence", ""),
            }
        )
    elif stage == "stage2":
        cause = fields.get("cause")
        row.update(
            {
                "error_kind": fields.get("error_kind", ""),
                "cause": cause or "",
                "evidence": fields.get("evidence", ""),
                "occluder_visible_in_frame": fields.get("occluder_visible_in_frame", ""),
                "confidence": fields.get("confidence", ""),
                "hallucinated_occluder": is_hallucinated_occluder(record.cause, cause),
            }
        )
    else:
        row["n_moments"] = fields.get("n_moments", "")
    return row


def run_stage(
    backend: Any,
    records: Sequence[ClipRecord],
    stage: str,
    output_root: Path,
    *,
    resume: bool = True,
    sampling: str | None = None,
) -> list[dict[str, Any]]:
    prompt = _prompt_for(stage)
    rows: list[dict[str, Any]] = []
    for record in records:
        dest = raw_path(
            output_root, backend.model_id, stage, record.clip_id, sampling=sampling
        )
        if resume and dest.is_file():
            text = dest.read_text(encoding="utf-8")
            result = GenerateResult(
                text=text, n_frames_used=None, elapsed_sec=0.0, extra={"resumed": True}
            )
        else:
            if record.video_path is None or not record.video_path.is_file():
                raise FileNotFoundError(
                    f"Missing video for {record.clip_id}: {record.video_path}"
                )
            result = backend.generate(record.video_path, prompt)
            write_raw(dest, result.text)
        parsed, status = _parse_stage(stage, result.text)
        row = _row_for(record, stage, result, parsed, status, dest)
        row["sampling"] = sampling or "uniform"
        rows.append(row)
    extra_fields = [
        "clip_id",
        "event_id",
        "n_frames_used",
        "elapsed_sec",
        "parse_status",
        "n_moments",
        "raw_path",
    ]
    fields = {"stage1": STAGE1_FIELDS, "stage2": STAGE2_FIELDS}.get(stage, extra_fields)
    write_csv(
        parsed_csv_path(output_root, backend.model_id, stage, sampling=sampling),
        rows,
        fields,
    )
    return rows


def _print_smoke_report(
    records: Sequence[ClipRecord],
    frame_rows: Sequence[Mapping[str, Any]],
    stage1_rows: Sequence[Mapping[str, Any]],
    stage2_rows: Sequence[Mapping[str, Any]],
) -> int:
    n = len(records)
    missing_video = [
        record.clip_id
        for record in records
        if not (record.video_path and record.video_path.is_file())
    ]
    multi_frame = 0
    for record in records:
        count = record.n_encoded_frames
        if record.video_path and record.video_path.is_file():
            probed = probe_n_frames(record.video_path)
            count = probed if probed is not None else count
        if count is not None and count > 1:
            multi_frame += 1
    n_moments_ok = sum(
        1
        for row in frame_rows
        if row.get("parse_status") == "ok" and int(row.get("n_moments") or 0) > 1
    )
    stage1_ok = sum(1 for row in stage1_rows if row.get("parse_status") == "ok")
    stage2_ok = sum(1 for row in stage2_rows if row.get("parse_status") == "ok")
    no_box = sum(
        1
        for row in stage1_rows
        if row.get("box_behaviour") == "no_highlighted_box_found"
    )
    elapsed = [
        float(row.get("elapsed_sec") or 0)
        for row in list(stage1_rows) + list(stage2_rows)
    ]
    mean = (sum(elapsed) / len(elapsed)) if elapsed else 0.0
    estimate = mean * 284 * 2
    print(f"smoke clips: {n}")
    print(f"missing videos: {len(missing_video)} {missing_video[:5]}")
    print(f"mp4s with >1 frame: {multi_frame}/{n}")
    print(f"frame_count parsed n_moments>1: {n_moments_ok}/{n}")
    print(f"stage1 parse ok: {stage1_ok}/{n}")
    print(f"stage2 parse ok: {stage2_ok}/{n}")
    print(f"no_highlighted_box_found: {no_box}/{n}")
    print(
        f"mean sec/call: {mean:.2f}; 284 clips x 2 stages ≈ {estimate / 3600:.1f} h"
    )
    if missing_video:
        return 2
    if n and no_box == n:
        print("FAIL: highlighted box not visible at current max_pixels. Raise resolution.")
        return 3
    if n and (stage1_ok < n or stage2_ok < n):
        print("WARN: parse failures on smoke set. Tighten the prompt before the full run.")
        return 4
    return 0


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--clip-map", type=Path, default=Path(DEFAULT_CLIP_MAP))
    parser.add_argument("--clips-dir", type=Path, default=Path(DEFAULT_CLIPS_DIR))
    parser.add_argument("--meta-dir", type=Path, default=Path(DEFAULT_CLIPS_META_DIR))
    parser.add_argument("--hashed-dir", type=Path, default=Path(DEFAULT_HASHED_DIR))
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--backend", default="dry-run", help="dry-run | transformers | vllm | gemini"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n-frames", type=int, default=FRAMES)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Qwen max_pixels override. Default is 768*28*28; W2/W4 use 1280*28*28.",
    )
    parser.add_argument("--clip-id", action="append", default=[], help="Repeatable clip_id filter")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--human-csv", type=Path, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the VLM probe on annotator clips")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="10-clip wiring check")
    _add_shared(smoke)
    run = sub.add_parser("run", help="Full or filtered probe run")
    _add_shared(run)
    run.add_argument(
        "--stages", default="stage1,stage2", help="Comma list: frame_count,stage1,stage2"
    )
    run.add_argument(
        "--sampling",
        choices=["uniform", "centred"],
        default=None,
        help="centred writes under raw/.../centred/ and does not overwrite v1 uniform results",
    )
    v3 = sub.add_parser("v3", help="W1: 100 stills from 50 FN events (present vs absent)")
    _add_shared(v3)
    v3.add_argument("--n-events", type=int, default=V3_N_EVENTS)
    v2 = sub.add_parser("v2", help="Perception control: first frame of 10 day/night clips")
    _add_shared(v2)
    v2.add_argument("--n", type=int, default=V2_N)
    v4 = sub.add_parser("v4", help="Stage 1 event-centred preview on 20 clips, then compare")
    _add_shared(v4)
    v4.add_argument("--n", type=int, default=V4_N)
    v4.add_argument("--stages", default="stage1")
    compare = sub.add_parser(
        "compare", help="Rebuild uniform-vs-centred tables from parsed CSVs"
    )
    compare.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    compare.add_argument("--model", default=DEFAULT_MODEL)
    plan = sub.add_parser("v2-plan", help="Run V3, then V2, then V4 in that order")
    _add_shared(plan)
    plan.add_argument("--n-events", type=int, default=V3_N_EVENTS)
    plan.add_argument("--n-v2", type=int, default=V2_N)
    plan.add_argument("--n-v4", type=int, default=V4_N)
    w1 = sub.add_parser("w1", help="Extend V3 to 100 stills with Wilson CIs")
    _add_shared(w1)
    w1.add_argument("--n-events", type=int, default=V3_N_EVENTS)
    w2 = sub.add_parser("w2", help="Same W1 stills at MAX_PIXELS_HIGH")
    _add_shared(w2)
    w2.add_argument("--n-events", type=int, default=V3_N_EVENTS)
    w3 = sub.add_parser("w3", help="Yes-bias / d' / criterion on W1 output (no inference)")
    w3.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    w3.add_argument("--model", default=DEFAULT_MODEL)
    w4 = sub.add_parser("w4", help="Re-score V2's 10 clips at the high pixel budget")
    _add_shared(w4)
    w4.add_argument("--n", type=int, default=V2_N)
    wplan = sub.add_parser(
        "w-plan", help="W1 then W2 then W3 then W4. Does not run the 284-clip centred re-run."
    )
    _add_shared(wplan)
    wplan.add_argument("--n-events", type=int, default=V3_N_EVENTS)
    wplan.add_argument("--n-v2", type=int, default=V2_N)
    still_fix = sub.add_parser(
        "still-fix",
        help="X3 diagnose + relabel V3 stills from rendered pixels; no new inference",
    )
    _add_shared(still_fix)
    still_fix.add_argument(
        "--tracker-dir",
        type=Path,
        default=Path("tracker_output/tracktrack"),
    )
    frozen = sub.add_parser(
        "frozen-17",
        help="Re-run Stage 1/2 on the 17 re-rendered clips; does not overwrite v1",
    )
    _add_shared(frozen)
    frozen.add_argument(
        "--void-json",
        type=Path,
        default=Path("out/probe_clips/summary/void_a1_clip_ids.json"),
    )
    frozen.add_argument("--stages", default="stage1,stage2")
    summary = sub.add_parser("summarize", help="Rebuild summary tables from parsed CSVs")
    summary.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    summary.add_argument("--model", default=DEFAULT_MODEL)
    summary.add_argument("--human-csv", type=Path, default=None)
    return parser


def _catalog_from_args(args: argparse.Namespace) -> list[ClipRecord]:
    return load_catalog(
        _must_resolve(args.clip_map),
        clips_dir=_must_resolve(args.clips_dir),
        meta_dir=_must_resolve(args.meta_dir),
        hashed_dir=_must_resolve(args.hashed_dir),
    )


def cmd_smoke(args: argparse.Namespace) -> int:
    records = select_smoke_clips(_catalog_from_args(args), n=SMOKE_N)
    backend = build_backend(
        args.backend, args.model, n_frames=args.n_frames, max_pixels=_pixels(args)
    )
    output = _must_resolve(args.output_root)
    resume = not args.overwrite
    frame_rows = run_stage(backend, records, "frame_count", output, resume=resume)
    stage1_rows = run_stage(backend, records, "stage1", output, resume=resume)
    stage2_rows = run_stage(backend, records, "stage2", output, resume=resume)
    write_summaries(output, stage1_rows=stage1_rows, stage2_rows=stage2_rows)
    write_json(
        output / "summary" / "smoke_report.json",
        {
            "clip_ids": [record.clip_id for record in records],
            "causes": sorted({record.cause for record in records}),
            "backend": backend.name,
            "model": backend.model_id,
        },
    )
    return _print_smoke_report(records, frame_rows, stage1_rows, stage2_rows)


def cmd_run(args: argparse.Namespace) -> int:
    records = _select_clips(
        _catalog_from_args(args),
        smoke=False,
        clip_ids=args.clip_id,
        limit=args.limit,
    )
    sampling = getattr(args, "sampling", None)
    n_frames = CENTRED_FRAMES if sampling == SAMPLING_CENTRED else args.n_frames
    backend = build_backend(
        args.backend, args.model, n_frames=n_frames, max_pixels=_pixels(args)
    )
    output = _must_resolve(args.output_root)
    stages = [item.strip() for item in args.stages.split(",") if item.strip()]
    resume = not args.overwrite
    stage1_rows = None
    stage2_rows = None
    started = time.perf_counter()
    for stage in stages:
        if sampling == SAMPLING_CENTRED and stage in {"stage1", "stage2"}:
            rows = run_centred_stage(backend, records, stage, output, resume=resume)
        else:
            rows = run_stage(backend, records, stage, output, resume=resume)
        if stage == "stage1":
            stage1_rows = rows
        elif stage == "stage2":
            stage2_rows = rows
    human_path = _resolve(args.human_csv)
    human = read_csv(human_path) if human_path and human_path.is_file() else []
    if sampling == SAMPLING_CENTRED:
        compare_conditions(
            output,
            backend.model_id,
            centred_stage1=stage1_rows,
            centred_stage2=stage2_rows,
        )
    else:
        write_summaries(
            output,
            stage1_rows=stage1_rows,
            stage2_rows=stage2_rows,
            human_rows=human or None,
            human_could_not_rate=human_could_not_rate(human) if human else None,
        )
    print(
        f"ran {len(records)} clips, stages={stages}, sampling={sampling or 'uniform'}, "
        f"{time.perf_counter() - started:.1f}s"
    )
    return 0


def cmd_v3(args: argparse.Namespace) -> int:
    records = _catalog_from_args(args)
    backend = build_backend(
        args.backend, args.model, n_frames=1, max_pixels=_pixels(args)
    )
    output = _must_resolve(args.output_root)
    rows = run_v3(
        backend,
        records,
        output,
        resume=not args.overwrite,
        n_events=args.n_events,
    )
    write_v2_summaries(output, v3_rows=rows)
    write_w_summaries(output, w1_rows=rows)
    return print_v3_report(rows)


def cmd_v2(args: argparse.Namespace) -> int:
    records = _catalog_from_args(args)
    backend = build_backend(
        args.backend, args.model, n_frames=1, max_pixels=_pixels(args)
    )
    output = _must_resolve(args.output_root)
    rows = run_v2(
        backend, records, output, resume=not args.overwrite, n=args.n
    )
    write_v2_summaries(output, v2_rows=rows)
    return print_v2_report(rows)


def cmd_v4(args: argparse.Namespace) -> int:
    records = _catalog_from_args(args)
    backend = build_backend(
        args.backend, args.model, n_frames=CENTRED_FRAMES, max_pixels=_pixels(args)
    )
    output = _must_resolve(args.output_root)
    stages = [item.strip() for item in args.stages.split(",") if item.strip()]
    report = run_v4_preview(
        backend,
        records,
        output,
        resume=not args.overwrite,
        n=args.n,
        stages=stages,
    )
    return print_v4_decision(report, require_complete=True)


def cmd_compare(args: argparse.Namespace) -> int:
    output = _must_resolve(args.output_root)
    report = compare_conditions(output, args.model)
    print(f"wrote comparison tables under {output / 'summary'}")
    return print_v4_decision(report)


def cmd_v2_plan(args: argparse.Namespace) -> int:
    records = _catalog_from_args(args)
    output = _must_resolve(args.output_root)
    resume = not args.overwrite
    backend = build_backend(
        args.backend, args.model, n_frames=CENTRED_FRAMES, max_pixels=_pixels(args)
    )
    print("=== V3 frame-level binary ===")
    v3_rows = run_v3(
        backend, records, output, resume=resume, n_events=args.n_events
    )
    write_v2_summaries(output, v3_rows=v3_rows)
    code = print_v3_report(v3_rows)
    if code == 3:
        print("Stopping v2-plan: V3 is at chance. Fix perception before V2/V4.")
        return code
    print("=== V2 perception control ===")
    v2_rows = run_v2(backend, records, output, resume=resume, n=args.n_v2)
    write_v2_summaries(output, v2_rows=v2_rows, v3_rows=v3_rows)
    code = max(code, print_v2_report(v2_rows))
    if code == 3:
        print("Stopping v2-plan: V2 did not find the highlight. Fix perception before V4.")
        return code
    print("=== V4 event-centred Stage 1 preview ===")
    report = run_v4_preview(
        backend, records, output, resume=resume, n=args.n_v4, stages=("stage1",)
    )
    write_v2_summaries(output, v2_rows=v2_rows, v3_rows=v3_rows)
    return max(code, print_v4_decision(report, require_complete=True))


def cmd_w1(args: argparse.Namespace) -> int:
    return cmd_v3(args)


def cmd_w2(args: argparse.Namespace) -> int:
    records = _catalog_from_args(args)
    backend = build_backend(
        args.backend,
        args.model,
        n_frames=1,
        max_pixels=_pixels(args, MAX_PIXELS_HIGH),
    )
    output = _must_resolve(args.output_root)
    high_rows = run_v3(
        backend,
        records,
        output,
        resume=not args.overwrite,
        n_events=args.n_events,
        stage="v3_binary_high",
    )
    w1_path = parsed_csv_path(output, args.model, "v3_binary")
    w1_rows = read_csv(w1_path) if w1_path.is_file() else []
    write_w_summaries(output, w1_rows=w1_rows or None, w2_rows=high_rows)
    if not w1_rows:
        print("W2 finished, but W1 parsed CSV is missing. Run w1 first for the paired comparison.")
        return 0
    return print_w2_report(w1_rows, high_rows)


def cmd_w3(args: argparse.Namespace) -> int:
    output = _must_resolve(args.output_root)
    path = parsed_csv_path(output, args.model, "v3_binary")
    if not path.is_file():
        raise FileNotFoundError(f"W3 needs W1 output: {path}")
    rows = read_csv(path)
    write_w_summaries(output, w1_rows=rows)
    return print_w3_report(rows)


def cmd_w4(args: argparse.Namespace) -> int:
    records = _catalog_from_args(args)
    backend = build_backend(
        args.backend,
        args.model,
        n_frames=1,
        max_pixels=_pixels(args, MAX_PIXELS_HIGH),
    )
    output = _must_resolve(args.output_root)
    high_rows = run_v2(
        backend,
        records,
        output,
        resume=not args.overwrite,
        n=args.n,
        stage="v2_perception_high",
    )
    low_path = parsed_csv_path(output, args.model, "v2_perception")
    low_rows = read_csv(low_path) if low_path.is_file() else None
    write_w_summaries(output, v2_low_rows=low_rows, v2_high_rows=high_rows)
    return print_w4_report(low_rows, high_rows)


def cmd_w_plan(args: argparse.Namespace) -> int:
    records = _catalog_from_args(args)
    output = _must_resolve(args.output_root)
    resume = not args.overwrite
    backend = build_backend(
        args.backend,
        args.model,
        n_frames=1,
        max_pixels=_pixels(args, MAX_PIXELS),
    )
    print("=== W1 100 stills (Wilson CIs, per-sequence) ===")
    w1_rows = run_v3(
        backend, records, output, resume=resume, n_events=args.n_events
    )
    write_v2_summaries(output, v3_rows=w1_rows)
    write_w_summaries(output, w1_rows=w1_rows)
    print_v3_report(w1_rows)

    print("=== W2 same stills at MAX_PIXELS_HIGH ===")
    backend.max_pixels = _pixels(args, MAX_PIXELS_HIGH)
    w2_rows = run_v3(
        backend,
        records,
        output,
        resume=resume,
        n_events=args.n_events,
        stage="v3_binary_high",
    )
    write_w_summaries(output, w1_rows=w1_rows, w2_rows=w2_rows)
    print_w2_report(w1_rows, w2_rows)

    print("=== W3 yes-bias / d' / criterion ===")
    print_w3_report(w1_rows)

    print("=== W4 V2 high-budget rescore ===")
    v2_low_path = parsed_csv_path(output, args.model, "v2_perception")
    v2_low = read_csv(v2_low_path) if v2_low_path.is_file() else None
    v2_high = run_v2(
        backend,
        records,
        output,
        resume=resume,
        n=args.n_v2,
        stage="v2_perception_high",
    )
    write_w_summaries(
        output,
        w1_rows=w1_rows,
        w2_rows=w2_rows,
        v2_low_rows=v2_low,
        v2_high_rows=v2_high,
    )
    print_w4_report(v2_low, v2_high)
    print("w-plan did not launch the 284-clip centred re-run.")
    return 0


def cmd_still_fix(args: argparse.Namespace) -> int:
    from .still_fix import run_still_fix

    records = _catalog_from_args(args)
    output = _must_resolve(args.output_root)
    tracker = _must_resolve(args.tracker_dir)
    report = run_still_fix(
        output,
        records,
        model_id=args.model,
        tracker_dir=tracker if tracker.is_dir() else None,
    )
    diagnosis = report.get("diagnosis") or {}
    w1 = report.get("w1") or {}
    cell = w1.get("present_cell") or {}
    print(diagnosis.get("sentence") or "")
    print(
        f"relabelled {report.get('n_relabelled')} stills; "
        f"present {w1.get('present_accuracy')} CI {w1.get('present_accuracy_ci')} "
        f"{(w1.get('present_cell_transition') or {})} "
        f"after={cell.get('code')}"
    )
    print((report.get("paper") or {}).get("revised_claim") or "")
    return 0


def _load_void_clip_ids(path: Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [str(item).strip() for item in (payload.get("clip_ids") or []) if str(item).strip()]
    if not ids:
        raise ValueError(f"No clip_ids in {path}")
    return ids


def cmd_frozen_17(args: argparse.Namespace) -> int:
    output = _must_resolve(args.output_root)
    clip_ids = list(args.clip_id) if args.clip_id else _load_void_clip_ids(
        _must_resolve(args.void_json)
    )
    records = _select_clips(
        _catalog_from_args(args),
        smoke=False,
        clip_ids=clip_ids,
        limit=args.limit,
    )
    backend = build_backend(
        args.backend, args.model, n_frames=args.n_frames, max_pixels=_pixels(args)
    )
    stages = [item.strip() for item in args.stages.split(",") if item.strip()]
    resume = not args.overwrite
    started = time.perf_counter()
    stage1_rows: list[dict[str, Any]] | None = None
    stage2_rows: list[dict[str, Any]] | None = None
    for stage in stages:
        rows = run_stage(
            backend,
            records,
            stage,
            output,
            resume=resume,
            sampling=SAMPLING_FROZEN,
        )
        if stage == "stage1":
            stage1_rows = rows
        elif stage == "stage2":
            stage2_rows = rows
    s1_path = parsed_csv_path(output, backend.model_id, "stage1")
    s2_path = parsed_csv_path(output, backend.model_id, "stage2")
    original_stage1 = read_csv(s1_path) if s1_path.is_file() else []
    original_stage2 = read_csv(s2_path) if s2_path.is_file() else []
    report = write_frozen_17_report(
        output,
        clip_ids=[record.clip_id for record in records],
        original_stage1=original_stage1,
        frozen_stage1=stage1_rows or [],
        original_stage2=original_stage2,
        frozen_stage2=stage2_rows or [],
    )
    print(json.dumps(report, indent=2))
    print(
        f"froze {len(records)} clips, stages={stages}, "
        f"{time.perf_counter() - started:.1f}s; "
        f"v1_degeneracy.json left in place"
    )
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    output = _must_resolve(args.output_root)
    stage1_path = parsed_csv_path(output, args.model, "stage1")
    stage2_path = parsed_csv_path(output, args.model, "stage2")
    stage1_rows = read_csv(stage1_path) if stage1_path.is_file() else None
    stage2_rows = read_csv(stage2_path) if stage2_path.is_file() else None
    human_path = _resolve(args.human_csv)
    human = read_csv(human_path) if human_path and human_path.is_file() else []
    write_summaries(
        output,
        stage1_rows=stage1_rows,
        stage2_rows=stage2_rows,
        human_rows=human or None,
        human_could_not_rate=human_could_not_rate(human) if human else None,
    )
    print(f"wrote summaries under {output / 'summary'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        return cmd_smoke(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "summarize":
        return cmd_summarize(args)
    if args.command == "v3":
        return cmd_v3(args)
    if args.command == "v2":
        return cmd_v2(args)
    if args.command == "v4":
        return cmd_v4(args)
    if args.command == "compare":
        return cmd_compare(args)
    if args.command == "v2-plan":
        return cmd_v2_plan(args)
    if args.command == "w1":
        return cmd_w1(args)
    if args.command == "w2":
        return cmd_w2(args)
    if args.command == "w3":
        return cmd_w3(args)
    if args.command == "w4":
        return cmd_w4(args)
    if args.command == "w-plan":
        return cmd_w_plan(args)
    if args.command == "still-fix":
        return cmd_still_fix(args)
    if args.command == "frozen-17":
        return cmd_frozen_17(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    sys.exit(main())
