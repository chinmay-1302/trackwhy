"""V2 probe experiments: perception stills, binary frames, event-centred video."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .backends import GenerateResult
from .catalog import (
    ClipRecord,
    select_centred_preview_clips,
    select_fn_binary_clips,
    select_perception_clips,
)
from .config import (
    CENTRED_FRAMES,
    MAX_PIXELS,
    SAMPLING_CENTRED,
    SAMPLING_UNIFORM,
    STAYS_ON_PERSON_DROP,
    V2_N,
    V3_N_EVENTS,
    V4_N,
)
from .parse import parse_stage1, parse_stage2, parse_v2_perception, parse_v3_binary
from .prompts import STAGE1_PROMPT, STAGE2_PROMPT, V2_PERCEPTION_PROMPT, V3_BINARY_PROMPT
from .sampling import (
    encoded_indices_for,
    extract_stills,
    read_still_bgr,
    sample_frames,
    still_pair,
    write_sampled_video,
)
from .still_fix import label_still_image
from .score import (
    V2_REQUIRED,
    V3_REQUIRED,
    V4_REQUIRED,
    assert_complete,
    build_v4_decision,
    is_hallucinated_occluder,
    signal_detection_summary,
    stage1_correct,
    stays_on_person_rate,
    track_number_match,
    v2_perception_summary,
    v3_binary_summary,
    w2_resolution_report,
    w4_followup,
)
from .summarize import (
    STAGE1_FIELDS,
    STAGE2_FIELDS,
    V2_FIELDS,
    V3_FIELDS,
    parsed_csv_path,
    raw_path,
    read_csv,
    write_csv,
    write_json,
    write_raw,
    write_v2_summaries,
    write_w_summaries,
)


def stills_dir(output_root: Path) -> Path:
    return output_root / "stills"


def sampled_dir(output_root: Path) -> Path:
    return output_root / "sampled" / SAMPLING_CENTRED


def _generate(
    backend: Any,
    media: Path,
    prompt: str,
    dest: Path,
    *,
    resume: bool,
) -> GenerateResult:
    if resume and dest.is_file():
        return GenerateResult(
            text=dest.read_text(encoding="utf-8"),
            n_frames_used=None,
            elapsed_sec=0.0,
            extra={"resumed": True},
        )
    result = backend.generate(media, prompt)
    write_raw(dest, result.text)
    return result


def backend_max_pixels(backend: Any, default: int = MAX_PIXELS) -> int:
    value = getattr(backend, "max_pixels", default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def run_v3(
    backend: Any,
    records: Sequence[ClipRecord],
    output_root: Path,
    *,
    resume: bool = True,
    n_events: int = V3_N_EVENTS,
    stage: str = "v3_binary",
) -> list[dict[str, Any]]:
    selected = select_fn_binary_clips(records, n=n_events)
    dest_stills = stills_dir(output_root)
    max_pixels_used = backend_max_pixels(backend)
    rows: list[dict[str, Any]] = []
    relabel_log: list[dict[str, Any]] = []
    for record in selected:
        if record.video_path is None or not record.video_path.is_file():
            raise FileNotFoundError(f"Missing video for {record.clip_id}")
        if record.failure_window is None:
            continue
        before_item, inside_item = still_pair(record.clip_frames, record.failure_window)
        stills = extract_stills(
            record.video_path,
            dest_stills,
            record.clip_id,
            before_index=before_item[1] if before_item else None,
            inside_index=inside_item[1] if inside_item else None,
            write_first=False,
        )
        for role, true_present, key, pair in (
            ("before", True, "before", before_item),
            ("inside", False, "inside", inside_item),
        ):
            image = stills.get(key)
            if image is None or pair is None:
                continue
            dest = raw_path(
                output_root, backend.model_id, stage, f"{record.clip_id}_{role}"
            )
            result = _generate(
                backend, image, V3_BINARY_PROMPT, dest, resume=resume
            )
            parsed, status = parse_v3_binary(result.text)
            pred = None if not parsed else parsed.get("highlighted_box_present")
            is_present, box_area, log_row = label_still_image(
                read_still_bgr(image),
                clip_id=record.clip_id,
                frame=pair[0],
                role=role,
                event_id=record.event_id,
                sequence=record.sequence,
                image_path=str(image),
            )
            if log_row:
                relabel_log.append(log_row)
            correct = None if pred is None else pred is is_present
            rows.append(
                {
                    "clip_id": record.clip_id,
                    "event_id": record.event_id,
                    "sequence": record.sequence,
                    "frame": pair[0],
                    "role": role,
                    "is_present": is_present,
                    "true_present": is_present,
                    "predicted": pred if pred is not None else "",
                    "highlighted_box_present": pred if pred is not None else "",
                    "correct": correct,
                    "box_area_px": box_area,
                    "max_pixels_used": max_pixels_used,
                    "role_present": true_present,
                    "relabelled": bool(log_row),
                    "parse_status": status,
                    "elapsed_sec": round(result.elapsed_sec, 3),
                    "image_path": str(image),
                    "raw_path": str(dest),
                }
            )
    write_csv(
        parsed_csv_path(output_root, backend.model_id, stage), rows, V3_FIELDS
    )
    if relabel_log and stage == "v3_binary":
        from .still_fix import RELABEL_FIELDS

        write_csv(
            output_root / "summary" / "v3_relabel_log.csv",
            relabel_log,
            RELABEL_FIELDS,
        )
    return rows


def run_v2(
    backend: Any,
    records: Sequence[ClipRecord],
    output_root: Path,
    *,
    resume: bool = True,
    n: int = V2_N,
    stage: str = "v2_perception",
) -> list[dict[str, Any]]:
    selected = select_perception_clips(records, n=n)
    dest_stills = stills_dir(output_root)
    rows: list[dict[str, Any]] = []
    for record in selected:
        if record.video_path is None or not record.video_path.is_file():
            raise FileNotFoundError(f"Missing video for {record.clip_id}")
        stills = extract_stills(record.video_path, dest_stills, record.clip_id)
        image = stills["first"]
        dest = raw_path(output_root, backend.model_id, stage, record.clip_id)
        result = _generate(backend, image, V2_PERCEPTION_PROMPT, dest, resume=resume)
        parsed, status = parse_v2_perception(result.text)
        fields = parsed or {}
        expected = record.highlight_before
        pred_number = fields.get("track_number")
        rows.append(
            {
                "clip_id": record.clip_id,
                "event_id": record.event_id,
                "sequence": record.sequence,
                "true_track_number": expected if expected is not None else "",
                "track_number": pred_number or "",
                "track_number_match": track_number_match(pred_number, expected),
                "position": fields.get("position", ""),
                "clothing": fields.get("clothing", ""),
                "found_highlighted_box": fields.get("found_highlighted_box", ""),
                "parse_status": status,
                "elapsed_sec": round(result.elapsed_sec, 3),
                "image_path": str(image),
                "raw_path": str(dest),
            }
        )
    write_csv(
        parsed_csv_path(output_root, backend.model_id, stage), rows, V2_FIELDS
    )
    return rows


def centred_media(
    record: ClipRecord, output_root: Path, *, resume: bool = True
) -> tuple[Path, Any]:
    sampled = sample_frames(record.clip_frames, record.failure_window, n=CENTRED_FRAMES)
    indices = encoded_indices_for(
        record.clip_frames,
        sampled.source_frames,
        segments=record.segments,
        cut_between_segments=record.cut_between_segments,
    )
    dest = sampled_dir(output_root) / f"{record.clip_id}.mp4"
    if not (resume and dest.is_file()):
        if record.video_path is None or not record.video_path.is_file():
            raise FileNotFoundError(f"Missing video for {record.clip_id}")
        write_sampled_video(record.video_path, indices, dest)
    return dest, sampled


def run_centred_stage(
    backend: Any,
    records: Sequence[ClipRecord],
    stage: str,
    output_root: Path,
    *,
    resume: bool = True,
) -> list[dict[str, Any]]:
    prompt = STAGE1_PROMPT if stage == "stage1" else STAGE2_PROMPT
    parse_fn = parse_stage1 if stage == "stage1" else parse_stage2
    rows: list[dict[str, Any]] = []
    for record in records:
        media, sampled = centred_media(record, output_root, resume=resume)
        dest = raw_path(
            output_root,
            backend.model_id,
            stage,
            record.clip_id,
            sampling=SAMPLING_CENTRED,
        )
        result = _generate(backend, media, prompt, dest, resume=resume)
        parsed, status = parse_fn(result.text)
        fields = parsed or {}
        row: dict[str, Any] = {
            "clip_id": record.clip_id,
            "event_id": record.event_id,
            "cause": record.cause,
            "true_cause": record.cause,
            "error_type": record.error_type,
            "stage1_bucket": record.stage1_bucket,
            "sampling": SAMPLING_CENTRED,
            "n_frames_used": result.n_frames_used or sampled.n_total,
            "n_frames_in_window": sampled.n_in_window,
            "n_frames_total": sampled.n_total,
            "elapsed_sec": round(result.elapsed_sec, 3),
            "parse_status": status,
            "video_path": str(media),
            "raw_path": str(dest),
        }
        if stage == "stage1":
            behaviour = fields.get("box_behaviour")
            row.update(
                {
                    "box_behaviour": behaviour or "",
                    "stage1_correct": stage1_correct(record, behaviour),
                    "disappears_at": fields.get("disappears_at", ""),
                    "person_still_visible_after": fields.get(
                        "person_still_visible_after", ""
                    ),
                    "person_description": fields.get("person_description", ""),
                    "confidence": fields.get("confidence", ""),
                }
            )
        else:
            cause = fields.get("cause")
            row.update(
                {
                    "error_kind": fields.get("error_kind", ""),
                    "cause": cause or "",
                    "evidence": fields.get("evidence", ""),
                    "occluder_visible_in_frame": fields.get(
                        "occluder_visible_in_frame", ""
                    ),
                    "confidence": fields.get("confidence", ""),
                    "hallucinated_occluder": is_hallucinated_occluder(
                        record.cause, cause
                    ),
                }
            )
        rows.append(row)
    fields = STAGE1_FIELDS if stage == "stage1" else STAGE2_FIELDS
    write_csv(
        parsed_csv_path(
            output_root, backend.model_id, stage, sampling=SAMPLING_CENTRED
        ),
        rows,
        fields,
    )
    return rows


def load_uniform_rows(
    output_root: Path, model_id: str, stage: str
) -> list[dict[str, str]]:
    path = parsed_csv_path(output_root, model_id, stage)
    if path.is_file():
        return read_csv(path)
    uniform = parsed_csv_path(output_root, model_id, stage, sampling=SAMPLING_UNIFORM)
    return read_csv(uniform) if uniform.is_file() else []


def compare_conditions(
    output_root: Path,
    model_id: str,
    *,
    centred_stage1: Sequence[dict[str, Any]] | None = None,
    centred_stage2: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    uniform_s1 = load_uniform_rows(output_root, model_id, "stage1")
    uniform_s2 = load_uniform_rows(output_root, model_id, "stage2")
    centred_s1 = centred_stage1
    if centred_s1 is None:
        path = parsed_csv_path(
            output_root, model_id, "stage1", sampling=SAMPLING_CENTRED
        )
        centred_s1 = read_csv(path) if path.is_file() else []
    centred_s2 = centred_stage2
    if centred_s2 is None:
        path = parsed_csv_path(
            output_root, model_id, "stage2", sampling=SAMPLING_CENTRED
        )
        centred_s2 = read_csv(path) if path.is_file() else []
    v2_path = parsed_csv_path(output_root, model_id, "v2_perception")
    v3_path = parsed_csv_path(output_root, model_id, "v3_binary")
    report = write_v2_summaries(
        output_root,
        v2_rows=read_csv(v2_path) if v2_path.is_file() else None,
        v3_rows=read_csv(v3_path) if v3_path.is_file() else None,
        uniform_stage1=uniform_s1 or None,
        centred_stage1=centred_s1 or None,
        uniform_stage2=uniform_s2 or None,
        centred_stage2=centred_s2 or None,
    )
    uniform_stays = stays_on_person_rate(uniform_s1) if uniform_s1 else None
    centred_stays = stays_on_person_rate(centred_s1) if centred_s1 else None
    v3_accuracy = (report.get("v3") or {}).get("accuracy")
    if not isinstance(v3_accuracy, (int, float)):
        v3_accuracy = None
    decision = build_v4_decision(
        uniform_stays,
        centred_stays,
        drop=STAYS_ON_PERSON_DROP,
        v3_accuracy=v3_accuracy,
    )
    write_json(output_root / "summary" / "v4_decision.json", decision)
    report["v4_decision"] = decision
    return report


def print_v3_report(rows: Sequence[dict[str, Any]]) -> int:
    summary = v3_binary_summary(rows)
    assert_complete(summary, V3_REQUIRED, "V3")
    present_ci = summary.get("present_accuracy_ci") or [None, None]
    reading = summary.get("present_cell") or {}
    print(
        f"V3/W1 binary: {summary['n_correct']}/{summary['n_parsed']} correct "
        f"(present {summary['present_accuracy']}, absent {summary['absent_accuracy']})"
    )
    print(
        f"present CI [{present_ci[0]}, {present_ci[1]}] "
        f"reading={reading.get('code')}"
    )
    for sequence, block in (summary.get("by_sequence") or {}).items():
        print(
            f"  {sequence}: accuracy={block.get('accuracy')} "
            f"present={block.get('present_accuracy')} absent={block.get('absent_accuracy')}"
        )
        if reading.get("sentence"):
            print(reading["sentence"])
        visible_n = summary.get("n_present_with_box")
        if visible_n not in (None, summary.get("n_present")):
            print(
                f"present stills with measurable box: "
                f"{summary.get('present_with_box_accuracy')} "
                f"({visible_n}/{summary.get('n_present')}) "
                f"CI {summary.get('present_with_box_ci')} "
                f"reading={((summary.get('present_with_box_cell') or {}).get('code'))}"
            )
    if summary.get("near_perfect"):
        print("V3: near-perfect. Perception of presence/absence is fine.")
        return 0
    if summary.get("at_chance"):
        print(
            "V3: at chance. The model cannot see the highlight. "
            "Raise max_pixels / thicken the box before anything else."
        )
        return 3
    print("V3: between chance and ceiling. Inspect stills before a full re-run.")
    return 0


def print_w2_report(
    low_rows: Sequence[dict[str, Any]], high_rows: Sequence[dict[str, Any]]
) -> int:
    report = w2_resolution_report(low_rows, high_rows)
    print(
        "W2 resolution: "
        f"low_present={report['low']['present_accuracy']} "
        f"high_present={report['high']['present_accuracy']} "
        f"delta={report['present_accuracy_delta']} "
        f"code={report['code']}"
    )
    print(report["sentence"])
    return 0


def print_w3_report(rows: Sequence[dict[str, Any]]) -> int:
    sdt = signal_detection_summary(rows)
    print(
        "W3 SDT: "
        f"say_yes_rate={sdt['say_yes_rate']} "
        f"d_prime={sdt['d_prime']} "
        f"criterion={sdt['criterion']} "
        f"bias={sdt['bias']}"
    )
    return 0


def print_w4_report(
    low_rows: Sequence[dict[str, Any]] | None = None,
    high_rows: Sequence[dict[str, Any]] | None = None,
) -> int:
    payload = w4_followup(low_rows, high_rows)
    print(f"W4: {payload['sentence']}")
    if payload.get("low"):
        print(
            "W4 low: "
            f"found={payload['low'].get('found_rate')} "
            f"track_number={payload['low'].get('track_number_accuracy')}"
        )
    if payload.get("high"):
        print(
            "W4 high: "
            f"found={payload['high'].get('found_rate')} "
            f"track_number={payload['high'].get('track_number_accuracy')} "
            f"delta={payload.get('track_number_accuracy_delta')}"
        )
    return 0


def print_v2_report(rows: Sequence[dict[str, Any]]) -> int:
    summary = v2_perception_summary(rows)
    assert_complete(summary, V2_REQUIRED, "V2")
    print(
        f"V2 perception: found={summary['n_found']}/{summary['n_parsed']} "
        f"track_number={summary['n_track_number_match']}/{summary['n_parsed']}"
    )
    n = int(summary["n_parsed"] or 0)
    found = int(summary["n_found"] or 0)
    if n and found / n < 0.5:
        print("V2: highlight often not found. Raise max_pixels / thicken the box.")
        return 3
    if n and int(summary["n_track_number_match"] or 0) / n >= 0.5:
        print("V2: can read the number on most clips. Perception is fine.")
    else:
        print("V2: sees the box but not the label — that is enough to proceed.")
    return 0


def print_v4_decision(report: dict[str, Any], *, require_complete: bool = False) -> int:
    decision = report.get("v4_decision") or {}
    if require_complete:
        assert_complete(decision, V4_REQUIRED, "V4")
    print(
        "V4 stays_on_person: "
        f"uniform={decision.get('uniform_stays_on_person_rate')} "
        f"centred={decision.get('centred_stays_on_person_rate')} "
        f"full_rerun_warranted={decision.get('full_rerun_warranted')}"
    )
    if decision.get("reason"):
        print(decision["reason"])
    print(decision.get("note") or "")
    return 0


def run_v4_preview(
    backend: Any,
    records: Sequence[ClipRecord],
    output_root: Path,
    *,
    resume: bool = True,
    n: int = V4_N,
    stages: Sequence[str] = ("stage1",),
) -> dict[str, Any]:
    selected = select_centred_preview_clips(records, n=n)
    centred_s1 = None
    centred_s2 = None
    if "stage1" in stages:
        centred_s1 = run_centred_stage(
            backend, selected, "stage1", output_root, resume=resume
        )
    if "stage2" in stages:
        centred_s2 = run_centred_stage(
            backend, selected, "stage2", output_root, resume=resume
        )
    return compare_conditions(
        output_root,
        backend.model_id,
        centred_stage1=centred_s1,
        centred_stage2=centred_s2,
    )
