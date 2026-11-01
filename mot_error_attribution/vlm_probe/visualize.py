"""Replay probe clips with a Stage 1 + Stage 2 results bar under each frame."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from ..probe_clips import FPS, encode_h264, pad_even
from .config import DEFAULT_MODEL, DEFAULT_OUTPUT_ROOT, SAMPLING_FROZEN
from .score import STAGE1_EXPECTED, map_rule_cause
from .summarize import model_dir_name, parsed_csv_path, read_csv

BG = (28, 28, 28)
FG = (230, 230, 230)
MUTED = (170, 170, 170)
GREEN = (70, 190, 80)
RED = (50, 50, 220)
SEPARATOR = (70, 70, 70)

PAD_X = 16
PAD_Y = 12
LINE_GAP = 6
HEADER_SCALE = 0.55
BODY_SCALE = 0.48
THICKNESS = 1
MAX_DESC_LINES = 3
MAX_EVIDENCE_LINES = 3
MAX_BAR_HEIGHT = 320


@dataclass(frozen=True)
class JoinedClip:
    clip_id: str
    event_id: str
    error_type: str
    stage1_bucket: str
    video_path: Path | None
    box_behaviour: str
    disappears_at: str
    person_still_visible_after: str
    person_description: str
    stage1_confidence: str
    true_cause: str
    pred_cause: str
    error_kind: str
    evidence: str
    hallucinated_occluder: str
    stage2_confidence: str
    stage1_parse_status: str
    stage2_parse_status: str


@dataclass(frozen=True)
class BarLine:
    text: str
    color: tuple[int, int, int]
    scale: float = BODY_SCALE
    max_lines: int | None = None


def _cell(row: Mapping[str, Any] | None, key: str) -> str:
    if not row:
        return ""
    value = row.get(key)
    if value is None:
        return ""
    return str(value).strip()


def expected_behaviours_for_bucket(bucket: str) -> frozenset[str]:
    return STAGE1_EXPECTED.get(bucket, frozenset())


def stage1_is_correct(bucket: str, box_behaviour: str | None) -> bool | None:
    expected = expected_behaviours_for_bucket(bucket)
    if not expected or not box_behaviour:
        return None
    if box_behaviour == "no_highlighted_box_found":
        return False
    return box_behaviour in expected


def stage2_is_correct(true_cause: str, pred_cause: str | None) -> bool | None:
    if not true_cause or not pred_cause:
        return None
    return pred_cause == map_rule_cause(true_cause)


def verdict_color(correct: bool | None) -> tuple[int, int, int]:
    if correct is True:
        return GREEN
    if correct is False:
        return RED
    return MUTED


def verdict_label(correct: bool | None) -> str:
    if correct is True:
        return "OK"
    if correct is False:
        return "WRONG"
    return "n/a"


def is_mistake(clip: JoinedClip) -> bool:
    if stage1_is_correct(clip.stage1_bucket, clip.box_behaviour or None) is False:
        return True
    if not clip.true_cause or not clip.pred_cause:
        return False
    return clip.pred_cause != map_rule_cause(clip.true_cause)


def join_stage_rows(
    stage1_rows: Sequence[Mapping[str, Any]],
    stage2_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[JoinedClip]:
    by_id = {_cell(row, "clip_id"): row for row in (stage2_rows or []) if _cell(row, "clip_id")}
    joined: list[JoinedClip] = []
    for row in stage1_rows:
        clip_id = _cell(row, "clip_id")
        if not clip_id:
            continue
        stage2 = by_id.get(clip_id)
        video = _cell(row, "video_path") or _cell(stage2, "video_path")
        joined.append(
            JoinedClip(
                clip_id=clip_id,
                event_id=_cell(row, "event_id") or _cell(stage2, "event_id"),
                error_type=_cell(row, "error_type") or _cell(stage2, "error_type"),
                stage1_bucket=_cell(row, "stage1_bucket"),
                video_path=Path(video) if video else None,
                box_behaviour=_cell(row, "box_behaviour"),
                disappears_at=_cell(row, "disappears_at"),
                person_still_visible_after=_cell(row, "person_still_visible_after"),
                person_description=_cell(row, "person_description"),
                stage1_confidence=_cell(row, "confidence"),
                true_cause=_cell(stage2, "true_cause") or _cell(row, "cause"),
                pred_cause=_cell(stage2, "cause"),
                error_kind=_cell(stage2, "error_kind"),
                evidence=_cell(stage2, "evidence"),
                hallucinated_occluder=_cell(stage2, "hallucinated_occluder"),
                stage2_confidence=_cell(stage2, "confidence"),
                stage1_parse_status=_cell(row, "parse_status"),
                stage2_parse_status=_cell(stage2, "parse_status"),
            )
        )
    return joined


def select_joined(
    clips: Sequence[JoinedClip],
    *,
    clip_ids: Sequence[str] | None = None,
    limit: int | None = None,
    mistakes_only: bool = False,
) -> list[JoinedClip]:
    selected = list(clips)
    if clip_ids:
        wanted = [item.strip() for item in clip_ids if item.strip()]
        by_id = {clip.clip_id: clip for clip in selected}
        missing = [clip_id for clip_id in wanted if clip_id not in by_id]
        if missing:
            raise ValueError(f"Unknown clip_id(s): {sorted(missing)}")
        selected = [by_id[clip_id] for clip_id in wanted]
    if mistakes_only:
        selected = [clip for clip in selected if is_mistake(clip)]
    if limit is not None:
        selected = selected[:limit]
    return selected


def parse_clip_ids(values: Sequence[str]) -> list[str]:
    ids: list[str] = []
    for value in values:
        for item in value.split(","):
            clip_id = item.strip()
            if clip_id:
                ids.append(clip_id)
    return ids


def wrap_text(
    text: str,
    max_width: int,
    *,
    scale: float = BODY_SCALE,
    thickness: int = THICKNESS,
    max_lines: int | None = None,
) -> list[str]:
    import cv2

    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    font = cv2.FONT_HERSHEY_SIMPLEX
    words = cleaned.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        width, _ = cv2.getTextSize(candidate, font, scale, thickness)[0]
        if width <= max_width or not current:
            current = candidate
            continue
        lines.append(current)
        current = word
    if current:
        lines.append(current)
    if max_lines is not None and len(lines) > max_lines:
        kept = lines[:max_lines]
        last = kept[-1]
        while True:
            suffix = last.rstrip() + "..."
            width, _ = cv2.getTextSize(suffix, font, scale, thickness)[0]
            if width <= max_width or len(last) <= 4:
                kept[-1] = suffix
                break
            last = last[:-1]
        lines = kept
    return lines


def _ascii(text: str) -> str:
    return (text or "").encode("ascii", "replace").decode("ascii")


def bar_lines(clip: JoinedClip, model_id: str) -> list[BarLine]:
    s1_ok = stage1_is_correct(clip.stage1_bucket, clip.box_behaviour or None)
    s2_ok = stage2_is_correct(clip.true_cause, clip.pred_cause or None)
    expected = " | ".join(sorted(expected_behaviours_for_bucket(clip.stage1_bucket))) or "?"
    mapped = map_rule_cause(clip.true_cause) if clip.true_cause else "?"
    header = (
        f"{clip.clip_id}  {clip.error_type}  {clip.stage1_bucket}  "
        f"{clip.event_id}    {model_id}"
    )
    s1 = (
        f"S1  pred: {clip.box_behaviour or '?'}  [{verdict_label(s1_ok)}]  "
        f"expected: {expected}    conf: {clip.stage1_confidence or '?'}  "
        f"disappears_at={clip.disappears_at or '?'}  "
        f"still_visible={clip.person_still_visible_after or '?'}"
    )
    s2 = (
        f"S2  pred: {clip.pred_cause or '?'}  [{verdict_label(s2_ok)}]  "
        f"GT: {clip.true_cause or '?'} -> {mapped}    "
        f"kind={clip.error_kind or '?'}  conf: {clip.stage2_confidence or '?'}"
    )
    if clip.hallucinated_occluder.lower() in {"true", "1", "yes"}:
        s2 += "    hallucinated_occluder=True"
    lines = [
        BarLine(_ascii(header), FG, HEADER_SCALE, max_lines=2),
        BarLine(_ascii(s1), verdict_color(s1_ok), max_lines=2),
        BarLine(_ascii(s2), verdict_color(s2_ok), max_lines=2),
    ]
    person = _ascii(clip.person_description)
    if person:
        lines.append(
            BarLine("person: " + person, MUTED, max_lines=MAX_DESC_LINES)
        )
    evidence = _ascii(clip.evidence)
    if evidence:
        lines.append(
            BarLine("evidence: " + evidence, MUTED, max_lines=MAX_EVIDENCE_LINES)
        )
    return lines


def _rewrap_lines(lines: Sequence[BarLine], max_width: int) -> list[BarLine]:
    out: list[BarLine] = []
    for line in lines:
        wrapped = wrap_text(
            line.text,
            max_width,
            scale=line.scale,
            max_lines=line.max_lines,
        )
        if not wrapped:
            continue
        out.append(BarLine(wrapped[0], line.color, line.scale, line.max_lines))
        indent = "  "
        for extra in wrapped[1:]:
            out.append(BarLine(indent + extra, line.color, line.scale, line.max_lines))
    return out


def render_bar(clip: JoinedClip, width: int, model_id: str) -> np.ndarray:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    usable = max(32, width - 2 * PAD_X)
    lines = _rewrap_lines(bar_lines(clip, model_id), usable)
    metrics: list[tuple[BarLine, int, int]] = []
    for line in lines:
        (_tw, th), baseline = cv2.getTextSize(line.text, font, line.scale, THICKNESS)
        metrics.append((line, th, baseline))
    content_h = PAD_Y
    for _line, th, baseline in metrics:
        content_h += th + baseline + LINE_GAP
    content_h += PAD_Y
    height = min(max(content_h, 64), MAX_BAR_HEIGHT)
    if height % 2:
        height += 1
    bar = np.full((height, width, 3), BG, dtype=np.uint8)
    cv2.line(bar, (0, 0), (width, 0), SEPARATOR, 1)
    y = PAD_Y
    for line, th, baseline in metrics:
        y += th
        if y + baseline > height - 4:
            break
        cv2.putText(
            bar,
            line.text,
            (PAD_X, y),
            font,
            line.scale,
            line.color,
            THICKNESS,
            cv2.LINE_AA,
        )
        y += baseline + LINE_GAP
    return bar


def attach_bar(frame: np.ndarray, bar: np.ndarray) -> np.ndarray:
    if bar.shape[1] != frame.shape[1]:
        raise ValueError(f"Bar width {bar.shape[1]} != frame width {frame.shape[1]}")
    return np.vstack([frame, bar])


def iter_visualized_frames(
    video_path: Path,
    clip: JoinedClip,
    model_id: str,
) -> tuple[Iterator[np.ndarray], int, int]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        raise RuntimeError(f"Empty video: {video_path}")
    bar = render_bar(clip, width=first.shape[1], model_id=model_id)
    stacked = attach_bar(first, bar)
    even_w = stacked.shape[1] + (stacked.shape[1] % 2)
    even_h = stacked.shape[0] + (stacked.shape[0] % 2)

    def frames() -> Iterator[np.ndarray]:
        try:
            yield pad_even(stacked, even_w, even_h)
            while True:
                more, frame = cap.read()
                if not more or frame is None:
                    break
                yield pad_even(attach_bar(frame, bar), even_w, even_h)
        finally:
            cap.release()

    return frames(), even_w, even_h


def visualize_clip(clip: JoinedClip, output_path: Path, model_id: str) -> Path:
    if clip.video_path is None or not clip.video_path.is_file():
        raise FileNotFoundError(f"Missing video for {clip.clip_id}: {clip.video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames, width, height = iter_visualized_frames(clip.video_path, clip, model_id)
    encode_h264(frames, output_path, width, height, fps=FPS)
    return output_path


def default_viz_dir(output_root: Path, model_id: str) -> Path:
    return output_root / "viz" / model_dir_name(model_id)


def load_joined(
    output_root: Path, model_id: str, *, sampling: str | None = None
) -> list[JoinedClip]:
    stage1_path = parsed_csv_path(output_root, model_id, "stage1", sampling=sampling)
    stage2_path = parsed_csv_path(output_root, model_id, "stage2", sampling=sampling)
    if not stage1_path.is_file():
        raise FileNotFoundError(f"Missing Stage 1 parsed CSV: {stage1_path}")
    stage2_rows = read_csv(stage2_path) if stage2_path.is_file() else []
    return join_stage_rows(read_csv(stage1_path), stage2_rows)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay probe clips with Stage 1 + Stage 2 VLM results under each frame"
    )
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--clip-ids",
        action="append",
        default=[],
        help="Comma-separated clip_id filter (repeatable)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--mistakes-only",
        action="store_true",
        help="Keep clips where Stage 1 is wrong or Stage 2 pred != mapped GT",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <output-root>/viz/<model>",
    )
    parser.add_argument(
        "--sampling",
        default=None,
        help="parsed CSV suffix (e.g. frozen). Default is the original v1 CSVs",
    )
    parser.add_argument(
        "--void-json",
        type=Path,
        default=None,
        help="If set, visualize the clip_ids listed in this JSON",
    )
    parser.add_argument(
        "--frozen-17",
        action="store_true",
        help="Re-render the 17 re-rendered clips using frozen Stage 1/2 answers",
    )
    return parser


def _clip_ids_from_void(path: Path) -> list[str]:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [str(item).strip() for item in (payload.get("clip_ids") or []) if str(item).strip()]
    if not ids:
        raise ValueError(f"No clip_ids in {path}")
    return ids


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = _must_resolve(args.output_root)
    model_id = args.model
    sampling = args.sampling
    clip_ids = parse_clip_ids(args.clip_ids)
    if args.frozen_17:
        sampling = sampling or SAMPLING_FROZEN
        void_path = _must_resolve(
            args.void_json or Path("out/probe_clips/summary/void_a1_clip_ids.json")
        )
        clip_ids = clip_ids or _clip_ids_from_void(void_path)
    elif args.void_json:
        clip_ids = clip_ids or _clip_ids_from_void(_must_resolve(args.void_json))
    clips = select_joined(
        load_joined(output_root, model_id, sampling=sampling),
        clip_ids=clip_ids or None,
        limit=args.limit,
        mistakes_only=args.mistakes_only,
    )
    dest_root = _must_resolve(args.output_dir) if args.output_dir else default_viz_dir(
        output_root, model_id
    )
    dest_root.mkdir(parents=True, exist_ok=True)
    if not clips:
        print("no clips selected")
        return 0
    written = 0
    failed: list[str] = []
    for clip in clips:
        dest = dest_root / f"{clip.clip_id}.mp4"
        try:
            visualize_clip(clip, dest, model_id)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"{clip.clip_id}: {exc}")
            failed.append(clip.clip_id)
            continue
        written += 1
        print(f"wrote {dest}")
    print(f"visualized {written}/{len(clips)} clips under {dest_root}")
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
