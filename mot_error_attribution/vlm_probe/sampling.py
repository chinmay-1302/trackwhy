"""Event-centred frame sampling and still extraction. Does not alter the clip set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import CENTRED_FRAMES, MIN_FRAMES_IN, MIN_HIGHLIGHT_PX

CUT_MARKER_FRAMES = 1


@dataclass(frozen=True)
class SampledFrames:
    source_frames: list[int]
    encoded_indices: list[int]
    n_in_window: int
    n_total: int
    window: tuple[int, int] | None


def uniform_pick(items: Sequence[int], n: int) -> list[int]:
    """Evenly spaced unique picks, preserving original order."""
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    last = len(items) - 1
    chosen: list[int] = []
    used: set[int] = set()
    for i in range(n):
        idx = int(round(i * last / (n - 1))) if n > 1 else last // 2
        if idx in used:
            for delta in range(1, len(items)):
                lo, hi = idx - delta, idx + delta
                if 0 <= lo <= last and lo not in used:
                    idx = lo
                    break
                if 0 <= hi <= last and hi not in used:
                    idx = hi
                    break
        used.add(idx)
        chosen.append(idx)
    return [items[i] for i in sorted(used)]


def sample_frames(
    clip_frames: Sequence[int],
    failure_window: tuple[int, int] | None,
    n: int = CENTRED_FRAMES,
    min_in: int = MIN_FRAMES_IN,
) -> SampledFrames:
    frames = [int(frame) for frame in clip_frames]
    if not frames:
        return SampledFrames([], [], 0, 0, failure_window)
    if failure_window is None:
        picked = uniform_pick(frames, n)
        encoded = [frames.index(frame) for frame in picked]
        return SampledFrames(picked, encoded, 0, len(picked), None)
    fs, fe = int(failure_window[0]), int(failure_window[1])
    inside = [frame for frame in frames if fs <= frame <= fe]
    before = [frame for frame in frames if frame < fs]
    after = [frame for frame in frames if frame > fe]
    n_in = min(len(inside), max(min_in, n // 2)) if inside else 0
    if inside and len(inside) < min_in:
        n_in = len(inside)
    picked_in = uniform_pick(inside, n_in) if inside else []
    n_out = max(0, n - len(picked_in))
    n_before = n_out // 2
    n_after = n_out - n_before
    if len(before) < n_before:
        n_after += n_before - len(before)
        n_before = len(before)
    if len(after) < n_after:
        n_before = min(len(before), n_before + (n_after - len(after)))
        n_after = len(after)
    picked_out = uniform_pick(before, n_before) + uniform_pick(after, n_after)
    picked = sorted(picked_in + picked_out)
    encoded = encoded_indices_for(frames, picked)
    return SampledFrames(
        source_frames=picked,
        encoded_indices=encoded,
        n_in_window=len(picked_in),
        n_total=len(picked),
        window=(fs, fe),
    )


def encoded_index_for(
    clip_frames: Sequence[int],
    source_frame: int,
    *,
    segments: Sequence[Sequence[int]] | None = None,
    cut_between_segments: bool = False,
) -> int:
    frames = [int(frame) for frame in clip_frames]
    if not cut_between_segments or not segments or len(segments) <= 1:
        return frames.index(int(source_frame))
    encoded = 0
    target = int(source_frame)
    for index, segment in enumerate(segments):
        if index:
            encoded += CUT_MARKER_FRAMES
        as_list = [int(frame) for frame in segment]
        if target in as_list:
            return encoded + as_list.index(target)
        encoded += len(as_list)
    return frames.index(target)


def encoded_indices_for(
    clip_frames: Sequence[int],
    source_frames: Sequence[int],
    *,
    segments: Sequence[Sequence[int]] | None = None,
    cut_between_segments: bool = False,
) -> list[int]:
    return [
        encoded_index_for(
            clip_frames,
            frame,
            segments=segments,
            cut_between_segments=cut_between_segments,
        )
        for frame in source_frames
    ]


def still_indices(
    clip_frames: Sequence[int],
    failure_window: tuple[int, int],
) -> tuple[int | None, int | None]:
    """Encoded indices for (before-window, inside-window) stills."""
    pair = still_pair(clip_frames, failure_window)
    return pair[0][1] if pair[0] else None, pair[1][1] if pair[1] else None


def still_pair(
    clip_frames: Sequence[int],
    failure_window: tuple[int, int],
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """((source_frame, encoded_index) before window, (source, encoded) inside)."""
    frames = [int(frame) for frame in clip_frames]
    fs, fe = int(failure_window[0]), int(failure_window[1])
    before = [frame for frame in frames if frame < fs]
    inside = [frame for frame in frames if fs <= frame <= fe]
    before_frame = before[-1] if before else None
    inside_frame = inside[len(inside) // 2] if inside else None
    before_item = (
        (before_frame, frames.index(before_frame)) if before_frame is not None else None
    )
    inside_item = (
        (inside_frame, frames.index(inside_frame)) if inside_frame is not None else None
    )
    return before_item, inside_item


def rendered_highlight_area(
    frame_img: np.ndarray,
    highlight_rgb: tuple[int, int, int] = (255, 255, 0),
    tol: int = 30,
) -> int:
    """Count pixels matching the highlight colour in the rendered still.

    OpenCV stills are BGR; the comparison is in RGB as specified.
    """
    if frame_img.ndim != 3 or frame_img.shape[2] < 3:
        return 0
    rgb = frame_img[:, :, :3][..., ::-1]
    distance = np.abs(
        rgb.astype(np.int16) - np.array(highlight_rgb, dtype=np.int16)
    ).sum(axis=2)
    return int((distance <= int(tol)).sum())


def still_label(frame_img: np.ndarray) -> str:
    return "present" if rendered_highlight_area(frame_img) > MIN_HIGHLIGHT_PX else "absent"


def highlight_box_area_px(frame: np.ndarray) -> int:
    """Pixel count of the yellow highlight, or 0 if none is visible."""
    area = rendered_highlight_area(frame)
    return area if area > MIN_HIGHLIGHT_PX else 0


def box_area_from_still(path: Path) -> int:
    import cv2

    frame = cv2.imread(str(path))
    if frame is None:
        return 0
    return highlight_box_area_px(frame)


def read_still_bgr(path: Path) -> np.ndarray:
    import cv2

    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Failed to read still: {path}")
    return frame


def read_video_frame(path: Path, index: int) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {index} from {path}")
        return frame
    finally:
        capture.release()


def write_still(path: Path, frame: np.ndarray) -> Path:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Failed to write still: {path}")
    return path


def extract_stills(
    video_path: Path,
    dest_dir: Path,
    stem: str,
    *,
    first_index: int = 0,
    before_index: int | None = None,
    inside_index: int | None = None,
    write_first: bool = True,
) -> dict[str, Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    if write_first:
        out["first"] = write_still(
            dest_dir / f"{stem}_first.jpg", read_video_frame(video_path, first_index)
        )
    if before_index is not None:
        out["before"] = write_still(
            dest_dir / f"{stem}_before.jpg", read_video_frame(video_path, before_index)
        )
    if inside_index is not None:
        out["inside"] = write_still(
            dest_dir / f"{stem}_inside.jpg", read_video_frame(video_path, inside_index)
        )
    return out


def write_sampled_video(
    video_path: Path,
    encoded_indices: Sequence[int],
    output_path: Path,
    *,
    fps: int = 10,
) -> Path:
    from mot_error_attribution.probe_clips import encode_h264, pad_even

    frames = [read_video_frame(video_path, index) for index in encoded_indices]
    if not frames:
        raise RuntimeError(f"No frames sampled from {video_path}")
    height, width = frames[0].shape[:2]
    even_w = width + (width % 2)
    even_h = height + (height % 2)
    padded = [pad_even(frame, even_w, even_h) for frame in frames]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encode_h264(padded, output_path, even_w, even_h, fps=fps)
    return output_path
