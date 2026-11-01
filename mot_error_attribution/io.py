from __future__ import annotations

import configparser
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import EVALUATION_SEQUENCES, PipelineConfig
from .geometry import xywh_to_xyxy
from .models import GTBox, SequenceData, TrkBox


SEQUENCE_PATTERN = re.compile(r"^MOT20-\d{2}$")
MOTCHALLENGE_CLASS_NAMES = {
    1: "pedestrian",
    2: "person_on_vehicle",
    3: "car",
    4: "bicycle",
    5: "motorbike",
    6: "non_mot_vehicle",
    7: "static_person",
    8: "distractor",
    9: "occluder",
    10: "occluder_on_ground",
    11: "occluder_full",
    12: "reflection",
    13: "crowd",
}


def _sequence_from_file_name(file_name: str) -> str:
    for part in Path(file_name).parts:
        if SEQUENCE_PATTERN.match(part):
            return part
    raise ValueError(f"Cannot find a MOT20 sequence name in image path: {file_name}")


def _resolve_image_path(dataset_root: Path, file_name: str) -> Path:
    relative = Path(file_name)
    sequence = _sequence_from_file_name(file_name)
    frame_name = relative.name
    candidates = (
        dataset_root / relative,
        dataset_root.parent / relative,
        dataset_root / "train" / sequence / "img1" / frame_name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    # Preserve a useful expected path even when clips are intentionally disabled.
    return (dataset_root / "train" / sequence / "img1" / frame_name).resolve()


def _read_seqinfo(dataset_root: Path, sequence: str) -> tuple[int, int, int, float]:
    path = dataset_root / "train" / sequence / "seqinfo.ini"
    if not path.is_file():
        raise FileNotFoundError(f"Missing seqinfo.ini for {sequence}: {path}")
    parser = configparser.ConfigParser()
    parser.read(path)
    section = parser["Sequence"]
    return (
        int(section["seqLength"]),
        int(section["imWidth"]),
        int(section["imHeight"]),
        float(section["frameRate"]),
    )


def _read_gt(
    dataset_root: Path,
    sequence: str,
    selected_frames: set[int],
) -> tuple[
    dict[int, list[GTBox]],
    dict[int, int],
    dict[int, int],
    dict[tuple[int, int], int],
]:
    path = dataset_root / "train" / sequence / "gt" / "gt.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing MOT20 ground truth for {sequence}: {path}")
    by_frame: dict[int, list[GTBox]] = defaultdict(list)
    full_hist: Counter[int] = Counter()
    selected_hist: Counter[int] = Counter()
    selected_flag_class: Counter[tuple[int, int]] = Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) < 9:
                raise ValueError(f"Malformed GT row at {path}:{line_number}: {row}")
            frame = int(float(row[0]))
            gt_id = int(float(row[1]))
            flag = int(float(row[6]))
            cls = int(float(row[7]))
            visibility = float(row[8])
            full_hist[cls] += 1
            if frame not in selected_frames:
                continue
            box = GTBox(
                frame=frame,
                gt_id=gt_id,
                bbox=xywh_to_xyxy(tuple(map(float, row[2:6]))),
                cls=cls,
                visibility=visibility,
                flag=flag,
            )
            by_frame[frame].append(box)
            selected_hist[cls] += 1
            selected_flag_class[(cls, flag)] += 1
    for frame in selected_frames:
        by_frame.setdefault(frame, [])
    return (
        dict(by_frame),
        dict(sorted(full_hist.items())),
        dict(sorted(selected_hist.items())),
        dict(sorted(selected_flag_class.items())),
    )


def load_sequences(
    tracker_output: str | Path,
    dataset_root: str | Path,
    config: PipelineConfig,
) -> tuple[dict[str, SequenceData], dict[str, Any]]:
    """Load tracker geometry and GT from COCO JSON or MOT-format text files.

    Confidence, appearance, detector, and ReID fields are deliberately never read.
    MOT20-01 and MOT20-03 are held-out evaluation sequences. Full-sequence mode
    evaluates every dataset frame and treats tracker frames with no row as empty.
    ``legacy_half`` exists only to reproduce the fixed v1 frame set for Pass A.
    """
    tracker_path = Path(tracker_output)
    root = Path(dataset_root)
    image_rows_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tracker_all: dict[tuple[str, int], list[TrkBox]] = defaultdict(list)
    source_format: str
    if tracker_path.is_dir() or tracker_path.suffix.lower() == ".txt":
        source_format = "mot_text"
        text_files = (
            sorted(tracker_path.glob("*.txt"))
            if tracker_path.is_dir()
            else [tracker_path]
        )
        if not text_files:
            raise ValueError(f"No MOT-format .txt files found in {tracker_path}")
        for text_path in text_files:
            sequence = text_path.stem
            if not SEQUENCE_PATTERN.match(sequence):
                raise ValueError(
                    f"MOT tracker filename must be a sequence name such as MOT20-01.txt: {text_path}"
                )
            if sequence not in EVALUATION_SEQUENCES:
                continue
            represented_frames: set[int] = set()
            with text_path.open(newline="", encoding="utf-8") as handle:
                for line_number, row in enumerate(csv.reader(handle), start=1):
                    if not row:
                        continue
                    if len(row) < 6:
                        raise ValueError(
                            f"Malformed MOT tracker row at {text_path}:{line_number}: {row}"
                        )
                    frame = int(float(row[0]))
                    tracker_all[(sequence, frame)].append(
                        TrkBox(
                            frame=frame,
                            trk_id=int(float(row[1])),
                            bbox=xywh_to_xyxy(tuple(map(float, row[2:6]))),
                        )
                    )
                    represented_frames.add(frame)
            if not represented_frames:
                raise ValueError(f"MOT tracker file is empty: {text_path}")
            image_rows_by_sequence[sequence] = [
                {
                    "frame_id": frame,
                    "file_name": f"MOT20/train/{sequence}/img1/{frame:06d}.jpg",
                }
                for frame in sorted(represented_frames)
            ]
    elif tracker_path.suffix.lower() == ".json":
        source_format = "coco_json"
        with tracker_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
            raise ValueError("Tracker JSON must be COCO-style and contain an 'images' list")
        if not isinstance(payload.get("annotations"), list):
            raise ValueError("Tracker JSON must contain an 'annotations' list")
        image_lookup: dict[int, tuple[str, int, str]] = {}
        seen_sequence_frames: set[tuple[str, int]] = set()
        for image in payload["images"]:
            sequence = _sequence_from_file_name(str(image["file_name"]))
            if sequence not in EVALUATION_SEQUENCES:
                continue
            frame = int(image["frame_id"])
            image_id = int(image["id"])
            if image_id in image_lookup:
                raise ValueError(f"Duplicate image id in tracker JSON: {image_id}")
            key = (sequence, frame)
            if key in seen_sequence_frames:
                raise ValueError(
                    f"Duplicate image frame in tracker JSON: {sequence} frame {frame}"
                )
            seen_sequence_frames.add(key)
            image_lookup[image_id] = (sequence, frame, str(image["file_name"]))
            image_rows_by_sequence[sequence].append(image)
        unknown_image_ids: set[int] = set()
        for annotation in payload["annotations"]:
            image_id = int(annotation["image_id"])
            image_record = image_lookup.get(image_id)
            if image_record is None:
                unknown_image_ids.add(image_id)
                continue
            sequence, frame, _ = image_record
            tracker_all[(sequence, frame)].append(
                TrkBox(
                    frame=frame,
                    trk_id=int(annotation["track_id"]),
                    bbox=xywh_to_xyxy(annotation["bbox"]),
                )
            )
        if unknown_image_ids:
            preview = sorted(unknown_image_ids)[:10]
            raise ValueError(f"Tracker annotations reference unknown image ids: {preview}")
    elif tracker_path.suffix.lower() in {".pickle", ".pkl"}:
        raise ValueError(
            "Detection pickles do not contain tracker IDs. Supply finalized TrackTrack MOT text files instead."
        )
    else:
        raise ValueError(
            f"Unsupported tracker source {tracker_path}; expected a COCO JSON, MOT .txt, or directory of MOT .txt files"
        )

    sequences: dict[str, SequenceData] = {}
    coverage: dict[str, Any] = {}
    for sequence, image_rows in sorted(image_rows_by_sequence.items()):
        num_frames, width, height, fps = _read_seqinfo(root, sequence)
        available = sorted({int(row["frame_id"]) for row in image_rows})
        invalid = [frame for frame in available if frame < 1 or frame > num_frames]
        if invalid:
            raise ValueError(
                f"Tracker source contains out-of-range frames for {sequence}: {invalid[:10]}"
            )
        if config.frame_scope == "legacy_half":
            selected = list(range(num_frames // 2 + 1, num_frames + 1))
        else:
            selected = list(range(1, num_frames + 1))
        dropped: list[int] = []
        if not selected:
            raise ValueError(f"No eligible frames remain for {sequence}")
        selected_set = set(selected)
        gt_by_frame, full_hist, selected_hist, selected_flag_class = _read_gt(
            root, sequence, selected_set
        )
        tracker_by_frame = {
            frame: list(tracker_all.get((sequence, frame), [])) for frame in selected
        }
        for frame, boxes in tracker_by_frame.items():
            ids = [box.trk_id for box in boxes]
            duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})
            if duplicate_ids:
                raise ValueError(
                    f"Tracker emits duplicate IDs in {sequence} frame {frame}: {duplicate_ids}"
                )
        row_by_frame = {int(row["frame_id"]): row for row in image_rows}
        image_paths = {
            frame: str(
                _resolve_image_path(
                    root,
                    str(
                        row_by_frame.get(
                            frame,
                            {
                                "file_name": (
                                    f"MOT20/train/{sequence}/img1/{frame:06d}.jpg"
                                )
                            },
                        )["file_name"]
                    ),
                )
            )
            for frame in selected
        }
        sequences[sequence] = SequenceData(
            name=sequence,
            num_frames=num_frames,
            width=width,
            height=height,
            fps=fps,
            dataset_root=str(root.resolve()),
            available_frames=tuple(available),
            selected_frames=tuple(selected),
            dropped_non_mining_frames=tuple(dropped),
            image_paths=image_paths,
            gt_by_frame=gt_by_frame,
            tracker_by_frame=tracker_by_frame,
            class_histogram_full=full_hist,
            class_histogram_selected=selected_hist,
            flag_class_histogram_selected=selected_flag_class,
        )
        coverage[sequence] = {
            "sequence_length": num_frames,
            "tracker_nonempty_frame_count": len(available),
            "tracker_nonempty_frame_min": min(available),
            "tracker_nonempty_frame_max": max(available),
            "tracker_empty_selected_frame_count": len(set(selected) - set(available)),
            "selected_frame_count": len(selected),
            "selected_frame_min": min(selected),
            "selected_frame_max": max(selected),
        }

    absent_expected = sorted(set(EVALUATION_SEQUENCES) - set(sequences))
    observed_classes = sorted(
        {
            class_id
            for sequence in sequences.values()
            for class_id in sequence.class_histogram_full
        }
    )
    configured_classes = {
        class_id for values in config.classes.values() for class_id in values
    }
    audit = {
        "tracker_source": str(tracker_path.resolve()),
        "tracker_source_format": source_format,
        "dataset_root": str(root.resolve()),
        "sequences_in_tracker_source": sorted(sequences),
        "expected_sequences_absent_from_tracker_source": absent_expected,
        "frame_policy": (
            "legacy_v1_half_validation"
            if config.frame_scope == "legacy_half"
            else "mot20-01+mot20-03_full_sequences"
        ),
        "coverage": coverage,
        "motchallenge_class_reference": {
            str(key): value for key, value in MOTCHALLENGE_CLASS_NAMES.items()
        },
        "observed_gt_classes": observed_classes,
        "observed_classes_missing_from_config": sorted(
            set(observed_classes) - configured_classes
        ),
        "detector_training_frame_provenance": (
            "mot20-02+mot20-05+crowdhuman; 01/03 held out"
        ),
    }
    return sequences, audit
