from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CLASSES: dict[str, list[int]] = {
    "TARGET_CLASSES": [1],
    "STATIC_PERSON": [7],
    "DISTRACTOR": [8, 12],
    "OCCLUDER_CLASSES": [9, 10, 11],
    "CROWD": [13],
    "ON_VEHICLE": [2],
    # MOTChallenge class 6 is ignored by TrackEval for MOT20. It is kept
    # separate because it is not an attribution cause in the requested taxonomy.
    "NON_MOT_VEHICLE": [6],
}

DEFAULT_THRESHOLDS: dict[str, float | int] = {
    "MATCH_IOU": 0.5,
    "IOA_OCCLUDER": 0.90,
    # Selected by the v4 sweep over {0.05, 0.10, 0.15, 0.20, 0.30}.
    "IOA_INTERACTION": 0.20,
    "AMBIGUITY_MARGIN": 0.10,
    "FRAME_VOTE_CONFIDENCE": 0.60,
    "VIS_DROP_DELTA": 0.25,
    "CROWD_OVERLAP": 0.50,
    "AGG_COVERAGE_THRESH": 0.90,
    "AGG_MIN_OCCLUDERS": 2,
    "DENSITY_RADIUS_PX": 150,
    "HIJACK_IOU": 0.30,
    "EVENT_WINDOW": 15,
    # Histogram knee is at 1–2 frames; 3 is the first length past that spike.
    "MIN_FN_RUN_FRAMES": 3,
    "FRAG_MIN_GAP": 3,
    "EDGE_MARGIN_PX": 20,
    # Own constant. Must not share a value with EVENT_WINDOW.
    "REENTRY_MIN_ABSENT_FRAMES": 25,
}

DEFAULT_IOA_OCCLUDER_BY_CLASS: dict[str, float] = {
    "large_object": 0.90,
    "person": 0.70,
}

DEFAULT_IOA_SWEEP_THRESHOLDS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30)

DEFAULT_VIS_BANDS: dict[str, tuple[float, float]] = {
    "FULL": (0.00, 0.10),
    "PARTIAL": (0.10, 0.80),
    "CLEAR": (0.80, 1.01),
}

CAUSES = (
    "full_occl_static",
    "full_occl_pedestrian",
    "full_occl_crowd_aggregate",
    "partial_occl_pedestrian",
    "partial_occl_static",
    "track_drift_to_static",
    "reentry",
    "visible_miss",
    "unattributed_fn",
    "unattributed_idsw",
)

EVALUATION_SEQUENCES = ("MOT20-01", "MOT20-03")


def validate_vis_bands(bands: dict[str, tuple[float, float]]) -> None:
    if tuple(bands) != ("FULL", "PARTIAL", "CLEAR"):
        raise ValueError("VIS_BANDS must be ordered FULL, PARTIAL, CLEAR")
    intervals = list(bands.items())
    if abs(float(intervals[0][1][0])) > 1e-12:
        raise ValueError("VIS_BANDS must start at 0.0")
    for (left_name, (_, left_hi)), (right_name, (right_lo, _)) in zip(
        intervals, intervals[1:]
    ):
        if abs(float(left_hi) - float(right_lo)) > 1e-12:
            raise ValueError(
                f"VIS_BANDS are not contiguous between {left_name} and {right_name}"
            )
    if float(intervals[-1][1][1]) <= 1.0:
        raise ValueError("VIS_BANDS must cover visibility=1.0")
    if any(float(lo) >= float(hi) for _, (lo, hi) in intervals):
        raise ValueError("Every VIS_BANDS interval must have lo < hi")


def vis_band(value: float, bands: dict[str, tuple[float, float]]) -> str:
    validate_vis_bands(bands)
    for name, (lo, hi) in bands.items():
        if float(lo) <= value < float(hi):
            return name
    raise ValueError(f"Visibility {value} falls outside VIS_BANDS")


@dataclass
class PipelineConfig:
    classes: dict[str, list[int]] = field(default_factory=lambda: deepcopy(DEFAULT_CLASSES))
    thresholds: dict[str, float | int] = field(
        default_factory=lambda: deepcopy(DEFAULT_THRESHOLDS)
    )
    vis_bands: dict[str, tuple[float, float]] = field(
        default_factory=lambda: deepcopy(DEFAULT_VIS_BANDS)
    )
    ioa_occluder_by_class: dict[str, float] = field(
        default_factory=lambda: deepcopy(DEFAULT_IOA_OCCLUDER_BY_CLASS)
    )
    ioa_sweep_thresholds: tuple[float, ...] = DEFAULT_IOA_SWEEP_THRESHOLDS
    tracker_name: str = "yolox_tracktrack"
    # "legacy_half" exists only for the v2 redistribution gate (Pass A).
    frame_scope: str = "full"
    render_clips: bool = True
    clip_codec: str = "mp4v"
    clip_workers: int = 4
    human_sample_per_cause: int = 25
    random_seed: int = 2026
    difficulty_recall_weight: float = 0.4
    difficulty_idsw_weight: float = 0.4
    difficulty_frag_weight: float = 0.2
    difficulty_idsw_rate_cap: float = 2.0
    difficulty_frag_rate_cap: float = 3.0
    min_track_frames: int = 30
    top_k_identities: int = 25

    @classmethod
    def from_json(cls, path: str | Path | None) -> "PipelineConfig":
        cfg = cls()
        if path is None:
            return cfg
        with Path(path).open(encoding="utf-8") as handle:
            supplied: dict[str, Any] = json.load(handle)
        if "classes" in supplied:
            cfg.classes.update(supplied.pop("classes"))
        if "thresholds" in supplied:
            cfg.thresholds.update(supplied.pop("thresholds"))
        if "vis_bands" in supplied:
            cfg.vis_bands.update(
                {
                    name: tuple(map(float, interval))
                    for name, interval in supplied.pop("vis_bands").items()
                }
            )
        if "ioa_occluder_by_class" in supplied:
            cfg.ioa_occluder_by_class.update(
                {
                    name: float(value)
                    for name, value in supplied.pop("ioa_occluder_by_class").items()
                }
            )
        if "ioa_sweep_thresholds" in supplied:
            cfg.ioa_sweep_thresholds = tuple(
                map(float, supplied.pop("ioa_sweep_thresholds"))
            )
        for key, value in supplied.items():
            if not hasattr(cfg, key):
                raise ValueError(f"Unknown configuration key: {key}")
            setattr(cfg, key, value)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        missing_classes = set(DEFAULT_CLASSES) - set(self.classes)
        missing_thresholds = set(DEFAULT_THRESHOLDS) - set(self.thresholds)
        if missing_classes:
            raise ValueError(f"Missing class groups: {sorted(missing_classes)}")
        if missing_thresholds:
            raise ValueError(f"Missing thresholds: {sorted(missing_thresholds)}")
        validate_vis_bands(self.vis_bands)
        if set(self.ioa_occluder_by_class) != {"large_object", "person"}:
            raise ValueError(
                "ioa_occluder_by_class must contain large_object and person"
            )
        if any(
            value <= 0 or value > 1
            for value in self.ioa_occluder_by_class.values()
        ):
            raise ValueError("Class-dependent IoA thresholds must be in (0, 1]")
        if not self.ioa_sweep_thresholds or any(
            value <= 0 or value > 1 for value in self.ioa_sweep_thresholds
        ):
            raise ValueError("ioa_sweep_thresholds must be non-empty and in (0, 1]")
        if self.frame_scope not in {"full", "legacy_half"}:
            raise ValueError("frame_scope must be 'full' or 'legacy_half'")
        if len(self.clip_codec) != 4:
            raise ValueError("clip_codec must be a four-character OpenCV codec")
        if self.clip_workers < 1:
            raise ValueError("clip_workers must be positive")
        if self.human_sample_per_cause < 0:
            raise ValueError("human_sample_per_cause must be non-negative")
        if self.difficulty_idsw_rate_cap <= 0 or self.difficulty_frag_rate_cap <= 0:
            raise ValueError("difficulty rate caps must be positive")
        if int(self.thresholds["FRAG_MIN_GAP"]) < int(
            self.thresholds["MIN_FN_RUN_FRAMES"]
        ):
            raise ValueError(
                "FRAG_MIN_GAP must be >= MIN_FN_RUN_FRAMES so a filtered FN "
                "is not admitted as a fragmentation"
            )
        if int(self.thresholds["REENTRY_MIN_ABSENT_FRAMES"]) == int(
            self.thresholds["EVENT_WINDOW"]
        ):
            raise ValueError(
                "REENTRY_MIN_ABSENT_FRAMES must not equal EVENT_WINDOW"
            )
        if self.min_track_frames < 1:
            raise ValueError("min_track_frames must be positive")
        weights = (
            self.difficulty_recall_weight
            + self.difficulty_idsw_weight
            + self.difficulty_frag_weight
        )
        if abs(weights - 1.0) > 1e-9:
            raise ValueError("difficulty weights must sum to 1.0")

    def as_dict(self) -> dict[str, Any]:
        return {
            "classes": deepcopy(self.classes),
            "thresholds": deepcopy(self.thresholds),
            "vis_bands": deepcopy(self.vis_bands),
            "ioa_occluder_by_class": deepcopy(self.ioa_occluder_by_class),
            "ioa_sweep_thresholds": list(self.ioa_sweep_thresholds),
            "tracker_name": self.tracker_name,
            "frame_scope": self.frame_scope,
            "render_clips": self.render_clips,
            "clip_codec": self.clip_codec,
            "clip_workers": self.clip_workers,
            "human_sample_per_cause": self.human_sample_per_cause,
            "random_seed": self.random_seed,
            "difficulty_recall_weight": self.difficulty_recall_weight,
            "difficulty_idsw_weight": self.difficulty_idsw_weight,
            "difficulty_frag_weight": self.difficulty_frag_weight,
            "difficulty_idsw_rate_cap": self.difficulty_idsw_rate_cap,
            "difficulty_frag_rate_cap": self.difficulty_frag_rate_cap,
            "min_track_frames": self.min_track_frames,
            "top_k_identities": self.top_k_identities,
        }
