from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class GTBox:
    frame: int
    gt_id: int
    bbox: BBox
    cls: int
    visibility: float
    flag: int


@dataclass(frozen=True)
class TrkBox:
    frame: int
    trk_id: int
    bbox: BBox


@dataclass
class ErrorEvent:
    event_id: str
    sequence: str
    tracker: str
    error_type: str
    frame_start: int
    frame_end: int
    focal_gt_id: int
    partner_gt_id: Optional[int] = None
    role: Optional[str] = None
    cause: Optional[str] = None
    cause_confidence: Optional[str] = None
    evidence: dict[str, Any] = field(default_factory=dict)
    prev_trk_id: Optional[int] = None
    new_trk_id: Optional[int] = None
    static_person_present: Optional[bool] = None
    static_person_max_ioa: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SequenceData:
    name: str
    num_frames: int
    width: int
    height: int
    fps: float
    dataset_root: str
    available_frames: tuple[int, ...]
    selected_frames: tuple[int, ...]
    dropped_non_mining_frames: tuple[int, ...]
    image_paths: dict[int, str]
    gt_by_frame: dict[int, list[GTBox]]
    tracker_by_frame: dict[int, list[TrkBox]]
    class_histogram_full: dict[int, int]
    class_histogram_selected: dict[int, int]
    flag_class_histogram_selected: dict[tuple[int, int], int]

    @property
    def selected_frame_set(self) -> set[int]:
        return set(self.selected_frames)

    def gt_for_id(self, gt_id: int, frames: set[int] | None = None) -> list[GTBox]:
        wanted = self.selected_frame_set if frames is None else frames
        return [
            box
            for frame in sorted(wanted)
            for box in self.gt_by_frame.get(frame, [])
            if box.gt_id == gt_id
        ]
