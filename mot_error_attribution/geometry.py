from __future__ import annotations

import numpy as np

from .models import BBox


def xywh_to_xyxy(box: list[float] | tuple[float, ...]) -> BBox:
    x, y, width, height = map(float, box)
    return (x, y, x + width, y + height)


def iou(a: BBox, b: BBox) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > np.finfo(float).eps else 0.0


def ioa(a: BBox, b: BBox) -> float:
    """Intersection over area of ``a`` (the target box)."""
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    return intersection / area_a if area_a > np.finfo(float).eps else 0.0


def box_area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_box(a: BBox, b: BBox) -> BBox | None:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def rectangle_union_area(rectangles: list[BBox]) -> float:
    """Exact union area for axis-aligned rectangles using an x-axis sweep."""
    valid = [rectangle for rectangle in rectangles if box_area(rectangle) > 0]
    if not valid:
        return 0.0
    x_coordinates = sorted({value for box in valid for value in (box[0], box[2])})
    total = 0.0
    for left, right in zip(x_coordinates, x_coordinates[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (box[1], box[3])
            for box in valid
            if box[0] < right and box[2] > left
        )
        if not intervals:
            continue
        covered_y = 0.0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                covered_y += end - start
                start, end = next_start, next_end
        covered_y += end - start
        total += (right - left) * covered_y
    return total


def near_frame_edge(box: BBox, width: int, height: int, margin: float) -> bool:
    return (
        box[0] <= margin
        or box[1] <= margin
        or (width - box[2]) <= margin
        or (height - box[3]) <= margin
    )


def in_frame_fraction(box: BBox, width: int, height: int) -> float:
    x1, y1, x2, y2 = (float(value) for value in box)
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0.0:
        return 0.0
    ix1 = max(0.0, x1)
    iy1 = max(0.0, y1)
    ix2 = min(float(width), x2)
    iy2 = min(float(height), y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return ((ix2 - ix1) * (iy2 - iy1)) / area


def border_edges(
    box: BBox, width: int, height: int, margin: float
) -> tuple[str, ...]:
    edges: list[str] = []
    if box[0] <= margin:
        edges.append("left")
    if box[1] <= margin:
        edges.append("top")
    if box[2] >= width - margin:
        edges.append("right")
    if box[3] >= height - margin:
        edges.append("bottom")
    return tuple(edges)
