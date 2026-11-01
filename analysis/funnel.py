"""Load the shipped annotation-blind-spot funnel."""

from __future__ import annotations

import csv
from pathlib import Path


def load_funnel(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def headline(rows: list[dict[str, str]]) -> dict[str, int]:
    by_stage = {row["stage"]: int(row["count"]) for row in rows}
    return {
        "raw": by_stage["apparent_visible_miss_raw"],
        "final": by_stage["visible_miss_final"],
    }
