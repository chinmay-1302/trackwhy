"""Hash probe-clip names and build leak-free Label Studio video tasks."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable


FORBIDDEN_SOURCE_DIRS = ("audit", "withheld", "clips_meta")
MAP_COLUMNS = (
    "clip_id",
    "hashed_filename",
    "event_id",
    "source_filename",
    "cause",
    "sequence",
    "error_type",
)
TASK_DATA_KEYS = ("clip_id", "video")
DEFAULT_SEED = "2026"
DEFAULT_LOCAL_FILES_PREFIX = "label-studio/data/probe_clips"


def hashed_filename(event_id: str, seed: str = DEFAULT_SEED) -> str:
    digest = hashlib.sha256(f"{seed}:{event_id}".encode("utf-8")).hexdigest()[:16]
    return f"{digest}.mp4"


def event_id_from_clip_path(path: Path) -> str:
    if path.suffix.lower() != ".mp4":
        raise ValueError(f"Not an mp4 clip: {path}")
    return path.stem


def assert_clips_dir_is_safe(clips_dir: Path) -> None:
    name = clips_dir.name
    if name in FORBIDDEN_SOURCE_DIRS:
        raise ValueError(
            f"Refusing to read {clips_dir}; only out/probe_clips/clips is allowed."
        )
    for part in clips_dir.resolve().parts:
        if part in FORBIDDEN_SOURCE_DIRS:
            raise ValueError(
                f"Refusing path that includes {part!r}: {clips_dir}"
            )


def list_clip_paths(clips_dir: Path) -> list[Path]:
    assert_clips_dir_is_safe(clips_dir)
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"Missing clips directory: {clips_dir}")
    return sorted(path for path in clips_dir.glob("*.mp4") if path.is_file())


def load_manifest(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    by_id: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            event_id = (row.get("event_id") or "").strip()
            if event_id:
                by_id[event_id] = row
    return by_id


def assign_clip_rows(
    clip_paths: Iterable[Path],
    *,
    seed: str = DEFAULT_SEED,
    manifest: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    manifest = manifest or {}
    seen_hashes: dict[str, str] = {}
    pending: list[dict[str, str]] = []
    for clip_path in clip_paths:
        event_id = event_id_from_clip_path(clip_path)
        hashed = hashed_filename(event_id, seed=seed)
        previous = seen_hashes.get(hashed)
        if previous is not None and previous != event_id:
            raise ValueError(
                f"Hash collision for {hashed}: {previous!r} vs {event_id!r}"
            )
        seen_hashes[hashed] = event_id
        meta = manifest.get(event_id, {})
        pending.append(
            {
                "hashed_filename": hashed,
                "event_id": event_id,
                "source_filename": clip_path.name,
                "cause": meta.get("cause", ""),
                "sequence": meta.get("sequence", ""),
                "error_type": meta.get("error_type", ""),
                "_source_path": str(clip_path),
            }
        )
    pending.sort(key=lambda row: row["hashed_filename"])
    width = max(3, len(str(len(pending))))
    rows: list[dict[str, str]] = []
    for index, row in enumerate(pending, start=1):
        assigned = dict(row)
        assigned["clip_id"] = f"C{index:0{width}d}"
        rows.append(assigned)
    return rows


def write_clip_map(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MAP_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in MAP_COLUMNS})


def read_clip_map(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing clip map: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def place_video(source: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.samefile(source):
            return "exists"
        if dest.stat().st_size == source.stat().st_size:
            return "exists"
        dest.unlink()
    try:
        os.link(source, dest)
        return "link"
    except OSError:
        shutil.copy2(source, dest)
        return "copy"


def place_hashed_videos(
    rows: list[dict[str, str]],
    destinations: Iterable[Path],
) -> dict[str, int]:
    counts = {"link": 0, "copy": 0, "exists": 0}
    dest_roots = [Path(item) for item in destinations]
    for row in rows:
        source = Path(row["_source_path"])
        for dest_root in dest_roots:
            action = place_video(source, dest_root / row["hashed_filename"])
            counts[action] += 1
    return counts


def video_url(hashed_filename: str, local_files_prefix: str = DEFAULT_LOCAL_FILES_PREFIX) -> str:
    prefix = local_files_prefix.strip("/")
    return f"/data/local-files/?d={prefix}/{hashed_filename}"


def build_tasks(
    rows: list[dict[str, str]],
    *,
    local_files_prefix: str = DEFAULT_LOCAL_FILES_PREFIX,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["clip_id"]):
        tasks.append(
            {
                "data": {
                    "clip_id": row["clip_id"],
                    "video": video_url(row["hashed_filename"], local_files_prefix),
                }
            }
        )
    return tasks


def write_tasks_json(tasks: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")


def write_import_notes(path: Path) -> None:
    path.write_text(
        """probe-clips Label Studio import

Hashed MP4s are served from the existing mydata mount. Do not recreate
the container and do not set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT.

1. Create a new project (e.g. probe-clips).
   Paste labelstudio-template.xml as the labeling interface.

2. Settings -> Cloud Storage -> Add Source Storage -> Local files
   Absolute local path: /label-studio/data/probe_clips
   Do not click Sync.

   Label Studio runs as uid 1001. If you see Permission denied, grant
   that user read access to the hashed video directory.

3. Import tasks.json. That creates the video tasks. clip_map.csv is
   private — do not import it and do not give it to annotators.

4. Open one task and confirm the Video tag plays and seeks at 10 fps.
""",
        encoding="utf-8",
    )


def hash_probe_clips(
    clips_dir: Path,
    output_root: Path,
    *,
    seed: str = DEFAULT_SEED,
    manifest_path: Path | None = None,
    copy_to: Path | None = None,
) -> dict[str, Any]:
    rows = assign_clip_rows(
        list_clip_paths(clips_dir),
        seed=seed,
        manifest=load_manifest(manifest_path),
    )
    map_path = output_root / "clip_map.csv"
    write_clip_map(rows, map_path)
    destinations = [output_root / "videos"]
    if copy_to is not None:
        destinations.append(copy_to)
    placed = place_hashed_videos(rows, destinations)
    return {
        "n_clips": len(rows),
        "seed": seed,
        "clip_map": str(map_path),
        "videos": str(output_root / "videos"),
        "copy_to": str(copy_to) if copy_to else None,
        "placed": placed,
    }


def export_probe_clip_tasks(
    map_path: Path,
    output_root: Path,
    *,
    local_files_prefix: str = DEFAULT_LOCAL_FILES_PREFIX,
    template_path: Path | None = None,
) -> dict[str, Any]:
    rows = read_clip_map(map_path)
    tasks = build_tasks(rows, local_files_prefix=local_files_prefix)
    tasks_path = output_root / "tasks.json"
    write_tasks_json(tasks, tasks_path)
    write_import_notes(output_root / "IMPORT.txt")
    if template_path is not None and template_path.is_file():
        dest = output_root / "labelstudio-template.xml"
        if template_path.resolve() != dest.resolve():
            shutil.copy2(template_path, dest)
    return {
        "n_tasks": len(tasks),
        "tasks": str(tasks_path),
        "local_files_prefix": local_files_prefix,
    }
