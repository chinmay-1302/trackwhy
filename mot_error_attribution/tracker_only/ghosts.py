"""P4: extract and validate ghost-track IDs from annotator captions."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping, Sequence

from .io import GHOST_CLIP_IDS
from .verify import named_track_ids

GHOST_EXTRACTION_PROMPT = """You extract tracker IDs from a human caption about a MOT clip.

The caption may name a ghost detection (a box that is not a real person) and a real person.

Return JSON only, using this schema:
{
  "clip_id": "<clip id>",
  "ghost_track_id": <integer or null>,
  "real_target_track_id": <integer or null>,
  "ghost_confidence": "high|medium|low",
  "real_target_confidence": "high|medium|low",
  "ghost_evidence_sentence": "<exact sentence the ghost ID came from, or empty>",
  "real_target_evidence_sentence": "<exact sentence the target ID came from, or empty>"
}

Rules:
- Use null rather than a guess. Do not infer an ID from context.
- An ID is allowed only if the caption explicitly writes that number.
- The evidence sentence must be copied from the caption.
- If the caption does not name a ghost ID, ghost_track_id must be null.
"""

GHOST_VALIDATION_FIELDS = (
    "clip_id",
    "ghost_track_id",
    "real_target_track_id",
    "p1_highlight_track_id",
    "ghost_confidence",
    "real_target_confidence",
    "c1_no_gt_match",
    "c2_persistent",
    "c3_real_target_matches_p1",
    "c3_real_target_matches_gt",
    "ghost_not_renderable",
    "needs_review",
    "regex_ghost_track_id",
    "regex_real_target_track_id",
    "source_disagreement",
    "ghost_evidence_sentence",
    "real_target_evidence_sentence",
    "free_text",
)

ID_NEAR_GHOST = re.compile(
    r"(?is)(?:ghost detection|a ghost|not a real person|not a person).{0,80}?\bID\s*:?\s*(\d+)\b"
    r"|\bID\s*:?\s*(\d+)\b.{0,100}?(?:ghost detection|not a real person|not a person|just some (?:random )?ghost)"
)
ID_NEAR_REAL = re.compile(
    r"(?is)(?:real target person|comes out from behind|actual target).{0,80}?\bID\s*:?\s*(\d+)\b"
    r"|\bID\s*:?\s*(\d+)\b.{0,40}?(?:is detected and tracked)"
)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def sentence_containing(text: str, track_id: int | None) -> str:
    if track_id is None:
        return ""
    needle = str(int(track_id))
    for raw in re.split(r"(?<=[.!?])\s+", text or ""):
        if re.search(rf"\bID\s*:?\s*{re.escape(needle)}\b", raw, re.IGNORECASE):
            return raw.strip()
    return ""


def regex_extract(clip_id: str, free_text: str) -> dict[str, Any]:
    text = free_text or ""
    ghost = None
    real = None
    ghost_match = ID_NEAR_GHOST.search(text)
    if ghost_match:
        ghost = int(next(group for group in ghost_match.groups() if group))
    real_match = ID_NEAR_REAL.search(text)
    if real_match:
        real = int(next(group for group in real_match.groups() if group))
        if real == ghost:
            real = None
    named = named_track_ids(text)
    if ghost is None and len(named) == 1 and re.search(r"ghost", text, re.IGNORECASE):
        ghost = named[0]
    if real is None and ghost is not None:
        others = [item for item in named if item != ghost]
        if len(others) == 1:
            real = others[0]
    return {
        "clip_id": clip_id,
        "ghost_track_id": ghost,
        "real_target_track_id": real,
        "ghost_confidence": "high" if ghost is not None else "low",
        "real_target_confidence": "high" if real is not None else "low",
        "ghost_evidence_sentence": sentence_containing(text, ghost),
        "real_target_evidence_sentence": sentence_containing(text, real),
        "source": "regex",
    }


def parse_llm_extraction(clip_id: str, payload: Mapping[str, Any], free_text: str) -> dict[str, Any]:
    ghost = _optional_int(payload.get("ghost_track_id"))
    real = _optional_int(payload.get("real_target_track_id"))
    named = set(named_track_ids(free_text))
    if ghost is not None and ghost not in named:
        ghost = None
    if real is not None and real not in named:
        real = None
    return {
        "clip_id": clip_id,
        "ghost_track_id": ghost,
        "real_target_track_id": real,
        "ghost_confidence": str(payload.get("ghost_confidence") or "low"),
        "real_target_confidence": str(payload.get("real_target_confidence") or "low"),
        "ghost_evidence_sentence": str(
            payload.get("ghost_evidence_sentence") or sentence_containing(free_text, ghost)
        ),
        "real_target_evidence_sentence": str(
            payload.get("real_target_evidence_sentence")
            or sentence_containing(free_text, real)
        ),
        "source": "llm",
    }


def extract_with_gemini(clip_id: str, free_text: str, model: str) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=model,
        contents=(
            GHOST_EXTRACTION_PROMPT
            + f"\n\nclip_id: {clip_id}\n\ncaption:\n{free_text}\n"
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    payload = json.loads(response.text or "{}")
    if not isinstance(payload, dict):
        raise ValueError(f"Gemini returned a non-object for {clip_id}")
    return parse_llm_extraction(clip_id, payload, free_text)


def extract_ghosts(
    captions: Mapping[str, str],
    *,
    skip_llm: bool = False,
    model: str = "gemini-3.7-flash",
    cached: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cache = dict(cached or {})
    use_llm = (not skip_llm) and bool(os.environ.get("GEMINI_API_KEY"))
    for clip_id in GHOST_CLIP_IDS:
        free_text = captions.get(clip_id, "")
        regex_row = regex_extract(clip_id, free_text)
        llm_row = None
        cached_row = cache.get(clip_id)
        if (
            cached_row
            and not skip_llm
            and str(cached_row.get("source") or "") in {"llm", "cache"}
        ):
            llm_row = dict(cached_row)
            llm_row.setdefault("source", "cache")
        elif use_llm:
            llm_row = extract_with_gemini(clip_id, free_text, model)
        chosen = llm_row or regex_row
        chosen["regex_ghost_track_id"] = regex_row["ghost_track_id"]
        chosen["regex_real_target_track_id"] = regex_row["real_target_track_id"]
        chosen["free_text"] = free_text
        if llm_row is not None:
            chosen["source_disagreement"] = (
                llm_row.get("ghost_track_id") != regex_row["ghost_track_id"]
                or llm_row.get("real_target_track_id") != regex_row["real_target_track_id"]
            )
        else:
            chosen["source_disagreement"] = False
        rows.append(chosen)
    return rows


def track_lifetime(
    track_id: int | None,
    clip_frames: Sequence[int],
    tracker_by_frame: Mapping[int, Sequence[int]],
) -> list[int]:
    if track_id is None:
        return []
    wanted = int(track_id)
    return [
        int(frame)
        for frame in clip_frames
        if wanted in set(map(int, tracker_by_frame.get(int(frame), ())))
    ]


def ghost_matches_any_gt(
    track_id: int | None,
    clip_frames: Sequence[int],
    matched_gt_by_track_frame: Mapping[tuple[int, int], int | None],
) -> bool:
    if track_id is None:
        return False
    wanted = int(track_id)
    for frame in clip_frames:
        matched = matched_gt_by_track_frame.get((int(frame), wanted))
        if matched not in (None, ""):
            return True
    return False


def real_target_matches_focal(
    track_id: int | None,
    focal_gt_id: int,
    matching_by_frame: Sequence[Mapping[str, Any]],
) -> bool:
    if track_id is None:
        return False
    wanted = int(track_id)
    return any(
        row.get("matched_prediction_track_id") not in (None, "")
        and int(row["matched_prediction_track_id"]) == wanted
        for row in matching_by_frame
    )


def validate_ghost(
    extraction: Mapping[str, Any],
    *,
    derivation: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    ghost_id = _optional_int(extraction.get("ghost_track_id"))
    real_id = _optional_int(extraction.get("real_target_track_id"))
    p1_id = _optional_int(derivation.get("highlight_track_id"))
    lifetime = track_lifetime(
        ghost_id, bundle["clip_frames"], bundle["tracker_by_frame"]
    )
    c1 = ghost_id is not None and not ghost_matches_any_gt(
        ghost_id, bundle["clip_frames"], bundle["matched_gt_by_track_frame"]
    )
    c2 = len(lifetime) >= 5
    c3_p1 = real_id is not None and p1_id is not None and int(real_id) == int(p1_id)
    c3_gt = real_target_matches_focal(
        real_id, int(bundle["focal_gt_id"]), bundle["matching_by_frame"]
    )
    low = (
        str(extraction.get("ghost_confidence") or "") == "low"
        or str(extraction.get("real_target_confidence") or "") == "low"
    )
    needs_review = (not c1) or (not c3_p1) or (not c3_gt) or low or ghost_id is None
    return {
        "clip_id": extraction["clip_id"],
        "ghost_track_id": "" if ghost_id is None else ghost_id,
        "real_target_track_id": "" if real_id is None else real_id,
        "p1_highlight_track_id": "" if p1_id is None else p1_id,
        "ghost_confidence": extraction.get("ghost_confidence", ""),
        "real_target_confidence": extraction.get("real_target_confidence", ""),
        "c1_no_gt_match": c1,
        "c2_persistent": c2,
        "c3_real_target_matches_p1": c3_p1,
        "c3_real_target_matches_gt": c3_gt,
        "ghost_not_renderable": not c2,
        "needs_review": needs_review,
        "regex_ghost_track_id": extraction.get("regex_ghost_track_id", ""),
        "regex_real_target_track_id": extraction.get("regex_real_target_track_id", ""),
        "source_disagreement": bool(extraction.get("source_disagreement")),
        "ghost_evidence_sentence": extraction.get("ghost_evidence_sentence", ""),
        "real_target_evidence_sentence": extraction.get(
            "real_target_evidence_sentence", ""
        ),
        "free_text": extraction.get("free_text", ""),
        "c1_or_c2_fail": (not c1) or (not c2),
    }


GHOST_DISAGREEMENT_FIELDS = (
    "clip_id",
    "ghost_track_id",
    "ghost_evidence_sentence",
    "matched_gt_ids",
    "n_frames_matched_to_gt",
    "example_source_frame",
    "example_bbox_xyxy",
    "example_bbox_xywh",
)


def ghost_gt_disagreement(
    extraction: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Record TrackEval GT assignments on the annotator's ghost box.

    C1 assumed a ghost has no GT match. MOT20 often assigns a person GT to
    that box; that disagreement is the finding, not an extraction error.
    """
    ghost_id = _optional_int(extraction.get("ghost_track_id"))
    if ghost_id is None:
        return None
    gt_ids: list[int] = []
    n_matched = 0
    example_frame = ""
    example_xyxy = ""
    example_xywh = ""
    clip_frames = set(map(int, bundle.get("clip_frames") or ()))
    for row in bundle.get("annotation", {}).get("prediction_annotations") or ():
        if int(row.get("track_id") or -1) != ghost_id:
            continue
        if clip_frames and int(row["source_frame"]) not in clip_frames:
            continue
        matched = row.get("matched_gt_id")
        if matched in (None, ""):
            continue
        n_matched += 1
        gt = int(matched)
        if gt not in gt_ids:
            gt_ids.append(gt)
        if example_frame == "":
            example_frame = int(row["source_frame"])
            xyxy = row.get("bbox_xyxy") or []
            xywh = row.get("bbox_xywh") or []
            example_xyxy = ";".join(str(value) for value in xyxy)
            example_xywh = ";".join(str(value) for value in xywh)
    if n_matched == 0:
        return None
    return {
        "clip_id": extraction["clip_id"],
        "ghost_track_id": ghost_id,
        "ghost_evidence_sentence": extraction.get("ghost_evidence_sentence", ""),
        "matched_gt_ids": ";".join(str(item) for item in gt_ids),
        "n_frames_matched_to_gt": n_matched,
        "example_source_frame": example_frame,
        "example_bbox_xyxy": example_xyxy,
        "example_bbox_xywh": example_xywh,
    }


def ghost_class_decision(validations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n_fail = sum(1 for row in validations if row.get("c1_or_c2_fail"))
    drop = n_fail > 3
    return {
        "n_ghost_clips": len(validations),
        "n_c1_or_c2_fail": n_fail,
        "drop_ghost_class": drop,
        "keep_t_halves": True,
    }
