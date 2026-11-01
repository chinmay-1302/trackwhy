"""Regime-B prompt: regime A's LONG_REASONING_PROMPT_V2 without probe/highlight scoring."""

from __future__ import annotations

import hashlib

TRACKER_ONLY_PROMPT = """You are evaluating the output of a Multi-Object Tracker (MOT) from a short video clip.

The intended target is indicated by a colored or highlighted bounding box during at least part of the clip. Other ordinary tracker bounding boxes and track IDs may also be visible.

Your task is to follow the highlighted TRACK through the entire clip and classify that track's behavior.

Use physical appearance, position, motion, expected motion path, surrounding people, ordinary tracker bounding boxes, track IDs, and temporal continuity.

The colored highlight identifies the track under evaluation. Follow that track ID. If the highlighted box does not correspond to a real person and persists, classify ghost_track.

Think through the video internally before answering, but output only the final JSON.

STAGE 1: ESTABLISH THE HIGHLIGHTED TRACK

Determine which tracker box is highlighted. Read its ID. Follow that same highlighted track across the clip.

If the highlighted box does not correspond to a real person, the primary error is ghost_track. Do not treat the ID of a ghost detection as a real person's identity. Do not infer an identity change between a ghost detection and a later real person.

Do not switch to a different track because another person is easier to see.

STAGE 2: BUILD THE TRACK TIMELINE

Follow the highlighted track across the entire clip.

Internally determine for each important interval:
- whether the highlighted track exists,
- whether it sits on a real person,
- whether that person stays the same,
- whether the highlighted box disappears,
- whether an unhighlighted box later appears on the same person,
- where the highlighted ID goes if it leaves the original person.

Do not stop following the track when its box or colored highlight disappears.

If the person under the track becomes occluded, continue reasoning about that physical person's expected motion path.

Check whether the person passes behind a tree, pole, pillar, standee, wall, crowd, another person, image boundary, or another occluder.

Use the full temporal sequence.

STAGE 3: DETERMINE ANNOTATION STATUS

Choose exactly one annotation status.

valid: the highlighted track can be followed well enough to classify tracker behavior.

target_not_identifiable: you cannot reliably determine which track is highlighted.

ambiguous: the highlighted track can be identified, but the tracker event itself cannot be classified reliably from the available visual evidence.

STAGE 4: CLASSIFY THE PRIMARY TRACKER ERROR

For a valid clip, choose exactly one primary error:
no_error
miss
identity_change
drift
ghost_track

Do not use no_error as a default.

Only choose no_error after explicitly ruling out ghost_track, drift, identity change, and miss.

Use this decision procedure.

Step 0: Ghost check.

If the highlighted track does not correspond to a real person (empty space or part of a static object) and persists, primary_error = ghost_track.

Step 1: Drift check.

Ask whether the highlighted track continues on a DIFFERENT physical person.

If yes, primary_error = drift.

Use drift_subtype = to_moving_person if the track moves onto another person who is moving.

Use drift_subtype = to_standing_person if the track moves onto another person who is standing or stationary.

Step 2: Identity-change check.

If there is no drift, ask whether the SAME physical person is tracked under the highlighted ID and later under a different unhighlighted ID, or whether the highlighted ID terminates and the person reappears with a new ID.

If yes, primary_error = identity_change.

Before choosing the subtype, check:
1. Where did the highlighted ID go?
2. Where did the new ID come from?

Use identity_transfer when the person changes to an ID that was already assigned to another physical person.

Use fragmentation when the highlighted ID terminates early and the same person later appears with a newly created unhighlighted ID.

A detection gap followed by the same person receiving a different ID should normally be identity_change with subtype fragmentation rather than only miss.

Step 3: Miss check.

If there is no ghost, no drift, and no identity change, ask whether there is any relevant interval where the highlighted track is absent from the person it was following, and no other box appears on that person.

If yes, primary_error = miss.

For this evaluation, a miss can occur when the person is fully visible, partly visible, heavily occluded, or completely not visible.

A temporary detection gap followed by recovery with the same highlighted ID can be a miss.

Step 4: No-error check.

Only if ghost_track, drift, identity change, and miss have all been ruled out, use primary_error = no_error.

Use no_error when the highlighted track remains correctly associated with the same real person throughout the relevant clip.

STAGE 5: MISS VISIBILITY

Only when primary_error = miss, identify the interval where the highlighted track is absent from the physical person.

Judge visibility using the missed interval itself.

Choose exactly one target visibility.

fully_visible: the physical person remains clearly visually observable and identifiable during the miss.

partly_visible: the physical person itself is substantially obscured, but some visible portion remains.

not_visible: the physical person is completely or almost completely hidden during the relevant missed interval.

uncertain: you cannot reliably determine the person's visibility during the miss.

For non-miss cases, target_visibility must be null.

STAGE 6: MISS VISUAL CONDITIONS

Only when primary_error = miss, select all visual conditions clearly relevant to the missed interval.

Multiple labels may apply.

Allowed values:

static_occlusion: a static scene object substantially occludes the person.

single_person_occlusion: one other person substantially occludes the person.

crowd_occlusion: multiple people or a dense crowd substantially occlude the person.

truncation: part of the person lies outside the image or frame boundary.

small_or_far: the person is visually very small or far away.

dark: poor illumination makes the person difficult to see.

background_blend: the person visually blends with the background because of appearance, contrast, or texture.

crowd_clutter: many nearby people make the person difficult to distinguish even when they do not substantially cover the person.

no_obvious_visual_trigger: the person is missed, but none of the listed conditions clearly explains the visible difficulty.

For non-miss cases, visual_conditions must be [].

STAGE 7: CONFIDENCE

Return one confidence label.

high: clear visual and temporal evidence supports the classification.

medium: one classification is more likely than alternatives, but some uncertainty remains.

low: evidence is weak or difficult to interpret.

OUTPUT

Return only valid JSON using exactly this structure:

{
  "annotation_status": "valid|ambiguous|target_not_identifiable",
  "primary_error": "no_error|miss|identity_change|drift|ghost_track|null",
  "identity_change_subtype": "fragmentation|identity_transfer|null",
  "drift_subtype": "to_moving_person|to_standing_person|null",
  "target_visibility": "fully_visible|partly_visible|not_visible|uncertain|null",
  "visual_conditions": [],
  "confidence": "high|medium|low",
  "explanation": "One concise sentence describing the highlighted track's event and the main visible evidence."
}

Final rules:

1. Follow the highlighted track.
2. If the highlighted track is not a real person and persists, use ghost_track.
3. When the highlighted box disappears, continue following the same physical person's expected motion path.
4. Do not choose no_error until you have checked the complete clip for ghost_track, drift, identity change, and miss.
5. When an ID changes, first check where the old ID goes, then check where the new ID came from.
6. If the highlighted ID moves to another person, use drift.
7. Else if the new ID previously belonged to another person, use identity_transfer.
8. Else if the same person receives a new or restarted ID, use fragmentation.
9. Else if the highlighted track is absent for a relevant interval and no other box appears on that person, use miss.
10. A completely occluded person can still have primary_error = miss.
11. Visibility and visual conditions are independent.
12. If annotation_status is not valid, set primary_error, identity_change_subtype, drift_subtype, and target_visibility to null, and visual_conditions to [].
13. Use identity_change_subtype only for identity_change.
14. Use drift_subtype only for drift.
15. Use target_visibility and visual_conditions only for miss.
16. Do not invent unreadable track IDs.
17. Return only allowed labels.
18. Return no markdown or text outside the JSON.
"""


def regime_b_prompt(*, include_ghost: bool = True) -> str:
    """Return the regime-B prompt, omitting ghost_track when that class is dropped."""
    if include_ghost:
        return TRACKER_ONLY_PROMPT
    text = TRACKER_ONLY_PROMPT
    text = text.replace(
        " Follow that track ID. If the highlighted box does not correspond to a real person and persists, classify ghost_track.",
        " Follow that track ID.",
    )
    text = text.replace(
        "\nIf the highlighted box does not correspond to a real person, the primary error is ghost_track. Do not treat the ID of a ghost detection as a real person's identity. Do not infer an identity change between a ghost detection and a later real person.\n",
        "\n",
    )
    text = text.replace(
        "Only choose no_error after explicitly ruling out ghost_track, drift, identity change, and miss.",
        "Only choose no_error after explicitly ruling out drift, identity change, and miss.",
    )
    text = text.replace(
        "identity_change\ndrift\nghost_track\n",
        "identity_change\ndrift\n",
    )
    text = text.replace(
        "Step 0: Ghost check.\n\nIf the highlighted track does not correspond to a real person (empty space or part of a static object) and persists, primary_error = ghost_track.\n\nStep 1: Drift check.",
        "Step 1: Drift check.",
    )
    text = text.replace(
        "Only if ghost_track, drift, identity change, and miss have all been ruled out, use primary_error = no_error.",
        "Only if drift, identity change, and miss have all been ruled out, use primary_error = no_error.",
    )
    text = text.replace(
        '"primary_error": "no_error|miss|identity_change|drift|ghost_track|null"',
        '"primary_error": "no_error|miss|identity_change|drift|null"',
    )
    text = text.replace(
        "2. If the highlighted track is not a real person and persists, use ghost_track.\n3. When the highlighted box disappears, continue following the same physical person's expected motion path.\n4. Do not choose no_error until you have checked the complete clip for ghost_track, drift, identity change, and miss.\n",
        "2. When the highlighted box disappears, continue following the same physical person's expected motion path.\n3. Do not choose no_error until you have checked the complete clip for drift, identity change, and miss.\n",
    )
    return text


def prompt_sha256(text: str = TRACKER_ONLY_PROMPT) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
