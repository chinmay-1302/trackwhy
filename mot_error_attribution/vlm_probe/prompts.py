"""Stage 1 and Stage 2 prompts. Independent: never chain Stage 1 into Stage 2."""

from __future__ import annotations

FRAME_COUNT_PROMPT = """You are watching a short surveillance video.

How many distinct moments (separate points in time) can you see in this video?
Answer in JSON only, no other text:
{"n_moments": <integer>}
"""

STAGE1_PROMPT = """You are watching a short surveillance video of a busy pedestrian area.

An automatic tracking system has drawn boxes around people it is following.
Most boxes are thin. ONE box is drawn thick and highlighted — that is the
track we care about.

Watch the whole clip, then answer:

1. Describe the person inside the highlighted box: their clothing, what they
   are carrying, and which direction they are walking.
2. Describe what happens to the highlighted box over the course of the clip.
   Does it stay on the same person the whole time? Does it disappear? Does it
   reappear? Does it move onto someone else?
3. If the highlighted box disappears at some point, say roughly when, and say
   whether you can still see the person it was following after it disappears.

Answer in JSON only, no other text:
{
  "person_description": "<one sentence>",
  "box_behaviour": "stays_on_person" | "disappears" | "disappears_and_returns" | "moves_to_different_person" | "no_highlighted_box_found",
  "disappears_at": "<early|middle|late|not_applicable>",
  "person_still_visible_after": "yes" | "no" | "partly" | "not_applicable",
  "confidence": "high" | "medium" | "low"
}
"""

STAGE2_PROMPT = """You are watching a short surveillance video of a busy pedestrian area.

An automatic tracking system has drawn boxes around people it is following.
Most boxes are thin. ONE box is drawn thick and highlighted — that is the
track we care about. At some point the tracking system makes a mistake with
that track.

Your job is to work out WHY the mistake happened, based only on what you can
see in the video.

First decide what kind of mistake it was:
- MISS: the highlighted box disappears (it may or may not come back)
- IDENTITY CHANGE: the box comes back with a different number
- DRIFT: the box slides off the person and settles on someone else
- NO ERROR: nothing appears to go wrong
- COULD NOT FIND: you cannot tell which track is highlighted

Then choose exactly one cause from this list:

  hidden_by_object            - fully hidden behind a large object (vehicle,
                                pillar, sign, barrier, tree)
  hidden_by_one_person        - fully hidden behind one other moving person
  hidden_by_standing_person   - fully hidden behind one stationary person
  hidden_by_crowd             - fully hidden by several people together, no
                                single one of whom would have hidden them
  partly_blocked_by_person    - partly blocked by a moving person
  partly_blocked_by_static    - partly blocked by a standing person or object
  partly_outside_the_frame    - partly outside the edge of the video
  clearly_visible_but_missed  - plainly visible, nothing in the way at all
  left_and_returned           - walked out of the picture (or behind something
                                large) and came back
  swapped_during_crossing     - another person crossed in front at the moment
                                the number changed
  changed_for_no_visible_reason - the number changed with nothing visible
                                happening
  drifted_to_standing_person  - the box ended up on a stationary person
  drifted_to_moving_person    - the box ended up on a moving person
  cannot_determine            - you genuinely cannot tell

Important:
- "clearly_visible_but_missed" and "changed_for_no_visible_reason" are correct
  answers when nothing is in the way. Do not invent an obstacle that is not
  there.
- Only say someone or something was blocking the person if you can actually
  see it in the video.
- "cannot_determine" is a valid answer. Guessing is worse.

Answer in JSON only, no other text:
{
  "error_kind": "miss" | "identity_change" | "drift" | "no_error" | "could_not_find",
  "cause": "<one value from the list above>",
  "evidence": "<one sentence describing what you saw that supports this>",
  "occluder_visible_in_frame": true | false,
  "confidence": "high" | "medium" | "low"
}
"""

V2_PERCEPTION_PROMPT = """Look at the FIRST frame of this video only.

Several thin boxes are drawn around people. Exactly one box is drawn thick
and highlighted in yellow.

1. Read the number printed above the thick highlighted box.
2. Describe where in the frame it is (e.g. "upper left", "centre right").
3. What colour clothing is the person inside it wearing?

Answer in JSON only:
{"track_number": "<digits or unreadable>",
 "position": "<short phrase>",
 "clothing": "<short phrase>",
 "found_highlighted_box": true | false}
"""

V3_BINARY_PROMPT = """This is a single frame from a surveillance video. Some people have thin boxes
drawn around them. Is there ALSO a box drawn thick and highlighted in yellow?

Answer JSON only: {"highlighted_box_present": true | false}
"""
