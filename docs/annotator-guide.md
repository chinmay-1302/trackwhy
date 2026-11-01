# Annotator guide

This is the text and schema observers received. They saw a tracker-only
clip. They did not see ground-truth boxes, metric names, or the attributor
output.

## Briefing

A highlighted box with a number is the track the program was following. At
some point something goes wrong with it — it disappears, its number changes,
or it slides onto somebody else. When it disappears, you will have to find
the person yourself; that is part of the task.

Watch the whole clip once before answering. Scrubbing is allowed.

## Questions

1. What kind of mistake happened?

   - `miss` — the box disappears (the same number returns, or never returns)
   - `identity_change` — the box returns with a different number
   - `drift` — the box slides off onto somebody else
   - `no_error_visible` — nothing looks wrong
   - `could_not_identify_target` — you could not tell who the target was
   - `cannot_determine` — you understood the target but could not classify the event

2. If `miss`: while the box was gone, could you see the target?

   - `not_at_all` / `partly` / `clearly` / `cannot_tell`

3. If `miss`: what was going on?

   - `hidden_by_object`
   - `hidden_by_one_person`
   - `hidden_by_standing_person`
   - `hidden_by_crowd`
   - `partly_blocked_by_person`
   - `partly_blocked_by_static`
   - `partly_outside_the_frame`
   - `clearly_visible_but_missed`
   - `cannot_determine`

   One person vs crowd: delete the single biggest blocker in your head. Still
   hidden? That is crowd. Standing vs moving: compare the first and last
   frame. Check `partly_outside_the_frame` before `clearly_visible_but_missed`.

4. If `identity_change` or `drift`: use the matching cause list in
   `docs/labelstudio_config.xml`.

5. Confidence: `high` / `medium` / `low`.

The Label Studio project XML is `docs/labelstudio_config.xml`. Clip IDs were
opaque `C001`–`C284` codes. Event IDs were not shown.

## Pilot briefing

The 30-clip detection pilot used a shorter briefing and no taxonomy:

> The highlighted box is a track produced by an automatic tracker. Tell us
> whether it correctly follows one single person for the whole clip.

Primary-error labels plus `dont_know` were accepted and reduced to a binary
detection judgement.
