# Annotation schema

All files in this directory are CC BY 4.0. They reference MOT20 by frame
index and track ID. They do not contain MOT20 pixels.

## `events.csv`

1,164 mined events after the v4 attribution and the occluder / reentry /
min-fn3 postpasses.

| Column | Values |
| --- | --- |
| `event_id` | `{sequence}__{tracker}__{error_type}__{hash}` |
| `sequence` | `MOT20-01` or `MOT20-03` |
| `tracker` | tracker name used at mining time |
| `error_type` | `FN`, `IDSW`, `FRAG_NO_SWITCH`, `FP_ID_HIJACK` |
| `cause` | rule label; see `attributor/README.md` |
| `cause_confidence` | rule-internal confidence string |
| `role` | `target` / `partner` when a pair is involved |
| `frame_start`, `frame_end` | inclusive MOT20 frame indices |
| `focal_gt_id`, `partner_gt_id` | MOT20 ground-truth IDs |
| `prev_trk_id`, `new_trk_id` | tracker IDs at an identity switch |
| `static_person_present`, `static_person_max_ioa` | proximity to class-7 identities |
| `evidence.*` | JSON or scalar diagnostics (IoA, visibility, votes). No bounding boxes. |

## `manifest.csv`

284 released clips.

| Column | Values |
| --- | --- |
| `clip_id` | opaque `C001`–`C284` |
| `tracker_only_clip_id` | same code, or `C###-T` for the 12 standee / ghost clips |
| `sequence` | `MOT20-01` or `MOT20-03` |
| `event_id` | row in `events.csv` |
| `error_type`, `cause` | mined labels for that event |
| `highlight_track_id` | tracker ID pinned in the supervision-free render |
| `focal_gt_id` | ground-truth ID of the physical target |
| `first_match_frame` | first MOT20 frame where highlight and GT match |
| `frame_start`, `frame_end`, `n_clip_frames` | clip window on the MOT20 timeline |
| `primary_error` | expert regime-B label |
| `scored_primary` | whether the clip enters the VLM primary-error denominator |
| `annotation_status` | `valid` / `ambiguous` / `target_not_identifiable` |

## `clip_causes.csv`

`clip_id` → mined `event_id`, `error_type`, and `cause`. Used to compute
rule-versus-observer κ.

## `funnel.csv`

The 282 → 22 visible-miss cascade from the paper. Counts are frozen; they
are not re-derived from MOT20 by the table path.

## `gt_v3.json`

Expert labels for the ground-truth-informed (regime A) evaluation, 283
clips. C205 is absent because it was unjudgeable. Label Studio provenance,
free text, and source filenames have been stripped. Each record keeps
`clip_id`, `primary_error`, subtypes, `target_visibility`,
`visual_conditions`, `probe_issue`, `annotation_status`, and the
`evaluation` eligibility flags.

`evaluation.include_in_core_error_eval` plus `primary_error` in
`{miss, identity_change, no_error}` yields the paper's *n* = 257.

## `gt_tracker_only.json`

Expert labels for the supervision-free (regime B) evaluation, 284 clips.
`scored_primary` is true for 255 clips in `{miss, identity_change, no_error}`.

## `human/observer_a.csv` and `human/observer_b.csv`

Forward, policy-naive annotations from two independent observers on all 284
clips. Cohen's κ in the paper is computed on the 282 clips where both
recorded a complete `human_cause`.

| Column | Values |
| --- | --- |
| `clip_id` | `C001`–`C284` |
| `error_kind` | `miss`, `identity_change`, `drift`, `no_error_visible`, `could_not_identify_target`, `cannot_determine` |
| `visibility` | miss only: `not_at_all`, `partly`, `clearly`, `cannot_tell` |
| `cause_miss` | `hidden_by_object`, `hidden_by_one_person`, `hidden_by_standing_person`, `hidden_by_crowd`, `partly_blocked_by_person`, `partly_blocked_by_static`, `partly_outside_the_frame`, `clearly_visible_but_missed`, `cannot_determine` |
| `cause_idsw` | `swapped_during_crossing`, `left_and_returned`, `changed_for_no_visible_reason`, … |
| `cause_drift` | `drifted_to_standing_person`, `drifted_to_moving_person` |
| `hard_to_spot` | optional multi-label on clearly-visible misses |
| `confidence` | `high` / `medium` / `low` |
| `human_cause` | the branch-specific cause, or `error_kind` when that kind is itself the cause |
| `incomplete`, `incomplete_reason` | validator flags |
| `contradiction`, `contradiction_reason` | visibility vs cause mismatches |
| `void_excluded` | reserved; unused in the released tables |

Observer names and free-text comments are not released.

## `human/verifier_forward.csv` and `human/verifier_informed.csv`

The paper's "47 of 170" figure. `verifier_forward.csv` is the v2.1 pass
(170 clips; 76 labelled `no_error`). `verifier_informed.csv` is the v3
pass on the 283-clip set. Exactly 47 of those 170 rows flip
`no_error` → `miss` when convention knowledge is supplied.

## `human/pilot.csv`

Thirty-clip binary detection pilot. `observer` is `naive` or `author`.
The paper already states that one pilot observer is the corpus author.
`file_id` is the blinded `P01`–`P30` code; `clip_id` is the released
`C###`. `dont_know` is an abstention.
