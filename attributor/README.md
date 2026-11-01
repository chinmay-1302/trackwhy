# Attributor

The attributor assigns a scene condition to each failure counted by TrackEval
on MOT20-01 and MOT20-03. It is anchored to `CLR_FN` and `IDSW` by an exact
closure check in `mot_error_attribution/matching.py`:

```python
assert local_fn == official_fn and local_idsw == official_idsw
```

On the released TrackTrack run this is 34,724 false-negative identity-frames
and 253 identity switches.

## What the rules claim

The rules identify the scene conditions that **co-occur** with a counted
failure. They do not name the tracker's internal failure mechanism, which is
not observable from its output. They read ground-truth geometry, the
`visibility` field, and class labels. They do not read detector confidences,
appearance features, or tracker-internal state.

The one exception is `track_drift_to_static`, which looks at where the
tracker box sits during an `FP_ID_HIJACK` event.

## Released causes

| Cause | What it tests |
| --- | --- |
| `full_occl_static` | FULL visibility band; a static occluder wins the IoA vote |
| `full_occl_pedestrian` | FULL band; a pedestrian wins |
| `full_occl_crowd_aggregate` | FULL band explained by crowd coverage, not one person |
| `partial_occl_pedestrian` | PARTIAL band; highest-IoA interactor is a pedestrian |
| `partial_occl_static` | PARTIAL band; highest-IoA interactor is static |
| `track_drift_to_static` | Tracker box hijacked onto a static identity |
| `reentry` | Identity switch after a long absence consistent with leaving and returning |
| `gt_annotation_gap` | Reentry-shaped gap better explained as a hole in MOT20 annotation |
| `visible_miss` | CLEAR visibility, no interacting occluder, counted FN |
| `visible_miss_unconfirmed` | Postpass: miss sits on unannotated structure |
| `visible_miss_truncated` | Postpass: miss is at the frame boundary |
| `excluded_micro_gap` | One- or two-frame FN/FRAG runs, held out of the scored pool |
| `unattributed_fn` / `unattributed_idsw` | Residual events; no rule fired |

`rules.py` exposes one function per cause. Assignment is still performed by
the single priority function `attribute_event`.

## Entry points

```bash
# needs a local MOT20 copy and tracker output
python attributor/mine_events.py \
  --tracker-output /path/to/tracktrack \
  --dataset-root /path/to/MOT20 \
  --out /tmp/events.csv \
  --check-release
```

`--check-release` compares `event_id` and `cause` against the frozen 1,164-row
`annotations/events.csv`. Exact cause reproduction also needs the occluder-mask
and reentry postpasses (`--occluder-export`). The TrackEval closure assertion
runs either way.
