# Reproducing TrackWhy

Three paths, in increasing cost.

## 1. Tables from shipped annotations

Needs: this repository and the Python packages in `requirements.txt`.
Does not need MOT20, a tracker, or a GPU.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python analysis/tables.py --out /tmp/tables
```

This regenerates the paper's headline numbers:

- Observer A / B `no_error_visible` rates (41% / 44%)
- Cohen's κ: observers 0.435, rules vs A 0.177, rules vs B 0.132
- 47 of 170 verifier flips `no_error` → `miss`
- Funnel 282 → 22 remaining visible misses
- Pilot *d′* = 1.90 / 2.64 and criterion +0.37 / −0.54
- Per-model miss / identity-change / no-error recall

Runtime is about one minute.

## 2. Attribution from MOT20 and shipped tracker output

Needs: MOT20 sequences MOT20-01 and MOT20-03 (not redistributed here) and
the TrackTrack MOT-format result files used for the paper.

```bash
pip install -e '.[attributor]'
python attributor/mine_events.py \
  --tracker-output /path/to/tracktrack \
  --dataset-root /path/to/MOT20 \
  --out /tmp/events.csv \
  --check-release
```

`mine_events.py` asserts that the local FN and IDSW counts equal TrackEval
`CLR_FN` and `IDSW` (34,724 and 253 on these two sequences). That assertion
lives in the matching code, not only in this document.

`--check-release` further requires `event_id` and `cause` to match
`annotations/events.csv`. Exact cause reproduction uses the occluder-mask
and reentry postpasses (`--occluder-export`). Without that export the
closure check still runs; the frozen 1,164-row CSV is the released pool.

Tracker: TrackTrack with the authors' published MOT20 config, detections
from the authors' YOLOX-X MOT20 checkpoint (training held out 01 and 03).

## 3. End to end

Needs: MOT20, the public TrackTrack implementation, and the YOLOX-X MOT20
weights. This path will drift. The paper used the authors' released
TrackTrack code and the published MOT20 config as-is. We do not pin a
commit hash here because hosted checkpoints and dependency stacks change;
treat any new tracker dump as a new run and compare it to
`annotations/events.csv` rather than expecting a bit match.

## Rendering

Clips keep every source frame and play at **10 fps** against a **25 fps**
source. Windows are 40 frames before and after the event, capped at 200
frames, which produces clips of **81–201 frames (median 93)**.

The two regimes differ at the failure:

- **Ground-truth-informed.** The highlight follows the correct physical
  person and re-anchors to whichever tracker box the ground truth matches
  when the tracker loses or switches the target. This makes the target easy
  to follow and the identity change easy to miss.
- **Supervision-free.** The highlight is pinned to one raw tracker identity
  at its first match and never re-anchors. This makes the track termination
  unmissable and the re-association the viewer's problem.

```bash
python renderer/render_clips.py \
  --tracker-output /path/to/tracktrack \
  --dataset-root /path/to/MOT20 \
  --events-root /path/to/events_jsonl \
  --output-root /tmp/clips \
  --regime supervision_free
```

Rendered MP4s are derivative works of MOT20 (CC BY-NC-SA 3.0) and are
released in a separate Hugging Face dataset, not in this repository.

## Model outputs

The files under `model_outputs/` will not reproduce from a fresh decode.
Hosted and local weights change. Decoding settings and prompt hashes are in
`model_outputs/README.md`. The scoring plan SHA-256 is
`e674837f9a1df909a83be32fcb783411c57bbd3382e30fa165f757a4dbc24d69`.
Permutation nulls used seed 2026.
