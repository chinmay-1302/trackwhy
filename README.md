# TrackWhy

Code, annotations and model outputs for *What Counts as a Tracking Failure?
Measuring Disagreement Between Metrics, Human Observers, and Vision–Language
Models* (VLM4RWD @ NeurIPS 2026).

Paper: <https://openreview.net/forum?id=TRJ7gFUnta>
([PDF](https://openreview.net/pdf?id=TRJ7gFUnta))

Project page: <https://chinmay-1302.github.io/trackwhy-page/>

Clip corpus: <https://huggingface.co/datasets/chinmay-1302/TrackWhy>
(separate repository — CC BY-NC-SA 3.0; see Licensing)

## What this is

A benchmark that asks whether a tracking failure counted by a standard
metric is one a human observer can see, and whether a vision–language model
can name it. It contains:

- a rule-based attributor that assigns scene conditions to metric-counted
  failures, anchored to TrackEval's `CLR_FN` and `IDSW` counts by an exact
  closure check
- corrected expert labels for MOT20-01 and MOT20-03 addressing annotation
  contamination (truncation, unannotated structure, flicker, localisation
  drift, a stationary standee, and crowd-class frames)
- human annotations from two observers plus a verifier pass, over 284 clips
- outputs from four vision–language models under one decoding setting

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python analysis/tables.py --out /tmp/tables
```

That command regenerates the paper's tables from the shipped annotations
and model outputs. It does not download MOT20.

To re-derive the event pool from scratch you need MOT20 (not redistributed
here) and tracker output. See [`docs/reproduce.md`](docs/reproduce.md).

## Licensing

| Contents | Licence |
| --- | --- |
| Code (`attributor/`, `renderer/`, `analysis/`, `mot_error_attribution/`) | MIT |
| Annotations, model outputs, manifest | CC BY 4.0 |
| Rendered clips (separate repository) | CC BY-NC-SA 3.0 |

The clips are derivative works of MOT20 and carry MOT20's CC BY-NC-SA 3.0
licence. The code, our annotations and the model outputs are not derived
from MOT20's image content — they reference it by frame index and track ID
— and carry the licences above. If you use any of this, MOT20 must be
cited and attributed.

```bibtex
@article{dendorfer2020mot20,
  author = {Dendorfer, Patrick and Rezatofighi, Hamid and Milan, Anton and Shi, Javen and Cremers, Daniel and Reid, Ian and Roth, Stefan and Leal-Taixé, Laura},
  title = {MOT20: A benchmark for multi object tracking in crowded scenes},
  year = {2020},
  doi = {10.48550/arXiv.2003.09003}
}

@inproceedings{mhatre2026what,
  title={What Counts as a Tracking Failure? Measuring Disagreement Between Metrics, Human Observers, and Vision-Language Models},
  author={Chinmay Mhatre and Sankalp Gawali and Punit Rathore},
  booktitle={Grounded and Faithful Vision-Language Models for Real-World Deployment 2026},
  year={2026},
  url={https://openreview.net/forum?id=TRJ7gFUnta}
}
```

## What the attributor does not claim

The rules identify the scene conditions co-occurring with a counted
failure, not the tracker's internal failure mechanism, which is not
observable from its output. The rules read ground-truth geometry, the
`visibility` field and class labels only — never detector confidences,
appearance features, or tracker-internal state. The one exception is
`track_drift_to_static`, documented in [`attributor/README.md`](attributor/README.md).
