# Model outputs

Raw structured responses, one JSON file per clip, for four models and five
conditions. Decoding settings below are what produced these files. Rerunning
the same hosted or local weights will not bit-match them: models change.

Prompt text is in `prompts/`. SHA-256 of the shipped files:

| Prompt | Bytes | SHA-256 |
| --- | ---: | --- |
| `prompts/gt_informed.txt` | 18,026 | `620a1b5f024ff21151ff390259858c5bdc73361ab3cfc306ffc3f53743167fca` |
| `prompts/supervision_free.txt` | 17,990 | `5381d19e8f5749663364adfc4ab4c75538b8971639a21261c2c724dd305f0ba6` |

Those hashes match Appendix A.1 / A.2 of the paper.

## Models

| Directory | Paper name |
| --- | --- |
| `Qwen3.8-27B/` | Qwen3.8-27B |
| `Qwen3-VL-8B/` | Qwen3-VL-8B |
| `InternVL3.5-8B/` | InternVL3.5-8B |
| `LLaVA-OneVision-2-8B/` | LLaVA-OneVision-2-8B |

## Conditions

| Directory | Regime | Prompt |
| --- | --- | --- |
| `02_inside_top_left` | ground-truth-informed | `gt_informed.txt` |
| `09_target_only_inside_top_left` | ground-truth-informed | `gt_informed.txt` |
| `18_target_only_inside_center_with_target_trajectory` | ground-truth-informed | `gt_informed.txt` |
| `02_inside_top_left_tracker_only` | supervision-free | `supervision_free.txt` |
| `02_inside_top_left_tracker_only_base_prompt` | supervision-free | `gt_informed.txt` |

Each condition directory has 284 files `C001.json` … `C284.json`.

## Decoding

The paper states that decoding is greedy throughout.

| Model | Setting used for these files |
| --- | --- |
| Qwen3.8-27B | `temperature=0.0`, greedy |
| Qwen3-VL-8B | `do_sample=False` |
| InternVL3.5-8B | `do_sample=False`, `max_new_tokens=512` |
| LLaVA-OneVision-2-8B | greedy first; if the parse is invalid, one retry at `temperature=0.7`, `top_p=0.8` |

Scoring treats a missing or unparsable `primary_error` as a prediction
failure and keeps the clip in the denominator. A schema-invalid extra field
does not override a valid `primary_error`.
