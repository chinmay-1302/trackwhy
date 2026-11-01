#!/usr/bin/env python3
"""Regenerate the paper's headline tables from the shipped files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.funnel import headline, load_funnel
from analysis.human_agreement import (
    observer_table,
    rule_observer_kappa,
    verifier_flips,
    load_observer,
    paired_clip_ids,
)
from analysis.sdt import binary_from_answer, dprime_and_c
from analysis.vlm_eval import evaluate_all


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_close(name: str, got: float, expected: float, tol: float = 0.005) -> None:
    if abs(got - expected) > tol:
        raise AssertionError(f"{name}: got {got:.6f}, expected {expected} ± {tol}")


def clip_id_checks(annotations: Path, model_root: Path) -> None:
    manifest = {row["clip_id"] for row in load_csv(annotations / "manifest.csv")}
    observer_a = {row["clip_id"] for row in load_csv(annotations / "human" / "observer_a.csv")}
    observer_b = {row["clip_id"] for row in load_csv(annotations / "human" / "observer_b.csv")}
    if manifest != observer_a or manifest != observer_b:
        raise AssertionError("manifest clip_id set does not match observer exports")
    if len(manifest) != 284:
        raise AssertionError(f"manifest has {len(manifest)} clips, expected 284")
    informed = {row["clip_id"] for row in load_csv(annotations / "human" / "verifier_informed.csv")}
    if informed - manifest:
        raise AssertionError("verifier_informed contains clip ids missing from the manifest")
    sample = next(model_root.glob("*/*/*.json"))
    condition_dir = sample.parent
    model_clips = {path.stem for path in condition_dir.glob("C*.json")}
    if manifest != model_clips:
        raise AssertionError(
            f"{condition_dir} clip ids do not match the manifest "
            f"({len(model_clips)} vs {len(manifest)})"
        )


def pilot_scores(rows: list[dict[str, str]]) -> dict[str, dict]:
    by_observer: dict[str, dict] = {}
    for observer in ("naive", "author"):
        items = [row for row in rows if row["observer"] == observer]
        positives = [row for row in items if row["true_class"] != "no_error"]
        negatives = [row for row in items if row["true_class"] == "no_error"]
        hits = 0
        false_alarms = 0
        abstentions = 0
        for row in positives:
            decision = binary_from_answer(row["answer"])
            if decision is None:
                abstentions += 1
                continue
            if decision == "positive":
                hits += 1
        for row in negatives:
            decision = binary_from_answer(row["answer"])
            if decision is None:
                abstentions += 1
                continue
            if decision == "positive":
                false_alarms += 1
        stats = dprime_and_c(hits, len(positives), false_alarms, len(negatives))
        by_observer[observer] = {
            "hits": hits,
            "n_pos": len(positives),
            "false_alarms": false_alarms,
            "n_neg": len(negatives),
            "abstentions": abstentions,
            **stats,
        }
    return by_observer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reproduce TrackWhy paper tables")
    parser.add_argument("--annotations", default=str(ROOT / "annotations"))
    parser.add_argument("--model-outputs", default=str(ROOT / "model_outputs"))
    parser.add_argument("--out", default=str(ROOT / "out" / "tables"))
    args = parser.parse_args(argv)
    annotations = Path(args.annotations)
    model_root = Path(args.model_outputs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    clip_id_checks(annotations, model_root)

    observer_a = load_observer(annotations / "human" / "observer_a.csv")
    observer_b = load_observer(annotations / "human" / "observer_b.csv")
    human = observer_table(observer_a, observer_b)
    clip_causes = {
        row["clip_id"]: row for row in load_csv(annotations / "clip_causes.csv")
    }
    paired = paired_clip_ids(observer_a, observer_b)
    rules_a = rule_observer_kappa(observer_a, clip_causes, paired)
    rules_b = rule_observer_kappa(observer_b, clip_causes, paired)
    forward = load_csv(annotations / "human" / "verifier_forward.csv")
    informed = load_csv(annotations / "human" / "verifier_informed.csv")
    flips = verifier_flips(forward, informed)
    funnel_rows = load_funnel(annotations / "funnel.csv")
    funnel = headline(funnel_rows)
    pilot = pilot_scores(load_csv(annotations / "human" / "pilot.csv"))
    gt_v3 = json.loads((annotations / "gt_v3.json").read_text(encoding="utf-8"))
    gt_tracker = json.loads(
        (annotations / "gt_tracker_only.json").read_text(encoding="utf-8")
    )
    model_rows = evaluate_all(model_root, gt_v3, gt_tracker)

    write_csv(
        out / "human_agreement.csv",
        [
            {
                "comparison": "observer_a_vs_observer_b",
                "n": human["n_paired"],
                "agreement": f"{human['agreement']:.3f}",
                "kappa": f"{human['kappa']:.3f}",
            },
            {
                "comparison": "rules_vs_observer_a",
                "n": rules_a["n"],
                "agreement": f"{rules_a['agreement']:.3f}",
                "kappa": f"{rules_a['kappa']:.3f}",
            },
            {
                "comparison": "rules_vs_observer_b",
                "n": rules_b["n"],
                "agreement": f"{rules_b['agreement']:.3f}",
                "kappa": f"{rules_b['kappa']:.3f}",
            },
        ],
    )
    write_csv(
        out / "observability.csv",
        [
            {
                "observer": "A",
                "no_error_visible_rate": f"{human['observer_a_no_error_rate']:.3f}",
            },
            {
                "observer": "B",
                "no_error_visible_rate": f"{human['observer_b_no_error_rate']:.3f}",
            },
        ],
    )
    write_csv(out / "funnel.csv", funnel_rows)
    write_csv(
        out / "verifier_reclassification.csv",
        [
            {
                "n_forward": flips["n_forward"],
                "n_no_error_forward": flips["n_no_error_forward"],
                "n_flips_no_error_to_miss": flips["n_flips_to_miss"],
            }
        ],
    )
    write_csv(
        out / "pilot_sdt.csv",
        [
            {
                "observer": name,
                "hits": f"{row['hits']}/{row['n_pos']}",
                "false_alarms": f"{row['false_alarms']}/{row['n_neg']}",
                "dprime": f"{row['dprime']:.2f}",
                "criterion": f"{row['criterion']:+.2f}",
                "abstentions": row["abstentions"],
            }
            for name, row in pilot.items()
        ],
    )
    write_csv(
        out / "model_cells.csv",
        [
            {
                "model": row["model"],
                "condition": row["condition"],
                "n": row["n"],
                "kappa": f"{row['kappa']:.3f}",
                "miss": f"{row['miss']:.3f}",
                "identity_change": f"{row['identity_change']:.3f}",
                "no_error": f"{row['no_error']:.3f}",
            }
            for row in model_rows
        ],
    )

    assert human["n_paired"] == 282
    assert_close("kappa_ab", human["kappa"], 0.435)
    assert_close("kappa_rules_a", rules_a["kappa"], 0.177, 0.003)
    assert_close("kappa_rules_b", rules_b["kappa"], 0.132)
    assert abs(human["observer_a_no_error_rate"] - 0.41) < 0.02
    assert abs(human["observer_b_no_error_rate"] - 0.44) < 0.02
    assert flips["n_forward"] == 170
    assert flips["n_flips_to_miss"] == 47
    assert funnel["raw"] == 282
    assert funnel["final"] == 22
    assert_close("naive_dprime", pilot["naive"]["dprime"], 1.90, 0.02)
    assert_close("author_dprime", pilot["author"]["dprime"], 2.64, 0.02)
    assert_close("naive_c", pilot["naive"]["criterion"], 0.37, 0.02)
    assert_close("author_c", pilot["author"]["criterion"], -0.54, 0.02)

    expected_cells = {
        ("Qwen3.8-27B", "02"): (257, 0.000, 0.500, 0.840),
        ("Qwen3.8-27B", "09"): (257, 0.007, 0.500, 0.900),
        ("Qwen3.8-27B", "18"): (257, 0.000, 0.500, 0.900),
        ("Qwen3.8-27B", "tracker-only"): (255, 0.007, 0.831, 0.596),
        ("Qwen3.8-27B", "tracker-only, base"): (255, 0.000, 0.385, 0.830),
        ("InternVL3.5-8B", "09"): (257, 0.028, 0.000, 0.720),
        ("Qwen3-VL-8B", "02"): (257, 0.000, 0.000, 1.000),
        ("LLaVA-OneVision-2-8B", "02"): (257, 0.000, 0.000, 1.000),
    }
    by_cell = {(row["model"], row["condition"]): row for row in model_rows}
    for key, (n, miss, ident, no_error) in expected_cells.items():
        row = by_cell[key]
        assert row["n"] == n, (key, row["n"], n)
        assert_close(f"{key}_miss", row["miss"], miss, 0.003)
        assert_close(f"{key}_id", row["identity_change"], ident, 0.003)
        assert_close(f"{key}_ne", row["no_error"], no_error, 0.003)

    print("paper tables reproduced")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
