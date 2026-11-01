"""Signal-detection helpers: d-prime and criterion with the log-linear correction."""

from __future__ import annotations

from statistics import NormalDist


def loglinear_rate(k: int, n: int) -> float:
    return (k + 0.5) / (n + 1.0)


def dprime_and_c(hits: int, n_pos: int, false_alarms: int, n_neg: int) -> dict[str, float]:
    hit_rate = loglinear_rate(hits, n_pos)
    fa_rate = loglinear_rate(false_alarms, n_neg)
    normal = NormalDist()
    z_h = normal.inv_cdf(hit_rate)
    z_f = normal.inv_cdf(fa_rate)
    return {
        "dprime": z_h - z_f,
        "criterion": -0.5 * (z_h + z_f),
        "hit_rate_corrected": hit_rate,
        "fa_rate_corrected": fa_rate,
    }


def binary_from_answer(answer: str) -> str | None:
    text = " ".join(str(answer or "").strip().lower().split())
    if text in {"dont_know", "don't_know", "dontknow", "unknown"}:
        return None
    if text in {"no_error", "yes", "y", "fine", "clean", "correct"}:
        return "negative"
    if text in {
        "miss",
        "identity_change",
        "drift",
        "no",
        "n",
        "broken",
        "error",
    }:
        return "positive"
    return "positive" if text and text != "no_error" else "negative"
