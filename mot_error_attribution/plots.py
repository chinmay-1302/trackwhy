from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mot-error-attribution-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


CAUSE_COLORS = {
    "full_occl_static": "#8c564b",
    "full_occl_pedestrian": "#1f4e79",
    "full_occl_crowd_aggregate": "#9c755f",
    "partial_occl_pedestrian": "#4f81bd",
    "partial_occl_static": "#f28e2b",
    "track_drift_to_static": "#d62728",
    "reentry": "#9467bd",
    "visible_miss": "#edc949",
    "unattributed_fn": "#59a14f",
    "unattributed_idsw": "#76b7b2",
}


def _label(value: str) -> str:
    return value.replace("_", " ")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _style_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color="#dddddd", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def _save(
    figure,
    output_dir: Path,
    stem: str,
    description: str,
    sources: list[str],
    manifest: list[dict[str, Any]],
    *,
    tight_layout: bool = True,
) -> None:
    if tight_layout:
        figure.tight_layout()
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    figure.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    manifest.append(
        {
            "name": stem,
            "description": description,
            "sources": sources,
            "files": [png.name, pdf.name],
        }
    )


def _plot_cause_distributions(
    summary_dir: Path, output_dir: Path, manifest: list[dict[str, Any]]
) -> None:
    source = summary_dir / "cause_distribution.csv"
    rows = _read_csv(source)
    sequences = sorted({row["sequence"] for row in rows})
    causes = list(dict.fromkeys(row["cause"] for row in rows))
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        counts[row["sequence"]][row["cause"]] = int(row["count"])

    figure, axes = plt.subplots(
        1, len(sequences), figsize=(7.2 * len(sequences), 6.4), sharey=True
    )
    axes = np.atleast_1d(axes)
    y = np.arange(len(causes))
    for axis, sequence in zip(axes, sequences):
        values = [counts[sequence].get(cause, 0) for cause in causes]
        bars = axis.barh(
            y,
            values,
            color=[CAUSE_COLORS.get(cause, "#999999") for cause in causes],
        )
        axis.set_title(sequence)
        axis.set_xlabel("Canonical events")
        axis.set_yticks(y, [_label(cause) for cause in causes])
        axis.invert_yaxis()
        axis.bar_label(bars, padding=3, fontsize=8)
        axis.margins(x=0.14)
        _style_axis(axis)
    figure.suptitle("Rule-attributed TrackTrack events by cause", fontsize=15, y=1.01)
    _save(
        figure,
        output_dir,
        "cause_distribution_counts",
        "Absolute canonical event counts by cause and sequence; panels use independent x scales.",
        [source.name],
        manifest,
    )

    figure, axis = plt.subplots(figsize=(11.5, 6.4))
    group_height = 0.78 / max(1, len(sequences))
    offsets = (np.arange(len(sequences)) - (len(sequences) - 1) / 2) * group_height
    hatches = ["", "///", "xx", ".."]
    for index, sequence in enumerate(sequences):
        total = sum(counts[sequence].values())
        values = [100.0 * counts[sequence].get(cause, 0) / total for cause in causes]
        bars = axis.barh(
            y + offsets[index],
            values,
            height=group_height * 0.88,
            color=[CAUSE_COLORS.get(cause, "#999999") for cause in causes],
            edgecolor="#333333",
            linewidth=0.4,
            hatch=hatches[index % len(hatches)],
            label=sequence,
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.1f}%" if value >= 1 else "" for value in values],
            padding=2,
            fontsize=7,
        )
    axis.set_yticks(y, [_label(cause) for cause in causes])
    axis.invert_yaxis()
    axis.set_xlabel("Share of sequence corpus (%)")
    axis.set_title("Cause distribution normalized within each sequence")
    axis.legend(frameon=False)
    axis.margins(x=0.13)
    _style_axis(axis)
    _save(
        figure,
        output_dir,
        "cause_distribution_percent",
        "Within-sequence percentages make MOT20-01 and MOT20-03 directly comparable.",
        [source.name],
        manifest,
    )


def _plot_validation(
    summary_dir: Path, output_dir: Path, manifest: list[dict[str, Any]]
) -> None:
    source = summary_dir / "validation_report.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    sequences = sorted(report["sequences"])
    included = [
        report["sequences"][sequence]["corpus"]["included_fn_frames"]
        for sequence in sequences
    ]
    excluded = [
        report["sequences"][sequence]["corpus"]["excluded_crowd_fn_frames"]
        for sequence in sequences
    ]
    idsw = [report["sequences"][sequence]["trackeval"]["IDSW"] for sequence in sequences]
    canonical = [
        report["sequences"][sequence]["corpus"]["canonical_event_count"]
        for sequence in sequences
    ]

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    x = np.arange(len(sequences))
    axes[0].bar(x, included, color="#4e79a7", label="Corpus FN frames")
    axes[0].bar(x, excluded, bottom=included, color="#bab0ac", label="Excluded crowd")
    axes[0].set_xticks(x, sequences)
    axes[0].set_ylabel("Frames")
    axes[0].set_title("FN closure")
    axes[0].legend(frameon=False, fontsize=8)
    for index, total in enumerate(np.asarray(included) + np.asarray(excluded)):
        axes[0].text(index, total, f"{total:,}", ha="center", va="bottom", fontsize=8)

    bars = axes[1].bar(x, idsw, color="#e15759")
    axes[1].bar_label(bars, padding=3)
    axes[1].set_xticks(x, sequences)
    axes[1].set_ylabel("Switches")
    axes[1].set_title("TrackEval ID switches")

    bars = axes[2].bar(x, canonical, color="#59a14f")
    axes[2].bar_label(bars, padding=3)
    axes[2].set_xticks(x, sequences)
    axes[2].set_ylabel("Events")
    axes[2].set_title("Canonical corpus size")
    for axis in axes:
        _style_axis(axis)
    figure.suptitle("Validation and corpus closure", fontsize=15)
    _save(
        figure,
        output_dir,
        "validation_and_closure",
        "TrackEval FN/IDSW totals, crowd exclusions, and canonical corpus counts.",
        [source.name],
        manifest,
    )


def _plot_residuals(
    summary_dir: Path, output_dir: Path, manifest: list[dict[str, Any]]
) -> None:
    source = summary_dir / "residual_diagnostics.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))

    visibility_order = ["FULL", "PARTIAL", "CLEAR", "missing"]
    visibility = data["by_min_visibility_band"]
    labels = [label for label in visibility_order if label in visibility]
    values = [visibility[label] for label in labels]
    bars = axes[0, 0].bar(labels, values, color="#4e79a7")
    axes[0, 0].bar_label(bars, padding=3, fontsize=8)
    axes[0, 0].set_title("Residuals by minimum visibility")
    axes[0, 0].set_ylabel("Events")
    axes[0, 0].tick_params(axis="x", rotation=25)

    causes = data["by_cause"]
    cause_labels = list(causes)
    bars = axes[0, 1].bar(
        [_label(label) for label in cause_labels],
        [causes[label] for label in cause_labels],
        color=[CAUSE_COLORS.get(label, "#999999") for label in cause_labels],
    )
    axes[0, 1].bar_label(bars, padding=3, fontsize=8)
    axes[0, 1].set_title("Residual type")
    axes[0, 1].tick_params(axis="x", rotation=15)

    reasons = data["by_residual_reason"]
    reason_labels = list(reasons)
    bars = axes[1, 0].barh(
        [_label(label) for label in reason_labels],
        [reasons[label] for label in reason_labels],
        color=["#59a14f", "#f28e2b"][: len(reason_labels)],
    )
    axes[1, 0].bar_label(bars, padding=3, fontsize=8)
    axes[1, 0].set_title("Why attribution stopped")
    axes[1, 0].set_xlabel("Events")

    near = data["near_interaction_threshold"]
    near_labels = list(near)
    bars = axes[1, 1].barh(
        [_label(label) for label in near_labels],
        [near[label] for label in near_labels],
        color=[
            "#76b7b2",
            "#4e79a7",
            "#59a14f",
            "#bab0ac",
            "#9c755f",
            "#ffbe7d",
            "#f28e2b",
        ][: len(near_labels)],
    )
    axes[1, 1].bar_label(bars, padding=3, fontsize=8)
    axes[1, 1].set_title("Residual interaction proximity")
    axes[1, 1].set_xlabel("Events")
    for axis in axes.flat:
        _style_axis(axis)
    figure.suptitle(
        f"Residual diagnostic ({data['residual_event_count']} unattributed events)",
        fontsize=15,
    )
    _save(
        figure,
        output_dir,
        "residual_diagnostics",
        "Visibility, type, stop reason, and near-threshold overlap diagnostics for unattributed events.",
        [source.name],
        manifest,
    )


def _identity_files(identity_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in identity_dir.glob("*.csv")
        if not path.stem.endswith("__never_detected")
    )


def _plot_identity_difficulty(
    identity_dir: Path, output_dir: Path, manifest: list[dict[str, Any]]
) -> None:
    files = _identity_files(identity_dir)
    rows_by_sequence: dict[str, list[dict[str, str]]] = {}
    for path in files:
        sequence = path.name.split("__", 1)[0]
        rows_by_sequence[sequence] = _read_csv(path)
    sequences = sorted(rows_by_sequence)

    figure = plt.figure(
        figsize=(8.7 * len(sequences) + 0.8, 9.5), layout="constrained"
    )
    grid = figure.add_gridspec(
        1, len(sequences) + 1, width_ratios=[1.0] * len(sequences) + [0.035]
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(len(sequences))]
    colorbar_axis = figure.add_subplot(grid[0, -1])
    cmap = plt.get_cmap("RdYlGn")
    norm = Normalize(0, 1)
    global_max_difficulty = max(
        (
            float(row["difficulty"])
            for rows in rows_by_sequence.values()
            for row in [
                item
                for item in rows
                if item.get("ranking_eligible", "").lower() == "true"
            ][:25]
        ),
        default=0.0,
    )
    difficulty_limit = max(0.68, global_max_difficulty * 1.36)
    for axis, sequence in zip(axes, sequences):
        rows = [
            row
            for row in rows_by_sequence[sequence]
            if row.get("ranking_eligible", "").lower() == "true"
        ][:25]
        difficulty = [float(row["difficulty"]) for row in rows]
        recall = [float(row["recall"]) for row in rows]
        labels = [f"GT {row['gt_id']}" for row in rows]
        y = np.arange(len(rows))
        bars = axis.barh(y, difficulty, color=[cmap(norm(value)) for value in recall])
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_title(sequence)
        axis.set_xlabel("Difficulty score")
        axis.set_xlim(0, difficulty_limit)
        for bar, row in zip(bars, rows):
            axis.text(
                bar.get_width() + 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"R={float(row['recall']):.2f}, S={row['n_idsw']}",
                va="center",
                fontsize=7,
            )
        _style_axis(axis)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    figure.colorbar(
        scalar,
        cax=colorbar_axis,
        label="Tracker recall for identity",
    )
    figure.suptitle("Top-25 hardest identities per sequence", fontsize=15)
    _save(
        figure,
        output_dir,
        "identity_difficulty_top25",
        "Top identities by configured length-normalized difficulty; labels show recall and ID switches.",
        [f"../identities/{path.name}" for path in files],
        manifest,
        tight_layout=False,
    )

    figure = plt.figure(
        figsize=(7.5 * len(sequences) + 0.8, 5.5), layout="constrained"
    )
    grid = figure.add_gridspec(
        1, len(sequences) + 1, width_ratios=[1.0] * len(sequences) + [0.035]
    )
    axes = []
    for index in range(len(sequences)):
        shared_axis = axes[0] if axes else None
        axes.append(
            figure.add_subplot(grid[0, index], sharex=shared_axis, sharey=shared_axis)
        )
    colorbar_axis = figure.add_subplot(grid[0, -1])
    global_max_difficulty = max(
        (
            float(row["difficulty"])
            for rows in rows_by_sequence.values()
            for row in rows
            if row.get("ranking_eligible", "").lower() == "true"
        ),
        default=0.0,
    )
    difficulty_norm = Normalize(0, max(0.6, global_max_difficulty))
    for axis, sequence in zip(axes, sequences):
        rows = [
            row
            for row in rows_by_sequence[sequence]
            if row.get("ranking_eligible", "").lower() == "true"
        ]
        recall = np.asarray([float(row["recall"]) for row in rows])
        rates = np.asarray([float(row["idsw_rate_per_100_frames"]) for row in rows])
        difficulty = np.asarray([float(row["difficulty"]) for row in rows])
        lengths = np.asarray([int(row["T"]) for row in rows])
        sizes = 18 + 90 * np.sqrt(lengths / max(1, lengths.max()))
        scatter = axis.scatter(
            recall,
            rates,
            s=sizes,
            c=difficulty,
            cmap="magma",
            norm=difficulty_norm,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
        )
        axis.set_title(sequence)
        axis.set_xlabel("Tracker recall")
        _style_axis(axis)
    axes[0].set_ylabel("ID switches per 100 GT frames")
    scalar = plt.cm.ScalarMappable(norm=difficulty_norm, cmap="magma")
    scalar.set_array([])
    figure.colorbar(scalar, cax=colorbar_axis, label="Difficulty")
    figure.suptitle("Identity-level recall and switch rate", fontsize=15)
    _save(
        figure,
        output_dir,
        "identity_recall_vs_switch_rate",
        "Each point is one GT identity; area represents GT track length and color represents difficulty.",
        [f"../identities/{path.name}" for path in files],
        manifest,
        tight_layout=False,
    )


def _plot_visible_miss(
    summary_dir: Path, output_dir: Path, manifest: list[dict[str, Any]]
) -> None:
    source = summary_dir / "visible_miss_reference.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    sequences = sorted(data["sequences"])

    area_edges = data["area_percentile_bins"]
    area_labels = [
        f"{int(left)}–{int(right)}" for left, right in zip(area_edges, area_edges[1:])
    ]
    figure, axes = plt.subplots(
        1, len(sequences), figsize=(7.2 * len(sequences), 5.2), sharey=True
    )
    axes = np.atleast_1d(axes)
    for axis, sequence in zip(axes, sequences):
        record = data["sequences"][sequence]
        all_counts = np.asarray(record["all_area_percentile_hist"], dtype=float)
        visible_counts = np.asarray(record["visible_area_percentile_hist"], dtype=float)
        all_share = 100 * all_counts / max(1, all_counts.sum())
        visible_share = 100 * visible_counts / max(1, visible_counts.sum())
        x = np.arange(len(area_labels))
        width = 0.42
        axis.bar(x - width / 2, all_share, width, color="#bab0ac", label="All target GT boxes")
        axis.bar(x + width / 2, visible_share, width, color="#edc949", label="Visible-miss events")
        axis.set_xticks(x, area_labels, rotation=35, ha="right")
        axis.set_xlabel("Within-sequence area percentile")
        axis.set_title(sequence)
        _style_axis(axis)
    axes[0].set_ylabel("Population share (%)")
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Visible-miss scale distribution vs. all target GT boxes", fontsize=15)
    _save(
        figure,
        output_dir,
        "visible_miss_area_percentile",
        "Area-percentile distribution of visible-miss events compared with all evaluated target GT boxes.",
        [source.name, "visible_miss_analysis.csv"],
        manifest,
    )

    density_edges = data["density_bins"]
    density_labels = [
        f"{int(left)}–{int(right)}" for left, right in zip(density_edges, density_edges[1:])
    ]
    density_labels[-1] += "+"
    figure, axes = plt.subplots(
        1, len(sequences), figsize=(7.2 * len(sequences), 5.2), sharey=True
    )
    axes = np.atleast_1d(axes)
    for axis, sequence in zip(axes, sequences):
        record = data["sequences"][sequence]
        all_counts = np.asarray(record["all_density_hist"], dtype=float)
        visible_counts = np.asarray(record["visible_density_hist"], dtype=float)
        all_counts[-1] += float(record["all_density_overflow"])
        visible_counts[-1] += float(record["visible_density_overflow"])
        all_share = 100 * all_counts / max(1, all_counts.sum())
        visible_share = 100 * visible_counts / max(1, visible_counts.sum())
        x = np.arange(len(density_labels))
        width = 0.42
        axis.bar(x - width / 2, all_share, width, color="#bab0ac", label="All target GT boxes")
        axis.bar(x + width / 2, visible_share, width, color="#edc949", label="Visible-miss events")
        tick_indices = np.arange(0, len(density_labels), 3)
        axis.set_xticks(tick_indices, [density_labels[index] for index in tick_indices], rotation=35, ha="right")
        axis.set_xlabel("Other target centres within 150 px")
        axis.set_title(sequence)
        _style_axis(axis)
    axes[0].set_ylabel("Population share (%)")
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Visible-miss local density vs. all target GT boxes", fontsize=15)
    _save(
        figure,
        output_dir,
        "visible_miss_local_density",
        "Local-density distribution of visible-miss events compared with all evaluated target GT boxes.",
        [source.name, "visible_miss_analysis.csv"],
        manifest,
    )


def _plot_cause_by_tracker(
    summary_dir: Path, output_dir: Path, manifest: list[dict[str, Any]]
) -> None:
    source = summary_dir / "cause_distribution_by_tracker.csv"
    rows = _read_csv(source)
    trackers = sorted({row["tracker"] for row in rows})
    groups = ["detection-caused", "association-caused"]
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))
    for axis, group in zip(axes, groups):
        selected = [row for row in rows if row["cause_group"] == group]
        causes = list(dict.fromkeys(row["cause"] for row in selected))
        x = np.arange(len(causes))
        width = 0.78 / max(1, len(trackers))
        for index, tracker in enumerate(trackers):
            lookup = {
                row["cause"]: int(row["count"])
                for row in selected
                if row["tracker"] == tracker
            }
            axis.bar(
                x + (index - (len(trackers) - 1) / 2) * width,
                [lookup.get(cause, 0) for cause in causes],
                width,
                label=tracker,
            )
        axis.set_xticks(x, [_label(cause) for cause in causes], rotation=35, ha="right")
        axis.set_ylabel("Canonical events")
        axis.set_title(group)
        axis.legend(frameon=False)
        _style_axis(axis)
    figure.suptitle("Cause distribution by tracker and mechanism group", fontsize=15)
    _save(
        figure,
        output_dir,
        "cause_distribution_by_tracker",
        "Detection- and association-caused event counts by tracker; additional trackers appear as paired bars.",
        [source.name],
        manifest,
    )


def _plot_class_histogram(
    summary_dir: Path, output_dir: Path, manifest: list[dict[str, Any]]
) -> None:
    source = summary_dir / "class_histogram.csv"
    rows = _read_csv(source)
    sequences = sorted({row["sequence"] for row in rows})
    figure, axes = plt.subplots(
        1, len(sequences), figsize=(6.7 * len(sequences), 4.8), sharey=True
    )
    axes = np.atleast_1d(axes)
    for axis, sequence in zip(axes, sequences):
        selected = [row for row in rows if row["sequence"] == sequence]
        labels = [f"class {row['class_id']}" for row in selected]
        full = np.asarray([int(row["full_sequence_rows"]) for row in selected])
        mining = np.asarray([int(row["selected_frame_rows"]) for row in selected])
        x = np.arange(len(labels))
        width = 0.38
        axis.bar(x - width / 2, full, width, label="Full GT sequence", color="#bab0ac")
        axis.bar(x + width / 2, mining, width, label="Mining frames", color="#4e79a7")
        axis.set_xticks(x, labels, rotation=30, ha="right")
        axis.set_yscale("log")
        axis.set_title(sequence)
        axis.set_ylabel("GT rows (log scale)")
        _style_axis(axis)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Observed MOT20 GT class histogram", fontsize=15)
    _save(
        figure,
        output_dir,
        "class_histogram",
        "Observed GT row counts by class for full sequences and authorized mining frames.",
        [source.name],
        manifest,
    )


def _plot_v4_rule_audits(
    summary_dir: Path, output_dir: Path, manifest: list[dict[str, Any]]
) -> None:
    sweep_source = summary_dir / "ioa_sweep.csv"
    if sweep_source.is_file():
        rows = [row for row in _read_csv(sweep_source) if row["sequence"] == "ALL"]
        x = np.asarray([float(row["ioa_interaction"]) for row in rows])
        figure, axis = plt.subplots(figsize=(8.4, 5.2))
        axis.plot(
            x,
            [int(row["partial_rules_total"]) for row in rows],
            marker="o",
            label="Partial rules",
            color="#4f81bd",
        )
        axis.plot(
            x,
            [int(row["visible_miss"]) for row in rows],
            marker="o",
            label="Visible miss",
            color="#edc949",
        )
        axis.plot(
            x,
            [int(row["residual_remaining"]) for row in rows],
            marker="o",
            label="Residual",
            color="#59a14f",
        )
        selected = next((row for row in rows if row["selected_threshold"] == "True"), None)
        if selected is not None:
            axis.axvline(
                float(selected["ioa_interaction"]),
                color="#d62728",
                linestyle="--",
                linewidth=1.2,
                label=f"Chosen {float(selected['ioa_interaction']):.2f}",
            )
        axis.set_xlabel("IOA_INTERACTION")
        axis.set_ylabel("Canonical events")
        axis.set_title("IoA interaction-threshold sweep")
        axis.legend(frameon=False)
        _style_axis(axis)
        _save(
            figure,
            output_dir,
            "ioa_sweep",
            "Trade-off between partial-occlusion capture, visible-miss purity, and residual size.",
            [sweep_source.name],
            manifest,
        )

    static_source = summary_dir / "static_person_presence_vs_drift.csv"
    if static_source.is_file():
        rows = [row for row in _read_csv(static_source) if row["sequence"] != "ALL"]
        labels = [row["sequence"] for row in rows]
        x = np.arange(len(rows))
        width = 0.38
        figure, axis = plt.subplots(figsize=(7.8, 5.0))
        axis.bar(
            x - width / 2,
            [int(row["static_person_present_events"]) for row in rows],
            width,
            label="Static person present",
            color="#f28e2b",
        )
        axis.bar(
            x + width / 2,
            [int(row["track_drift_to_static_events"]) for row in rows],
            width,
            label="Track drift to static",
            color="#d62728",
        )
        axis.set_xticks(x, labels)
        axis.set_ylabel("Canonical events")
        axis.set_title("Static-person proximity versus identity drift")
        axis.legend(frameon=False)
        _style_axis(axis)
        _save(
            figure,
            output_dir,
            "static_person_presence_vs_drift",
            "GT-only static-person proximity compared with tracker-coordinate drift events.",
            [static_source.name],
            manifest,
        )

    delta_source = summary_dir / "partial_selection_delta.csv"
    if delta_source.is_file():
        rows = _read_csv(delta_source)
        figure, axis = plt.subplots(figsize=(8.0, 4.8))
        if rows:
            labels = [
                f"{row['sequence']}\n{_label(row['order_based_cause'])} →\n{_label(row['score_based_cause'])}"
                for row in rows
            ]
            bars = axis.bar(
                np.arange(len(rows)),
                [int(row["changed_event_count"]) for row in rows],
                color="#f28e2b",
            )
            axis.set_xticks(np.arange(len(rows)), labels)
            axis.bar_label(bars, padding=3)
        else:
            axis.text(0.5, 0.5, "No order-sensitive partial events", ha="center", va="center")
            axis.set_xticks([])
        axis.set_ylabel("Events changed")
        axis.set_title("Partial-cause sensitivity to score-based selection")
        _style_axis(axis)
        _save(
            figure,
            output_dir,
            "partial_selection_delta",
            "Events whose partial cause changes when comparable focal-IoA scores replace fixed rule order.",
            [delta_source.name, "ordering_suppression.csv"],
            manifest,
        )


def generate_summary_plots(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    summary_dir = root / "summary"
    identity_dir = root / "identities"
    plot_dir = summary_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.titlesize": 15,
            "savefig.dpi": 220,
        }
    )
    _plot_cause_distributions(summary_dir, plot_dir, manifest)
    _plot_validation(summary_dir, plot_dir, manifest)
    _plot_residuals(summary_dir, plot_dir, manifest)
    _plot_identity_difficulty(identity_dir, plot_dir, manifest)
    _plot_visible_miss(summary_dir, plot_dir, manifest)
    _plot_cause_by_tracker(summary_dir, plot_dir, manifest)
    _plot_class_histogram(summary_dir, plot_dir, manifest)
    _plot_v4_rule_audits(summary_dir, plot_dir, manifest)
    result = {"plot_count": len(manifest), "plots": manifest}
    (plot_dir / "plot_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
