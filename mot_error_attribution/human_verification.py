from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable


CAUSE_MIGRATIONS = {"static_person_hijack": "track_drift_to_static"}


def normalize_cause_label(value: str) -> str:
    label = value.strip()
    return CAUSE_MIGRATIONS.get(label, label)


def migrate_annotation_vocabulary(source: Path, destination: Path) -> int:
    """Write a one-time vocabulary-migrated copy of an annotation CSV."""
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    cause_fields = {
        "annotator_1_cause",
        "annotator_2_cause",
        "adjudicated_cause",
        "cause",
    }
    changed = 0
    for row in rows:
        for field in cause_fields & set(row):
            original = row[field]
            migrated = normalize_cause_label(original)
            if migrated != original:
                row[field] = migrated
                changed += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return changed


def cohen_kappa(left: Iterable[str], right: Iterable[str]) -> float:
    left_values = list(left)
    right_values = list(right)
    if len(left_values) != len(right_values) or not left_values:
        raise ValueError("Cohen's kappa requires two non-empty, equal-length label lists")
    labels = set(left_values) | set(right_values)
    n = len(left_values)
    observed = sum(a == b for a, b in zip(left_values, right_values)) / n
    left_counts = Counter(left_values)
    right_counts = Counter(right_values)
    expected = sum(left_counts[label] * right_counts[label] for label in labels) / (n * n)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def score_human_annotations(
    annotations_csv: Path,
    answer_key_jsonl: Path,
    output_csv: Path,
) -> None:
    import json

    rules = {}
    with answer_key_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                event = json.loads(line)
                rules[event["event_id"]] = normalize_cause_label(event["cause"])
    records = []
    with annotations_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            event_id = row["event_id"]
            if event_id not in rules:
                raise ValueError(f"Unknown event_id in human annotations: {event_id}")
            human = normalize_cause_label(row.get("adjudicated_cause", ""))
            if not human:
                first = normalize_cause_label(row.get("annotator_1_cause", ""))
                second = normalize_cause_label(row.get("annotator_2_cause", ""))
                if first and first == second:
                    human = first
            if not human:
                continue
            records.append((event_id, rules[event_id], human, row))
    if not records:
        raise ValueError("No adjudicated or agreeing human labels were found")

    causes = sorted(set(rules.values()))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = ["category", "n", "observed_agreement", "cohen_kappa"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        rule_labels = [record[1] for record in records]
        human_labels = [record[2] for record in records]
        writer.writerow(
            {
                "category": "ALL",
                "n": len(records),
                "observed_agreement": sum(
                    a == b for a, b in zip(rule_labels, human_labels)
                )
                / len(records),
                "cohen_kappa": cohen_kappa(rule_labels, human_labels),
            }
        )
        for cause in causes:
            left = [str(rule == cause) for rule in rule_labels]
            right = [str(human == cause) for human in human_labels]
            writer.writerow(
                {
                    "category": cause,
                    "n": len(records),
                    "observed_agreement": sum(a == b for a, b in zip(left, right))
                    / len(records),
                    "cohen_kappa": cohen_kappa(left, right),
                }
            )
