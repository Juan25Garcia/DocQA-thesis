
"""Descriptive and question-clustered analysis of retrieval confusions."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path


FEATURES = (
    "same_document",
    "same_page",
    "same_type",
    "same_parent",
    "sibling",
)


def load_confusions(path: str | Path) -> list[dict]:
    """Load and validate the confusion CSV."""
    with Path(path).open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "example_id", "margin", "confusion",
            "positive_score", "candidate_score", *FEATURES,
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        rows = []
        for row in reader:
            parsed = dict(row)

            for key in ("margin", "positive_score", "candidate_score"):
                parsed[key] = float(row[key])

            for key in ("confusion", *FEATURES):
                value = row[key].strip().lower()
                if value in ("true", "false"):
                    parsed[key] = value == "true"
                elif value in ("", "none", "null") and key != "confusion":
                    parsed[key] = None
                else:
                    raise ValueError(f"Invalid {key}: {row[key]!r}")

            rows.append(parsed)

    if not rows:
        raise ValueError("The confusion CSV is empty.")

    # Check that the recorded outcome matches the score definition.
    for row in rows:
        expected_margin = (
            row["positive_score"] - row["candidate_score"]
        )
        if abs(row["margin"] - expected_margin) > 1e-5:
            raise ValueError("Margin does not match retrieval scores.")
        if row["confusion"] != (row["margin"] < 0):
            raise ValueError("Confusion flag does not match margin.")

    return rows


def confusion_rate(rows: list[dict], feature: str, value: bool):
    """Return (confusions, total, rate) for one feature value."""
    subset = [r for r in rows if r[feature] is value]
    total = len(subset)
    errors = sum(r["confusion"] for r in subset)
    return errors, total, errors / total if total else None


def rate_difference(rows: list[dict], feature: str):
    """Difference: P(confusion | True) - P(confusion | False)."""
    true_rate = confusion_rate(rows, feature, True)[2]
    false_rate = confusion_rate(rows, feature, False)[2]

    if true_rate is None or false_rate is None:
        return None

    return true_rate - false_rate


def clustered_bootstrap(
    rows: list[dict],
    feature: str,
    iterations: int = 2000,
    seed: int = 42,
):
    """Percentile CI from resampling entire questions with replacement."""
    if iterations < 1:
        raise ValueError("iterations must be positive")

    groups = defaultdict(list)
    for row in rows:
        groups[row["example_id"]].append(row)

    question_ids = sorted(groups)
    rng = random.Random(seed)
    estimates = []

    for _ in range(iterations):
        sampled_ids = rng.choices(
            question_ids, k=len(question_ids)
        )
        sample = [
            row
            for question_id in sampled_ids
            for row in groups[question_id]
        ]

        estimate = rate_difference(sample, feature)
        if estimate is not None:
            estimates.append(estimate)

    if not estimates:
        return None

    estimates.sort()

    def percentile(p: float) -> float:
        position = (len(estimates) - 1) * p
        lower = int(position)
        upper = min(lower + 1, len(estimates) - 1)
        weight = position - lower
        return (
            estimates[lower] * (1 - weight)
            + estimates[upper] * weight
        )

    return {
        "lower": percentile(0.025),
        "upper": percentile(0.975),
        "valid_samples": len(estimates),
    }


def summarize(
    rows: list[dict],
    iterations: int = 2000,
    seed: int = 42,
) -> list[dict]:
    """Summarize associations for all structural features."""
    summary = []

    for feature in FEATURES:
        true_errors, true_total, true_rate = confusion_rate(
            rows, feature, True
        )
        false_errors, false_total, false_rate = confusion_rate(
            rows, feature, False
        )

        difference = rate_difference(rows, feature)
        interval = clustered_bootstrap(
            rows, feature, iterations, seed
        )

        summary.append({
            "feature": feature,
            "true_errors": true_errors,
            "true_total": true_total,
            "true_rate": true_rate,
            "false_errors": false_errors,
            "false_total": false_total,
            "false_rate": false_rate,
            "rate_difference": difference,
            "ci_lower": interval["lower"] if interval else None,
            "ci_upper": interval["upper"] if interval else None,
            "valid_bootstrap_samples": (
                interval["valid_samples"] if interval else 0
            ),
        })

    return summary


def save_summary(summary: list[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(summary[0].keys())
        )
        writer.writeheader()
        writer.writerows(summary)
