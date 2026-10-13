
"""Paired, question-clustered bootstrap for confusion model metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


METRICS = {
    "average_precision": "higher",
    "roc_auc": "higher",
    "brier_score": "lower",
    "log_loss": "lower",
}

REQUIRED_COLUMNS = {
    "example_id",
    "confusion",
    "prob_semantic",
    "prob_semantic_structural",
}


def parse_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)

    normalized = str(value).strip().lower()

    if normalized in {"true", "1"}:
        return 1

    if normalized in {"false", "0"}:
        return 0

    raise ValueError(f"Invalid confusion value: {value!r}")


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Prediction CSV is empty.")

    if df["example_id"].isna().any():
        raise ValueError("Missing question IDs.")

    df["confusion"] = df["confusion"].map(parse_bool)

    probability_columns = [
        "prob_semantic",
        "prob_semantic_structural",
    ]

    for column in probability_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

        values = df[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite probabilities in {column}"
            )

        if ((values < 0) | (values > 1)).any():
            raise ValueError(
                f"Probabilities outside [0, 1] in {column}"
            )

    if df["confusion"].nunique() != 2:
        raise ValueError(
            "Both confusion classes are required."
        )

    return df


def calculate_metrics(y, probabilities):
    """Evaluate predictions; AUC requires both classes."""
    return {
        "average_precision": average_precision_score(
            y, probabilities
        ),
        "roc_auc": roc_auc_score(
            y, probabilities
        ),
        "brier_score": brier_score_loss(
            y, probabilities
        ),
        "log_loss": log_loss(
            y,
            probabilities,
            labels=[0, 1],
        ),
    }


def metric_differences(frame):
    y = frame["confusion"].to_numpy(dtype=int)

    if np.unique(y).size < 2:
        return None

    semantic = calculate_metrics(
        y,
        frame["prob_semantic"].to_numpy(dtype=float),
    )

    structural = calculate_metrics(
        y,
        frame[
            "prob_semantic_structural"
        ].to_numpy(dtype=float),
    )

    return {
        metric: structural[metric] - semantic[metric]
        for metric in METRICS
    }


def bootstrap(
    df: pd.DataFrame,
    iterations: int = 2000,
    seed: int = 42,
):
    """Resample whole questions, preserving their comparisons."""
    rng = np.random.default_rng(seed)

    question_groups = [
        group
        for _, group in df.groupby(
            "example_id",
            sort=True,
        )
    ]

    n_questions = len(question_groups)

    if n_questions < 2:
        raise ValueError(
            "At least two questions are required."
        )

    observed = metric_differences(df)

    if observed is None:
        raise ValueError(
            "Observed data require both classes."
        )

    samples = {
        metric: []
        for metric in METRICS
    }

    skipped = 0

    for _ in range(iterations):
        selected = rng.integers(
            0,
            n_questions,
            size=n_questions,
        )

        # Repeated questions remain repeated:
        # each draw contributes its complete comparison set.
        sample = pd.concat(
            [
                question_groups[index]
                for index in selected
            ],
            ignore_index=True,
        )

        differences = metric_differences(sample)

        if differences is None:
            skipped += 1
            continue

        for metric, difference in differences.items():
            samples[metric].append(difference)

    rows = []

    for metric, direction in METRICS.items():
        values = np.asarray(
            samples[metric],
            dtype=float,
        )

        if values.size == 0:
            raise RuntimeError(
                "No valid bootstrap samples."
            )

        lower, upper = np.percentile(
            values,
            [2.5, 97.5],
        )

        # Positive differences are improvements for
        # AP/AUC; negative differences for Brier/LogLoss.
        if direction == "higher":
            improvement_fraction = np.mean(values > 0)
        else:
            improvement_fraction = np.mean(values < 0)

        rows.append({
            "metric": metric,
            "observed_difference": observed[metric],
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "improvement_fraction": float(
                improvement_fraction
            ),
            "valid_iterations": int(values.size),
            "skipped_iterations": skipped,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/evaluation/"
            "confusion_model_predictions.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/evaluation/"
            "confusion_model_bootstrap.csv"
        ),
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.iterations < 100:
        raise ValueError(
            "Use at least 100 bootstrap iterations."
        )

    df = load_predictions(args.input)

    print("\n=== CONFUSION MODEL BOOTSTRAP ===")
    print(f"Questions: {df['example_id'].nunique()}")
    print(f"Comparisons: {len(df)}")
    print(f"Confusions: {int(df['confusion'].sum())}")
    print(f"Iterations: {args.iterations}")
    print(f"Seed: {args.seed}")

    results = bootstrap(
        df,
        iterations=args.iterations,
        seed=args.seed,
    )

    print("\n=== RESULTS (MODEL B - MODEL A) ===")

    for row in results.itertuples():
        print(f"\n{row.metric}")
        print(
            f"  Observed difference: "
            f"{row.observed_difference:+.5f}"
        )
        print(
            f"  Bootstrap 95% interval: "
            f"[{row.ci_lower:+.5f}, "
            f"{row.ci_upper:+.5f}]"
        )
        print(
            f"  Fraction favoring B: "
            f"{row.improvement_fraction:.2%}"
        )

    print(
        "\nValid iterations: "
        f"{results['valid_iterations'].iloc[0]}"
    )

    print(
        "Skipped iterations: "
        f"{results['skipped_iterations'].iloc[0]}"
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        args.output,
        index=False,
    )

    print(f"\nOutput: {args.output.resolve()}")

    print(
        "\nNote: intervals describe variability from "
        "resampling questions with fixed out-of-fold "
        "predictions. Model refitting is not included."
    )


if __name__ == "__main__":
    main()
