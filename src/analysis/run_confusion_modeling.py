
"""Compare semantic and semantic-structural confusion models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_SETS = {
    "semantic": ["block_similarity"],
    "semantic_structural": [
        "block_similarity",
        "same_page",
    ],
}


def parse_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)

    value = str(value).strip().lower()

    if value in {"true", "1"}:
        return 1

    if value in {"false", "0"}:
        return 0

    raise ValueError(f"Invalid boolean: {value!r}")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "example_id",
        "confusion",
        "same_page",
        "block_similarity",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Input CSV is empty.")

    df["confusion"] = df["confusion"].map(parse_bool)
    df["same_page"] = df["same_page"].map(parse_bool)

    df["block_similarity"] = pd.to_numeric(
        df["block_similarity"],
        errors="raise",
    )

    if not np.isfinite(
        df["block_similarity"].to_numpy()
    ).all():
        raise ValueError("Invalid block similarities.")

    return df


def make_model():
    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                max_iter=2000,
                solver="lbfgs",
                class_weight=None,
                random_state=42,
            ),
        ),
    ])


def cross_validated_predictions(
    df: pd.DataFrame,
    features: list[str],
    n_splits: int,
) -> np.ndarray:
    X = df[features].to_numpy(dtype=float)
    y = df["confusion"].to_numpy(dtype=int)
    groups = df["example_id"].to_numpy()

    predictions = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    cv = GroupKFold(n_splits=n_splits)

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups),
        start=1,
    ):
        y_train = y[train_idx]

        # Logistic regression requires both classes.
        if len(np.unique(y_train)) < 2:
            raise ValueError(
                f"Fold {fold} has only one training class. "
                "Use more questions with confusions."
            )

        model = make_model()

        model.fit(
            X[train_idx],
            y_train,
        )

        predictions[test_idx] = (
            model.predict_proba(X[test_idx])[:, 1]
        )

        print(
            f"  Fold {fold}: "
            f"train={len(train_idx)}, "
            f"test={len(test_idx)}, "
            f"test_confusions={int(y[test_idx].sum())}"
        )

    if not np.isfinite(predictions).all():
        raise RuntimeError(
            "Some comparisons were not evaluated."
        )

    return predictions


def evaluate(y, probabilities):
    result = {
        "average_precision": average_precision_score(
            y,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y,
            probabilities,
        ),
        "log_loss": log_loss(
            y,
            probabilities,
            labels=[0, 1],
        ),
    }

    return result


def print_metrics(name, metrics):
    print(f"\n{name}")

    for metric, value in metrics.items():
        print(f"  {metric}: {value:.5f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/evaluation/"
            "retrieval_confusions_semantic.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    df = load_data(args.input)

    y = df["confusion"].to_numpy(dtype=int)
    groups = df["example_id"].to_numpy()

    n_groups = len(np.unique(groups))

    if args.folds < 2 or args.folds > n_groups:
        raise ValueError(
            f"--folds must be between 2 and {n_groups}"
        )

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Both confusion classes are required."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n=== CONFUSION MODELING ===")
    print(f"Questions: {n_groups}")
    print(f"Comparisons: {len(df)}")
    print(f"Confusions: {int(y.sum())}")
    print(f"Prevalence: {y.mean():.4%}")
    print(f"GroupKFold: {args.folds} folds")

    all_metrics = {}
    prediction_frame = df[
        [
            "example_id",
            "confusion",
            "same_page",
            "block_similarity",
        ]
    ].copy()

    for name, features in FEATURE_SETS.items():
        print(f"\nTraining {name}...")

        predictions = cross_validated_predictions(
            df,
            features,
            args.folds,
        )

        metrics = evaluate(y, predictions)

        all_metrics[name] = metrics

        prediction_frame[
            f"prob_{name}"
        ] = predictions

        print_metrics(name, metrics)

    semantic = all_metrics["semantic"]
    structural = all_metrics["semantic_structural"]

    print("\n=== MODEL COMPARISON (B - A) ===")

    for metric in semantic:
        difference = (
            structural[metric] - semantic[metric]
        )

        print(
            f"  {metric}: {difference:+.5f}"
        )

    print(
        "\nHigher is better: average_precision, roc_auc"
    )
    print(
        "Lower is better: brier_score, log_loss"
    )

    predictions_path = (
        args.output_dir / "confusion_model_predictions.csv"
    )

    metrics_path = (
        args.output_dir / "confusion_model_metrics.csv"
    )

    prediction_frame.to_csv(
        predictions_path,
        index=False,
    )

    metrics_df = pd.DataFrame.from_dict(
        all_metrics,
        orient="index",
    )

    metrics_df.index.name = "model"

    metrics_df.to_csv(metrics_path)

    print(f"\nPredictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")

    print(
        "\nNote: exploratory grouped cross-validation. "
        "These results do not establish causality "
        "or independent benchmark generalization."
    )


if __name__ == "__main__":
    main()
