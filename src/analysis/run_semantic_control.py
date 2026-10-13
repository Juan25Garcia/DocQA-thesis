
"""Exploratory confusion analysis stratified by semantic similarity."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "example_id",
    "confusion",
    "same_page",
    "block_similarity",
}


def parse_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    normalized = str(value).strip().lower()

    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False

    raise ValueError(f"Invalid boolean value: {value!r}")


def summarize_group(group):
    result = {}

    for same_page in (True, False):
        subset = group[group["same_page"] == same_page]
        count = len(subset)
        errors = int(subset["confusion"].sum())

        result[same_page] = (
            errors,
            count,
            errors / count if count else float("nan"),
        )

    true_rate = result[True][2]
    false_rate = result[False][2]

    difference = (
        (true_rate - false_rate) * 100
        if np.isfinite(true_rate) and np.isfinite(false_rate)
        else float("nan")
    )

    return result, difference


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/evaluation/retrieval_confusions_semantic.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/evaluation/semantic_control_summary.csv"
        ),
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    if args.bins < 2:
        raise ValueError("--bins must be at least 2")

    df = pd.read_csv(args.input)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["confusion"] = df["confusion"].map(parse_bool)
    df["same_page"] = df["same_page"].map(parse_bool)

    df["block_similarity"] = pd.to_numeric(
        df["block_similarity"],
        errors="raise",
    )

    if not np.isfinite(df["block_similarity"]).all():
        raise ValueError("Non-finite similarities detected")

    # Quantile bins: approximately equal comparison counts.
    df["similarity_bin"] = pd.qcut(
        df["block_similarity"],
        q=args.bins,
        duplicates="drop",
    )

    results = []

    print("\n=== SEMANTICALLY STRATIFIED CONFUSIONS ===")
    print(f"Questions: {df['example_id'].nunique()}")
    print(f"Comparisons: {len(df)}")
    print(f"Confusions: {int(df['confusion'].sum())}")

    for interval, group in df.groupby(
        "similarity_bin",
        observed=True,
    ):
        rates, difference = summarize_group(group)

        true_errors, true_count, true_rate = rates[True]
        false_errors, false_count, false_rate = rates[False]

        print(f"\nSimilarity range: {interval}")
        print(
            f"  Same page: "
            f"{true_errors}/{true_count} "
            f"({true_rate * 100:.2f}%)"
        )
        print(
            f"  Different page: "
            f"{false_errors}/{false_count} "
            f"({false_rate * 100:.2f}%)"
        )
        print(f"  Difference: {difference:+.2f} pp")

        results.append({
            "similarity_bin": str(interval),
            "same_page_errors": true_errors,
            "same_page_total": true_count,
            "different_page_errors": false_errors,
            "different_page_total": false_count,
            "difference_pp": difference,
        })

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(results).to_csv(
        args.output,
        index=False,
    )

    print(f"\nOutput: {args.output.resolve()}")
    print(
        "\nNote: exploratory stratification only; "
        "not a formal adjusted hypothesis test."
    )


if __name__ == "__main__":
    main()
