
"""CLI for exploratory structural confusion statistics."""

import argparse
from pathlib import Path

from src.analysis.confusion_statistics import (
    load_confusions,
    save_summary,
    summarize,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/evaluation/retrieval_confusions.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/evaluation/confusion_statistics.csv"
        ),
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_confusions(args.input)
    results = summarize(
        rows,
        iterations=args.bootstrap,
        seed=args.seed,
    )

    questions = len({r["example_id"] for r in rows})
    errors = sum(r["confusion"] for r in rows)

    print("\n=== CONFUSION STATISTICS ===")
    print(f"Questions: {questions}")
    print(f"Comparisons: {len(rows)}")
    print(f"Confusions: {errors}")
    print(f"Bootstrap iterations: {args.bootstrap}")

    print("\nRates and clustered 95% bootstrap intervals")
    print("(Differences expressed in percentage points)\n")

    for result in results:
        name = result["feature"]
        true_rate = result["true_rate"]
        false_rate = result["false_rate"]
        difference = result["rate_difference"]
        lower = result["ci_lower"]
        upper = result["ci_upper"]

        print(f"{name}:")
        print(
            f"  True:  {result['true_errors']}/"
            f"{result['true_total']}"
            f" ({true_rate:.2%})"
            if true_rate is not None else "  True: N/A"
        )
        print(
            f"  False: {result['false_errors']}/"
            f"{result['false_total']}"
            f" ({false_rate:.2%})"
            if false_rate is not None else "  False: N/A"
        )

        if difference is not None:
            print(f"  Difference: {difference * 100:+.2f} pp")
        if lower is not None:
            print(
                f"  Bootstrap interval: "
                f"[{lower * 100:+.2f}, "
                f"{upper * 100:+.2f}] pp"
            )
        print()

    save_summary(results, args.output)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
