"""Command-line entry point.

Examples:
    python -m pipeline.run_pipeline --quick
    python -m pipeline.run_pipeline
"""

import argparse
from dataclasses import replace

import pandas as pd

from pipeline.config import DEFAULT_CONFIG
from pipeline.workflow import run_training_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, tune and evaluate the customer-churn models."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke run with 1 random configuration and 2 CV folds.",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip the six baseline/ablation models.",
    )
    parser.add_argument(
        "--search-n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs used by RandomizedSearchCV (default: -1).",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="RandomizedSearchCV verbosity.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = (
        replace(
            DEFAULT_CONFIG,
            tuning_iterations=1,
            cv_splits=2,
        )
        if args.quick
        else DEFAULT_CONFIG
    )
    run = run_training_workflow(
        config,
        include_baselines=not args.skip_baselines,
        search_n_jobs=args.search_n_jobs,
        verbose=args.verbose,
    )

    pd.set_option("display.max_columns", 30)
    print("\nTuning summary")
    print(
        run.tuning.summary[
            ["Model", "CV PR-AUC mean", "CV PR-AUC std"]
        ].to_string(index=False)
    )
    print(f"\nSelected model: {run.selection.selected_model_name}")
    print(
        f"Operating threshold: "
        f"{run.selection.operating_threshold:.2f}"
    )
    print("\nExternal test")
    print(run.external.results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

