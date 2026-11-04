"""Summarize metrics across one or several runs (mean, std, min, max).

Reads results_metrics.csv and diagnostics.csv of each run and prints, for
the paper-comparable rows, the spread across runs (e.g. training seeds).
With a single run, std is empty: the table then simply shows its values.

Usage:
  uv run scripts/summarize_seeds.py --runs replica_v1 replica_s43 replica_s44 replica_s45 replica_s46
  uv run scripts/summarize_seeds.py --runs cb_satellite
"""

from __future__ import annotations

import argparse

import pandas as pd

from co2sat.utils import data_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()
    root = data_dir("processed", "model", "runs")

    res = pd.concat(
        [pd.read_csv(root / r / "results_metrics.csv").assign(run=r) for r in args.runs]
    )
    plant = res[(res["level"].isin(["plant", "national"])) & (res["set"] != "P1-val")]
    print("=== Daily R2 (plant) and national relative bias ===")
    print(
        plant.groupby(["set", "target", "level"])[["R2", "rel_bias"]]
        .agg(["mean", "std", "min", "max"])
        .round(3)
        .to_string()
    )

    diag = pd.concat(
        [pd.read_csv(root / r / "diagnostics.csv").assign(run=r) for r in args.runs]
    )
    matched = diag[diag["group"].isin(["mean_co2>=3000", "seen", "never_seen"])]
    print("\n=== CO2 plant-mean metrics (matched universe, seen/never seen) ===")
    print(
        matched.groupby(["period", "group"])[["R2", "MAPE", "WAPE", "rel_bias"]]
        .agg(["mean", "std"])
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
