"""Within-plant diagnostic: does a model track day-to-day changes?

Plant-day R2 mixes two things: differences BETWEEN plants (a 2000 MW coal
plant emits far more than a 50 MW peaker, which capacity alone predicts)
and changes WITHIN a plant from one day to the next (the only thing a
satellite can add). This script removes each plant's own period mean from
both observed and predicted values and scores what is left.

For each run and test period it reports, on plants with at least
--min-days days in the period:
  - within_R2: R2 of daily anomalies (observed - plant mean) predicted by
    (predicted - plant mean). 0 = no daily skill (a constant per plant
    scores exactly 0); 1 = perfect tracking.
  - median_corr: median, across plants, of the correlation between
    observed and predicted daily values (NaN for constant predictions).

Usage:
  uv run scripts/diagnose_within.py --runs replica_v1 replica_s43 naive_history cb_satellite
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from co2sat.utils import data_dir

TARGETS = ["co2_metric_tons", "gross_load_mwh"]


def within_scores(df: pd.DataFrame, min_days: int) -> dict[str, float | int]:
    """Score daily anomalies around each plant's period mean.

    Args:
        df (pd.DataFrame): columns facility_id, yt (observed), yp (predicted)
            for one period and one target.
        min_days (int): minimum number of days for a plant to be kept.

    Returns:
        dict[str, float | int]: n_plants, within_R2 and median_corr.
    """
    counts = df.groupby("facility_id")["yt"].transform("size")
    d = df[counts >= min_days].copy()
    g = d.groupby("facility_id")
    at = d["yt"] - g["yt"].transform("mean")
    ap = d["yp"] - g["yp"].transform("mean")
    ss_tot = float((at**2).sum())
    within_r2 = 1.0 - float(((at - ap) ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan

    def corr(x: pd.DataFrame) -> float:
        if x["yp"].std() < 1e-9 or x["yt"].std() < 1e-9:
            return np.nan
        return float(np.corrcoef(x["yt"], x["yp"])[0, 1])

    corrs = pd.Series([corr(x) for _, x in g[["yt", "yp"]]], dtype=float)
    return {
        "n_plants": int(d["facility_id"].nunique()),
        "within_R2": within_r2,
        "median_corr": float(corrs.median()) if corrs.notna().any() else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--min-days", type=int, default=10)
    args = ap.parse_args()
    root = data_dir("processed", "model", "runs")

    rows = []
    for run in args.runs:
        pred = pd.read_parquet(root / run / "predictions.parquet")
        for per in ("P2", "P3", "P4"):
            sub = pred[pred["period"] == per]
            for t in TARGETS:
                df = pd.DataFrame(
                    {
                        "facility_id": sub["facility_id"].to_numpy(),
                        "yt": sub[f"{t}_true"].to_numpy(),
                        "yp": sub[f"{t}_pred"].to_numpy(),
                    }
                )
                rows.append(
                    {
                        "run": run,
                        "period": per,
                        "target": t,
                        **within_scores(df, args.min_days),
                    }
                )
    out = pd.DataFrame(rows)
    print(
        out.pivot_table(
            index=["target", "period"],
            columns="run",
            values=["within_R2", "median_corr"],
        )
        .round(3)
        .to_string()
    )
    out.to_csv(root / "within_plant_diagnostics.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
