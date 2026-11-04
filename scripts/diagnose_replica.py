"""Phase 4 diagnostics on a trained replica (no retraining).

Reads predictions.parquet (written by evaluate_model.py --dump-predictions)
and reports, per test period:
  1. plant-level metrics on the period mean of each plant (Mo et al.
     definition) for growing emission thresholds, to find the plant
     universe comparable to the paper (590 / 513 / 592 plants);
  2. the same metrics split between plants seen during training and
     plants never seen.

Usage:
  uv run scripts/diagnose_replica.py --run-name replica_v1
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from loguru import logger

from co2sat.utils import data_dir

THRESHOLDS_T_PER_DAY = [0, 10, 50, 100, 250, 500, 1000, 2000, 3000]
PAPER_N_PLANTS = {"P2": 590, "P3": 513, "P4": 592}


def plant_mean_metrics(pm: pd.DataFrame) -> dict[str, float | int]:
    """Compute paper-style metrics on per-plant period means.

    Args:
        pm (pd.DataFrame): one row per plant with columns 'yt' (observed
            mean) and 'yp' (predicted mean).

    Returns:
        dict[str, float | int]: n plants, R2, MAPE (fraction), WAPE and
        relative bias; NaN when fewer than two plants remain.
    """
    yt, yp = pm["yt"].to_numpy(), pm["yp"].to_numpy()
    n = len(yt)
    if n < 2:
        return {
            "n_plants": n,
            "R2": np.nan,
            "MAPE": np.nan,
            "WAPE": np.nan,
            "rel_bias": np.nan,
        }
    resid = yt - yp
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    nz = np.abs(yt) > 1e-9
    return {
        "n_plants": n,
        "R2": 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else np.nan,
        "MAPE": float(np.abs(resid[nz] / yt[nz]).mean()),
        "WAPE": float(np.abs(resid).sum() / np.abs(yt).sum()),
        "rel_bias": float((yp - yt).sum() / yt.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="replica_v1")
    args = ap.parse_args()

    run_dir = data_dir("processed", "model", "runs", args.run_name)
    pred = pd.read_parquet(run_dir / "predictions.parquet")
    seen = set(pred.loc[pred["split"] == "train", "facility_id"])
    logger.info(f"{len(seen)} plants seen during training")

    rows = []
    for per in ("P2", "P3", "P4"):
        sub = pred[pred["period"] == per]
        pm = sub.groupby("facility_id").agg(
            yt=("co2_metric_tons_true", "mean"), yp=("co2_metric_tons_pred", "mean")
        )
        pm["seen"] = pm.index.isin(seen)
        # 1. universe sensitivity: keep plants above a mean-emission threshold
        for thr in THRESHOLDS_T_PER_DAY:
            kept = pm[pm["yt"] >= thr]
            rows.append(
                {"period": per, "group": f"mean_co2>={thr}", **plant_mean_metrics(kept)}
            )
        # 2. seen vs never-seen plants (all plants)
        for flag, name in ((True, "seen"), (False, "never_seen")):
            rows.append(
                {
                    "period": per,
                    "group": name,
                    **plant_mean_metrics(pm[pm["seen"] == flag]),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(run_dir / "diagnostics.csv", index=False, float_format="%.4f")
    print(out.to_string(index=False))
    print(f"\nPaper plant counts for reference: {PAPER_N_PLANTS}")
    logger.info(f"Saved -> {run_dir / 'diagnostics.csv'}")


if __name__ == "__main__":
    main()
