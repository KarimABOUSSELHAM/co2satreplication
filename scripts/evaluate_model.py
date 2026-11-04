"""Phase 4: evaluate the trained replica (paper Section 2.3.2 metrics).

Levels, per the paper:
  - plant:    one sample point = one plant-day
  - state:    daily totals per state (state joined from
              epa_daily_with_attributes.parquet)
  - national: daily totals over all plants

Metrics R2, RMSE, MAPE per target (co2_metric_tons, gross_load_mwh), per
evaluation set (P1-val, P2, P3, P4), after de-normalization to physical
units with the stats of normalization_stats.json.

Usage:
  uv run scripts/evaluate_model.py --run-name replica_v1
  uv run scripts/evaluate_model.py --run-name replica_v1 --dump-predictions
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from loguru import logger

from co2sat.model import CNNBiLSTMAttention
from co2sat.utils import data_dir

TARGETS = ["co2_metric_tons", "gross_load_mwh"]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Compute point-accuracy metrics in physical units.

    Args:
        y_true (np.ndarray): observed values.
        y_pred (np.ndarray): predicted values.

    Returns:
        dict[str, float | int]: R2, RMSE, MAE, MAPE (fraction, on non-zero
        observations), WAPE, bias (mean of pred - true), relative bias
        (sum of errors / sum of observations) and the sample count n.
    """
    resid = y_true - y_pred
    ss_res = float((resid**2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    nz = np.abs(y_true) > 1e-9
    denom = float(np.abs(y_true).sum())
    return {
        "R2": r2,
        "RMSE": float(np.sqrt((resid**2).mean())),
        "MAE": float(np.abs(resid).mean()),
        "MAPE": float(np.abs(resid[nz] / y_true[nz]).mean())
        if nz.any()
        else float("nan"),
        "WAPE": float(np.abs(resid).sum()) / denom if denom > 0 else float("nan"),
        "bias": float((y_pred - y_true).mean()),
        "rel_bias": float((y_pred - y_true).sum()) / float(y_true.sum())
        if y_true.sum() != 0
        else float("nan"),
        "n": int(len(y_true)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="replica_v1")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--dump-predictions", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdir = data_dir("processed", "model")
    run_dir = mdir / "runs" / args.run_name
    cfg = json.loads((run_dir / "run_config.json").read_text())
    stats = json.loads((mdir / "normalization_stats.json").read_text())

    X_dyn = np.load(mdir / "X_dyn.npy").astype(np.float32)
    X_stat = np.load(mdir / "X_stat.npy").astype(np.float32)
    y_raw = np.load(mdir / "y.npy").astype(np.float64)
    samples = pd.read_parquet(mdir / "samples.parquet")

    d_mean = np.asarray(stats["dynamic"]["mean"], dtype=np.float32)
    d_std = np.asarray(stats["dynamic"]["std"], dtype=np.float32)
    s_mean = np.asarray(stats["static"]["mean"], dtype=np.float32)
    s_std = np.asarray(stats["static"]["std"], dtype=np.float32)
    t_mean = np.asarray(stats["targets"]["mean"], dtype=np.float64)
    t_std = np.asarray(stats["targets"]["std"], dtype=np.float64)

    band_idx = (
        [int(b) - 1 for b in cfg["bands_1based"].split(",")]
        if cfg.get("bands_1based")
        else None
    )
    if band_idx is not None:
        X_dyn = X_dyn[:, band_idx, :]
        d_mean, d_std = d_mean[band_idx], d_std[band_idx]
    X_dyn = (X_dyn - d_mean[None, :, None]) / d_std[None, :, None]
    X_stat = (X_stat - s_mean[None, :]) / s_std[None, :]

    model = CNNBiLSTMAttention(n_bands=X_dyn.shape[1]).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()

    preds_z = np.empty_like(y_raw)
    with torch.no_grad():
        for lo in range(0, len(y_raw), args.batch_size):
            hi = min(lo + args.batch_size, len(y_raw))
            xd = torch.from_numpy(X_dyn[lo:hi]).to(device)
            xs = torch.from_numpy(X_stat[lo:hi]).to(device)
            preds_z[lo:hi] = model(xd, xs).cpu().numpy()
    p_phys = preds_z * t_std[None, :] + t_mean[None, :]  # de-normalize

    # ---- state join (from the attributes file; samples has no state col) ----
    epa = pd.read_parquet(
        data_dir("processed", "epa_daily_with_attributes.parquet"),
        columns=["facility_id", "state"],
    ).drop_duplicates("facility_id")
    samples = samples.merge(epa, on="facility_id", how="left")
    if samples["state"].isna().any():
        logger.warning(f"{samples['state'].isna().sum()} samples without state")

    if args.dump_predictions:
        dump = samples[["facility_id", "date", "period", "split", "state"]].copy()
        for ti, t in enumerate(TARGETS):
            dump[f"{t}_true"] = y_raw[:, ti]
            dump[f"{t}_pred"] = p_phys[:, ti]
        dump.to_parquet(run_dir / "predictions.parquet", index=False)
        logger.info(f"Dumped -> {run_dir / 'predictions.parquet'}")

    eval_sets: dict[str, np.ndarray] = {
        "P1-val": (samples["split"] == "val").to_numpy()
    }
    for per in ("P2", "P3", "P4"):
        eval_sets[per] = (samples["period"] == per).to_numpy()

    rows = []
    for name, mask in eval_sets.items():
        idx = np.flatnonzero(mask)
        sub = samples.iloc[idx]
        for ti, tname in enumerate(TARGETS):
            yt, yp = y_raw[idx, ti], p_phys[idx, ti]
            rows.append(
                {"set": name, "target": tname, "level": "plant", **metrics(yt, yp)}
            )
            df = pd.DataFrame(
                {
                    "date": sub["date"].values,
                    "state": sub["state"].values,
                    "yt": yt,
                    "yp": yp,
                }
            )
            df["facility_id"] = sub["facility_id"].values
            pm = df.groupby("facility_id", observed=True)[["yt", "yp"]].mean()
            rows.append(
                {
                    "set": name,
                    "target": tname,
                    "level": "plant_mean",
                    **metrics(pm["yt"].to_numpy(), pm["yp"].to_numpy()),
                }
            )
            g = df.groupby(["date", "state"], observed=True)[["yt", "yp"]].sum()
            rows.append(
                {
                    "set": name,
                    "target": tname,
                    "level": "state",
                    **metrics(g["yt"].to_numpy(), g["yp"].to_numpy()),
                }
            )
            n = df.groupby("date", observed=True)[["yt", "yp"]].sum()
            rows.append(
                {
                    "set": name,
                    "target": tname,
                    "level": "national",
                    **metrics(n["yt"].to_numpy(), n["yp"].to_numpy()),
                }
            )

    out = pd.DataFrame(rows)
    out_path = run_dir / "results_metrics.csv"
    out.to_csv(out_path, index=False, float_format="%.4f")
    print(out.to_string(index=False))
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
