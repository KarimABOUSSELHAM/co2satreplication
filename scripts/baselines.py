"""Phase 4, step 2: simple comparison models for the replica.

Four baselines, trained on the same split as the deep model (P1 train
plants; early stopping on P1 val plants; P2-P4 as test):

  naive_fuel_cf   capacity x mean daily output per MW of capacity of the
                  plant's dominant fuel, learned on the train split.
                  No satellite, no plant history.
  naive_history   the plant's own mean over P1 (train and val plants),
                  carried forward to P2-P4; plants without P1 history fall
                  back to naive_fuel_cf. Needs past measurements, so it is
                  an upper reference that does NOT exist in Morocco.
  cb_static       CatBoost on the 11 static features only (no satellite).
  cb_satellite    CatBoost on the 11 static features + 4 daily statistics
                  (mean, std, min, max) of each of the 16 bands.

Each baseline is written as a run folder (data/processed/model/runs/<name>)
with the same files as evaluate_model.py (predictions.parquet,
results_metrics.csv), so diagnose_replica.py and summarize_seeds.py work
on baselines unchanged.

Usage:
  uv run scripts/baselines.py
  uv run scripts/baselines.py --seed 43 --suffix _s43
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from loguru import logger

from co2sat.utils import data_dir

TARGETS = ["co2_metric_tons", "gross_load_mwh"]
FUEL_COLS = ["coal_ratio", "gas_ratio", "oil_ratio", "other_ratio"]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Compute point-accuracy metrics in physical units.

    Args:
        y_true (np.ndarray): observed values.
        y_pred (np.ndarray): predicted values.

    Returns:
        dict[str, float | int]: R2, RMSE, MAE, MAPE (fraction), WAPE, bias,
        relative bias and n, with the same definitions as evaluate_model.py.
    """
    resid = y_true - y_pred
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    nz = np.abs(y_true) > 1e-9
    denom = float(np.abs(y_true).sum())
    total = float(y_true.sum())
    return {
        "R2": 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan"),
        "RMSE": float(np.sqrt((resid**2).mean())),
        "MAE": float(np.abs(resid).mean()),
        "MAPE": float(np.abs(resid[nz] / y_true[nz]).mean())
        if nz.any()
        else float("nan"),
        "WAPE": float(np.abs(resid).sum()) / denom if denom > 0 else float("nan"),
        "bias": float((y_pred - y_true).mean()),
        "rel_bias": float((y_pred - y_true).sum()) / total
        if total != 0
        else float("nan"),
        "n": int(len(y_true)),
    }


def band_statistics(x_dyn: np.ndarray, band_names: list[str]) -> pd.DataFrame:
    """Summarize each band's 24 hourly values into four daily statistics.

    Args:
        x_dyn (np.ndarray): raw dynamic features, shape (N, 16, 24).
        band_names (list[str]): names of the 16 bands, in array order.

    Returns:
        pd.DataFrame: N rows, 64 columns '<band>_{mean,std,min,max}'.
    """
    feats = {}
    for i, b in enumerate(band_names):
        feats[f"{b}_mean"] = x_dyn[:, i, :].mean(axis=1)
        feats[f"{b}_std"] = x_dyn[:, i, :].std(axis=1)
        feats[f"{b}_min"] = x_dyn[:, i, :].min(axis=1)
        feats[f"{b}_max"] = x_dyn[:, i, :].max(axis=1)
    return pd.DataFrame(feats)


def fit_fuel_cf(stat: pd.DataFrame, y: np.ndarray, train: np.ndarray) -> pd.DataFrame:
    """Learn the mean daily output per MW of capacity for each dominant fuel.

    Args:
        stat (pd.DataFrame): static features (needs capacity_mw and the fuel
            ratio columns).
        y (np.ndarray): raw targets, shape (N, 2).
        train (np.ndarray): boolean mask of training rows.

    Returns:
        pd.DataFrame: one row per dominant fuel, one column per target, value
        = sum(target) / sum(capacity) over training rows of that fuel.
    """
    fuel = stat[FUEL_COLS].idxmax(axis=1)
    df = pd.DataFrame(y, columns=TARGETS)
    df["capacity_mw"] = stat["capacity_mw"].to_numpy()
    df["fuel"] = fuel.to_numpy()
    g = df[train].groupby("fuel")[["capacity_mw", *TARGETS]].sum()
    return g[TARGETS].div(g["capacity_mw"], axis=0)


def predict_fuel_cf(stat: pd.DataFrame, rates: pd.DataFrame) -> np.ndarray:
    """Apply the per-fuel output rates to every row.

    Args:
        stat (pd.DataFrame): static features.
        rates (pd.DataFrame): output of fit_fuel_cf.

    Returns:
        np.ndarray: predictions, shape (N, 2). Fuels absent from training
        get the overall mean rate.
    """
    fuel = stat[FUEL_COLS].idxmax(axis=1)
    r = rates.reindex(fuel.to_numpy()).fillna(rates.mean())
    return r.to_numpy() * stat["capacity_mw"].to_numpy()[:, None]


def fit_catboost(
    x: pd.DataFrame, y: np.ndarray, train: np.ndarray, val: np.ndarray, seed: int
) -> np.ndarray:
    """Train one CatBoost regressor per target and predict every row.

    Args:
        x (pd.DataFrame): feature table, N rows.
        y (np.ndarray): raw targets, shape (N, 2).
        train (np.ndarray): boolean mask of training rows.
        val (np.ndarray): boolean mask of validation rows (early stopping).
        seed (int): random seed.

    Returns:
        np.ndarray: predictions for all N rows, shape (N, 2).
    """
    out = np.empty_like(y, dtype=np.float64)
    for ti, t in enumerate(TARGETS):
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=3000,
            learning_rate=0.05,
            depth=6,
            random_seed=seed,
            od_type="Iter",
            od_wait=100,
            verbose=False,
        )
        model.fit(
            x[train], y[train, ti], eval_set=(x[val], y[val, ti]), use_best_model=True
        )
        logger.info(f"  {t}: best iteration {model.get_best_iteration()}")
        out[:, ti] = model.predict(x)
    return out


def evaluate(
    name: str, pred: np.ndarray, y: np.ndarray, samples: pd.DataFrame, meta: dict
) -> None:
    """Write predictions and metrics for one baseline as a run folder.

    Args:
        name (str): run name (folder under data/processed/model/runs).
        pred (np.ndarray): predictions, shape (N, 2).
        y (np.ndarray): raw targets, shape (N, 2).
        samples (pd.DataFrame): sample table with facility_id, date, period,
            split and state columns.
        meta (dict): description stored in run_config.json.
    """
    run_dir = data_dir("processed", "model", "runs", name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(meta, indent=2))

    dump = samples[["facility_id", "date", "period", "split", "state"]].copy()
    for ti, t in enumerate(TARGETS):
        dump[f"{t}_true"] = y[:, ti]
        dump[f"{t}_pred"] = pred[:, ti]
    dump.to_parquet(run_dir / "predictions.parquet", index=False)

    sets = {"P1-val": (samples["split"] == "val").to_numpy()}
    for per in ("P2", "P3", "P4"):
        sets[per] = (samples["period"] == per).to_numpy()

    rows = []
    for set_name, mask in sets.items():
        sub = dump[mask]
        for t in TARGETS:
            df = sub[["facility_id", "date", "state"]].assign(
                yt=sub[f"{t}_true"].to_numpy(), yp=sub[f"{t}_pred"].to_numpy()
            )
            levels = {
                "plant": df,
                "plant_mean": df.groupby("facility_id")[["yt", "yp"]].mean(),
                "state": df.groupby(["date", "state"])[["yt", "yp"]].sum(),
                "national": df.groupby("date")[["yt", "yp"]].sum(),
            }
            for lvl, d in levels.items():
                rows.append(
                    {
                        "set": set_name,
                        "target": t,
                        "level": lvl,
                        **metrics(d["yt"].to_numpy(), d["yp"].to_numpy()),
                    }
                )
    res = pd.DataFrame(rows)
    res.to_csv(run_dir / "results_metrics.csv", index=False, float_format="%.4f")
    show = res[
        (res["target"] == "co2_metric_tons")
        & (res["level"].isin(["plant", "national"]))
    ]
    logger.info(
        f"{name} (CO2):\n{show[['set', 'level', 'R2', 'WAPE', 'rel_bias']].to_string(index=False)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suffix", default="", help="appended to run names, e.g. _s43")
    args = ap.parse_args()

    mdir = data_dir("processed", "model")
    x_dyn = np.load(mdir / "X_dyn.npy").astype(np.float32)
    x_stat = np.load(mdir / "X_stat.npy").astype(np.float32)
    y = np.load(mdir / "y.npy").astype(np.float64)
    samples = pd.read_parquet(mdir / "samples.parquet")
    stats = json.loads((mdir / "normalization_stats.json").read_text())
    stat = pd.DataFrame(x_stat, columns=stats["static"]["cols"])

    epa = pd.read_parquet(
        data_dir("processed", "epa_daily_with_attributes.parquet"),
        columns=["facility_id", "state"],
    ).drop_duplicates("facility_id")
    samples = samples.merge(epa, on="facility_id", how="left")

    train = (samples["split"] == "train").to_numpy()
    val = (samples["split"] == "val").to_numpy()
    p1 = (samples["period"] == "P1").to_numpy()
    logger.info(f"train {train.sum():,} | val {val.sum():,} | all {len(samples):,}")

    # 1. naive_fuel_cf
    rates = fit_fuel_cf(stat, y, train)
    logger.info(f"Daily output per MW of capacity, by dominant fuel:\n{rates.round(3)}")
    pred_cf = predict_fuel_cf(stat, rates)
    evaluate(
        "naive_fuel_cf" + args.suffix,
        pred_cf,
        y,
        samples,
        {
            "model": "naive_fuel_cf",
            "uses_satellite": False,
            "uses_plant_history": False,
        },
    )

    # 2. naive_history (P1 mean of each plant, fallback to fuel rate)
    hist = (
        pd.DataFrame(y[p1], columns=TARGETS)
        .assign(facility_id=samples.loc[p1, "facility_id"].to_numpy())
        .groupby("facility_id")[TARGETS]
        .mean()
    )
    pred_hist = hist.reindex(samples["facility_id"]).to_numpy().copy()
    no_hist = np.isnan(pred_hist[:, 0])
    pred_hist[no_hist] = pred_cf[no_hist]
    logger.info(
        f"naive_history: {no_hist.mean():.1%} of samples without P1 history (fallback)"
    )
    evaluate(
        "naive_history" + args.suffix,
        pred_hist,
        y,
        samples,
        {
            "model": "naive_history",
            "uses_satellite": False,
            "uses_plant_history": True,
            "share_fallback": float(no_hist.mean()),
        },
    )

    # 3. cb_static
    pred_cbs = fit_catboost(stat, y, train, val, args.seed)
    evaluate(
        "cb_static" + args.suffix,
        pred_cbs,
        y,
        samples,
        {"model": "catboost", "features": "11 static", "seed": args.seed},
    )

    # 4. cb_satellite
    x_full = pd.concat(
        [stat, band_statistics(x_dyn, stats["dynamic"]["bands"])], axis=1
    )
    pred_cbx = fit_catboost(x_full, y, train, val, args.seed)
    evaluate(
        "cb_satellite" + args.suffix,
        pred_cbx,
        y,
        samples,
        {
            "model": "catboost",
            "features": "11 static + 64 band statistics",
            "seed": args.seed,
        },
    )

    logger.info(
        "Done. Run diagnose_replica.py on each baseline for the matched-universe MAPE."
    )


if __name__ == "__main__":
    main()
