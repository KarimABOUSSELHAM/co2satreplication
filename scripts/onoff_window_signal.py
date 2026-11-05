"""On/off test on WINDOW features: does the plant's surroundings react?

Same logic and controls as onoff_raw_signal.py (effect size d of running vs
off days, chance levels SHUF and CIRC), applied to contrast features built
from the 5 x 5 window statistics of extract_goes16_window.py:

  c          plant pixel (the Phase 2 / Mo et al. input, for reference)
  c-ring     plant pixel minus its border: a local hot or bright spot
  mx-m5      window maximum minus window mean: a bright or warm feature
             anywhere in the window (a plume drifting downwind)
  m5-mn      window mean minus window minimum: a dark or cold feature
             anywhere in the window (a cold plume top in the IR)
  sd5        window standard deviation: any local structure

On/off labels come from a dataset built with --include-off-days (default
model_off), joined on (facility_id, date) as in the raw test.

Usage:
  uv run scripts/onoff_window_signal.py
  uv run scripts/onoff_window_signal.py --bands 2 3 7 13 14 --big-mw 1000
"""

from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from loguru import logger

from co2sat.utils import data_dir

CONTRASTS = ["c", "c-ring", "mx-m5", "m5-mn", "sd5"]
warnings.filterwarnings("ignore", category=RuntimeWarning)


def build_features(
    win: pd.DataFrame, bands: list[int]
) -> tuple[pd.DataFrame, list[str]]:
    """Derive the contrast features from the window statistics.

    Args:
        win (pd.DataFrame): window statistics, one row per plant-scan.
        bands (list[int]): ABI bands to use.

    Returns:
        tuple[pd.DataFrame, list[str]]: facility_id, date, hour and the
        feature columns; the list of feature names.
    """
    feats = {}
    for b in bands:
        p = f"b{b:02d}_"
        feats[f"B{b:02d} c"] = win[p + "c"]
        feats[f"B{b:02d} c-ring"] = win[p + "c"] - win[p + "ring"]
        feats[f"B{b:02d} mx-m5"] = win[p + "mx5"] - win[p + "m5"]
        feats[f"B{b:02d} m5-mn"] = win[p + "m5"] - win[p + "mn5"]
        feats[f"B{b:02d} sd5"] = win[p + "sd5"]
    df = pd.concat([win[["facility_id", "date", "hour"]], pd.DataFrame(feats)], axis=1)
    return df, list(feats)


def effect_sizes(x: np.ndarray, on: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware standardized on-minus-off differences for one plant-period.

    Args:
        x (np.ndarray): features, shape (n_days, n_features, 24), NaN allowed.
        on (np.ndarray): boolean, True for running days.

    Returns:
        tuple[np.ndarray, np.ndarray]: per feature, the effect size on the
        daily mean and at the hour with the largest |d|.
    """

    def d(v: np.ndarray) -> np.ndarray:
        a, b = v[on], v[~on]
        na = np.sum(~np.isnan(a), axis=0)
        nb = np.sum(~np.isnan(b), axis=0)
        with np.errstate(all="ignore"):
            pooled = np.sqrt(
                (
                    (na - 1) * np.nanvar(a, axis=0, ddof=1)
                    + (nb - 1) * np.nanvar(b, axis=0, ddof=1)
                )
                / (na + nb - 2)
            )
            res = (np.nanmean(a, axis=0) - np.nanmean(b, axis=0)) / pooled
        res[(na < 2) | (nb < 2) | ~np.isfinite(res)] = np.nan
        return res

    with np.errstate(all="ignore"):
        daily = np.nanmean(x, axis=2)
    d_daily = d(daily)
    d_hourly = d(x)
    filled = np.where(np.isnan(d_hourly), 0.0, d_hourly)
    idx = np.abs(filled).argmax(axis=1)
    d_best = d_hourly[np.arange(d_hourly.shape[0]), idx]
    return d_daily, d_best


def summarize(rows: list[dict], label: str) -> pd.DataFrame:
    """Summarize effect sizes per feature.

    Args:
        rows (list[dict]): one dict per plant-period and feature.
        label (str): subset name.

    Returns:
        pd.DataFrame: per feature, n, med_d, med|d|, SHUF, CIRC, and the
        best-hour med|d| with its CIRC.
    """
    df = pd.DataFrame(rows)
    g = df.groupby("feature", sort=False)

    def med_abs(col: str) -> pd.Series:
        return g[col].apply(lambda s: s.abs().median())

    out = pd.DataFrame(
        {
            "n": g["d_daily"].count(),
            "med_d": g["d_daily"].median(),
            "med|d|": med_abs("d_daily"),
            "SHUF": med_abs("null_daily"),
            "CIRC": med_abs("circ_daily"),
            "hour_med|d|": med_abs("d_hour"),
            "hour_CIRC": med_abs("circ_hour"),
        }
    )
    out["excess_vs_CIRC"] = out["med|d|"] - out["CIRC"]
    print(f"\n=== {label} ===")
    print(out.round(2).to_string())
    return out.assign(subset=label)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-from", default="model_off")
    ap.add_argument("--bands", type=int, nargs="+", default=[2, 3, 7, 13, 14])
    ap.add_argument("--min-days", type=int, default=3)
    ap.add_argument("--big-mw", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    win = pd.read_parquet(data_dir("processed", "window_features.parquet"))
    win["date"] = pd.to_datetime(win["date"])
    feat_df, feat_names = build_features(win, args.bands)

    ldir = data_dir("processed", args.labels_from)
    samples = pd.read_parquet(ldir / "samples.parquet")
    samples["date"] = pd.to_datetime(samples["date"])
    x_stat = np.load(ldir / "X_stat.npy")
    cols = json.loads((ldir / "normalization_stats.json").read_text())["static"]["cols"]
    samples["capacity_mw"] = x_stat[:, cols.index("capacity_mw")]
    samples["coal_ratio"] = x_stat[:, cols.index("coal_ratio")]
    lab = samples[
        ["facility_id", "date", "period", "is_off", "capacity_mw", "coal_ratio"]
    ]

    data = feat_df.merge(lab, on=["facility_id", "date"])
    logger.info(f"Joined rows: {len(data):,} | plants: {data['facility_id'].nunique()}")

    rng = np.random.default_rng(args.seed)
    subsets: dict[str, list[dict]] = {
        "ALL": [],
        "COAL": [],
        f">= {args.big_mw:.0f} MW": [],
    }
    n_pp = 0
    for (_, _), grp in data.groupby(["facility_id", "period"]):
        days = grp[["date", "is_off"]].drop_duplicates("date").sort_values("date")
        on = ~days["is_off"].to_numpy()
        if on.sum() < args.min_days or (~on).sum() < args.min_days:
            continue
        n_pp += 1
        cube = np.full((len(days), len(feat_names), 24), np.nan, dtype=np.float32)
        pos = {d: i for i, d in enumerate(days["date"])}
        di = grp["date"].map(pos).to_numpy()
        hi = grp["hour"].to_numpy()
        cube[di, :, hi] = grp[feat_names].to_numpy()
        d_daily, d_hour = effect_sizes(cube, on)
        n_daily, _ = effect_sizes(cube, rng.permutation(on))
        c_daily, c_hour = effect_sizes(cube, np.roll(on, int(rng.integers(1, len(on)))))
        recs = [
            {
                "feature": f,
                "d_daily": d_daily[k],
                "d_hour": d_hour[k],
                "null_daily": n_daily[k],
                "circ_daily": c_daily[k],
                "circ_hour": c_hour[k],
            }
            for k, f in enumerate(feat_names)
        ]
        subsets["ALL"] += recs
        if grp["coal_ratio"].iloc[0] >= 0.5:
            subsets["COAL"] += recs
        if grp["capacity_mw"].iloc[0] >= args.big_mw:
            subsets[f">= {args.big_mw:.0f} MW"] += recs

    logger.info(f"Plant-periods with >= {args.min_days} running and off days: {n_pp}")
    tables = [summarize(r, k) for k, r in subsets.items() if r]
    out = pd.concat(tables)
    out.to_csv(data_dir("processed", "onoff_window_signal.csv"), float_format="%.4f")
    logger.info("Saved -> data/processed/onoff_window_signal.csv")


if __name__ == "__main__":
    main()
