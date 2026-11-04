"""Assemble model-ready training data: gap rule + split + z-score stats.

Applies the Sub-Phase 2.5 gap rule (NULL-aware: missing hours and NULL
values counted together per band; <=2 absent -> linear interpolation along
hours, else drop), the paper's split protocol (P1 plants 80/20 train/val,
P2-P4 entirely test), and computes normalization statistics on the training
split only (paper eq 17).

Two optional variants (both OFF by default = the replication protocol):

  --include-off-days  also keep plant-days where the plant did not run at all
                      (CO2 = 0 and gross load = 0 after summing its units).
                      The replication drops them, which removes the clearest
                      contrast a satellite could see (running vs off).
  --align-local       re-index the 24 satellite hours on the plant's LOCAL
                      standard day (the day EPA reports), instead of the UTC
                      day. The UTC offset is approximated from longitude
                      (round(-lon / 15) hours). The last day of each period
                      lacks its final 5-8 local hours (they fall on the next
                      UTC day, not extracted) and is dropped by the gap rule.

Outputs go to data/processed/<--out-name> (default 'model'). With
--reuse-split-from, training plants are taken from an existing dataset so
that variants are compared on the same train/val plants.

Usage:
  uv run scripts/build_training_data.py                                   # replication
  uv run scripts/build_training_data.py --out-name model_off --include-off-days --reuse-split-from model_base
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from loguru import logger

from co2sat.utils import data_dir

SEED = 42
PERIODS = {
    "P1": ("2021-04-01", "2021-05-19"),
    "P2": ("2021-09-01", "2021-09-30"),
    "P3": ("2022-04-01", "2022-04-29"),
    "P4": ("2022-09-01", "2022-09-28"),
}
STATIC_COLS = [
    "capacity_mw",
    "latitude",
    "longitude",
    "coal_ratio",
    "gas_ratio",
    "oil_ratio",
    "other_ratio",
    "altitude_m",
    "zenith_angle",
    "edgar_co2_surround",
    "consumption_surround",
]
BAND_COLS = [f"band_{b:02d}" for b in range(1, 17)]
KEY = ["facility_id", "date"]


def assign_period(dates: pd.Series) -> pd.Series:
    """Assign the period label to each date.

    Args:
        dates (pd.Series): datetime values.

    Returns:
        pd.Series: 'P1'..'P4', or NA for dates outside all periods.
    """
    out = pd.Series(pd.NA, index=dates.index, dtype="object")
    for name, (a, b) in PERIODS.items():
        out[dates.between(a, b)] = name
    return out


def load_labels(include_off_days: bool) -> pd.DataFrame:
    """Load the labelled plant-days of the four periods.

    Args:
        include_off_days (bool): if True, add the plant-days where the plant
            did not run (CO2 = 0 and load = 0), for the same set of plants.

    Returns:
        pd.DataFrame: facility_id, date, co2_metric_tons, gross_load_mwh,
        period and is_off (bool).
    """
    epa = pd.read_parquet(data_dir("processed", "epa_daily_with_attributes.parquet"))
    epa["date"] = pd.to_datetime(epa["date"])
    cols = ["facility_id", "date", "co2_metric_tons", "gross_load_mwh"]
    on = epa.loc[
        (epa["co2_metric_tons"] > 0) & (epa["gross_load_mwh"] > 0), cols
    ].drop_duplicates(KEY)
    on["is_off"] = False
    frames = [on]
    if include_off_days:
        allday = pd.read_parquet(data_dir("processed", "epa_daily.parquet"))
        allday["date"] = pd.to_datetime(allday["date"])
        off = allday.loc[
            allday["facility_id"].isin(on["facility_id"].unique())
            & (allday["co2_metric_tons"].fillna(0) == 0)
            & (allday["gross_load_mwh"].fillna(0) == 0),
            cols,
        ].fillna(0.0)
        off = off.drop_duplicates(KEY)
        off["is_off"] = True
        frames.append(off)
    labels = pd.concat(frames, ignore_index=True)
    labels["period"] = assign_period(labels["date"])
    labels = labels.dropna(subset=["period"]).reset_index(drop=True)
    logger.info(
        f"Labelled in-period samples: {len(labels):,} "
        f"(off days: {int(labels['is_off'].sum()):,})"
    )
    return labels


def load_dynamic(statics: pd.DataFrame, align_local: bool) -> pd.DataFrame:
    """Load hourly satellite features, optionally re-indexed on local days.

    Args:
        statics (pd.DataFrame): static features (needs facility_id and
            longitude, used for the UTC offset).
        align_local (bool): if True, convert (UTC date, UTC hour) into
            (local standard date, local hour).

    Returns:
        pd.DataFrame: facility_id, date, hour and the 16 band columns.
    """
    dyn = pd.read_parquet(data_dir("processed", "dynamic_features.parquet"))
    dyn["date"] = pd.to_datetime(dyn["date"])
    if align_local:
        off_h = (
            statics.set_index("facility_id")["longitude"]
            .mul(-1 / 15)
            .round()
            .astype(int)
        )
        utc = dyn["date"] + pd.to_timedelta(dyn["hour"], unit="h")
        local = utc - pd.to_timedelta(
            dyn["facility_id"].map(off_h).to_numpy(), unit="h"
        )
        dyn["date"] = local.dt.normalize()
        dyn["hour"] = local.dt.hour
        logger.info(
            f"Aligned on local standard days; UTC offsets used: {sorted(off_h.unique())}"
        )
    return dyn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="model")
    ap.add_argument("--include-off-days", action="store_true")
    ap.add_argument("--align-local", action="store_true")
    ap.add_argument(
        "--reuse-split-from",
        default=None,
        help="dataset folder (under data/processed) whose train plants are reused",
    )
    args = ap.parse_args()

    # ---- 1. Labels and statics ----
    labels = load_labels(args.include_off_days)
    statics = pd.read_parquet(data_dir("processed", "static_features.parquet"))

    # ---- 2. Dynamic features restricted to labelled keys ----
    dyn = load_dynamic(statics, args.align_local)
    dyn = dyn.merge(labels[KEY], on=KEY)

    # ---- 3. Gap rule, vectorized: pivot to (samples x 24) per band ----
    matrices, keep_masks = {}, []
    for bc in BAND_COLS:
        wide = dyn.pivot(index=KEY, columns="hour", values=bc)
        wide = wide.reindex(columns=range(24))
        keep_masks.append(wide.isna().sum(axis=1) <= 2)
        matrices[bc] = wide.interpolate(axis=1, limit_direction="both")
    keep = pd.concat(keep_masks, axis=1).all(axis=1)
    logger.info(f"Plant-days dropped by gap rule: {(~keep).sum()}")

    kept_index = matrices[BAND_COLS[0]].index[keep]
    X_dyn = np.stack(
        [matrices[bc].loc[kept_index].values for bc in BAND_COLS], axis=1
    ).astype(np.float32)  # (N, 16, 24)
    assert not np.isnan(X_dyn).any(), "NaN survived interpolation"

    # ---- 4. Align labels + statics ----
    samples = labels.set_index(KEY).loc[kept_index].reset_index()
    samples = samples.merge(statics, on="facility_id", how="left")
    X_stat = samples[STATIC_COLS].values.astype(np.float32)
    y = samples[["co2_metric_tons", "gross_load_mwh"]].values.astype(np.float32)
    assert not np.isnan(X_stat).any(), "NaN in static features"

    # ---- 5. Split: paper protocol (or reused from another dataset) ----
    p1_plants = np.sort(samples.loc[samples["period"] == "P1", "facility_id"].unique())
    if args.reuse_split_from:
        base = pd.read_parquet(
            data_dir("processed", args.reuse_split_from, "samples.parquet")
        )
        train_plants = set(base.loc[base["split"] == "train", "facility_id"])
        logger.info(f"Train plants reused from {args.reuse_split_from}")
    else:
        rng = np.random.default_rng(SEED)
        shuffled = rng.permutation(p1_plants)
        train_plants = set(shuffled[: int(round(0.8 * len(p1_plants)))])
    samples["split"] = "test"
    in_p1 = samples["period"] == "P1"
    samples.loc[in_p1 & samples["facility_id"].isin(train_plants), "split"] = "train"
    samples.loc[in_p1 & ~samples["facility_id"].isin(train_plants), "split"] = "val"
    logger.info(
        f"P1 plants: {len(p1_plants)} -> {len(train_plants & set(p1_plants))} train"
    )
    logger.info(f"Split counts:\n{samples['split'].value_counts()}")

    # ---- 6. Z-score stats on TRAIN only ----
    tr = samples["split"].values == "train"
    stats = {
        "dynamic": {
            "bands": BAND_COLS,
            "mean": X_dyn[tr].mean(axis=(0, 2)).tolist(),
            "std": X_dyn[tr].std(axis=(0, 2)).tolist(),
        },
        "static": {
            "cols": STATIC_COLS,
            "mean": X_stat[tr].mean(axis=0).tolist(),
            "std": X_stat[tr].std(axis=0).tolist(),
        },
        "targets": {
            "cols": ["co2_metric_tons", "gross_load_mwh"],
            "mean": y[tr].mean(axis=0).tolist(),
            "std": y[tr].std(axis=0).tolist(),
        },
        "seed": SEED,
        "n_p1_plants": int(len(p1_plants)),
        "n_train_plants": int(len(train_plants)),
        "variant": {
            "include_off_days": args.include_off_days,
            "align_local": args.align_local,
            "reuse_split_from": args.reuse_split_from,
        },
    }

    # ---- 7. Save ----
    out = data_dir("processed", args.out_name)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X_dyn.npy", X_dyn)
    np.save(out / "X_stat.npy", X_stat)
    np.save(out / "y.npy", y)
    samples[["facility_id", "date", "period", "split", "is_off"]].to_parquet(
        out / "samples.parquet", index=False
    )
    (out / "normalization_stats.json").write_text(json.dumps(stats, indent=2))
    logger.info(
        f"Saved to {out}: X_dyn {X_dyn.shape}, X_stat {X_stat.shape}, y {y.shape}"
    )


if __name__ == "__main__":
    main()
