"""Assemble model-ready training data: gap rule + split + z-score stats.

Applies the Sub-Phase 2.5 gap rule (NULL-aware: missing hours and NULL
values counted together per band; <=2 absent -> linear interpolation along
hours, else drop), the paper's split protocol (P1 plants 80/20 train/val,
P2-P4 entirely test), and computes normalization statistics on the training
split only (paper eq 17).
"""

from __future__ import annotations

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
    """Assign period label to each date, or NaN if outside all periods."""
    out = pd.Series(pd.NA, index=dates.index, dtype="object")
    for name, (a, b) in PERIODS.items():
        out[dates.between(a, b)] = name
    return out


def main() -> None:
    # ---- 1. Labeled samples in-period ----
    epa = pd.read_parquet(data_dir("processed", "epa_daily_with_attributes.parquet"))
    epa["date"] = pd.to_datetime(epa["date"])
    labels = (
        epa.loc[
            (epa["co2_metric_tons"] > 0) & (epa["gross_load_mwh"] > 0),
            ["facility_id", "date", "co2_metric_tons", "gross_load_mwh"],
        ]
        .drop_duplicates(KEY)
        .copy()
    )
    labels["period"] = assign_period(labels["date"])
    labels = labels.dropna(subset=["period"]).reset_index(drop=True)
    logger.info(f"Labeled in-period samples: {len(labels)}")  # expect ~90,261

    # ---- 2. Dynamic features restricted to labeled keys ----
    dyn = pd.read_parquet(data_dir("processed", "dynamic_features.parquet"))
    dyn["date"] = pd.to_datetime(dyn["date"])
    dyn = dyn.merge(labels[KEY], on=KEY)

    # ---- 3. Gap rule, vectorized: pivot to (samples x 24) per band ----
    # Missing hours become absent columns after reindex; NULL values stay NaN.
    # Both are counted identically as 'absent' -> the 2.5 rule, NULL-aware.
    matrices, keep_masks = {}, []
    for bc in BAND_COLS:
        wide = dyn.pivot(index=KEY, columns="hour", values=bc)
        wide = wide.reindex(columns=range(24))
        keep_masks.append(wide.isna().sum(axis=1) <= 2)
        matrices[bc] = wide.interpolate(axis=1, limit_direction="both")
    keep = pd.concat(keep_masks, axis=1).all(axis=1)
    logger.info(f"Plant-days dropped by gap rule: {(~keep).sum()}")  # expect ~1000-1100

    kept_index = matrices[BAND_COLS[0]].index[keep]
    X_dyn = np.stack(
        [matrices[bc].loc[kept_index].values for bc in BAND_COLS], axis=1
    ).astype(np.float32)  # (N, 16, 24)
    assert not np.isnan(X_dyn).any(), "NaN survived interpolation"

    # ---- 4. Align labels + statics ----
    samples = labels.set_index(KEY).loc[kept_index].reset_index()
    statics = pd.read_parquet(data_dir("processed", "static_features.parquet"))
    samples = samples.merge(statics, on="facility_id", how="left")
    X_stat = samples[STATIC_COLS].values.astype(np.float32)
    y = samples[["co2_metric_tons", "gross_load_mwh"]].values.astype(np.float32)
    assert not np.isnan(X_stat).any(), "NaN in static features"

    # ---- 5. Split: paper protocol ----
    rng = np.random.default_rng(SEED)
    p1_plants = np.sort(samples.loc[samples["period"] == "P1", "facility_id"].unique())
    shuffled = rng.permutation(p1_plants)
    train_plants = set(shuffled[: int(round(0.8 * len(p1_plants)))])
    samples["split"] = "test"
    in_p1 = samples["period"] == "P1"
    samples.loc[in_p1 & samples["facility_id"].isin(train_plants), "split"] = "train"
    samples.loc[in_p1 & ~samples["facility_id"].isin(train_plants), "split"] = "val"
    logger.info(f"P1 plants: {len(p1_plants)} -> {len(train_plants)} train")
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
    }

    # ---- 7. Save ----
    out = data_dir("processed", "model")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X_dyn.npy", X_dyn)
    np.save(out / "X_stat.npy", X_stat)
    np.save(out / "y.npy", y)
    samples[["facility_id", "date", "period", "split"]].to_parquet(
        out / "samples.parquet", index=False
    )
    (out / "normalization_stats.json").write_text(json.dumps(stats, indent=2))
    logger.info(f"Saved: X_dyn {X_dyn.shape}, X_stat {X_stat.shape}, y {y.shape}")


if __name__ == "__main__":
    main()
