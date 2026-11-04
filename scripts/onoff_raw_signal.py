"""Raw-signal positive control: do satellite bands differ when a plant is OFF?

No model involved. For every (plant, period) with enough running days AND
enough off days, it compares the raw value of each band on running days vs
off days, as a standardized effect size:

    d = (mean_on - mean_off) / pooled within-plant standard deviation

computed on (a) the band's daily mean over the 24 hours and (b) the single
hour where the two groups differ most. Effect sizes are then summarized per
band across plant-periods, for all plants and for large plants only.

Two chance controls are computed the same way:
  _SHUF  on/off labels randomly permuted inside each plant-period;
  _CIRC  on/off sequence circularly shifted in time by a random lag. This
         keeps the runs of consecutive off days intact, so it also captures
         differences that come from the weather of those runs (off weeks
         that happen to be cooler, cloudier...). It is the stricter control.
A real plant signal must clearly exceed _CIRC, and be consistent in sign
(median_d clearly away from 0).

Reading guide: |d| around 0.2 is small, 0.5 medium, 0.8 large. The
'best hour' figures are inflated by picking the maximum of 24 hours: only
compare them with their own CIRC column.

Usage:
  uv run scripts/onoff_raw_signal.py --dataset model_off
  uv run scripts/onoff_raw_signal.py --dataset model_off --min-days 3 --big-mw 500
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from loguru import logger

from co2sat.utils import data_dir

BAND_LABELS = {
    1: "0.47 blue",
    2: "0.64 red (plume, daytime)",
    3: "0.86 veggie",
    4: "1.37 cirrus",
    5: "1.6 snow/ice",
    6: "2.2 cloud particle",
    7: "3.9 shortwave IR (heat)",
    8: "6.2 upper water vapour",
    9: "6.9 mid water vapour",
    10: "7.3 lower water vapour",
    11: "8.4 cloud-top phase",
    12: "9.6 ozone",
    13: "10.3 clean IR",
    14: "11.2 IR",
    15: "12.3 dirty IR",
    16: "13.3 CO2",
}


def effect_sizes(x: np.ndarray, on: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardized on-minus-off differences for one plant-period.

    Args:
        x (np.ndarray): raw dynamic features of the plant-period, shape
            (n_days, 16, 24).
        on (np.ndarray): boolean, True for running days.

    Returns:
        tuple[np.ndarray, np.ndarray]: per band (16,), the effect size on the
        daily mean, and the effect size at the hour with the largest |d|.
    """

    def d(values: np.ndarray) -> np.ndarray:
        a, b = values[on], values[~on]
        na, nb = len(a), len(b)
        pooled = np.sqrt(
            ((na - 1) * a.var(axis=0, ddof=1) + (nb - 1) * b.var(axis=0, ddof=1))
            / (na + nb - 2)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(pooled > 0, (a.mean(axis=0) - b.mean(axis=0)) / pooled, 0.0)

    d_daily = d(x.mean(axis=2))  # (16,)
    d_hourly = d(x)  # (16, 24)
    idx = np.abs(d_hourly).argmax(axis=1)
    d_best_hour = d_hourly[np.arange(16), idx]
    return d_daily, d_best_hour


def summarize(rows: list[dict], label: str) -> pd.DataFrame:
    """Summarize effect sizes per band.

    Args:
        rows (list[dict]): one dict per plant-period and band with keys
            band, d_daily, d_hour, null_daily, null_hour.
        label (str): name of the plant subset, printed in the table.

    Returns:
        pd.DataFrame: one row per band. med_d = median signed effect on the
        daily mean (a real plant signal is consistent in sign); med|d| =
        median absolute effect, to compare with its chance levels SHUF
        (random labels) and CIRC (time-shifted labels, stricter); hour_* =
        the same at the hour where on and off differ most.
    """
    df = pd.DataFrame(rows)
    g = df.groupby("band")

    def med_abs(col: str) -> np.ndarray:
        return g[col].apply(lambda s: s.abs().median()).to_numpy()

    out = pd.DataFrame(
        {
            "band_label": [BAND_LABELS[b] for b in g.size().index],
            "n": g.size().to_numpy(),
            "med_d": g["d_daily"].median().to_numpy(),
            "med|d|": med_abs("d_daily"),
            "SHUF": med_abs("null_daily"),
            "CIRC": med_abs("circ_daily"),
            "hour_med_d": g["d_hour"].median().to_numpy(),
            "hour_med|d|": med_abs("d_hour"),
            "hour_CIRC": med_abs("circ_hour"),
        },
        index=g.size().index,
    )
    print(f"\n=== {label} ===")
    print(out.round(2).to_string())
    return out.assign(subset=label)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", default="model_off", help="folder under data/processed"
    )
    ap.add_argument("--min-days", type=int, default=3, help="min running AND off days")
    ap.add_argument("--big-mw", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mdir = data_dir("processed", args.dataset)
    x_dyn = np.load(mdir / "X_dyn.npy").astype(np.float32)
    x_stat = np.load(mdir / "X_stat.npy").astype(np.float32)
    samples = pd.read_parquet(mdir / "samples.parquet")
    cols = json.loads((mdir / "normalization_stats.json").read_text())["static"]["cols"]
    if "is_off" not in samples.columns:
        raise SystemExit(
            f"{args.dataset} has no off days: build it with --include-off-days"
        )
    samples["capacity_mw"] = x_stat[:, cols.index("capacity_mw")]
    samples["coal_ratio"] = x_stat[:, cols.index("coal_ratio")]

    rng = np.random.default_rng(args.seed)
    rows_all, rows_big, rows_coal = [], [], []
    n_pp = 0
    for (_, _), grp in samples.groupby(["facility_id", "period"]):
        grp = grp.sort_values("date")
        on = ~grp["is_off"].to_numpy()
        if on.sum() < args.min_days or (~on).sum() < args.min_days:
            continue
        n_pp += 1
        x = x_dyn[grp.index.to_numpy()]
        d_daily, d_hour = effect_sizes(x, on)
        n_daily, n_hour = effect_sizes(x, rng.permutation(on))
        lag = int(rng.integers(1, len(on)))
        c_daily, c_hour = effect_sizes(x, np.roll(on, lag))
        recs = [
            {
                "band": b + 1,
                "d_daily": d_daily[b],
                "d_hour": d_hour[b],
                "null_daily": n_daily[b],
                "null_hour": n_hour[b],
                "circ_daily": c_daily[b],
                "circ_hour": c_hour[b],
            }
            for b in range(16)
        ]
        rows_all += recs
        if grp["capacity_mw"].iloc[0] >= args.big_mw:
            rows_big += recs
        if (
            grp["coal_ratio"].iloc[0] >= 0.5
            and grp["capacity_mw"].iloc[0] >= args.big_mw
        ):
            rows_coal += recs

    logger.info(f"Plant-periods with >= {args.min_days} running and off days: {n_pp}")
    tables = [summarize(rows_all, "ALL plant-periods")]
    if rows_big:
        tables.append(summarize(rows_big, f"plants >= {args.big_mw:.0f} MW"))
    if rows_coal:
        tables.append(summarize(rows_coal, f"COAL plants >= {args.big_mw:.0f} MW"))
    out = pd.concat(tables)
    out.to_csv(mdir / "onoff_raw_signal.csv", float_format="%.4f")
    logger.info(f"Saved -> {mdir / 'onoff_raw_signal.csv'}")


if __name__ == "__main__":
    main()
