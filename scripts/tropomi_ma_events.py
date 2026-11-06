"""Morocco: monthly NO2 enhancement per thermal site and the October 2021 test.

No hourly ground truth exists for Morocco, so the satellite signal is tested
on documented events:

  - the Algerian gas cut (31 Oct 2021): Tahaddart and Ain Beni Mathar stopped
    burning gas; their NO2 enhancement must fall, the coal sites' must not;
  - the seasonal cycle of the coal sites (summer peak demand);
  - any outage reported in the press (to be added by hand in EVENTS).

Outputs: data/processed/tropomi_ma_monthly.csv and figures/tropomi_ma_sites.png.

Usage:
  uv run scripts/tropomi_ma_events.py
"""

from __future__ import annotations

import argparse
import glob

import matplotlib
import numpy as np
import pandas as pd
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from co2sat.utils import data_dir  # noqa: E402

GAS_CUT = pd.Timestamp("2021-11-01")
# The GME pipeline closed on 31 Oct 2021; reverse flow of regasified LNG from
# Spain resumed in June 2022. The two gas plants were off for ~7 months.
OFF_START, OFF_END = pd.Timestamp("2021-11-01"), pd.Timestamp("2022-06-01")
EVENTS = {  # site_id -> list of (date, label); add press-reported outages here
    5: [(GAS_CUT, "Algerian gas cut")],
    6: [(GAS_CUT, "Algerian gas cut")],
    2: [(pd.Timestamp("2018-12-01"), "commissioning")],
}


def load(radius: int) -> pd.DataFrame:
    """All Moroccan CSVs of one radius.

    Args:
        radius (int): window radius (m).

    Returns:
        pd.DataFrame: site_id, date, no2_mean, no2_max, n_px, cloud.
    """
    files = sorted(
        glob.glob(str(data_dir("raw", "tropomi_ma") / f"tropomi_ma_*_r{radius}.csv"))
    )
    if not files:
        raise SystemExit(f"No file data/raw/tropomi_ma/tropomi_ma_*_r{radius}.csv")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True).rename(
        columns={"facility_id": "site_id"}
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna(subset=["no2_mean"]).drop_duplicates(["site_id", "date"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud-max", type=float, default=0.3)
    ap.add_argument("--stat", default="no2_max", choices=["no2_mean", "no2_max"])
    args = ap.parse_args()

    sites = pd.read_csv(data_dir("raw", "morocco_thermal_sites.csv"))
    s10, s30 = (
        load(10000),
        load(30000)[["site_id", "date", "no2_mean"]].rename(
            columns={"no2_mean": "bg_mean"}
        ),
    )
    df = s10.merge(s30, on=["site_id", "date"]).merge(sites, on="site_id")
    df["background"] = (df["bg_mean"] * 30000**2 - df["no2_mean"] * 10000**2) / (
        30000**2 - 10000**2
    )
    df["enh"] = df[args.stat] - df["background"]
    df = df[(df["cloud"] <= args.cloud_max) & (df["n_px"] >= 3)]
    df["month"] = df["date"].dt.to_period("M")
    logger.info(
        f"{len(df):,} clear-sky site-days, {df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}"
    )

    m = (
        df.groupby(["site_id", "name", "fuel", "month"], observed=True)
        .agg(
            enh=("enh", "median"),
            raw=(args.stat, "median"),
            background=("background", "median"),
            n_days=("enh", "size"),
        )
        .reset_index()
    )
    m["month"] = m["month"].dt.to_timestamp()
    m.to_csv(
        data_dir("processed", "tropomi_ma_monthly.csv"),
        index=False,
        float_format="%.6g",
    )

    print("\n=== Valid clear-sky days per site and year ===")
    print(df.groupby(["name", df["date"].dt.year])["enh"].size().unstack().to_string())

    print(
        "\n=== Gas cut: Nov 2021 - May 2022 (plants off) vs the same months one year earlier and one year later ==="
    )
    rows = []
    for sid, g in m.groupby("site_id"):
        off = g[(g["month"] >= OFF_START) & (g["month"] < OFF_END)]["enh"]
        prev = g[
            (g["month"] >= OFF_START - pd.DateOffset(years=1))
            & (g["month"] < OFF_END - pd.DateOffset(years=1))
        ]["enh"]
        nxt = g[
            (g["month"] >= OFF_START + pd.DateOffset(years=1))
            & (g["month"] < OFF_END + pd.DateOffset(years=1))
        ]["enh"]
        ref = pd.concat([prev, nxt]).median()
        rows.append(
            {
                "site": g["name"].iloc[0],
                "fuel": g["fuel"].iloc[0],
                "same_months_2020-21": 1e5 * prev.median(),
                "off_2021-22": 1e5 * off.median(),
                "same_months_2022-23": 1e5 * nxt.median(),
                "change_vs_ref_%": 100 * (off.median() / ref - 1) if ref else np.nan,
            }
        )
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print(
        "Reading: gas sites should drop >= 50 % vs the reference (same months, adjacent years); coal sites within +-20 %."
    )

    fig, axes = plt.subplots(len(sites), 1, figsize=(11, 2.2 * len(sites)), sharex=True)
    for ax, (sid, g) in zip(axes, m.groupby("site_id")):
        ax.plot(g["month"], 1e5 * g["enh"], lw=1.2, label="enhancement (window - ring)")
        ax.plot(
            g["month"],
            1e5 * g["background"],
            lw=0.8,
            color="grey",
            alpha=0.7,
            label="ring background",
        )
        for d, lab in EVENTS.get(sid, []):
            ax.axvline(d, color="red", ls="--", lw=0.8)
            ax.text(
                d,
                ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 0,
                lab,
                color="red",
                fontsize=7,
                rotation=90,
                va="top",
            )
        ax.set_ylabel("NO2 (1e-5 mol/m2)", fontsize=8)
        ax.set_title(
            f"{g['name'].iloc[0]} ({g['fuel'].iloc[0]}, {sites.set_index('site_id').loc[sid, 'capacity_mw']:.0f} MW)",
            fontsize=9,
            loc="left",
        )
        ax.axhline(0, color="k", lw=0.5)
    axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    out = data_dir("..", "figures") / "tropomi_ma_sites.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    logger.info(f"Figure -> {out}")


if __name__ == "__main__":
    main()
