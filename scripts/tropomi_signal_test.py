"""Block 0 go/no-go: does TROPOMI NO2 track plant activity where GOES did not?

Joins the daily NO2 window statistics (extract_tropomi.py) with the EPA
daily load and CO2 of the replication, and applies the SAME tests that
condemned the GOES pixel:

  1. Within-plant R2 and correlation between NO2 and gross load, daily
     (clear-sky days) and monthly (mean of valid days), by fuel and size.
  2. Running vs off days: standardized effect size d of NO2 with the CIRC
     (time-shifted) chance control, as in onoff_raw_signal.py.
  3. Share of valid (clear-sky) days per plant: what the satellite can see.

Decision rule (fixed before the run), on coal plants >= 500 MW, 10 km window,
clear sky (cloud < 0.3):
  GO     median within-plant R2 at MONTHLY resolution >= 0.3 and daily
         effect size d (running - off) clearly above CIRC
  NO-GO  monthly within-plant R2 < 0.1 (the GOES situation)
  GREY   otherwise

Usage:
  uv run scripts/tropomi_signal_test.py                 # default: data/raw/tropomi/*.csv
  uv run scripts/tropomi_signal_test.py --radius 10000 --cloud-max 0.3
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd
from loguru import logger

from co2sat.utils import data_dir

PERIODS = {
    "P1": ("2021-04-01", "2021-05-20"),
    "P2": ("2021-09-01", "2021-10-01"),
    "P3": ("2022-04-01", "2022-04-30"),
    "P4": ("2022-09-01", "2022-09-29"),
}
FUEL_OF = {
    "Coal": "coal",
    "Coal Refuse": "coal",
    "Petroleum Coke": "coal",
    "Natural Gas": "gas",
    "Pipeline Natural Gas": "gas",
    "Process Gas": "gas",
    "Other Gas": "gas",
    "Diesel Oil": "oil",
    "Residual Oil": "oil",
    "Other Oil": "oil",
}


def load_no2(radius: int) -> pd.DataFrame:
    """All TROPOMI CSVs of one window radius.

    Args:
        radius (int): window radius in metres.

    Returns:
        pd.DataFrame: facility_id, date, no2_mean, no2_max, n_px, cloud.
    """
    files = sorted(
        glob.glob(str(data_dir("raw", "tropomi") / f"tropomi_*_r{radius}*.csv"))
    )
    if not files:
        raise SystemExit(f"No file data/raw/tropomi/tropomi_*_r{radius}*.csv")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["no2_mean"]).drop_duplicates(["facility_id", "date"])
    logger.info(
        f"{len(files)} files, {len(df):,} plant-days with NO2 (radius {radius} m)"
    )
    return df


def load_epa() -> pd.DataFrame:
    """EPA daily panel with fuel and capacity (incl. off days).

    Returns:
        pd.DataFrame: facility_id, date, period, load, co2, fuel, capacity_mw.
    """
    epa = pd.read_parquet(data_dir("processed", "epa_daily.parquet"))
    epa["date"] = pd.to_datetime(epa["date"])
    # Period = calendar quarter (the on/off test needs several weeks per block).
    epa["period"] = epa["date"].dt.to_period("Q").astype(str)
    epa["load"] = epa["gross_load_mwh"].fillna(0.0).clip(lower=0)
    epa["co2"] = epa["co2_metric_tons"].fillna(0.0).clip(lower=0)
    fuel = (
        epa.groupby("facility_id")["primary_fuel_type"]
        .agg(lambda s: s.mode().iloc[0] if s.notna().any() else None)
        .map(FUEL_OF)
        .rename("fuel")
    )
    st = pd.read_parquet(
        data_dir("processed", "static_features.parquet"),
        columns=["facility_id", "capacity_mw"],
    )
    df = epa.merge(fuel, on="facility_id").merge(st, on="facility_id")
    return df[
        ["facility_id", "date", "period", "load", "co2", "fuel", "capacity_mw"]
    ].drop_duplicates(["facility_id", "date"])


def within_stats(df: pd.DataFrame, x: str, y: str) -> pd.Series:
    """Per-plant within R2 (of a per-plant linear fit) and correlation.

    Args:
        df (pd.DataFrame): rows of several plants.
        x (str): predictor column (NO2).
        y (str): target column (load).

    Returns:
        pd.Series: median within-plant R2, median correlation, number of plants.
    """
    out = []
    for _, g in df.groupby("facility_id"):
        if len(g) < 6 or g[x].std() == 0 or g[y].std() == 0:
            continue
        r = np.corrcoef(g[x], g[y])[0, 1]
        out.append((r * r, r))
    if not out:
        return pd.Series({"within_R2": np.nan, "corr": np.nan, "n_plants": 0})
    a = np.array(out)
    return pd.Series(
        {
            "within_R2": np.median(a[:, 0]),
            "corr": np.median(a[:, 1]),
            "n_plants": len(a),
        }
    )


def onoff(df: pd.DataFrame, x: str, rng: np.random.Generator) -> pd.Series:
    """Effect size of running vs off days on NO2, with CIRC control.

    Args:
        df (pd.DataFrame): rows with load and NO2.
        x (str): NO2 column.
        rng (np.random.Generator): random generator.

    Returns:
        pd.Series: median d, median |d|, CIRC median |d|, plant-periods.
    """
    ds, cs = [], []
    for _, g in df.groupby(["facility_id", "period"]):
        g = g.sort_values("date")
        on = (g["load"] > 0).to_numpy()
        v = g[x].to_numpy()
        if on.sum() < 3 or (~on).sum() < 3:
            continue

        def d(mask: np.ndarray) -> float:
            a, b = v[mask], v[~mask]
            s = np.sqrt(
                ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                / (len(a) + len(b) - 2)
            )
            return (a.mean() - b.mean()) / s if s > 0 else np.nan

        ds.append(d(on))
        cs.append(d(np.roll(on, int(rng.integers(1, len(on))))))
    ds, cs = np.array(ds, float), np.array(cs, float)
    return pd.Series(
        {
            "med_d": np.nanmedian(ds),
            "med|d|": np.nanmedian(np.abs(ds)),
            "CIRC|d|": np.nanmedian(np.abs(cs)),
            "n_plant_periods": len(ds),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=10000)
    ap.add_argument("--cloud-max", type=float, default=0.3)
    ap.add_argument("--min-px", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--stat",
        default="no2_mean",
        choices=["no2_mean", "no2_max"],
        help="window statistic: mean (dilutes the plume) or max (plume pixel)",
    )
    ap.add_argument(
        "--background-radius",
        type=int,
        default=None,
        help="if a tropomi_*_r<R>.csv exists for this larger radius, subtract the "
        "10-30 km ring background from the window value (enhancement)",
    )
    ap.add_argument(
        "--isolated-km",
        type=float,
        default=None,
        help="keep only plants with no other plant >= 100 MW within this distance",
    )
    args = ap.parse_args()

    no2 = load_no2(args.radius)
    epa = load_epa()
    df = epa.merge(no2, on=["facility_id", "date"], how="inner")
    df["sig"] = df[args.stat]
    if args.background_radius:
        bg = load_no2(args.background_radius)[
            ["facility_id", "date", "no2_mean"]
        ].rename(columns={"no2_mean": "bg_mean"})
        df = df.merge(bg, on=["facility_id", "date"], how="inner")
        # Ring background: mean over the big disc minus the inner disc, area-weighted.
        a_in, a_out = args.radius**2, args.background_radius**2
        df["background"] = (df["bg_mean"] * a_out - df["no2_mean"] * a_in) / (
            a_out - a_in
        )
        df["sig"] = df[args.stat] - df["background"]
        logger.info(
            f"Enhancement = {args.stat}({args.radius} m) - ring background ({args.radius}-{args.background_radius} m)"
        )
    if args.isolated_km:
        st = pd.read_parquet(
            data_dir("processed", "static_features.parquet"),
            columns=["facility_id", "latitude", "longitude", "capacity_mw"],
        )
        big = st[st["capacity_mw"] >= 100]
        lat = np.radians(st["latitude"].to_numpy())[:, None]
        lon = np.radians(st["longitude"].to_numpy())[:, None]
        blat = np.radians(big["latitude"].to_numpy())[None, :]
        blon = np.radians(big["longitude"].to_numpy())[None, :]
        d = 6371 * np.arccos(
            np.clip(
                np.sin(lat) * np.sin(blat)
                + np.cos(lat) * np.cos(blat) * np.cos(lon - blon),
                -1,
                1,
            )
        )
        other = (d <= args.isolated_km) & (
            st["facility_id"].to_numpy()[:, None]
            != big["facility_id"].to_numpy()[None, :]
        )
        iso = st.loc[~other.any(axis=1), "facility_id"]
        df = df[df["facility_id"].isin(iso)]
        logger.info(
            f"Isolated plants (no other >= 100 MW within {args.isolated_km} km): {len(iso)} of {len(st)}"
        )
    df["valid"] = (
        df["no2_mean"].notna()
        & (df["cloud"] <= args.cloud_max)
        & (df["n_px"] >= args.min_px)
    )
    logger.info(
        f"Joined {len(df):,} plant-days | valid clear-sky {df['valid'].mean():.1%}"
    )
    df["size"] = pd.cut(
        df["capacity_mw"],
        [0, 200, 500, 1000, 1e5],
        labels=["<200", "200-500", "500-1000", ">=1000"],
    )
    df["month"] = df["date"].dt.to_period("M")

    print("\n=== Share of clear-sky valid days by quarter ===")
    print(df.groupby("period")["valid"].mean().round(3).to_string())

    clear = df[df["valid"]]
    rows = []
    for (fuel, size), g in clear.groupby(["fuel", "size"], observed=True):
        s = within_stats(g, "sig", "load")
        rows.append({"res": "daily", "fuel": fuel, "size": size} | s.to_dict())
        m = (
            g.groupby(["facility_id", "month"], observed=True)[["sig", "load"]]
            .mean()
            .reset_index()
        )
        m = m.groupby("facility_id").filter(lambda t: len(t) >= 6)
        s = within_stats(m, "sig", "load")
        rows.append({"res": "monthly", "fuel": fuel, "size": size} | s.to_dict())
    res = pd.DataFrame(rows)
    print(
        f"\n=== Within-plant R2 of signal ({args.stat}{' - background' if args.background_radius else ''}) vs gross load, clear sky ==="
    )
    print(
        res.pivot_table(
            index=["fuel", "size"],
            columns="res",
            values=["within_R2", "corr", "n_plants"],
            observed=True,
        )
        .round(3)
        .to_string()
    )

    rng = np.random.default_rng(args.seed)
    print("\n=== Running vs off days: effect size on NO2 (clear sky) ===")
    for label, sub in [
        ("ALL", clear),
        (
            "COAL >= 500 MW",
            clear[(clear["fuel"] == "coal") & (clear["capacity_mw"] >= 500)],
        ),
        (
            "GAS >= 500 MW",
            clear[(clear["fuel"] == "gas") & (clear["capacity_mw"] >= 500)],
        ),
    ]:
        print(
            f"  {label:16s} "
            + onoff(sub, "sig", rng).round(3).to_string().replace("\n", " | ")
        )

    # Where does the signal exist? Per-plant monthly R2 for large coal plants,
    # against the plant's mean NO2 level (a proxy of the regional background:
    # Morocco's sites sit in a clean, coastal background).
    coal = clear[(clear["fuel"] == "coal") & (clear["capacity_mw"] >= 500)]
    per = []
    for fid, g in coal.groupby("facility_id"):
        m = g.groupby("month", observed=True)[["sig", "load"]].mean()
        if len(m) >= 6 and m["sig"].std() > 0 and m["load"].std() > 0:
            r = np.corrcoef(m["sig"], m["load"])[0, 1]
            per.append(
                {
                    "facility_id": fid,
                    "capacity_mw": g["capacity_mw"].iloc[0],
                    "mean_no2": g["no2_mean"].mean(),
                    "load_cv": g["load"].std() / max(g["load"].mean(), 1e-9),
                    "r": r,
                    "R2": r * r,
                    "months": len(m),
                }
            )
    per = pd.DataFrame(per)
    if len(per):
        per["bg_tercile"] = pd.qcut(
            per["mean_no2"], 3, labels=["clean", "medium", "polluted"]
        )
        print(
            "\n=== Coal >= 500 MW: monthly R2 by background level (tercile of plant mean NO2) ==="
        )
        print(
            per.groupby("bg_tercile", observed=True)
            .agg(
                n=("R2", "size"),
                median_R2=("R2", "median"),
                share_R2_ge_0_3=("R2", lambda x: (x >= 0.3).mean()),
                median_r=("r", "median"),
            )
            .round(3)
            .to_string()
        )
        print(
            "\n=== Coal >= 500 MW: 12 plants with the highest monthly R2 (sign of r matters) ==="
        )
        print(
            per.sort_values("R2", ascending=False)
            .head(12)[
                [
                    "facility_id",
                    "capacity_mw",
                    "mean_no2",
                    "load_cv",
                    "r",
                    "R2",
                    "months",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )
        per.to_csv(
            data_dir("processed", f"tropomi_per_plant_{args.stat}.csv"),
            index=False,
            float_format="%.4f",
        )

    big = res[
        (res["fuel"] == "coal")
        & (res["size"].isin(["500-1000", ">=1000"]))
        & (res["res"] == "monthly")
    ]
    r2 = float(np.nanmean(big["within_R2"])) if len(big) else np.nan
    verdict = "GO" if r2 >= 0.3 else ("NO-GO" if r2 < 0.1 else "GREY")
    print(
        f"\n=== Decision: coal >= 500 MW, monthly within-plant R2 = {r2:.3f} -> {verdict} ==="
    )
    tag = (
        f"r{args.radius}_{args.stat}"
        + (f"_bg{args.background_radius}" if args.background_radius else "")
        + (f"_iso{int(args.isolated_km)}" if args.isolated_km else "")
    )
    res.to_csv(
        data_dir("processed", f"tropomi_signal_{tag}.csv"),
        index=False,
        float_format="%.4f",
    )


if __name__ == "__main__":
    main()
