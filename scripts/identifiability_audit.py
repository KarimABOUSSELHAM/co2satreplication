"""Observability audit: how much does a system total say about each plant?

Question (Article 1, stage 2): if we only know the TOTAL fossil generation of
a system on a given day, how well can we recover what EACH plant produced?
We answer it on the EPA data, where the truth per plant is known, by
comparing allocation rules that use more and more system-level information.
No satellite, no machine learning: this measures the information content of
the observables, i.e. the ceiling any stage-2 method can reach.

The "system" is the US state (first approximation; balancing authorities in a
later version). The system total is taken from EPA itself (oracle total), so
the results are an upper bound for a method that would have to estimate it.

Allocation rules, from least to most informed:

  C   capacity x mean capacity factor of the fuel (national, over the
      period). Uses no system information. This is the rule applied today to
      the 96 % of plants without a visible plume (Climate TRACE, WRI).
  T   system total shared pro rata of capacity. Uses the total, not the fuel.
  MO  merit order on the system total: plants stacked by marginal cost
      (heat rate x fuel price), each filled up to availability x capacity
      x 24 h, until the total is met. Uses the total and an economic prior.
  F   fuel totals known (e.g. monthly or daily fuel statistics), shared pro
      rata of capacity within each fuel.
  FMO fuel totals known, merit order within each fuel.

Reading: within-plant R2 is the share of each plant's day-to-day variation
that the rule recovers. C scores 0 by construction. If F and FMO stay near 0,
system totals carry little plant-level information (the fear is confirmed).
If MO approaches FMO, the merit order recovers the fuel split from the total
alone, which is what matters for emissions.

Inputs (already in the project):
  data/processed/epa_daily.parquet      facility_id, state, date, co2, load,
                                        primary_fuel_type (all days, incl. 0)
  data/processed/static_features.parquet facility_id, capacity_mw

Usage:
  uv run scripts/identifiability_audit.py
  uv run scripts/identifiability_audit.py --availability 0.85 --min-plants 3
"""

from __future__ import annotations

import argparse

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
# CO2 emission factors, metric t CO2 per MMBtu (EPA, rounded).
EF = {"coal": 0.0954, "gas": 0.0531, "oil": 0.0740}
# Approximate delivered fuel prices to US power plants, USD/MMBtu, per period
# (EIA Electric Power Monthly orders of magnitude). Only the RANKING between
# fuels matters for the merit order; --price-sensitivity tests it.
PRICE = {
    "P1": {"coal": 2.0, "gas": 3.0, "oil": 14.0},
    "P2": {"coal": 2.0, "gas": 5.5, "oil": 16.0},
    "P3": {"coal": 2.3, "gas": 6.5, "oil": 22.0},
    "P4": {"coal": 2.6, "gas": 8.5, "oil": 24.0},
}
RULES = ["C", "T", "MO", "F", "FMO"]


def load_panel() -> pd.DataFrame:
    """Plant-day panel with capacity, fuel and period.

    Returns:
        pd.DataFrame: facility_id, state, date, period, load (MWh, 0 on off
        days), co2, capacity_mw, fuel.
    """
    epa = pd.read_parquet(data_dir("processed", "epa_daily.parquet"))
    epa["date"] = pd.to_datetime(epa["date"])
    epa["period"] = None
    for name, (start, end) in PERIODS.items():
        epa.loc[(epa["date"] >= start) & (epa["date"] < end), "period"] = name
    epa = epa.dropna(subset=["period"])
    epa["load"] = epa["gross_load_mwh"].fillna(0.0).clip(lower=0.0)
    epa["co2"] = epa["co2_metric_tons"].fillna(0.0).clip(lower=0.0)

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
    df = df[df["fuel"].notna() & (df["capacity_mw"] > 0)]
    df = df.drop_duplicates(["facility_id", "date"])
    logger.info(
        f"Panel: {len(df):,} plant-days | {df['facility_id'].nunique()} plants | "
        f"{df['state'].nunique()} states | off days {int((df['load'] == 0).sum()):,}"
    )
    return df[
        ["facility_id", "state", "date", "period", "load", "co2", "capacity_mw", "fuel"]
    ].reset_index(drop=True)


def marginal_cost(df: pd.DataFrame, price_scale: dict[str, float]) -> pd.Series:
    """Marginal cost per plant and period, USD/MWh.

    Heat rate is proxied by the plant's CO2 intensity over the period divided
    by the fuel emission factor (a static plant characteristic, as a
    nameplate heat rate would be in a country without data).

    Args:
        df (pd.DataFrame): panel.
        price_scale (dict[str, float]): multiplier per fuel (sensitivity).

    Returns:
        pd.Series: marginal cost aligned on df rows.
    """
    run = df[df["load"] > 0]
    inten = (
        run.groupby(["facility_id", "period"])[["co2", "load"]].sum().eval("co2 / load")
    )
    inten = inten.where(inten.between(0.2, 2.0))
    key = pd.MultiIndex.from_frame(df[["facility_id", "period"]])
    i = pd.Series(inten.reindex(key).to_numpy(), index=df.index)
    fuel_med = i.groupby(df["fuel"]).transform("median")
    i = i.fillna(fuel_med)
    hr = i / df["fuel"].map(EF)
    price = [
        PRICE[p][f] * price_scale.get(f, 1.0) for p, f in zip(df["period"], df["fuel"])
    ]
    return hr * np.asarray(price)


def merit_fill(cost: np.ndarray, cap_day: np.ndarray, total: float) -> np.ndarray:
    """Fill the cheapest units first up to a total.

    Args:
        cost (np.ndarray): marginal cost of each unit.
        cap_day (np.ndarray): maximum energy of each unit over the day (MWh).
        total (float): energy to supply (MWh).

    Returns:
        np.ndarray: energy allocated to each unit; if the total exceeds the
        stack, the excess is shared pro rata of capacity.
    """
    order = np.argsort(cost, kind="stable")
    out = np.zeros_like(cap_day)
    cum = np.cumsum(cap_day[order])
    prev = np.concatenate([[0.0], cum[:-1]])
    out[order] = np.clip(total - prev, 0.0, cap_day[order])
    excess = total - out.sum()
    if excess > 1e-6 and cap_day.sum() > 0:
        out += excess * cap_day / cap_day.sum()
    return out


def allocate(df: pd.DataFrame, availability: float) -> pd.DataFrame:
    """Apply every allocation rule.

    Args:
        df (pd.DataFrame): panel with a 'mc' column (marginal cost).
        availability (float): share of capacity usable in a day (merit order).

    Returns:
        pd.DataFrame: df with one column per rule.
    """
    df = df.copy()
    df["cap_day"] = df["capacity_mw"] * 24.0
    cf = df.groupby(["period", "fuel"])["load"].transform("sum") / df.groupby(
        ["period", "fuel"]
    )["cap_day"].transform("sum")
    df["C"] = cf * df["cap_day"]

    sd = ["state", "date"]
    sfd = ["state", "fuel", "date"]
    df["T"] = (
        df.groupby(sd)["load"].transform("sum")
        * df["cap_day"]
        / df.groupby(sd)["cap_day"].transform("sum")
    )
    df["F"] = (
        df.groupby(sfd)["load"].transform("sum")
        * df["cap_day"]
        / df.groupby(sfd)["cap_day"].transform("sum")
    )

    df["MO"] = 0.0
    df["FMO"] = 0.0
    for keys, idx in df.groupby(sd).groups.items():
        g = df.loc[idx]
        avail = availability * g["cap_day"].to_numpy()
        df.loc[idx, "MO"] = merit_fill(g["mc"].to_numpy(), avail, g["load"].sum())
    for keys, idx in df.groupby(sfd).groups.items():
        g = df.loc[idx]
        avail = availability * g["cap_day"].to_numpy()
        df.loc[idx, "FMO"] = merit_fill(g["mc"].to_numpy(), avail, g["load"].sum())
    return df


def scores(df: pd.DataFrame, rule: str) -> dict[str, float]:
    """Plant-level skill of one rule.

    Args:
        df (pd.DataFrame): allocated panel (one period).
        rule (str): column name of the rule.

    Returns:
        dict[str, float]: R2, within-plant R2, WAPE, share of plant-days
        whose on/off status is right, error on the state-day coal share.
    """
    y, p = df["load"].to_numpy(), df[rule].to_numpy()
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    ay = df["load"] - df.groupby("facility_id")["load"].transform("mean")
    ap = df[rule] - df.groupby("facility_id")[rule].transform("mean")
    within = 1 - ((ay - ap) ** 2).sum() / (ay**2).sum()
    wape = np.abs(y - p).sum() / y.sum()
    onoff = ((y > 0) == (p > 1.0)).mean()
    coal = df.assign(c_y=np.where(df["fuel"] == "coal", y, 0.0)).assign(
        c_p=np.where(df["fuel"] == "coal", p, 0.0)
    )
    g = coal.groupby(["state", "date"])[["c_y", "c_p", "load", rule]].sum()
    g = g[g["load"] > 0]
    share_err = (g["c_y"] / g["load"] - g["c_p"] / g[rule].replace(0, np.nan)).abs()
    return {
        "R2": r2,
        "within_R2": within,
        "WAPE": wape,
        "onoff_acc": onoff,
        "coal_share_MAE": float(share_err.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--availability", type=float, default=0.85)
    ap.add_argument("--min-plants", type=int, default=3, help="per state")
    ap.add_argument(
        "--price-sensitivity",
        action="store_true",
        help="rerun the merit order with gas at 0.5x and 2x",
    )
    args = ap.parse_args()

    df = load_panel()
    n_state = df.groupby("state")["facility_id"].nunique()
    df = df[df["state"].isin(n_state[n_state >= args.min_plants].index)]

    scenarios = {"base": {}}
    if args.price_sensitivity:
        scenarios |= {"gas_x0.5": {"gas": 0.5}, "gas_x2": {"gas": 2.0}}

    rows = []
    for scen, scale in scenarios.items():
        df["mc"] = marginal_cost(df, scale)
        alloc = allocate(df, args.availability)
        rules = RULES if scen == "base" else ["MO", "FMO"]
        for per, sub in alloc.groupby("period"):
            for rule in rules:
                rows.append(
                    {"scenario": scen, "period": per, "rule": rule} | scores(sub, rule)
                )
        logger.info(f"Scenario {scen} done")

    res = pd.DataFrame(rows)
    out = data_dir("processed", "identifiability_audit.csv")
    res.to_csv(out, index=False, float_format="%.4f")
    for metric in ["within_R2", "R2", "WAPE", "onoff_acc", "coal_share_MAE"]:
        print(f"\n=== {metric} ===")
        print(
            res.pivot_table(index=["scenario", "rule"], columns="period", values=metric)
            .reindex(RULES, level=1)
            .round(3)
            .to_string()
        )
    logger.info(f"Saved -> {out}")


if __name__ == "__main__":
    main()
