"""Monitoring design: which plants should be measured first?

The audits showed that an unmeasured plant's activity is largely
unobservable (satellite pixel: nothing; system totals: 20-37 % of its daily
variation at best). This script asks the next question: with a budget of
k plants that can be measured (continuous monitoring, on-site verification,
targeted observation), which plants should be chosen so that the emissions of
the whole fleet are best known?

Setting (per US state, daily, EPA data):
  - the state's total fossil generation is known (system observable);
  - measured plants: their generation and CO2 are known exactly;
  - unmeasured plants: the residual (total minus measured) is shared among
    them pro rata of capacity (the best simple rule of the audits), and
    converted to CO2 with each plant's intensity measured on P1.

Selection strategies (chosen on P1 data only, evaluated on P2-P4):
  random     average of 10 random orders (floor)
  largest    largest capacity first (common sense)
  variable   largest standard deviation of daily output on P1 first
  worst      largest P1 error under the no-measurement rule first
  greedy     at each step, the plant whose measurement most reduces the P1
             fleet CO2 error (lazy greedy; exact gain recomputed only for
             the state of the plant just added)

Metrics on P2-P4, for budgets of 0-50 % of plants:
  fleet_err   sum over plant-days of |CO2 - estimate| / total CO2: how well
              each plant's emissions are known (measured plants count 0)
  state_err   sum over state-days of |state CO2 - estimate| / total CO2: how
              well the inventory total is known

No-go criterion: if greedy is not clearly better than largest (e.g. the same
error with < 30 % fewer plants), the design problem is trivial.

Usage:
  uv run scripts/monitoring_design.py
  uv run scripts/monitoring_design.py --fuel      # fuel totals also known
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
# Fallback CO2 intensities (t/MWh) for fuels with no plant running in P1.
INTEN_DEFAULT = {"coal": 1.0, "gas": 0.45, "oil": 0.8}
BUDGETS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
STRATEGIES = ["random", "largest", "variable", "worst", "greedy"]


def load_panel() -> pd.DataFrame:
    """Plant-day panel (same source as the audits).

    Returns:
        pd.DataFrame: facility_id, state, date, period, load, co2,
        capacity_mw, fuel.
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
    return df.drop_duplicates(["facility_id", "date"]).reset_index(drop=True)


class StateBlock:
    """Dense day x plant matrices of one state and one period.

    Args:
        sub (pd.DataFrame): panel rows of the state and period.
        plants (np.ndarray): facility ids of the state (fixed order).
        cap (np.ndarray): capacity of each plant (MW).
        inten (np.ndarray): CO2 intensity of each plant (t/MWh, from P1).
        fuel (np.ndarray): fuel of each plant.
        by_fuel (bool): if True, residuals are shared within each fuel.
    """

    def __init__(self, sub, plants, cap, inten, fuel, by_fuel):
        L = sub.pivot(index="date", columns="facility_id", values="load")
        E = sub.pivot(index="date", columns="facility_id", values="co2")
        self.L = L.reindex(columns=plants).fillna(0.0).to_numpy()
        self.E = E.reindex(columns=plants).fillna(0.0).to_numpy()
        self.cap, self.inten = cap, inten
        self.groups = (
            [fuel == f for f in np.unique(fuel)]
            if by_fuel
            else [np.ones(len(plants), bool)]
        )

    def estimate(self, measured: np.ndarray) -> np.ndarray:
        """CO2 estimate of every plant-day given the measured set.

        Args:
            measured (np.ndarray): boolean, True for measured plants.

        Returns:
            np.ndarray: estimated CO2, days x plants.
        """
        est = np.where(measured[None, :], self.E, 0.0)
        for g in self.groups:
            un = g & ~measured
            if not un.any():
                continue
            resid = self.L[:, g].sum(1) - self.L[:, g & measured].sum(1)
            share = self.cap[un] / self.cap[un].sum()
            est[:, un] = (
                np.clip(resid, 0.0, None)[:, None]
                * share[None, :]
                * self.inten[un][None, :]
            )
        return est

    def errors(self, measured: np.ndarray) -> tuple[float, float]:
        """Fleet and state-total absolute CO2 errors (tonnes).

        Args:
            measured (np.ndarray): boolean, True for measured plants.

        Returns:
            tuple[float, float]: (sum of plant-day errors, sum of
            state-day total errors).
        """
        est = self.estimate(measured)
        return float(np.abs(self.E - est).sum()), float(
            np.abs(self.E.sum(1) - est.sum(1)).sum()
        )


def build_blocks(df: pd.DataFrame, by_fuel: bool) -> tuple[dict, pd.DataFrame]:
    """Blocks per (period, state) and the plant table.

    Args:
        df (pd.DataFrame): panel.
        by_fuel (bool): residual shared within fuel.

    Returns:
        tuple[dict, pd.DataFrame]: {(period, state): StateBlock} and a plant
        table (facility_id, state, capacity_mw, fuel, inten, sd_p1).
    """
    p1 = df[df["period"] == "P1"]
    run = p1[p1["load"] > 0].groupby("facility_id")[["co2", "load"]].sum()
    inten = (run["co2"] / run["load"]).where(lambda s: s.between(0.2, 2.0))
    plants = df.groupby("facility_id").agg(
        state=("state", "first"),
        capacity_mw=("capacity_mw", "first"),
        fuel=("fuel", "first"),
    )
    plants["inten"] = inten.reindex(plants.index)
    plants["inten"] = (
        plants["inten"]
        .fillna(plants.groupby("fuel")["inten"].transform("median"))
        .fillna(plants["fuel"].map(INTEN_DEFAULT))
    )
    plants["sd_p1"] = (
        p1.groupby("facility_id")["load"].std().reindex(plants.index).fillna(0.0)
    )
    plants = plants.reset_index()
    blocks = {}
    for (per, state), sub in df.groupby(["period", "state"]):
        pl = plants[plants["state"] == state]
        blocks[(per, state)] = StateBlock(
            sub,
            pl["facility_id"].to_numpy(),
            pl["capacity_mw"].to_numpy(),
            pl["inten"].to_numpy(),
            pl["fuel"].to_numpy(),
            by_fuel,
        )
    return blocks, plants


def greedy_order(blocks: dict, plants: pd.DataFrame) -> list[int]:
    """Lazy greedy ranking of plants on the P1 fleet CO2 error.

    Args:
        blocks (dict): StateBlock per (period, state).
        plants (pd.DataFrame): plant table.

    Returns:
        list[int]: facility ids, most valuable measurement first.
    """
    states = plants["state"].unique()
    masks = {s: np.zeros((plants["state"] == s).sum(), bool) for s in states}
    ids = {
        s: plants.loc[plants["state"] == s, "facility_id"].to_numpy() for s in states
    }
    base = {
        s: blocks[("P1", s)].errors(masks[s])[0] for s in states if ("P1", s) in blocks
    }

    def gains(s: str) -> list[tuple[float, str, int]]:
        if ("P1", s) not in blocks:
            return [(0.0, s, j) for j in range(len(ids[s])) if not masks[s][j]]
        out = []
        for j in np.flatnonzero(~masks[s]):
            m = masks[s].copy()
            m[j] = True
            out.append((base[s] - blocks[("P1", s)].errors(m)[0], s, j))
        return out

    cand = {s: gains(s) for s in states}
    order = []
    while any(cand.values()):
        g, s, j = max((c for lst in cand.values() for c in lst), key=lambda t: t[0])
        masks[s][j] = True
        order.append(ids[s][j])
        if ("P1", s) in blocks:
            base[s] = blocks[("P1", s)].errors(masks[s])[0]
        cand[s] = gains(s)
        if len(order) % 100 == 0:
            logger.info(f"greedy: {len(order)} plants ranked")
    return order


def evaluate(
    order: list[int], blocks: dict, plants: pd.DataFrame, per: str
) -> list[dict]:
    """Errors on one period for every budget, given a ranking.

    Args:
        order (list[int]): facility ids, first = measured first.
        blocks (dict): StateBlock per (period, state).
        plants (pd.DataFrame): plant table.
        per (str): period.

    Returns:
        list[dict]: one row per budget.
    """
    rank = {f: i for i, f in enumerate(order)}
    pos = plants["facility_id"].map(rank).to_numpy()
    total = sum(b.E.sum() for (p, _), b in blocks.items() if p == per)
    rows = []
    for b in BUDGETS:
        n = int(round(b * len(plants)))
        meas = pd.Series(pos < n, index=plants["facility_id"])
        fe = se = 0.0
        for (p, s), blk in blocks.items():
            if p != per:
                continue
            ids = plants.loc[plants["state"] == s, "facility_id"]
            f, t = blk.errors(meas.loc[ids].to_numpy())
            fe, se = fe + f, se + t
        rows.append(
            {
                "budget": b,
                "n_measured": n,
                "fleet_err": fe / total,
                "state_err": se / total,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuel", action="store_true", help="fuel totals also known")
    ap.add_argument("--min-plants", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_panel()
    n = df.groupby("state")["facility_id"].nunique()
    df = df[df["state"].isin(n[n >= args.min_plants].index)]
    blocks, plants = build_blocks(df, args.fuel)
    logger.info(f"{len(plants)} plants in {plants['state'].nunique()} states")

    p1_err = {}
    for s in plants["state"].unique():
        if ("P1", s) in blocks:
            blk = blocks[("P1", s)]
            e = np.abs(blk.E - blk.estimate(np.zeros(len(blk.cap), bool))).sum(0)
            p1_err.update(dict(zip(plants.loc[plants["state"] == s, "facility_id"], e)))
    orders = {
        "largest": plants.sort_values("capacity_mw", ascending=False)[
            "facility_id"
        ].tolist(),
        "variable": plants.sort_values("sd_p1", ascending=False)[
            "facility_id"
        ].tolist(),
        "worst": plants.assign(e=plants["facility_id"].map(p1_err).fillna(0))
        .sort_values("e", ascending=False)["facility_id"]
        .tolist(),
        "greedy": greedy_order(blocks, plants),
    }
    rng = np.random.default_rng(args.seed)
    rows = []
    for per in ["P2", "P3", "P4"]:
        for strat, order in orders.items():
            rows += [
                r | {"period": per, "strategy": strat}
                for r in evaluate(order, blocks, plants, per)
            ]
        for k in range(10):
            order = list(rng.permutation(plants["facility_id"].to_numpy()))
            rows += [
                r | {"period": per, "strategy": "random", "draw": k}
                for r in evaluate(order, blocks, plants, per)
            ]
    res = pd.DataFrame(rows)
    suffix = "_fuel" if args.fuel else ""
    res.to_csv(
        data_dir("processed", f"monitoring_design{suffix}.csv"),
        index=False,
        float_format="%.4f",
    )

    mean = (
        res.groupby(["strategy", "budget", "period"])[
            ["fleet_err", "state_err", "n_measured"]
        ]
        .mean()
        .reset_index()
    )
    for metric in ["fleet_err", "state_err"]:
        print(f"\n=== {metric} (mean of P2-P4; columns = share of plants measured) ===")
        t = (
            mean.groupby(["strategy", "budget"])[metric]
            .mean()
            .unstack("budget")
            .reindex(STRATEGIES)
        )
        print(t.round(3).to_string())
    print(
        "\n=== plants needed to reach a fleet_err target (mean of P2-P4, linear interpolation) ==="
    )
    curve = mean.groupby(["strategy", "budget"])[["fleet_err", "n_measured"]].mean()
    for target in [0.30, 0.20, 0.10]:
        line = []
        for strat in STRATEGIES:
            c = curve.loc[strat].sort_index()
            below = c[c["fleet_err"] <= target]
            if below.empty:
                line.append(f"{strat}: >{int(c['n_measured'].max())}")
                continue
            i = c.index.get_loc(below.index[0])
            if i == 0:
                line.append(f"{strat}: 0")
                continue
            a, b = c.iloc[i - 1], c.iloc[i]
            w = (a["fleet_err"] - target) / (a["fleet_err"] - b["fleet_err"])
            line.append(
                f"{strat}: {a['n_measured'] + w * (b['n_measured'] - a['n_measured']):.0f}"
            )
        print(f"target {target:.0%}: " + " | ".join(line))


if __name__ == "__main__":
    main()
