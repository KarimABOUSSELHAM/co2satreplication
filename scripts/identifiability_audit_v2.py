"""Observability audit, v2: soft merit order and a learned allocator.

v1 showed that the system total carries plant-level information (pro rata
rule T: within-plant R2 0.13-0.27; with fuel totals F: 0.19-0.37) but that a
hard merit order destroys it (MO: -0.5 to -1.1). v2 tests the two remedies
that the Article 1 stage 2 relies on:

  SMO   soft merit order: each plant runs at a capacity factor that decreases
        smoothly with its marginal cost, sigmoid((lambda - cost) / tau), where
        lambda is solved each state-day so that the plants sum to the total.
        tau -> 0 is the hard merit order of v1; tau -> infinity is the pro
        rata rule T. tau is tuned on P1 only, then frozen for P2-P4.
  SFMO  same, within each fuel, given fuel totals.
  ML    learned allocator (CatBoost): predicts each plant's share of the
        state-day total from characteristics that a country without data
        would have (fuel, capacity, cost rank, system load factor, calendar),
        never the plant identity. Trained on P1, evaluated on P2-P4 on
        states held out of training (5-fold grouped by state): this is
        transfer to unseen systems, the situation of Morocco.
  MLF   same as ML, with fuel totals known.

T and F from v1 are recomputed as references. Diagnostic: amp = standard
deviation of predicted daily anomalies / that of true anomalies; amp >> 1
means the rule over-reacts (the hard merit order's all-or-nothing).

Usage:
  uv run scripts/identifiability_audit_v2.py
  uv run scripts/identifiability_audit_v2.py --no-ml      # soft merit order only
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
EF = {"coal": 0.0954, "gas": 0.0531, "oil": 0.0740}
# Fallback heat rates (MMBtu/MWh) for fuels with no plant running in P1.
HR_DEFAULT = {"coal": 10.5, "gas": 8.0, "oil": 11.5}
PRICE = {
    "P1": {"coal": 2.0, "gas": 3.0, "oil": 14.0},
    "P2": {"coal": 2.0, "gas": 5.5, "oil": 16.0},
    "P3": {"coal": 2.3, "gas": 6.5, "oil": 22.0},
    "P4": {"coal": 2.6, "gas": 8.5, "oil": 24.0},
}
TAU_GRID = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]  # in units of the state-day cost std
SD = ["state", "date"]
SFD = ["state", "fuel", "date"]


def load_panel() -> pd.DataFrame:
    """Plant-day panel with capacity, fuel, period (same as v1).

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
    df = df.drop_duplicates(["facility_id", "date"]).reset_index(drop=True)
    df["cap_day"] = df["capacity_mw"] * 24.0
    logger.info(f"Panel: {len(df):,} plant-days | {df['facility_id'].nunique()} plants")
    return df


def add_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Marginal cost from a STATIC heat rate (CO2 intensity measured on P1).

    Unlike v1, the heat rate is measured on P1 only and reused for P2-P4,
    so the test periods carry no information from their own outcomes.

    Args:
        df (pd.DataFrame): panel.

    Returns:
        pd.DataFrame: df with columns hr (MMBtu/MWh) and mc (USD/MWh).
    """
    run = df[(df["load"] > 0) & (df["period"] == "P1")]
    inten = run.groupby("facility_id")[["co2", "load"]].sum().eval("co2 / load")
    inten = inten.where(inten.between(0.2, 2.0))
    df = df.copy()
    i = df["facility_id"].map(inten)
    i = i.fillna(i.groupby(df["fuel"]).transform("median"))
    df["hr"] = (i / df["fuel"].map(EF)).fillna(df["fuel"].map(HR_DEFAULT))
    df["mc"] = df["hr"] * [PRICE[p][f] for p, f in zip(df["period"], df["fuel"])]
    return df


def soft_fill(
    cost: np.ndarray, cap: np.ndarray, total: float, tau: float
) -> np.ndarray:
    """Soft merit order: capacity factor sigmoid((lambda - cost) / tau).

    Args:
        cost (np.ndarray): marginal costs.
        cap (np.ndarray): daily energy capacity of each plant (MWh).
        total (float): energy to supply (MWh).
        tau (float): softness, in cost units.

    Returns:
        np.ndarray: energy per plant, summing to min(total, sum(cap)).
    """
    if total <= 0 or cap.sum() <= 0:
        return np.zeros_like(cap)
    if total >= cap.sum():
        return cap.copy()
    lo, hi = cost.min() - 50 * tau, cost.max() + 50 * tau
    for _ in range(60):
        lam = 0.5 * (lo + hi)
        x = cap / (1.0 + np.exp(-np.clip((lam - cost) / tau, -60, 60)))
        if x.sum() > total:
            hi = lam
        else:
            lo = lam
    return x * total / x.sum()


def soft_rule(df: pd.DataFrame, keys: list[str], s: float) -> np.ndarray:
    """Apply the soft merit order on every group.

    Args:
        df (pd.DataFrame): panel with mc and cap_day.
        keys (list[str]): grouping (state-day, or state-fuel-day).
        s (float): tau as a multiple of the group's cost standard deviation.

    Returns:
        np.ndarray: allocation aligned on df rows.
    """
    out = np.zeros(len(df))
    for _, idx in df.groupby(keys).indices.items():
        c = df["mc"].to_numpy()[idx]
        tau = s * max(c.std(), 1.0)
        out[idx] = soft_fill(
            c, df["cap_day"].to_numpy()[idx], df["load"].to_numpy()[idx].sum(), tau
        )
    return out


def pro_rata(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Group total shared pro rata of capacity.

    Args:
        df (pd.DataFrame): panel.
        keys (list[str]): grouping.

    Returns:
        pd.Series: allocation.
    """
    g = df.groupby(keys)
    return g["load"].transform("sum") * df["cap_day"] / g["cap_day"].transform("sum")


def features(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Plant features available in a country without plant-level data.

    Args:
        df (pd.DataFrame): panel with mc.
        keys (list[str]): the system grouping whose total is known.

    Returns:
        pd.DataFrame: feature matrix (no plant identifier).
    """
    g = df.groupby(keys)
    cap_sys = g["cap_day"].transform("sum")
    return pd.DataFrame(
        {
            "fuel": df["fuel"],
            "capacity_mw": df["capacity_mw"],
            "cap_share": df["cap_day"] / cap_sys,
            "hr": df["hr"],
            "mc": df["mc"],
            "mc_rank": g["mc"].rank(pct=True),
            "mc_rel": df["mc"] / g["mc"].transform("mean"),
            "sys_lf": g["load"].transform("sum") / cap_sys,
            "n_plants": g["mc"].transform("size"),
            "dow": df["date"].dt.dayofweek,
            "month": df["date"].dt.month,
        }
    )


def ml_rule(df: pd.DataFrame, keys: list[str], folds: int, seed: int) -> np.ndarray:
    """Learned allocator, trained on P1, applied to held-out states.

    Args:
        df (pd.DataFrame): panel with mc and hr.
        keys (list[str]): system grouping whose total is known.
        folds (int): number of state folds.
        seed (int): random seed.

    Returns:
        np.ndarray: allocation aligned on df rows (NaN for P1 rows).
    """
    from catboost import CatBoostRegressor

    X = features(df, keys)
    tot = df.groupby(keys)["load"].transform("sum")
    y = (df["load"] / tot.replace(0, np.nan)).fillna(0.0)
    states = np.array(sorted(df["state"].unique()))
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(states), np.arange(len(states)) % folds))
    fold = df["state"].map(fold_of).to_numpy()
    out = np.full(len(df), np.nan)
    for k in range(folds):
        tr = (fold != k) & (df["period"] == "P1").to_numpy()
        te = (fold == k) & (df["period"] != "P1").to_numpy()
        m = CatBoostRegressor(
            iterations=600,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
            cat_features=["fuel"],
        )
        m.fit(X[tr], y[tr])
        out[te] = np.clip(m.predict(X[te]), 0.0, None)
        logger.info(f"ML fold {k + 1}/{folds} ({'+'.join(keys)}) done")
    sub = df.assign(p=out)
    s = sub.groupby(keys)["p"].transform("sum").replace(0, np.nan)
    return (sub["p"] / s * tot).fillna(0.0).to_numpy()


def scores(df: pd.DataFrame, rule: str) -> dict[str, float]:
    """Plant-level skill of one rule on one period.

    Args:
        df (pd.DataFrame): panel restricted to one period, with the rule column.
        rule (str): rule column.

    Returns:
        dict[str, float]: within-plant R2, R2, WAPE, amp, coal share MAE.
    """
    if df[rule].isna().any():
        raise ValueError(f"rule {rule} has {int(df[rule].isna().sum())} missing values")
    y, p = df["load"].to_numpy(), df[rule].to_numpy()
    ay = df["load"] - df.groupby("facility_id")["load"].transform("mean")
    ap = df[rule] - df.groupby("facility_id")[rule].transform("mean")
    g = (
        df.assign(
            cy=np.where(df["fuel"] == "coal", y, 0.0),
            cp=np.where(df["fuel"] == "coal", p, 0.0),
        )
        .groupby(SD)[["cy", "cp", "load", rule]]
        .sum()
    )
    g = g[g["load"] > 0]
    return {
        "within_R2": 1 - ((ay - ap) ** 2).sum() / (ay**2).sum(),
        "R2": 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
        "WAPE": np.abs(y - p).sum() / y.sum(),
        "amp": float(ap.std() / ay.std()),
        "coal_share_MAE": float(
            (g["cy"] / g["load"] - g["cp"] / g[rule].replace(0, np.nan)).abs().mean()
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-plants", type=int, default=3)
    ap.add_argument("--no-ml", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_panel()
    n = df.groupby("state")["facility_id"].nunique()
    df = add_cost(
        df[df["state"].isin(n[n >= args.min_plants].index)].reset_index(drop=True)
    )
    df["T"] = pro_rata(df, SD)
    df["F"] = pro_rata(df, SFD)

    p1 = df["period"] == "P1"
    best = {}
    for rule, keys in [("SMO", SD), ("SFMO", SFD)]:
        tune = []
        for s in TAU_GRID:
            alloc = soft_rule(df[p1], keys, s)
            w = scores(df[p1].assign(tmp=alloc), "tmp")["within_R2"]
            tune.append((s, w))
            logger.info(f"{rule} tau={s:<5} P1 within_R2={w:.3f}")
        best[rule] = max(tune, key=lambda t: t[1])[0]
        df[rule] = soft_rule(df, keys, best[rule])
    logger.info(f"Tau chosen on P1: {best}")

    rules = ["T", "F", "SMO", "SFMO"]
    if not args.no_ml:
        df["ML"] = ml_rule(df, SD, args.folds, args.seed)
        df["MLF"] = ml_rule(df, SFD, args.folds, args.seed)
        rules += ["ML", "MLF"]

    rows = []
    for per, sub in df[df["period"] != "P1"].groupby("period"):
        for rule in rules:
            rows.append({"period": per, "rule": rule} | scores(sub, rule))
    res = pd.DataFrame(rows)
    res.to_csv(
        data_dir("processed", "identifiability_audit_v2.csv"),
        index=False,
        float_format="%.4f",
    )
    for metric in ["within_R2", "R2", "WAPE", "amp", "coal_share_MAE"]:
        print(
            f"\n=== {metric} (P2-P4: tau and ML fitted on P1; ML on held-out states) ==="
        )
        print(
            res.pivot(index="rule", columns="period", values=metric)
            .reindex(rules)
            .round(3)
            .to_string()
        )


if __name__ == "__main__":
    main()
