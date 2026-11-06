"""Go/no-go, step 1: can a country's demand uncertainty be learned from the
other countries?

Target: 5-year electricity demand growth ratio, D(t+5)/D(t), from Ember
yearly data (~125 countries, base years 1995-2020).

Three interval constructions at nominal coverage 90 %, all evaluated on
countries held out of training (5 folds by country) and on a strict
temporal split (train base years <= 2012, test base years >= 2013):

  NAIVE   global empirical 5 %-95 % quantiles of the growth ratio, same
          interval for every country (what a planner does without a model)
  CQR     conformalized quantile regression: CatBoost quantile models
          (5 %, 95 %) on features known at t, then split-conformal
          correction so that 90 % coverage holds on exchangeable data
  CQR-W   same, with the conformal correction weighted by similarity between
          the calibration countries and the target (continent, income proxy:
          demand per capita is not in the file, so we use log demand, demand
          growth history and fossil share) -- the Tibshirani et al. 2019
          weighted conformal idea, light version

Decision rules (fixed before the run):
  GO      held-out coverage of CQR in [87 %, 93 %] on BOTH splits AND mean
          width <= 0.75 x NAIVE width
  NO-GO   coverage < 85 % or > 95 % on a split, OR width >= 0.90 x NAIVE
  GREY    otherwise

Also reported: coverage by continent (is Africa covered as well as Europe?)
and Morocco's own intervals vs realized growth.

Usage:
  python conformal_demand_test.py --file ember.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

H = 5
ALPHA = 0.10


def load_pairs(path: str) -> pd.DataFrame:
    """Country-year pairs (t, t+5) with features known at t.

    Args:
        path (str): Ember yearly CSV.

    Returns:
        pd.DataFrame: one row per country and base year.
    """
    d = pd.read_csv(path, low_memory=False)
    d = d[d["Area type"] == "Country or economy"]
    g = d.pivot_table(
        index=["Area", "Year"], columns="Electricity source", values="Generation (TWh)"
    )
    meta = d.groupby("Area")[["Continent", "OECD member"]].first()
    g = g.reset_index().merge(meta, left_on="Area", right_index=True)
    g = g.sort_values(["Area", "Year"]).set_index(["Area", "Year"])
    rows = []
    for (area, t), r in g.iterrows():
        keys = [(area, t + H), (area, t - 1), (area, t - H), (area, t - 2 * H)]
        if any(k not in g.index for k in keys):
            continue
        f, p1, p5, p10 = (g.loc[k] for k in keys)
        if not (
            r["Demand"] >= 1
            and f["Demand"] > 0
            and p5["Demand"] > 0
            and p10["Demand"] > 0
        ):
            continue
        rows.append(
            {
                "Area": area,
                "t": t,
                "continent": r["Continent"] if pd.notna(r["Continent"]) else "na",
                "oecd": bool(r["OECD member"]),
                "log_demand": np.log(r["Demand"]),
                "g1": r["Demand"] / p1["Demand"],
                "g5": r["Demand"] / p5["Demand"],
                "g10": p5["Demand"] / p10["Demand"],
                "fossil_share": r["Fossil"] / r["Total generation"]
                if r["Total generation"] > 0
                else np.nan,
                "imports_share": (r.get("Net imports", 0) or 0) / r["Demand"],
                "y": f["Demand"] / r["Demand"],
            }
        )
    return pd.DataFrame(rows).dropna()


FEATS = [
    "t",
    "continent",
    "oecd",
    "log_demand",
    "g1",
    "g5",
    "g10",
    "fossil_share",
    "imports_share",
]


def fit_quantiles(
    tr: pd.DataFrame, te: pd.DataFrame, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """CatBoost quantile models for the 5 % and 95 % quantiles.

    Args:
        tr (pd.DataFrame): training rows.
        te (pd.DataFrame): rows to predict.
        seed (int): random seed.

    Returns:
        tuple[np.ndarray, np.ndarray]: lower and upper quantile predictions.
    """
    from catboost import CatBoostRegressor

    out = []
    for q in (ALPHA / 2, 1 - ALPHA / 2):
        m = CatBoostRegressor(
            iterations=500,
            depth=4,
            learning_rate=0.05,
            loss_function=f"Quantile:alpha={q}",
            random_seed=seed,
            verbose=False,
            cat_features=["continent"],
        )
        m.fit(tr[FEATS], tr["y"])
        out.append(m.predict(te[FEATS]))
    lo, hi = out
    return np.minimum(lo, hi), np.maximum(lo, hi)


def similarity_weights(cal: pd.DataFrame, x: pd.Series) -> np.ndarray:
    """Weights of calibration rows for one target row (weighted conformal).

    Args:
        cal (pd.DataFrame): calibration rows.
        x (pd.Series): target row.

    Returns:
        np.ndarray: non-negative weights, larger for similar countries.
    """
    num = ["log_demand", "g5", "fossil_share"]
    z = (cal[num].to_numpy(dtype=float) - x[num].to_numpy(dtype=float)) / cal[
        num
    ].std().replace(0, 1).to_numpy(dtype=float)
    d2 = (z**2).sum(axis=1)
    same = (cal["continent"] == x["continent"]).to_numpy()
    return np.exp(-0.5 * d2) * np.where(same, 1.0, 0.3) + 1e-6


def weighted_quantile(scores: np.ndarray, w: np.ndarray, q: float) -> float:
    """Weighted empirical quantile (with the +inf point of conformal theory).

    Args:
        scores (np.ndarray): nonconformity scores.
        w (np.ndarray): weights.
        q (float): quantile level.

    Returns:
        float: quantile.
    """
    order = np.argsort(scores)
    s, w = scores[order], w[order]
    cum = np.cumsum(w) / (w.sum() + w.max())
    idx = np.searchsorted(cum, q)
    return float(s[idx]) if idx < len(s) else float(s[-1])


def conformalize(tr: pd.DataFrame, te: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Split-conformal CQR, plain and similarity-weighted, for held-out rows.

    Args:
        tr (pd.DataFrame): training rows (countries disjoint from te).
        te (pd.DataFrame): test rows.
        seed (int): random seed.

    Returns:
        pd.DataFrame: te with interval bounds for CQR and CQR-W.
    """
    rng = np.random.default_rng(seed)
    areas = tr["Area"].unique()
    cal_areas = set(rng.choice(areas, size=len(areas) // 3, replace=False))
    fit, cal = tr[~tr["Area"].isin(cal_areas)], tr[tr["Area"].isin(cal_areas)]
    lo_c, hi_c = fit_quantiles(fit, cal, seed)
    scores = np.maximum(lo_c - cal["y"].to_numpy(), cal["y"].to_numpy() - hi_c)
    lo_t, hi_t = fit_quantiles(fit, te, seed)
    n = len(scores)
    q_plain = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)))
    te = te.copy()
    te["cqr_lo"], te["cqr_hi"] = lo_t - q_plain, hi_t + q_plain
    qw = np.array(
        [
            weighted_quantile(scores, similarity_weights(cal, x), 1 - ALPHA)
            for _, x in te.iterrows()
        ]
    )
    te["cqrw_lo"], te["cqrw_hi"] = lo_t - qw, hi_t + qw
    return te


def naive(tr: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    """Global empirical interval, same for everyone.

    Args:
        tr (pd.DataFrame): training rows.
        te (pd.DataFrame): test rows.

    Returns:
        pd.DataFrame: te with naive bounds.
    """
    te = te.copy()
    te["naive_lo"], te["naive_hi"] = (
        np.quantile(tr["y"], ALPHA / 2),
        np.quantile(tr["y"], 1 - ALPHA / 2),
    )
    return te


def report(df: pd.DataFrame, label: str) -> dict[str, float]:
    """Coverage and width of the three intervals.

    Args:
        df (pd.DataFrame): rows with bounds.
        label (str): split name.

    Returns:
        dict[str, float]: coverage and width per method.
    """
    out = {}
    print(f"\n=== {label}: n = {len(df):,}, countries = {df['Area'].nunique()} ===")
    print(f"{'method':8s} {'coverage':>9s} {'width':>7s} {'width/naive':>12s}")
    wn = (df["naive_hi"] - df["naive_lo"]).mean()
    for m in ["naive", "cqr", "cqrw"]:
        cov = ((df["y"] >= df[f"{m}_lo"]) & (df["y"] <= df[f"{m}_hi"])).mean()
        wid = (df[f"{m}_hi"] - df[f"{m}_lo"]).mean()
        out[m] = {"cov": cov, "width": wid, "rel": wid / wn}
        print(f"{m:8s} {cov:9.1%} {wid:7.3f} {wid / wn:12.2f}")
    print("\nCoverage by continent (naive / cqr / cqrw):")
    for c, g in df.groupby("continent"):
        covs = [
            ((g["y"] >= g[f"{m}_lo"]) & (g["y"] <= g[f"{m}_hi"])).mean()
            for m in ["naive", "cqr", "cqrw"]
        ]
        print(f"  {c:15s} n={len(g):4d}  " + " / ".join(f"{c_:.0%}" for c_ in covs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_pairs(args.file)
    print(
        f"Pairs: {len(df):,} | countries: {df['Area'].nunique()} | base years {df['t'].min()}-{df['t'].max()}"
    )
    print(
        f"5-year demand growth ratio: median {df['y'].median():.3f}, 5 %-95 % {df['y'].quantile(0.05):.3f}-{df['y'].quantile(0.95):.3f}"
    )

    # Split A: countries held out (5 folds)
    rng = np.random.default_rng(args.seed)
    areas = df["Area"].unique()
    fold_of = dict(zip(rng.permutation(areas), np.arange(len(areas)) % 5))
    df["fold"] = df["Area"].map(fold_of)
    parts = []
    for k in range(5):
        tr, te = df[df["fold"] != k], df[df["fold"] == k]
        parts.append(naive(tr, conformalize(tr, te, args.seed)))
    a = pd.concat(parts)
    res_a = report(a, "Split A: countries never seen")

    # Split B: temporal (train t <= 2012, test t >= 2013), countries may be seen
    tr, te = df[df["t"] <= 2012], df[df["t"] >= 2013]
    b = naive(tr, conformalize(tr, te, args.seed))
    res_b = report(b, "Split B: base years >= 2013 (trained on <= 2012)")

    print("\n=== Decision (CQR) ===")
    ok_cov = all(0.87 <= r["cqr"]["cov"] <= 0.93 for r in (res_a, res_b))
    ok_w = all(r["cqr"]["rel"] <= 0.75 for r in (res_a, res_b))
    bad_cov = any(
        r["cqr"]["cov"] < 0.85 or r["cqr"]["cov"] > 0.95 for r in (res_a, res_b)
    )
    bad_w = any(r["cqr"]["rel"] >= 0.90 for r in (res_a, res_b))
    verdict = "GO" if ok_cov and ok_w else ("NO-GO" if bad_cov or bad_w else "GREY")
    print(
        f"  coverage in [87, 93] on both splits: {ok_cov} | width <= 0.75 x naive on both: {ok_w} -> {verdict}"
    )

    m = a[a["Area"] == "Morocco"].sort_values("t")
    if len(m):
        print(
            "\n=== Morocco, countries-never-seen split: 5-year demand growth ratio ==="
        )
        cols = [
            "t",
            "y",
            "naive_lo",
            "naive_hi",
            "cqr_lo",
            "cqr_hi",
            "cqrw_lo",
            "cqrw_hi",
        ]
        print(m[cols].set_index("t").round(3).to_string())
    a.to_csv("conformal_demand_pairs.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
