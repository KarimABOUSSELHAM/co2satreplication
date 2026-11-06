"""Go/no-go test: how much decarbonization does a capacity target buy?

Data: Ember yearly electricity data (CC-BY-4.0), ~200 countries, 2000-2025.

Setting: at year t, a country's capacity in year t+5 is taken as known (as if
its capacity plan were exactly delivered). What will its fossil share of
generation and its emissions intensity be at t+5?

Predictors, from naive to learned:
  CAP     the capacity-target view: fossil share of generation at t+5 =
          fossil share of CAPACITY at t+5 (what a capacity target implies
          when read as a decarbonization target)
  ENG     engineering accounting: clean generation = capacity(t+5) x the
          country's capacity factor at t, per technology; demand grows at
          its past 5-year trend; net imports stay at their level of t;
          fossil = demand - imports - clean; intensity = fossil share x
          fossil intensity at t
  ML      CatBoost on the same information (levels at t, capacities at t+5,
          past trends, region), countries held out (5-fold by country)
  HYB     CatBoost predicting the ENG error (physics + learned correction)

Decision rules, fixed before the first run:
  LEAK    the capacity view is materially misleading if the MAE of CAP on the
          fossil share is >= 2 x the MAE of ENG
  GO      ML or HYB reduce the MAE of ENG by >= 20 % on held-out countries,
          for the fossil share AND for the emissions intensity
  NO-GO   reduction < 10 %: the pipeline is accounting, not learning
  GREY    in between

Usage:
  python leak_test.py --file ember.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

H = 5
TECH = ["Solar", "Wind", "Hydro", "Bioenergy", "Other renewables", "Nuclear"]
FOSSIL = ["Coal", "Gas", "Other fossil"]


def load(path: str) -> pd.DataFrame:
    """Country-year wide table.

    Args:
        path (str): Ember yearly CSV.

    Returns:
        pd.DataFrame: one row per country-year, generation (TWh), capacity
        (GW), emissions (Mt) by source, demand and net imports.
    """
    d = pd.read_csv(path, low_memory=False)
    d = d[d["Area type"] == "Country or economy"]
    g = d.pivot_table(
        index=["Area", "Year"], columns="Electricity source", values="Generation (TWh)"
    )
    c = d.pivot_table(
        index=["Area", "Year"], columns="Electricity source", values="Capacity (GW)"
    )
    e = d.pivot_table(
        index=["Area", "Year"],
        columns="Electricity source",
        values="Emissions (MtCO2e)",
    )
    w = pd.concat(
        [g.add_prefix("g_"), c.add_prefix("c_"), e.add_prefix("e_")], axis=1
    ).reset_index()
    meta = d.groupby("Area")[["Continent", "OECD member"]].first()
    return w.merge(meta, left_on="Area", right_index=True)


def build(w: pd.DataFrame) -> pd.DataFrame:
    """Pairs (t, t+5) with features, targets and the CAP and ENG predictions.

    Args:
        w (pd.DataFrame): country-year table.

    Returns:
        pd.DataFrame: one row per country and base year t.
    """
    w = w.sort_values(["Area", "Year"]).set_index(["Area", "Year"])
    # tot = w["g_Total generation"]
    rows = []
    for (area, t), r in w.iterrows():
        if (area, t + H) not in w.index or (area, t - H) not in w.index:
            continue
        f = w.loc[(area, t + H)]
        p = w.loc[(area, t - H)]
        if not (r["g_Total generation"] >= 1 and f["g_Total generation"] >= 1):
            continue
        if (
            pd.isna(f.get("c_Fossil"))
            or pd.isna(f.get("c_Clean"))
            or pd.isna(r.get("e_Fossil"))
        ):
            continue
        row = {
            "Area": area,
            "t": t,
            "continent": r["Continent"],
            "oecd": bool(r["OECD member"]),
        }
        # Targets at t+5
        row["y_fossil_share"] = f["g_Fossil"] / f["g_Total generation"]
        row["y_intensity"] = (
            1000 * f["e_Fossil"] / f["g_Total generation"]
            if pd.notna(f["e_Fossil"])
            else np.nan
        )
        # CAP view
        cap_tot = f["c_Fossil"] + f["c_Clean"]
        row["CAP_fossil_share"] = f["c_Fossil"] / cap_tot if cap_tot > 0 else np.nan
        # ENG accounting
        clean = 0.0
        for k in TECH:
            cap_f, cap_t, gen_t = (
                f.get(f"c_{k}", 0) or 0,
                r.get(f"c_{k}", 0) or 0,
                r.get(f"g_{k}", 0) or 0,
            )
            cf = gen_t / (cap_t * 8.76) if cap_t and cap_t > 0.01 else np.nan
            row[f"cf_{k}"] = cf
            row[f"dcap_{k}"] = cap_f - cap_t
            clean += cap_f * 8.76 * (cf if pd.notna(cf) else 0.0)
        growth = (
            r["g_Demand"] / p["g_Demand"]
            if p["g_Demand"] and p["g_Demand"] > 0
            else np.nan
        )
        row["demand_growth_past"] = growth
        dem_f = r["g_Demand"] * (growth if pd.notna(growth) else 1.0)
        imp = r.get("g_Net imports", 0) or 0
        tot_f = max(dem_f - imp, 1e-6)
        fos_f = min(max(tot_f - clean, 0.0), tot_f)
        row["ENG_fossil_share"] = fos_f / tot_f
        fos_int_t = (
            1000 * r["e_Fossil"] / r["g_Fossil"] if r["g_Fossil"] > 0 else np.nan
        )
        row["fossil_int_t"] = fos_int_t
        row["ENG_intensity"] = (
            row["ENG_fossil_share"] * fos_int_t if pd.notna(fos_int_t) else np.nan
        )
        # Other features known at t (plus planned capacity at t+5)
        row["fossil_share_t"] = r["g_Fossil"] / r["g_Total generation"]
        row["coal_share_t"] = (r.get("g_Coal", 0) or 0) / r["g_Total generation"]
        row["gas_share_t"] = (r.get("g_Gas", 0) or 0) / r["g_Total generation"]
        row["intensity_t"] = 1000 * r["e_Fossil"] / r["g_Total generation"]
        row["imports_share_t"] = imp / r["g_Demand"] if r["g_Demand"] else np.nan
        row["log_demand_t"] = np.log(r["g_Demand"]) if r["g_Demand"] > 0 else np.nan
        row["clean_cap_share_f"] = f["c_Clean"] / cap_tot if cap_tot > 0 else np.nan
        row["dcap_clean_rel"] = (f["c_Clean"] - r["c_Clean"]) / max(
            r["c_Fossil"] + r["c_Clean"], 1e-3
        )
        row["dcap_fossil_rel"] = (f["c_Fossil"] - r["c_Fossil"]) / max(
            r["c_Fossil"] + r["c_Clean"], 1e-3
        )
        rows.append(row)
    return pd.DataFrame(rows)


FEATS = [
    "t",
    "continent",
    "oecd",
    "fossil_share_t",
    "coal_share_t",
    "gas_share_t",
    "intensity_t",
    "imports_share_t",
    "log_demand_t",
    "demand_growth_past",
    "clean_cap_share_f",
    "dcap_clean_rel",
    "dcap_fossil_rel",
    "fossil_int_t",
    "ENG_fossil_share",
    "CAP_fossil_share",
] + [f"cf_{k}" for k in TECH]


def grouped_predict(
    df: pd.DataFrame, target: str, base: str | None, folds: int = 5, seed: int = 0
) -> np.ndarray:
    """CatBoost with countries held out.

    Args:
        df (pd.DataFrame): pairs table.
        target (str): target column.
        base (str | None): if given, the model learns target - base (hybrid).
        folds (int): number of country folds.
        seed (int): random seed.

    Returns:
        np.ndarray: out-of-fold predictions.
    """
    from catboost import CatBoostRegressor

    X = df[FEATS].copy()
    X["continent"] = X["continent"].fillna("na").astype(str)
    y = df[target] - (df[base] if base else 0)
    areas = df["Area"].unique()
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(areas), np.arange(len(areas)) % folds))
    fold = df["Area"].map(fold_of).to_numpy()
    out = np.full(len(df), np.nan)
    for k in range(folds):
        tr, te = fold != k, fold == k
        m = CatBoostRegressor(
            iterations=800,
            depth=6,
            learning_rate=0.05,
            loss_function="MAE",
            random_seed=seed,
            verbose=False,
            cat_features=["continent"],
        )
        m.fit(X[tr], y[tr])
        out[te] = m.predict(X[te])
    return out + (df[base].to_numpy() if base else 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    args = ap.parse_args()

    df = build(load(args.file))
    df = df.dropna(
        subset=[
            "y_fossil_share",
            "y_intensity",
            "ENG_fossil_share",
            "ENG_intensity",
            "CAP_fossil_share",
        ]
    )
    print(
        f"Pairs (t, t+{H}): {len(df):,} | countries: {df['Area'].nunique()} | base years {df['t'].min()}-{df['t'].max()}"
    )

    df["ML_fossil_share"] = grouped_predict(df, "y_fossil_share", None).clip(0, 1)
    df["HYB_fossil_share"] = grouped_predict(
        df, "y_fossil_share", "ENG_fossil_share"
    ).clip(0, 1)
    df["ML_intensity"] = grouped_predict(df, "y_intensity", None).clip(0, None)
    df["HYB_intensity"] = grouped_predict(df, "y_intensity", "ENG_intensity").clip(
        0, None
    )
    df["PERSIST_fossil_share"] = df["fossil_share_t"]
    df["PERSIST_intensity"] = df["intensity_t"]

    print(
        "\n=== Fossil share of generation at t+5: mean absolute error (percentage points) ==="
    )
    res = {}
    for m in ["CAP", "PERSIST", "ENG", "ML", "HYB"]:
        res[m] = 100 * (df[f"{m}_fossil_share"] - df["y_fossil_share"]).abs().mean()
        bias = 100 * (df[f"{m}_fossil_share"] - df["y_fossil_share"]).mean()
        print(f"  {m:8s} MAE {res[m]:5.2f} pp   bias {bias:+5.2f} pp")
    print("\n=== Emissions intensity at t+5: mean absolute error (gCO2/kWh) ===")
    resi = {}
    for m in ["PERSIST", "ENG", "ML", "HYB"]:
        resi[m] = (df[f"{m}_intensity"] - df["y_intensity"]).abs().mean()
        bias = (df[f"{m}_intensity"] - df["y_intensity"]).mean()
        print(f"  {m:8s} MAE {resi[m]:6.1f}   bias {bias:+6.1f}")

    best_s = min(res["ML"], res["HYB"])
    best_i = min(resi["ML"], resi["HYB"])
    gain_s, gain_i = 1 - best_s / res["ENG"], 1 - best_i / resi["ENG"]
    print("\n=== Decision ===")
    print(
        f"  LEAK: CAP / ENG MAE on fossil share = {res['CAP'] / res['ENG']:.2f} (material if >= 2)"
    )
    print(f"  ML gain vs ENG: fossil share {gain_s:+.1%}, intensity {gain_i:+.1%}")
    verdict = (
        "GO"
        if min(gain_s, gain_i) >= 0.20
        else ("NO-GO" if max(gain_s, gain_i) < 0.10 else "GREY")
    )
    print(f"  Verdict: {verdict}")

    mar = df[df["Area"] == "Morocco"].sort_values("t")
    if len(mar):
        print("\n=== Morocco (fossil share at t+5, %) ===")
        cols = [
            "t",
            "y_fossil_share",
            "CAP_fossil_share",
            "ENG_fossil_share",
            "ML_fossil_share",
            "HYB_fossil_share",
        ]
        print((mar[cols].set_index("t") * 100).round(1).to_string())
    df.to_csv("leak_test_pairs.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
