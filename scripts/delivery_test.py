"""Go/no-go test: can machine learning predict which announced projects get built?

Data: LBNL "Queued Up" project-level file (US interconnection queues, every
request since the early 2000s, with queue date, status, operation and
withdrawal dates, technology, capacity, region, state).

Question: for a project entering the queue, predict whether it reaches
commercial operation within --horizon years. A model is only useful if it
beats the BASE RATE (share of projects of the same technology, region and
entry period that got built): if the base rate is as good, delivery risk is a
table of averages, not a machine learning problem (the trap of the previous
tests). The decisive metric is the within-cell AUC: can the model rank
projects that the base rate treats as identical?

Splits:
  temporal   train on entry years <= --split-year, test on the later years
  spatial    leave one region out (transfer to an unseen system, as for a
             country with few projects such as Morocco)

Models:
  rate_global  one rate for all projects
  rate_cell    rate per technology x region x entry period (the baseline)
  logit        logistic regression on the same features as the boosting
  catboost     gradient boosting

Features use only what is known at queue entry (no current status, no
interconnection-agreement status, no dates after entry).

Decision rule (set before the run):
  GO       catboost Brier skill vs rate_cell >= 0.10 AND within-cell AUC
           >= 0.65, on both splits
  NO-GO    Brier skill < 0.05 OR within-cell AUC < 0.58
  GREY     otherwise

Usage:
  uv run scripts/delivery_test.py --file data/raw/queued_up.xlsx --inspect
  uv run scripts/delivery_test.py --file data/raw/queued_up.xlsx
  uv run scripts/delivery_test.py --file ... --sheet data --horizon 6 --split-year 2014
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from loguru import logger

# Candidate column names (the LBNL file has changed slightly across editions).
CANDIDATES = {
    "status": ["q_status", "queue_status", "status"],
    "q_date": ["q_date", "queue_date", "date_entered_queue"],
    "on_date": ["on_date", "cod", "operation_date", "online_date"],
    "wd_date": ["wd_date", "withdrawn_date", "withdrawal_date"],
    "type": ["type_clean", "type", "resource_type", "type1"],
    "mw": ["mw1", "mw_1", "capacity_mw", "mw", "summer_mw"],
    "region": ["region", "entity", "iso", "rto"],
    "state": ["state"],
    "developer": ["developer", "dev", "owner"],
    "service": ["service", "service_type"],
    "poi": ["poi_name", "poi", "point_of_interconnection"],
}


def pick(cols: list[str], key: str) -> str | None:
    """Find the column matching one of the candidate names.

    Args:
        cols (list[str]): columns of the file (lower-cased).
        key (str): logical name.

    Returns:
        str | None: the matching column, or None.
    """
    return next((c for c in CANDIDATES[key] if c in cols), None)


def to_date(s: pd.Series) -> pd.Series:
    """Parse dates that may be strings, datetimes or Excel serial numbers.

    Args:
        s (pd.Series): raw date column.

    Returns:
        pd.Series: datetime64 values (NaT when unparseable).
    """
    num = pd.to_numeric(s, errors="coerce")
    serial = pd.to_datetime(
        num.where(num.between(20000, 80000)), unit="D", origin="1899-12-30"
    )
    text = pd.to_datetime(s.where(num.isna()), errors="coerce")
    return serial.fillna(text)


def load(path: str, sheet: str | None) -> pd.DataFrame:
    """Read the LBNL file and standardize column names.

    Args:
        path (str): xlsx or csv path.
        sheet (str | None): sheet name for xlsx (None = first sheet named
            like 'data', else the first sheet).

    Returns:
        pd.DataFrame: raw table with lower-case column names.
    """
    if path.endswith(".csv"):
        df = pd.read_csv(path, low_memory=False)
    else:
        xl = pd.ExcelFile(path)
        name = sheet or next(
            (s for s in xl.sheet_names if "complete queue" in s.lower()),
            next((s for s in xl.sheet_names if "data" in s.lower()), xl.sheet_names[0]),
        )
        logger.info(f"Reading sheet '{name}'")
        df = xl.parse(name, header=None)
        # The LBNL sheets carry title rows above the table: find the header row.
        known = {c for names in CANDIDATES.values() for c in names} | {"q_id"}
        hits = df.apply(
            lambda r: r.astype(str).str.strip().str.lower().isin(known).sum(), axis=1
        )
        h = int(hits.idxmax()) if hits.max() >= 3 else 0
        df.columns = df.iloc[h]
        df = df.iloc[h + 1 :].reset_index(drop=True)
        logger.info(f"Header found on row {h + 1}; {len(df):,} rows")
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def prepare(raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Build the modelling table: one row per request, outcome and features.

    Args:
        raw (pd.DataFrame): raw LBNL table.
        horizon (int): years after queue entry within which operation counts.

    Returns:
        pd.DataFrame: entry year, outcome 'built', and features.
    """
    cols = list(raw.columns)
    m = {k: pick(cols, k) for k in CANDIDATES}
    logger.info(f"Column mapping: {m}")
    for k in ["status", "q_date", "type", "mw", "region"]:
        if m[k] is None:
            raise SystemExit(
                f"Column for '{k}' not found; run with --inspect and adapt CANDIDATES."
            )
    df = pd.DataFrame(
        {
            "status": raw[m["status"]].astype(str).str.lower().str.strip(),
            "q_date": to_date(raw[m["q_date"]]),
            "type": raw[m["type"]].astype(str).str.lower().str.strip(),
            "mw": pd.to_numeric(raw[m["mw"]], errors="coerce"),
            "region": raw[m["region"]].astype(str).str.strip(),
        }
    )
    df["on_date"] = to_date(raw[m["on_date"]]) if m["on_date"] else pd.NaT
    df["state"] = raw[m["state"]].astype(str).str.strip() if m["state"] else "na"
    df["service"] = (
        raw[m["service"]].astype(str).str.lower().str.strip() if m["service"] else "na"
    )
    df["developer"] = (
        raw[m["developer"]].astype(str).str.lower().str.strip()
        if m["developer"]
        else ""
    )
    df["poi"] = raw[m["poi"]].astype(str).str.lower().str.strip() if m["poi"] else ""
    df = df.dropna(subset=["q_date", "mw"])
    df = df[df["mw"] > 0]
    df["year"] = df["q_date"].dt.year
    logger.info(f"Statuses: {df['status'].value_counts().head(8).to_dict()}")
    logger.info(
        f"Queue dates parsed: {df['q_date'].min():%Y-%m-%d} to {df['q_date'].max():%Y-%m-%d}"
    )

    deadline = df["q_date"] + pd.DateOffset(years=horizon)
    op = df["status"].str.contains("operat")
    on_ok = df["on_date"].notna() & (df["on_date"] <= deadline)
    # Operational without a date: counted as built (minor; reported).
    df["built"] = (on_ok | (op & df["on_date"].isna())).astype(int)
    logger.info(
        f"Operational without on_date counted as built: {int((op & df['on_date'].isna()).sum())}"
    )

    # Context known at entry: how crowded the queue was, developer track record.
    df = df.sort_values("q_date").reset_index(drop=True)
    df["n_region_year"] = df.groupby(["region", "year"])["mw"].transform("size")
    df["mw_region_year"] = df.groupby(["region", "year"])["mw"].transform("sum")
    df["n_state_year"] = df.groupby(["state", "year"])["mw"].transform("size")
    df["n_poi"] = df.groupby("poi")["mw"].cumcount() if m["poi"] else 0
    has_dev = df["developer"].ne("") & df["developer"].ne("nan")
    df["dev_prior"] = np.where(has_dev, df.groupby("developer").cumcount(), -1)
    df["log_mw"] = np.log1p(df["mw"])
    df["share_type_region_year"] = (
        df.groupby(["region", "year", "type"])["mw"].transform("size")
        / df["n_region_year"]
    )
    df["period"] = pd.cut(
        df["year"],
        [0, 2009, 2013, 2017, 2100],
        labels=["<=09", "10-13", "14-17", "18+"],
    ).astype(str)
    return df


FEATS = [
    "type",
    "region",
    "state",
    "service",
    "year",
    "log_mw",
    "n_region_year",
    "mw_region_year",
    "n_state_year",
    "n_poi",
    "dev_prior",
    "share_type_region_year",
]
CATS = ["type", "region", "state", "service"]


def clean_cats(X: pd.DataFrame) -> pd.DataFrame:
    """Categorical columns as plain strings, missing values as 'na'.

    With recent pandas, astype(str) keeps missing values as NaN, which
    CatBoost refuses in categorical features.

    Args:
        X (pd.DataFrame): feature matrix.

    Returns:
        pd.DataFrame: copy with clean categorical columns.
    """
    X = X.copy()
    for c in CATS:
        X[c] = X[c].astype(object).where(X[c].notna(), "na").map(str)
        X.loc[X[c].isin(["nan", "None", ""]), c] = "na"
    return X


def cell_rate(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """Base rate per technology x region x period, with fallbacks.

    Args:
        tr (pd.DataFrame): training rows.
        te (pd.DataFrame): test rows.

    Returns:
        np.ndarray: predicted probability for test rows.
    """
    g0 = tr["built"].mean()
    out = pd.Series(g0, index=te.index, dtype=float)
    for keys in (["type"], ["type", "region"], ["type", "region", "period"]):
        r = tr.groupby(keys)["built"].agg(["mean", "size"])
        r = r[r["size"] >= 20]["mean"]
        hit = te.set_index(keys).index.map(r.to_dict()).to_numpy(dtype=float)
        out = pd.Series(np.where(np.isnan(hit), out, hit), index=te.index)
    return out.to_numpy()


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    """Fit the four models and predict the test rows.

    Args:
        tr (pd.DataFrame): training rows.
        te (pd.DataFrame): test rows.
        seed (int): random seed.

    Returns:
        dict[str, np.ndarray]: probabilities per model.
    """
    from catboost import CatBoostClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num = [f for f in FEATS if f not in CATS]
    logit = make_pipeline(
        ColumnTransformer(
            [
                ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATS),
                (
                    "n",
                    make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                    num,
                ),
            ]
        ),
        LogisticRegression(max_iter=2000, C=0.5),
    )
    Xtr, Xte = clean_cats(tr[FEATS]), clean_cats(te[FEATS])
    logit.fit(Xtr, tr["built"])
    cb = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        random_seed=seed,
        verbose=False,
        cat_features=CATS,
    )
    cb.fit(Xtr, tr["built"])
    return {
        "rate_global": np.full(len(te), tr["built"].mean()),
        "rate_cell": cell_rate(tr, te),
        "logit": logit.predict_proba(Xte)[:, 1],
        "catboost": cb.predict_proba(Xte)[:, 1],
    }


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC (NaN if one class only).

    Args:
        y (np.ndarray): outcomes.
        p (np.ndarray): scores.

    Returns:
        float: AUC.
    """
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan


def evaluate(te: pd.DataFrame, preds: dict[str, np.ndarray]) -> list[dict]:
    """Metrics per model: Brier, Brier skill vs rate_cell, AUC, within-cell
    AUC, calibration gap, and built capacity predicted vs observed.

    Args:
        te (pd.DataFrame): test rows.
        preds (dict[str, np.ndarray]): probabilities per model.

    Returns:
        list[dict]: one row per model.
    """
    y = te["built"].to_numpy()
    ref = ((preds["rate_cell"] - y) ** 2).mean()
    cells = te.groupby(["type", "region", "period"]).indices
    rows = []
    for name, p in preds.items():
        brier = ((p - y) ** 2).mean()
        w_auc, w_n = [], []
        for idx in cells.values():
            a = auc(y[idx], p[idx])
            if not np.isnan(a) and len(idx) >= 30:
                w_auc.append(a)
                w_n.append(len(idx))
        bins = pd.qcut(p, 10, duplicates="drop")
        cal = (
            pd.DataFrame({"p": p, "y": y, "b": bins}).groupby("b", observed=True).mean()
        )
        rows.append(
            {
                "model": name,
                "n_test": len(y),
                "base_rate": y.mean(),
                "brier": brier,
                "brier_skill_vs_cell": 1 - brier / ref,
                "auc": auc(y, p),
                "within_cell_auc": np.average(w_auc, weights=w_n) if w_auc else np.nan,
                "calib_gap": float((cal["p"] - cal["y"]).abs().mean()),
                "mw_pred_over_obs": float(
                    (p * te["mw"]).sum() / max((y * te["mw"]).sum(), 1.0)
                ),
            }
        )
    return rows


def warm_test(
    df: pd.DataFrame, warm_year: int, seed: int, local_sizes: list[int] | None = None
) -> None:
    """Does a little local history rescue transfer to a new region?

    For each region R: test rows = projects of R entering after warm_year.
    cold: train on the OTHER regions' projects up to warm_year.
    warm: same, plus R's own projects up to warm_year (a country that has a
    short project history, like Morocco's registry).
    Same test rows for both, so the comparison is fair.

    Args:
        df (pd.DataFrame): modelling table.
        warm_year (int): last entry year available for training.
        seed (int): random seed.
        local_sizes (list[int] | None): if given, also train with only n
            randomly drawn local projects (3 draws each): how much local
            history is needed?
    """
    rows = []
    rng = np.random.default_rng(seed)
    past = df[df["year"] <= warm_year]
    regions = df["region"].value_counts()
    for reg in regions[regions >= 300].index:
        te = df[(df["region"] == reg) & (df["year"] > warm_year)]
        n_local = int((past["region"] == reg).sum())
        if len(te) < 100:
            continue
        others, local = past[past["region"] != reg], past[past["region"] == reg]
        runs = [("cold", others), ("warm", past)]
        for n in local_sizes or []:
            if n < n_local:
                for k in range(3):
                    sub = local.iloc[rng.choice(n_local, n, replace=False)]
                    runs.append((f"local_{n:04d}", pd.concat([others, sub])))
        for mode, tr in runs:
            rows += [
                r | {"region": reg, "mode": mode}
                for r in evaluate(te, fit_predict(tr, te, seed))
            ]
        logger.info(
            f"{reg}: {n_local} local projects up to {warm_year}, {len(te)} test projects"
        )
    res = pd.DataFrame(rows)
    res.to_csv(
        f"data/processed/delivery_test_warm{warm_year}.csv",
        index=False,
        float_format="%.4f",
    )
    metrics = [
        "brier_skill_vs_cell",
        "auc",
        "within_cell_auc",
        "calib_gap",
        "mw_pred_over_obs",
    ]
    summ = res.groupby(["mode", "model"]).apply(
        lambda g: pd.Series(
            {c: np.average(g[c].astype(float), weights=g["n_test"]) for c in metrics}
        ),
        include_groups=False,
    )
    print(
        f"\n=== Transfer test: train <= {warm_year}, test = held-out region after {warm_year} (weighted) ==="
    )
    print(summ.round(3).to_string())
    if local_sizes:
        cb = res[res["model"] == "catboost"]
        curve = cb.groupby(["mode", "region"])[
            ["brier_skill_vs_cell", "within_cell_auc", "n_test"]
        ].mean()
        curve = curve.groupby("mode").apply(
            lambda g: pd.Series(
                {
                    c: np.average(g[c], weights=g["n_test"])
                    for c in ["brier_skill_vs_cell", "within_cell_auc"]
                }
            ),
            include_groups=False,
        )
        print(
            "\n=== Learning curve (catboost): how many local projects are needed? ==="
        )
        print(
            "(local_N = only N local projects added, mean of 3 draws; regions with fewer than N are skipped for that N)"
        )
        print(curve.round(3).to_string())
    print("\nBy region (catboost):")
    print(
        res[res["model"] == "catboost"]
        .groupby(["region", "mode"])[["brier_skill_vs_cell", "within_cell_auc"]]
        .mean()
        .unstack("mode")
        .round(3)
        .to_string()
    )
    print(
        "\nNote: brier_skill_vs_cell compares each model with the cell rate fitted on the SAME training set."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--split-year", type=int, default=2014)
    ap.add_argument("--first-year", type=int, default=2005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--warm-year",
        type=int,
        default=None,
        help="transfer test: train on all regions up to this year, with (warm) or "
        "without (cold) the target region's own history; test the target region after it",
    )
    ap.add_argument(
        "--local-sizes",
        type=int,
        nargs="+",
        default=None,
        help="with --warm-year: learning curve on the number of local projects",
    )
    args = ap.parse_args()

    raw = load(args.file, args.sheet)
    if args.inspect:
        print(raw.shape)
        print(raw.dtypes.to_string())
        for c in raw.columns:
            if raw[c].dtype == object and raw[c].nunique() < 40:
                print(
                    f"\n--- {c} ---\n{raw[c].value_counts(dropna=False).head(15).to_string()}"
                )
        return

    df = prepare(raw, args.horizon)
    data_end = df["q_date"].max()
    last_year = data_end.year - args.horizon - 1
    df = df[(df["year"] >= args.first_year) & (df["year"] <= last_year)]
    logger.info(
        f"Requests {args.first_year}-{last_year}: {len(df):,} | built within {args.horizon} y: {df['built'].mean():.1%}"
    )
    print("\nBuilt share by technology (top 8):")
    print(
        df.groupby("type")["built"]
        .agg(["mean", "size"])
        .sort_values("size", ascending=False)
        .head(8)
        .round(3)
        .to_string()
    )

    if args.warm_year is not None:
        warm_test(df, args.warm_year, args.seed, args.local_sizes)
        return

    rows = []
    tr, te = df[df["year"] <= args.split_year], df[df["year"] > args.split_year]
    rows += [
        r | {"split": f"temporal (test > {args.split_year})"}
        for r in evaluate(te, fit_predict(tr, te, args.seed))
    ]
    spatial = []
    regions = df["region"].value_counts()
    for reg in regions[regions >= 300].index:
        tr, te = df[df["region"] != reg], df[df["region"] == reg]
        spatial += [
            r | {"region": reg} for r in evaluate(te, fit_predict(tr, te, args.seed))
        ]
        logger.info(f"spatial: {reg} done")
    sp = pd.DataFrame(spatial)
    w = (
        sp.groupby("model")
        .apply(
            lambda g: pd.Series(
                {
                    c: np.average(g[c].astype(float), weights=g["n_test"])
                    for c in [
                        "base_rate",
                        "brier",
                        "brier_skill_vs_cell",
                        "auc",
                        "within_cell_auc",
                        "calib_gap",
                        "mw_pred_over_obs",
                    ]
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    w["split"], w["n_test"] = (
        "spatial (leave-one-region-out, weighted)",
        sp.groupby("model")["n_test"].sum().values,
    )
    res = pd.concat([pd.DataFrame(rows), w], ignore_index=True)
    res.to_csv("data/processed/delivery_test.csv", index=False, float_format="%.4f")
    sp.to_csv(
        "data/processed/delivery_test_by_region.csv", index=False, float_format="%.4f"
    )
    cols = [
        "split",
        "model",
        "n_test",
        "base_rate",
        "brier",
        "brier_skill_vs_cell",
        "auc",
        "within_cell_auc",
        "calib_gap",
        "mw_pred_over_obs",
    ]
    print("\n" + res[cols].round(3).to_string(index=False))
    print("\nBy held-out region (catboost vs rate_cell):")
    print(
        sp[sp["model"].isin(["rate_cell", "catboost"])]
        .pivot(
            index="region",
            columns="model",
            values=["brier_skill_vs_cell", "within_cell_auc"],
        )
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
