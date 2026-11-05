"""Where does the satellite's importance come from? Permutation tests.

Takes an already trained replica (no retraining) and measures how much the
test-set R2 drops when the satellite input (the 16 x 24 band matrix) is
scrambled in three different ways, plus the static features for reference:

  none           the replica as trained (reference)
  sat_across     satellite matrices shuffled across ALL plant-days of a
                 period: breaks everything the image says about the plant
                 (where it is, what surrounds it, which day it is)
  sat_within     satellite matrices shuffled only among the days of the SAME
                 plant in the same period: keeps what the image says about
                 the plant and its place, breaks only the link with that
                 particular day's activity
  sat_days_mean  each day's matrix replaced by the plant's average matrix over
                 the period: same idea as sat_within, without randomness
  static_across  the 11 static features shuffled across plant-days

Reading: if sat_across hurts but sat_within and sat_days_mean do not, the
bands are "important" because they identify the plant and its
surroundings, not because they reveal what it emitted that day.

Usage:
  uv run scripts/permutation_importance.py --runs fix_s42 fix_s43 fix_s44 fix_s45 fix_s46
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from loguru import logger

from co2sat.model import CNNBiLSTMAttention
from co2sat.utils import data_dir

TARGETS = ["co2_metric_tons", "gross_load_mwh"]
SCHEMES = ["none", "sat_across", "sat_within", "sat_days_mean", "static_across"]


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination.

    Args:
        y_true (np.ndarray): observed values.
        y_pred (np.ndarray): predicted values.

    Returns:
        float: R2.
    """
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - float(((y_true - y_pred) ** 2).sum()) / ss_tot


def within_r2(y_true: np.ndarray, y_pred: np.ndarray, plant: np.ndarray) -> float:
    """R2 of daily anomalies around each plant's mean.

    Args:
        y_true (np.ndarray): observed values.
        y_pred (np.ndarray): predicted values.
        plant (np.ndarray): plant identifier of each row.

    Returns:
        float: within-plant R2.
    """
    df = pd.DataFrame({"p": plant, "t": y_true, "y": y_pred})
    at = df["t"] - df.groupby("p")["t"].transform("mean")
    ap = df["y"] - df.groupby("p")["y"].transform("mean")
    return 1.0 - float(((at - ap) ** 2).sum()) / float((at**2).sum())


def scramble(
    scheme: str,
    x_dyn: np.ndarray,
    x_stat: np.ndarray,
    plant: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return scrambled copies of the inputs of ONE period.

    Args:
        scheme (str): one of SCHEMES.
        x_dyn (np.ndarray): normalized satellite input, (n, bands, 24).
        x_stat (np.ndarray): normalized static input, (n, 11).
        plant (np.ndarray): plant identifier of each row.
        rng (np.random.Generator): random generator.

    Returns:
        tuple[np.ndarray, np.ndarray]: (x_dyn, x_stat) after scrambling.
    """
    xd, xs = x_dyn.copy(), x_stat.copy()
    if scheme == "sat_across":
        xd = xd[rng.permutation(len(xd))]
    elif scheme == "static_across":
        xs = xs[rng.permutation(len(xs))]
    elif scheme in ("sat_within", "sat_days_mean"):
        for p in np.unique(plant):
            idx = np.flatnonzero(plant == p)
            if scheme == "sat_within":
                xd[idx] = x_dyn[rng.permutation(idx)]
            else:
                xd[idx] = x_dyn[idx].mean(axis=0, keepdims=True)
    return xd, xs


def predict(
    model: torch.nn.Module, xd: np.ndarray, xs: np.ndarray, device: torch.device
) -> np.ndarray:
    """Run the model in batches.

    Args:
        model (torch.nn.Module): trained replica in eval mode.
        xd (np.ndarray): satellite input.
        xs (np.ndarray): static input.
        device (torch.device): device.

    Returns:
        np.ndarray: z-scored predictions, shape (n, 2).
    """
    out = []
    with torch.no_grad():
        for lo in range(0, len(xd), 2048):
            out.append(
                model(
                    torch.from_numpy(xd[lo : lo + 2048]).to(device),
                    torch.from_numpy(xs[lo : lo + 2048]).to(device),
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=3, help="random draws per scheme")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdir = data_dir("processed", "model")
    stats = json.loads((mdir / "normalization_stats.json").read_text())
    x_dyn = np.load(mdir / "X_dyn.npy").astype(np.float32)
    x_stat = np.load(mdir / "X_stat.npy").astype(np.float32)
    y = np.load(mdir / "y.npy").astype(np.float64)
    samples = pd.read_parquet(mdir / "samples.parquet")
    dm = np.asarray(stats["dynamic"]["mean"], np.float32)[None, :, None]
    ds = np.asarray(stats["dynamic"]["std"], np.float32)[None, :, None]
    sm = np.asarray(stats["static"]["mean"], np.float32)[None, :]
    ss = np.asarray(stats["static"]["std"], np.float32)[None, :]
    tm = np.asarray(stats["targets"]["mean"])[None, :]
    ts = np.asarray(stats["targets"]["std"])[None, :]
    x_dyn = (x_dyn - dm) / ds
    x_stat = (x_stat - sm) / ss

    rows = []
    for run in args.runs:
        run_dir = mdir / "runs" / run
        model = CNNBiLSTMAttention(n_bands=x_dyn.shape[1]).to(device)
        model.load_state_dict(
            torch.load(run_dir / "best_model.pt", map_location=device)
        )
        model.eval()
        rng = np.random.default_rng(0)
        for per in ("P2", "P3", "P4"):
            idx = np.flatnonzero((samples["period"] == per).to_numpy())
            plant = samples["facility_id"].to_numpy()[idx]
            for scheme in SCHEMES:
                reps = 1 if scheme in ("none", "sat_days_mean") else args.repeats
                for _ in range(reps):
                    xd, xs = scramble(scheme, x_dyn[idx], x_stat[idx], plant, rng)
                    pred = predict(model, xd, xs, device) * ts + tm
                    for ti, t in enumerate(TARGETS):
                        rows.append(
                            {
                                "run": run,
                                "period": per,
                                "scheme": scheme,
                                "target": t,
                                "R2": r2(y[idx, ti], pred[:, ti]),
                                "within_R2": within_r2(y[idx, ti], pred[:, ti], plant),
                            }
                        )
        logger.info(f"{run} done")

    res = pd.DataFrame(rows)
    res.to_csv(
        mdir / "runs" / "permutation_importance.csv", index=False, float_format="%.4f"
    )
    table = (
        res.groupby(["target", "period", "scheme"], sort=False)[["R2", "within_R2"]]
        .mean()
        .unstack("scheme")
    )
    base = table["R2"]["none"]
    drop = table["R2"].drop(columns="none").rsub(base, axis=0)
    print("=== Plant-level daily R2 (mean over runs and draws) ===")
    print(table["R2"][SCHEMES].round(3).to_string())
    print("\n=== R2 DROP vs the replica as trained (bigger = more important) ===")
    print(drop[SCHEMES[1:]].round(3).to_string())
    print("\n=== Within-plant R2 ===")
    print(table["within_R2"][SCHEMES].round(3).to_string())


if __name__ == "__main__":
    main()
