"""Phase 4: train the CNN-BiLSTM-Attention replica (paper Section 2.3.1).

Reads the RAW arrays written by build_training_data.py and applies the
z-score normalization at load time, using the training-split statistics
stored in normalization_stats.json (paper eq. 17).

Protocol: Adam lr=3e-4, batch=64, max 500 epochs, EarlyStopping on the
validation loss. Split comes from samples.parquet (column 'split', built
by build_training_data.py with the paper protocol, seed 42).

Documented deviations (replication log):
  - EarlyStopping patience unspecified in the paper -> patience=20 on the
    validation total loss, checkpoint at the best epoch.
  - Loss function unspecified in the paper and supplementary -> unweighted
    sum of MSE on the two z-scored targets (deviation #10).

Usage:
  uv run scripts/train_model.py                       # defaults
  uv run scripts/train_model.py --run-name ablation_seviri --bands 1,2,...
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from co2sat.model import CNNBiLSTMAttention
from co2sat.utils import data_dir

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_normalized(
    band_idx: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load raw arrays and apply train-split z-score from the stats file.

    band_idx: optional 0-based band indices to keep (ablations); stats are
    sliced accordingly so normalization stays per-band correct.
    """
    mdir = data_dir("processed", "model")
    X_dyn = np.load(mdir / "X_dyn.npy").astype(np.float32)
    X_stat = np.load(mdir / "X_stat.npy").astype(np.float32)
    y = np.load(mdir / "y.npy").astype(np.float32)
    samples = pd.read_parquet(mdir / "samples.parquet")
    stats = json.loads((mdir / "normalization_stats.json").read_text())

    d_mean = np.asarray(stats["dynamic"]["mean"], dtype=np.float32)
    d_std = np.asarray(stats["dynamic"]["std"], dtype=np.float32)
    s_mean = np.asarray(stats["static"]["mean"], dtype=np.float32)
    s_std = np.asarray(stats["static"]["std"], dtype=np.float32)
    t_mean = np.asarray(stats["targets"]["mean"], dtype=np.float32)
    t_std = np.asarray(stats["targets"]["std"], dtype=np.float32)

    if band_idx is not None:
        X_dyn = X_dyn[:, band_idx, :]
        d_mean, d_std = d_mean[band_idx], d_std[band_idx]
        logger.info(f"Band subset: {len(band_idx)} bands {band_idx}")

    X_dyn = (X_dyn - d_mean[None, :, None]) / d_std[None, :, None]
    X_stat = (X_stat - s_mean[None, :]) / s_std[None, :]
    y = (y - t_mean[None, :]) / t_std[None, :]
    assert len(samples) == len(X_dyn) == len(X_stat) == len(y)
    return X_dyn, X_stat, y, samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="replica_v1")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument(
        "--bands",
        default=None,
        help="comma-separated 1-based ABI band numbers to keep (ablations), "
        "e.g. '1,2,3,...'; default: all 16",
    )
    args = ap.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    band_idx = [int(b) - 1 for b in args.bands.split(",")] if args.bands else None
    X_dyn, X_stat, y, samples = load_normalized(band_idx)
    n_bands = X_dyn.shape[1]

    tr = (samples["split"] == "train").to_numpy()
    va = (samples["split"] == "val").to_numpy()
    logger.info(f"train {tr.sum():,} | val {va.sum():,} | device {device}")

    def subset(mask: np.ndarray) -> TensorDataset:
        idx = np.flatnonzero(mask)
        return TensorDataset(
            torch.from_numpy(X_dyn[idx]),
            torch.from_numpy(X_stat[idx]),
            torch.from_numpy(y[idx]),
        )

    dl_tr = DataLoader(
        subset(tr),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    dl_va = DataLoader(subset(va), batch_size=512, num_workers=args.num_workers)

    model = CNNBiLSTMAttention(n_bands=n_bands).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()
    out_dir = data_dir("processed", "model", "runs", args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_val() -> float:
        model.eval()
        tot = n = 0
        with torch.no_grad():
            for xd, xs, yy in dl_va:
                loss = mse(model(xd.to(device), xs.to(device)), yy.to(device))
                tot += loss.item() * len(yy)
                n += len(yy)
        return tot / n

    best_val, best_epoch, bad = float("inf"), -1, 0
    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "seconds"])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot = n = 0
        for xd, xs, yy in dl_tr:
            opt.zero_grad()
            loss = mse(model(xd.to(device), xs.to(device)), yy.to(device))
            loss.backward()
            opt.step()
            tot += loss.item() * len(yy)
            n += len(yy)
        tr_loss, va_loss, dt = tot / n, run_val(), time.time() - t0
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{tr_loss:.6f}", f"{va_loss:.6f}", f"{dt:.1f}"]
            )
        if va_loss < best_val:
            best_val, best_epoch, bad = va_loss, epoch, 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            logger.info(
                f"epoch {epoch:3d} train {tr_loss:.5f} val {va_loss:.5f} "
                f"({dt:.0f}s) <-- best"
            )
        else:
            bad += 1
            logger.info(
                f"epoch {epoch:3d} train {tr_loss:.5f} val {va_loss:.5f} ({dt:.0f}s)"
            )
        if bad >= args.patience:
            logger.info(
                f"EarlyStopping at {epoch} (best {best_epoch}, val {best_val:.5f})"
            )
            break

    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "run_name": args.run_name,
                "n_bands": n_bands,
                "bands_1based": args.bands,
                "epochs_max": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "patience": args.patience,
                "optimizer": "Adam",
                "loss": "unweighted sum of MSE on z-scored targets "
                "(unspecified in paper; deviation #10)",
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "device": str(device),
            },
            indent=2,
        )
    )
    logger.info(f"Done -> {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
