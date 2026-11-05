"""Windowed GOES-16 extraction: statistics of a 5 x 5 pixel window per plant.

The Phase 2 extraction (and Mo et al.) keeps ONE pixel per plant. A plume
drifts with the wind and may sit in a NEIGHBOURING pixel (see the Powerton
check). This script keeps, for every band, seven statistics of the 5 x 5
window centred on the plant pixel (about 10 x 10 km, more over the US):

  c     the plant pixel itself (same value as Phase 2)
  m3    mean of the 3 x 3 window
  m5    mean of the 5 x 5 window
  ring  mean of the 16 pixels of the 5 x 5 border (5 x 5 minus 3 x 3)
  mx5   maximum of the 5 x 5 window
  mn5   minimum of the 5 x 5 window
  sd5   standard deviation of the 5 x 5 window

Same mechanics as scripts/extract_goes16.py: one Parquet per scan-hour in
data/interim/goes16_window_cache/, existing files skipped (resumable), same
nearest-pixel rule. Runs locally or on Colab (needs only
data/processed/static_features.parquet). The full run downloads every
CONUS scan of the four periods (several hundred GB): run it on Colab, as for
Phase 2, then copy the cache folder back and consolidate locally.

Usage:
  uv run scripts/extract_goes16_window.py --min-mw 300            # extract
  uv run scripts/extract_goes16_window.py --hours 15-22 --min-mw 300
  uv run scripts/extract_goes16_window.py --consolidate           # merge cache
"""

from __future__ import annotations

import argparse
import datetime as dt
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from loguru import logger
from tqdm import tqdm

from co2sat.data.goes16 import find_scan_for_hour, make_goes_projection
from co2sat.utils import data_dir

PERIODS = [
    ("2021-04-01", "2021-05-20"),
    ("2021-09-01", "2021-10-01"),
    ("2022-04-01", "2022-04-30"),
    ("2022-09-01", "2022-09-29"),
]
STATS = ["c", "m3", "m5", "ring", "mx5", "mn5", "sd5"]
HALF = 2  # 5 x 5 window


def build_task_list(hours: list[int]) -> list[dt.datetime]:
    """Every (date, hour) of the four periods, restricted to the given hours.

    Args:
        hours (list[int]): UTC hours to keep.

    Returns:
        list[dt.datetime]: scan targets.
    """
    tasks: list[dt.datetime] = []
    for start, end in PERIODS:
        d, stop = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
        while d < stop:
            tasks.extend(dt.datetime(d.year, d.month, d.day, h) for h in hours)
            d += dt.timedelta(days=1)
    return tasks


def cache_path(target: dt.datetime) -> Path:
    """Cache file of one scan-hour.

    Args:
        target (dt.datetime): scan target.

    Returns:
        Path: Parquet path under data/interim/goes16_window_cache.
    """
    return (
        data_dir("interim", "goes16_window_cache")
        / f"{target:%Y-%m-%d}_H{target:%H}.parquet"
    )


def window_stats_for_scan(ds: xr.Dataset, plants: pd.DataFrame) -> pd.DataFrame:
    """Compute the seven window statistics of every band for every plant.

    Args:
        ds (xr.Dataset): an opened MCMIPC file.
        plants (pd.DataFrame): facility_id, latitude, longitude.

    Returns:
        pd.DataFrame: one row per plant, columns facility_id and
        b<band>_<stat> (16 bands x 7 statistics).
    """
    proj = make_goes_projection(ds)
    h = ds["goes_imager_projection"].attrs["perspective_point_height"]
    xs, ys = ds["x"].values, ds["y"].values
    x_m, y_m = proj(plants["longitude"].values, plants["latitude"].values)
    xi = np.abs(xs[None, :] - (x_m / h)[:, None]).argmin(axis=1)
    yi = np.abs(ys[None, :] - (y_m / h)[:, None]).argmin(axis=1)

    off = np.arange(-HALF, HALF + 1)
    yy = np.clip(yi[:, None, None] + off[None, :, None], 0, len(ys) - 1)
    xx = np.clip(xi[:, None, None] + off[None, None, :], 0, len(xs) - 1)
    inner = np.zeros((2 * HALF + 1, 2 * HALF + 1), dtype=bool)
    inner[HALF - 1 : HALF + 2, HALF - 1 : HALF + 2] = True

    out: dict[str, np.ndarray] = {"facility_id": plants["facility_id"].values}
    for b in range(1, 17):
        win = ds[f"CMI_C{b:02d}"].values[yy, xx].astype(np.float32)  # (n, 5, 5)
        flat = win.reshape(len(plants), -1)
        with np.errstate(all="ignore"):
            out[f"b{b:02d}_c"] = win[:, HALF, HALF]
            out[f"b{b:02d}_m3"] = np.nanmean(win[:, inner], axis=1)
            out[f"b{b:02d}_m5"] = np.nanmean(flat, axis=1)
            out[f"b{b:02d}_ring"] = np.nanmean(win[:, ~inner], axis=1)
            out[f"b{b:02d}_mx5"] = np.nanmax(flat, axis=1)
            out[f"b{b:02d}_mn5"] = np.nanmin(flat, axis=1)
            out[f"b{b:02d}_sd5"] = np.nanstd(flat, axis=1)
    return pd.DataFrame(out)


def process_scan_hour(target: dt.datetime, plants: pd.DataFrame) -> str:
    """Worker: extract one scan-hour for all plants into the cache.

    Args:
        target (dt.datetime): scan target.
        plants (pd.DataFrame): facility_id, latitude, longitude.

    Returns:
        str: 'done', 'cached', 'missing' or 'failed'.
    """
    out = cache_path(target)
    if out.exists():
        return "cached"
    fs = s3fs.S3FileSystem(anon=True)
    path = find_scan_for_hour(fs, target)
    if path is None:
        return "missing"
    try:
        with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
            fs.get(path, tmp.name)
            ds = xr.open_dataset(tmp.name)
            df = window_stats_for_scan(ds, plants)
            ds.close()
        df.insert(1, "date", target.date().isoformat())
        df.insert(2, "hour", target.hour)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False, compression="snappy")
        return "done"
    except Exception as e:  # noqa: BLE001 - one bad scan must not stop the batch
        logger.error(f"FAILED {target}: {type(e).__name__} - {e}")
        return "failed"


def parse_hours(spec: str) -> list[int]:
    """Parse an hour specification such as '0-23' or '15-22,3'.

    Args:
        spec (str): comma-separated hours or ranges.

    Returns:
        list[int]: sorted UTC hours.
    """
    hours: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = map(int, part.split("-"))
            hours.update(range(a, b + 1))
        else:
            hours.add(int(part))
    return sorted(hours)


def consolidate() -> None:
    """Merge the cache into data/processed/window_features.parquet."""
    files = sorted(data_dir("interim", "goes16_window_cache").glob("*.parquet"))
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    out = data_dir("processed", "window_features.parquet")
    df.sort_values(["facility_id", "date", "hour"]).to_parquet(out, index=False)
    logger.info(f"{len(files)} scan files -> {len(df):,} rows -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mw", type=float, default=300.0)
    ap.add_argument("--hours", default="0-23", help="UTC hours, e.g. '0-23' or '15-22'")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--limit", type=int, default=None, help="process at most N scans (test)"
    )
    ap.add_argument("--consolidate", action="store_true")
    args = ap.parse_args()

    if args.consolidate:
        consolidate()
        return

    st = pd.read_parquet(data_dir("processed", "static_features.parquet"))
    plants = st.loc[
        st["capacity_mw"] >= args.min_mw, ["facility_id", "latitude", "longitude"]
    ].reset_index(drop=True)
    tasks = build_task_list(parse_hours(args.hours))
    if args.limit:
        tasks = tasks[: args.limit]
    logger.info(
        f"Plants >= {args.min_mw:.0f} MW: {len(plants)} | scan-hours: {len(tasks)}"
    )

    counts = {"done": 0, "cached": 0, "missing": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_scan_hour, t, plants) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            counts[fut.result()] += 1
    logger.info(f"Run complete: {counts}")


if __name__ == "__main__":
    main()
