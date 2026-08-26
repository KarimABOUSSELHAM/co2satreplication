"""Batch GOES-16 MCMIPC extraction for all plants across the study periods.

Resumable: one Parquet per scan-hour in data/interim/goes16_cache/;
existing files are skipped, so the script can be killed and relaunched freely.
"""

from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import s3fs
from loguru import logger
from tqdm import tqdm

from co2sat.data.goes16 import extract_all_plants_for_scan, find_scan_for_hour
from co2sat.utils import data_dir, project_root

# Loguru: keep the default stderr sink (plays nicely with tqdm), add the file sink
log_dir = project_root() / "logs"
log_dir.mkdir(exist_ok=True)
logger.add(
    log_dir / "extract_goes16.log",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    enqueue=True,  # thread-safe writes from the worker pool
    rotation="10 MB",  # don't let a multi-day run grow an unbounded log
)

PERIODS = [
    ("2021-04-01", "2021-05-20"),
    ("2021-09-01", "2021-10-01"),
    ("2022-04-01", "2022-04-30"),
    ("2022-09-01", "2022-09-29"),
]

# PERIODS = [
#     ("2021-04-01", "2021-04-02"),  # PILOT — one day only; restore full list after
# ]
MAX_WORKERS = 4


def build_task_list() -> list[dt.datetime]:
    """Every (date, hour) across the four study periods: 3,264 scan-hours."""
    tasks: list[dt.datetime] = []
    for start, end in PERIODS:
        d = dt.date.fromisoformat(start)
        stop = dt.date.fromisoformat(end)
        while d < stop:
            tasks.extend(dt.datetime(d.year, d.month, d.day, h) for h in range(24))
            d += dt.timedelta(days=1)
    return tasks


def cache_path(target: dt.datetime) -> Path:
    return (
        data_dir("interim", "goes16_cache") / f"{target:%Y-%m-%d}_H{target:%H}.parquet"
    )


def process_scan_hour(target: dt.datetime, plants: pd.DataFrame) -> str:
    """Worker: extract all plants for one scan-hour. Returns a status string."""
    out = cache_path(target)
    if out.exists():
        return "cached"
    # Each worker gets its own fs handle: clean thread isolation
    fs = s3fs.S3FileSystem(anon=True)
    path = find_scan_for_hour(fs, target)
    if path is None:
        logger.warning("No scan for %s", target)
        return "missing"
    try:
        df = extract_all_plants_for_scan(fs, path, plants)
        df.insert(1, "date", target.date().isoformat())
        df.insert(2, "hour", target.hour)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False, compression="snappy")
        return "done"
    # Keep one bad scan from aborting the whole resumable batch.
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("FAILED %s: %s – %s", target, type(e).__name__, e)
        return "failed"


def load_plants() -> pd.DataFrame:
    return (
        pd.read_parquet(data_dir("processed", "epa_daily_with_attributes.parquet"))[
            ["facility_id", "latitude", "longitude"]
        ]
        .drop_duplicates("facility_id")
        .reset_index(drop=True)
    )


def main() -> None:
    plants = load_plants()
    tasks = build_task_list()
    logger.info(f"Plants: {len(plants)} | Scan-hours: {len(tasks)}")

    counts = {"done": 0, "cached": 0, "missing": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_scan_hour, t, plants): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            counts[fut.result()] += 1
    logger.info(f"Run complete: {counts}")


if __name__ == "__main__":
    main()
