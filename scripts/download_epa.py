"""
Entry-point for downloading EPA data.
"""

from loguru import logger
import os
from pathlib import Path

from dotenv import load_dotenv

from co2sat.data.epa import (
    list_bulk_files,
    select_daily_emissions,
    download_all,
)

load_dotenv(Path("notebooks/.env"))


def main() -> None:
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError("Missing API_KEY in environment")
    listing_url = os.environ.get("BASE")
    if listing_url is None:
        raise RuntimeError("Missing BASE in environment")
    logger.info(f"The listing URL is {listing_url}")
    bulk_base = os.environ.get("BULK")
    if bulk_base is None:
        raise RuntimeError("Missing BULK in environment")
    bulk_base = bulk_base.rstrip("/") + "/"
    logger.info(f"The bulk base URL is {bulk_base}")

    out_dir = Path("data/raw/epa_daily")

    items = list_bulk_files(api_key, listing_url)
    files = select_daily_emissions(items, years=(2021, 2022))
    total_mb = sum(f["megaBytes"] for f in files)
    logger.info(f"Will download {len(files)} files, ~{total_mb:.0f} MB total")

    paths = download_all(files, out_dir, bulk_base)
    logger.info(f"Downloaded {len(paths)} files to {out_dir}")


if __name__ == "__main__":
    main()
