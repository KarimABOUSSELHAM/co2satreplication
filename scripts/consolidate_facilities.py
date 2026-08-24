"""Consolidate EPA facility-attribute CSVs into a single Parquet."""

from loguru import logger

from co2sat.data.epa import consolidate_facilities
from co2sat.utils import project_root


def main() -> None:
    """Consolidate EPA facility-attribute CSVs into a single Parquet"""
    root = project_root()
    raw_dir = root / "data" / "raw" / "epa_facilities"
    out_path = root / "data" / "processed" / "epa_facilities.parquet"
    logger.info("Start consolidating facilities")
    consolidate_facilities(raw_dir, out_path)


if __name__ == "__main__":
    main()
