"""Consolidate GOES-16 cache files into the Phase 2 deliverable."""

from __future__ import annotations

import duckdb
from loguru import logger

from co2sat.utils import data_dir


def main() -> None:
    cache_glob = str(data_dir("interim", "goes16_cache") / "*.parquet")
    out_path = data_dir("processed", "dynamic_features.parquet")

    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT
                facility_id,
                CAST(date AS DATE) AS date,
                hour,
                {", ".join(f"band_{b:02d}" for b in range(1, 17))}
            FROM read_parquet('{cache_glob}')
            ORDER BY facility_id, date, hour
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)

    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()
    if n is None:
        raise RuntimeError(f"COUNT(*) returned no rows for {out_path}")
    n = n[0]
    logger.info(f"Wrote {n:,} rows to {out_path}")


if __name__ == "__main__":
    main()
