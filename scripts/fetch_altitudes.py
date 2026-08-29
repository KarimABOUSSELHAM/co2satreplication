"""Fetch plant altitudes from USGS EPQS with a resumable cache."""

import pandas as pd
from loguru import logger
from co2sat.data.static import fetch_altitude_epqs
from co2sat.utils import data_dir
from typing import cast

CACHE = data_dir("interim", "altitude_cache.parquet")

statics = pd.read_parquet(data_dir("processed", "static_features.parquet"))
done = (
    pd.read_parquet(CACHE)
    if CACHE.exists()
    else pd.DataFrame(columns=["facility_id", "altitude_m"])
)
todo = statics[~statics["facility_id"].isin(done["facility_id"])]
logger.info(f"{len(done)} cached, {len(todo)} to fetch")

rows = []
for i, row in enumerate(todo.itertuples(), 1):
    lat = cast(float, row.latitude)
    lon = cast(float, row.longitude)
    alt = fetch_altitude_epqs(lat, lon)
    rows.append({"facility_id": row.facility_id, "altitude_m": alt})
    if i % 50 == 0:
        pd.concat([done, pd.DataFrame(rows)]).to_parquet(CACHE, index=False)
        logger.info(f"{i}/{len(todo)}")
pd.concat([done, pd.DataFrame(rows)]).to_parquet(CACHE, index=False)
logger.info("done")
