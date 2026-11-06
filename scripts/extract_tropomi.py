"""Block 0: TROPOMI NO2 at every EPA plant, one value per day (Google Earth Engine).

Same plants and periods as the Mo et al. replication, so the satellite
test uses the same ground truth (EPA daily load and CO2). For each plant
and day, the script takes the daily mosaic of Sentinel-5P OFFL L3 NO2 and
reduces it over a circle around the plant:

  no2_mean / no2_max  tropospheric NO2 column (mol/m2) over the window
  no2_c               value at the plant pixel (window radius ~1 km)
  cloud               cloud fraction (mean over the window)
  n_px                number of valid pixels in the window

Windows: 1 km (pixel), 5 km, 10 km radius. The 10 km window follows the
plume on most days without wind correction (block 1 adds the wind).

The results land as CSV files in your Google Drive, folder 'tropomi_epa',
one file per period and window. Copy them to data/raw/tropomi/.

Setup (once):
  1. Sign up for Earth Engine with a Google account (non-commercial use):
     https://earthengine.google.com/  -> create a Cloud project when asked.
  2. uv add earthengine-api
  3. uv run earthengine authenticate      (opens the browser once)

Usage:
  uv run scripts/extract_tropomi.py --project ... --periods P1 --radius 10000 --limit 5   # smoke test
  uv run scripts/extract_tropomi.py --project YOUR_GEE_PROJECT_ID                        # full 2021-2022
Then watch the tasks at https://code.earthengine.google.com/tasks
"""

from __future__ import annotations

import argparse
import datetime as dt

import ee
import pandas as pd
from loguru import logger

from co2sat.utils import data_dir

PERIODS = {
    "P1": ("2021-04-01", "2021-05-20"),
    "P2": ("2021-09-01", "2021-10-01"),
    "P3": ("2022-04-01", "2022-04-30"),
    "P4": ("2022-09-01", "2022-09-29"),
    # Full years: the EPA daily file covers all of 2021-2022, and the monthly
    # test needs many months per plant. Prefer these two for block 0.
    "Y2021": ("2021-01-01", "2022-01-01"),
    "Y2022": ("2022-01-01", "2023-01-01"),
}
NO2 = "COPERNICUS/S5P/OFFL/L3_NO2"
BAND = "tropospheric_NO2_column_number_density"
SCALE = 1113.2  # m, native L3 grid (0.01 deg)


def plants_fc(limit: int | None) -> ee.FeatureCollection:
    """EPA plants as an Earth Engine FeatureCollection.

    Args:
        limit (int | None): keep only the first N plants (test).

    Returns:
        ee.FeatureCollection: points with facility_id.
    """
    st = pd.read_parquet(
        data_dir("processed", "static_features.parquet"),
        columns=["facility_id", "latitude", "longitude"],
    )
    if limit:
        st = st.head(limit)
    feats = [
        ee.Feature(
            ee.Geometry.Point([r.longitude, r.latitude]),
            {"facility_id": int(r.facility_id)},
        )
        for r in st.itertuples()
    ]
    logger.info(f"{len(feats)} plants")
    return ee.FeatureCollection(feats)


def daily_table(
    day: ee.Date, plants: ee.FeatureCollection, radius: float
) -> ee.FeatureCollection:
    """Reduce the NO2 mosaic of one day over every plant window.

    Args:
        day (ee.Date): the day (UTC).
        plants (ee.FeatureCollection): plant points.
        radius (float): window radius in metres.

    Returns:
        ee.FeatureCollection: one feature per plant with the statistics.
    """
    coll = (
        ee.ImageCollection(NO2)
        .filterDate(day, day.advance(1, "day"))
        .select([BAND, "cloud_fraction"])
    )
    # Some days have no scene at all (gaps in the archive): a mean over an
    # empty collection has no bands and makes reduceRegions fail. Guard it.
    empty = (
        ee.Image.constant([0, 0])
        .rename([BAND, "cloud_fraction"])
        .updateMask(ee.Image.constant(0))
    )
    img = ee.Image(ee.Algorithms.If(coll.size().gt(0), coll.mean(), empty))
    stats = img.reduceRegions(
        collection=plants.map(lambda f: f.buffer(radius)),
        reducer=ee.Reducer.mean()
        .combine(ee.Reducer.max(), sharedInputs=True)
        .combine(ee.Reducer.count(), sharedInputs=True),
        scale=SCALE,
    )
    return stats.map(
        lambda f: ee.Feature(
            None,
            {
                "facility_id": f.get("facility_id"),
                "date": day.format("YYYY-MM-dd"),
                "no2_mean": f.get(BAND + "_mean"),
                "no2_max": f.get(BAND + "_max"),
                "n_px": f.get(BAND + "_count"),
                "cloud": f.get("cloud_fraction_mean"),
            },
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Earth Engine Cloud project id")
    ap.add_argument("--periods", nargs="+", default=["Y2021", "Y2022"])
    ap.add_argument(
        "--radius",
        type=float,
        nargs="+",
        default=[1000, 5000, 10000],
        help="window radii in m; add 30000 for the ring background of the signal test",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--folder", default="tropomi_epa")
    args = ap.parse_args()

    ee.Initialize(project=args.project)
    plants = plants_fc(args.limit)
    for per in args.periods:
        start, end = PERIODS[per]
        n_days = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days
        days = ee.List.sequence(0, n_days - 1).map(
            lambda i: ee.Date(start).advance(i, "day")
        )
        for r in args.radius:
            table = ee.FeatureCollection(
                days.map(lambda d: daily_table(ee.Date(d), plants, r))
            ).flatten()
            name = f"tropomi_{per}_r{int(r)}"
            ee.batch.Export.table.toDrive(
                collection=table,
                description=name,
                folder=args.folder,
                fileNamePrefix=name,
                fileFormat="CSV",
                selectors=[
                    "facility_id",
                    "date",
                    "no2_mean",
                    "no2_max",
                    "n_px",
                    "cloud",
                ],
            ).start()
            logger.info(f"Export task started: {name} ({n_days} days)")
    logger.info(
        "Follow the tasks at https://code.earthengine.google.com/tasks ; files -> Drive/"
        + args.folder
    )


if __name__ == "__main__":
    main()
