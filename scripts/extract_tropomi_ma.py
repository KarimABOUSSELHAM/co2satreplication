"""TROPOMI NO2 over the six Moroccan thermal sites, 2019-2025 (Earth Engine).

Same mechanics as extract_tropomi.py (daily mosaic, window statistics), for
the sites listed in data/raw/morocco_thermal_sites.csv. Two windows: 10 km
(signal) and 30 km (ring background), one export per year.

Before running: check the six coordinates on a map (the CSV says VERIFY).
A wrong coordinate puts the window on the sea or on a town.
  uv run scripts/extract_tropomi_ma.py --print-links      # Google Maps links

Usage:
  uv run scripts/extract_tropomi_ma.py --project YOUR_GEE_PROJECT_ID
Files land in Drive/tropomi_ma as tropomi_ma_<year>_r<radius>.csv; copy them
to data/raw/tropomi_ma/.
"""

from __future__ import annotations

import argparse

import pandas as pd
from loguru import logger

from co2sat.utils import data_dir

SITES = data_dir("raw", "morocco_thermal_sites.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2019, 2026)))
    ap.add_argument("--radius", type=float, nargs="+", default=[10000, 30000])
    ap.add_argument("--print-links", action="store_true")
    ap.add_argument("--folder", default="tropomi_ma")
    args = ap.parse_args()

    # sep=None lets pandas detect ';' if the file was saved from Excel.
    sites = pd.read_csv(SITES, sep=None, engine="python")
    sites.columns = [c.strip().lower() for c in sites.columns]
    sites = sites.dropna(how="all")  # a trailing empty line is harmless
    bad = sites[sites[["site_id", "latitude", "longitude"]].isna().any(axis=1)]
    if len(bad):
        print("Rows with a missing site_id / latitude / longitude (fix the CSV):")
        print(bad.to_string())
        raise SystemExit(1)
    sites["site_id"] = sites["site_id"].astype(int)
    print(
        sites[
            ["site_id", "name", "fuel", "capacity_mw", "latitude", "longitude"]
        ].to_string(index=False)
    )
    if args.print_links:
        for r in sites.itertuples():
            print(
                f"{r.name:22s} https://www.google.com/maps/@{r.latitude},{r.longitude},5000m/data=!3m1!1e3"
            )
        return
    if not args.project:
        raise SystemExit("--project is required (or use --print-links)")

    import ee
    from extract_tropomi import daily_table  # same folder: scripts/

    ee.Initialize(project=args.project)
    fc = ee.FeatureCollection(
        [
            ee.Feature(
                ee.Geometry.Point([r.longitude, r.latitude]),
                {"facility_id": int(r.site_id)},
            )
            for r in sites.itertuples()
        ]
    )
    for y in args.years:
        start, n_days = f"{y}-01-01", 366 if y % 4 == 0 else 365
        days = ee.List.sequence(0, n_days - 1).map(
            lambda i: ee.Date(start).advance(i, "day")
        )
        for r in args.radius:
            table = ee.FeatureCollection(
                days.map(lambda d: daily_table(ee.Date(d), fc, r))
            ).flatten()
            name = f"tropomi_ma_{y}_r{int(r)}"
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
            logger.info(f"Export task started: {name}")


if __name__ == "__main__":
    main()
