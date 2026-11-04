"""Check that the extracted GOES-16 pixel really sits on the power plant.

For a few plants (default: the 3 largest coal plants), downloads ONE
MCMIPC scan, finds the pixel exactly as the Phase 2 extraction does
(extract_all_plants_for_scan), and:

  1. converts the pixel centre back to lon/lat and prints its distance to
     the plant coordinates (must stay under ~2 km: the nominal 2 km pixel grows
     to about 2.5-3 km over the US because of the oblique viewing angle);
  2. prints a Google Maps link to the plant coordinates, to confirm by eye
     that the EPA coordinates point at the plant itself;
  3. saves a figure with, for each plant, a 31 x 31 pixel window
     (about 60 x 60 km) in band 2 (red, daytime), band 7 (3.9 um, heat) and
     band 13 (10.3 um, clean IR), the extracted pixel framed in red.

Choose a clear-sky moment: 18 UTC is early afternoon in the eastern and
central US (daylight for band 2). Try another date if the windows are
cloudy.

Usage:
  uv run scripts/check_pixel_location.py
  uv run scripts/check_pixel_location.py --when 2021-04-20T18 --facility-ids 6146 3470 6166
"""

from __future__ import annotations

import argparse
import datetime as dt
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from loguru import logger
from matplotlib.patches import Rectangle

from co2sat.data.goes16 import find_scan_for_hour, make_goes_projection
from co2sat.utils import data_dir, project_root

BANDS = {2: "B02 red 0.64 um", 7: "B07 3.9 um (heat)", 13: "B13 10.3 um IR"}


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance between two points.

    Args:
        lon1 (float): longitude of point 1, degrees.
        lat1 (float): latitude of point 1, degrees.
        lon2 (float): longitude of point 2, degrees.
        lat2 (float): latitude of point 2, degrees.

    Returns:
        float: distance in km.
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(6371.0 * 2 * np.arcsin(np.sqrt(a)))


def pick_plants(ids: list[int] | None, n: int) -> pd.DataFrame:
    """Select the plants to check.

    Args:
        ids (list[int] | None): facility ids given on the command line.
        n (int): number of plants when no ids are given (largest coal).

    Returns:
        pd.DataFrame: facility_id, latitude, longitude, capacity_mw and,
        when available, facility_name.
    """
    st = pd.read_parquet(data_dir("processed", "static_features.parquet"))
    if ids:
        sel = st[st["facility_id"].isin(ids)]
    else:
        sel = st[st["coal_ratio"] >= 0.5].nlargest(n, "capacity_mw")
    names_path = data_dir("processed", "epa_facilities.parquet")
    if names_path.exists():
        names = pd.read_parquet(names_path, columns=["facility_id", "facility_name"])
        sel = sel.merge(
            names.drop_duplicates("facility_id"), on="facility_id", how="left"
        )
    return sel.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--when", default="2021-04-15T18", help="UTC date and hour")
    ap.add_argument("--facility-ids", type=int, nargs="*", default=None)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--half", type=int, default=15, help="half window, in pixels")
    args = ap.parse_args()

    plants = pick_plants(args.facility_ids, args.n)
    target = dt.datetime.strptime(args.when, "%Y-%m-%dT%H")
    fs = s3fs.S3FileSystem(anon=True)
    path = find_scan_for_hour(fs, target)
    if path is None:
        raise SystemExit(f"No scan found for {target}")
    logger.info(f"Scan: {path}")

    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        fs.get(path, tmp.name)
        ds = xr.open_dataset(tmp.name)
        proj = make_goes_projection(ds)
        h = ds["goes_imager_projection"].attrs["perspective_point_height"]
        xs, ys = ds["x"].values, ds["y"].values
        imgs = {b: ds[f"CMI_C{b:02d}"].values for b in BANDS}
        ds.close()

    fig, axes = plt.subplots(len(plants), len(BANDS), figsize=(11, 3.6 * len(plants)))
    axes = np.atleast_2d(axes)
    for i, p in plants.iterrows():
        x_m, y_m = proj(p["longitude"], p["latitude"])
        xi = int(np.abs(xs - x_m / h).argmin())
        yi = int(np.abs(ys - y_m / h).argmin())
        lon_c, lat_c = proj(xs[xi] * h, ys[yi] * h, inverse=True)
        dist = haversine_km(p["longitude"], p["latitude"], lon_c, lat_c)
        name = p.get("facility_name", "") or ""
        print(
            f"\n{int(p['facility_id'])} {name} | {p['capacity_mw']:.0f} MW | "
            f"plant ({p['latitude']:.4f}, {p['longitude']:.4f}) -> pixel centre "
            f"({lat_c:.4f}, {lon_c:.4f}) | distance {dist:.2f} km"
        )
        print(
            f"   Google Maps: https://www.google.com/maps/@{p['latitude']:.5f},"
            f"{p['longitude']:.5f},2000m/data=!3m1!1e3"
        )
        y0, y1 = max(yi - args.half, 0), yi + args.half + 1
        x0, x1 = max(xi - args.half, 0), xi + args.half + 1
        for j, (b, label) in enumerate(BANDS.items()):
            ax = axes[i, j]
            win = imgs[b][y0:y1, x0:x1]
            ax.imshow(win, cmap="gray" if b == 2 else "inferno")
            ax.add_patch(
                Rectangle(
                    (xi - x0 - 0.5, yi - y0 - 0.5), 1, 1, fill=False, ec="red", lw=2
                )
            )
            ax.set_title(
                f"{int(p['facility_id'])} {str(name)[:22]}\n{label}", fontsize=8
            )
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"Extracted pixel (red) — scan {target:%Y-%m-%d %H} UTC", fontsize=10)
    fig.tight_layout()
    out = project_root() / "figures" / f"pixel_check_{target:%Y%m%d_%H}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    logger.info(f"Figure -> {out}")


if __name__ == "__main__":
    main()
