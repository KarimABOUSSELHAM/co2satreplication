"""
Implementing satellite paper data pipeline.

Implements the satellite side of the Mo et al. (2025) replication:
locating scans on the NOAA S3 bucket, transforming plant coordinates to the ABI fixed grid, and extracting per-band pixel values.
"""

from __future__ import annotations

import tempfile
import datetime as dt
import re
import logging
import xarray as xr
from pyproj import Proj
import numpy as np
import s3fs
import pandas as pd

logger = logging.getLogger(__name__)
BUCKET_PREFIX = "noaa-goes16/ABI-L2-MCMIPC"

_MCMIP_FILENAME_RE = re.compile(
    r"OR_ABI-L2-MCMIPC-M\d_G16_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{3})_"
)


def parse_mcmip_filename(name: str) -> dt.datetime | None:
    """Extract the scan start time from an MCMIPC filename.

    Returns None if the name doesn't match the expected pattern.
    """
    m = _MCMIP_FILENAME_RE.search(name)
    if not m:
        return None
    year, day, hour, minute, _second = map(int, m.groups())
    return dt.datetime(year, 1, 1) + dt.timedelta(
        days=day - 1, hours=hour, minutes=minute
    )


def find_scan_for_hour(fs: s3fs.S3FileSystem, target: dt.datetime) -> str | None:
    """Return the S3 path of the MCMIPC scan starting closest to target.

    Lists the bucket directory for the target's year/day/hour and picks
    the scan whose start time is nearest. Returns None if the directory
    doesn't exist or contains no parseable files (satellite outage).
    """
    prefix = (
        f"{BUCKET_PREFIX}/{target.year}/"
        f"{target.timetuple().tm_yday:03d}/{target.hour:02d}/"
    )
    try:
        files = fs.ls(prefix)
    except FileNotFoundError:
        return None
    candidates = [(f, t) for f in files if (t := parse_mcmip_filename(f)) is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda c: abs((c[1] - target).total_seconds()))[0]


def make_goes_projection(ds: xr.Dataset) -> Proj:
    """Build a pyproj geostationary projection from GOES metadata.

    Moved from notebooks/07_coordinates_transformation.ipynb.
    """
    proj_info = ds["goes_imager_projection"]
    return Proj(
        proj="geos",
        h=proj_info.attrs["perspective_point_height"],
        lon_0=proj_info.attrs["longitude_of_projection_origin"],
        sweep=proj_info.attrs["sweep_angle_axis"],
    )


def lonlat_to_xy(
    lon: float, lat: float, proj: Proj, height: float
) -> tuple[float, float]:
    """Convert lon/lat (degrees) to ABI fixed-grid x, y (radians).

    Moved from notebooks/07_coordinates_transformation.ipynb.
    pyproj returns meters; dividing by satellite height gives scan-angle
    radians matching the file's x/y coordinate values.
    """
    x, y = proj(lon, lat, inverse=False)
    return x / height, y / height


def extract_bands_at_point(
    fs: s3fs.S3FileSystem, file_path: str, lon: float, lat: float
) -> dict[int, float]:
    """Open one MCMIPC file, return {band: value} at (lon, lat) for all 16 bands."""

    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        fs.get(file_path, tmp.name)  # one sequential bulk download
        ds = xr.open_dataset(tmp.name)  # local read, default engine
        proj = make_goes_projection(ds)
        h = ds["goes_imager_projection"].attrs["perspective_point_height"]
        x, y = lonlat_to_xy(lon, lat, proj, h)
        x_idx = int(np.abs(ds["x"].values - x).argmin())
        y_idx = int(np.abs(ds["y"].values - y).argmin())
        result = {
            b: float(ds[f"CMI_C{b:02d}"].values[y_idx, x_idx]) for b in range(1, 17)
        }
        ds.close()
        return result


def extract_day_matrix(
    fs: s3fs.S3FileSystem, date: dt.date, lon: float, lat: float
) -> np.ndarray:
    """16x24 matrix (bands x hours) for one location on one day. NaN = missing."""
    matrix = np.full((16, 24), np.nan)
    for hour in range(24):
        target = dt.datetime(date.year, date.month, date.day, hour)
        scan_path = find_scan_for_hour(fs, target)
        if scan_path is None:
            continue
        try:
            for band, v in extract_bands_at_point(fs, scan_path, lon, lat).items():
                matrix[band - 1, hour] = v
        except (OSError, ValueError, KeyError, IndexError) as e:
            logger.error(
                "Extraction failed for %s: %s – %s",
                target,
                type(e).__name__,
                e,
            )
    return matrix


def extract_all_plants_for_scan(
    fs: s3fs.S3FileSystem, file_path: str, plants: pd.DataFrame
) -> pd.DataFrame:
    """Download one MCMIPC file, extract all 16 bands for every plant.

    plants must have columns: facility_id, latitude, longitude.
    Returns one row per plant with band_01..band_16 columns.
    """
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
        fs.get(file_path, tmp.name)  # one bulk download serves all plants
        ds = xr.open_dataset(tmp.name)
        proj = make_goes_projection(ds)
        h = ds["goes_imager_projection"].attrs["perspective_point_height"]

        # Vectorized transform: all plants at once (pyproj accepts arrays)
        x_m, y_m = proj(plants["longitude"].values, plants["latitude"].values)
        x_rad, y_rad = x_m / h, y_m / h

        # Vectorized nearest-pixel search:
        # broadcasting (n_plants, 1) against (1, n_grid) -> argmin per plant
        x_idx = np.abs(ds["x"].values[None, :] - x_rad[:, None]).argmin(axis=1)
        y_idx = np.abs(ds["y"].values[None, :] - y_rad[:, None]).argmin(axis=1)

        out = {"facility_id": plants["facility_id"].values}
        for b in range(1, 17):
            # Fancy indexing: pairs (y_idx[i], x_idx[i]) -> one pixel per plant
            out[f"band_{b:02d}"] = ds[f"CMI_C{b:02d}"].values[y_idx, x_idx]
        ds.close()
        return pd.DataFrame(out)
