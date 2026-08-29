"""Static features for the satellite paper replication.

Sources per paper Table 3: EPA attributes (Phase 1), satellite zenith angle
(paper eqs 1-3), altitude (Mapzen->EPQS substitution), EDGAR v8.0
surroundings, Hu et al. 2022 consumption surroundings.
"""

from __future__ import annotations

from typing import Mapping, Union, Iterable
from pandas.api.extensions import ExtensionArray
from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
import time
import requests
from loguru import logger
import xarray as xr

# Paper's constants (section 2.1.3) — keep verbatim for replication
SAT_LON_DEG = -75.2
R_EARTH_KM = 6370.0
R_SAT_KM = 42156.0
# EPQS API for elevation (meters) at lat/lon; see https://epqs.nationalmap.gov/FAQ.html
EPQS_URL = "https://epqs.nationalmap.gov/v1/json"


def satellite_zenith_angle(
    lat_deg: Union[ArrayLike, ExtensionArray],
    lon_deg: Union[ArrayLike, ExtensionArray],
) -> np.ndarray:
    """Satellite zenith angle in degrees, per paper equations (1)-(3).

    gamma = arccos(cos(lat) * cos(sat_lon - lon))          (1)
    d = r * sqrt(1 + R^2/r^2 - 2*(R/r)*cos(gamma))         (2)
    SZA = arcsin(r * sin(gamma) / d) * 180/pi              (3)
    """
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    dlon = np.radians(SAT_LON_DEG - np.asarray(lon_deg, dtype=float))
    gamma = np.arccos(np.cos(lat) * np.cos(dlon))
    ratio = R_EARTH_KM / R_SAT_KM
    d = R_SAT_KM * np.sqrt(1 + ratio**2 - 2 * ratio * np.cos(gamma))
    sza = np.degrees(np.arcsin(R_SAT_KM * np.sin(gamma) / d))
    return sza


def build_epa_statics(epa_parquet_path) -> pd.DataFrame:
    """One row per facility: capacity, coords, fuel ratios, zenith angle."""
    cols = [
        "facility_id",
        "latitude",
        "longitude",
        "capacity_mw",
        "coal_ratio",
        "gas_ratio",
        "oil_ratio",
        "other_ratio",
    ]
    df = (
        pd.read_parquet(epa_parquet_path, columns=cols)
        .drop_duplicates("facility_id")
        .reset_index(drop=True)
    )
    df["zenith_angle"] = satellite_zenith_angle(
        df["latitude"].values, df["longitude"].values
    )
    return df


ParamsType = Mapping[
    str,
    Union[str, bytes, int, float, Iterable[str | bytes | int | float] | None],
]


def fetch_altitude_epqs(lat: float, lon: float, retries: int = 3) -> float | None:
    params: ParamsType = {
        "x": lon,
        "y": lat,
        "units": "Meters",
        "wkid": 4326,
    }

    for attempt in range(retries):
        try:
            r = requests.get(EPQS_URL, params=params, timeout=15)
            r.raise_for_status()
            return float(r.json()["value"])
        except (requests.RequestException, ValueError, KeyError) as e:
            logger.warning(f"EPQS attempt {attempt + 1} failed ({lat},{lon}): {e}")
            time.sleep(2 * (attempt + 1))
    return None


def extract_edgar_at_plants(
    nc_path,
    statics: pd.DataFrame,
    year: int = 2021,
) -> np.ndarray:
    """EDGAR value of the 0.1-deg cell containing each plant.

    'Surrounding' interpretation: containing cell (paper leaves it
    unquantified — deviation log #4). Handles 0-360 lon and corner
    registration automatically.
    """
    ds = xr.open_dataset(nc_path)
    var = list(ds.data_vars)[0]
    da = ds[var]

    # Time handling: full-timeseries file -> select the chosen year
    if "time" in da.dims:
        da = da.sel(time=str(year)).squeeze()
        if "time" in da.dims:  # monthly within the year
            da = da.sum("time")  # ton/cell/month -> ton/cell/year

    # Corner-registered coordinates -> shift to centers
    lon0 = float(da["lon"].values[0])
    if abs((lon0 * 10) % 1) < 1e-6:  # .0 decimals => corners
        da = da.assign_coords(lon=da["lon"].values + 0.05, lat=da["lat"].values + 0.05)

    # Longitude wrap
    lons = np.asarray(statics["longitude"].values, dtype=float)
    if float(da["lon"].max()) > 180:
        lons = lons % 360

    vals = da.sel(
        lat=xr.DataArray(statics["latitude"].values, dims="p"),
        lon=xr.DataArray(lons, dims="p"),
        method="nearest",
    ).values.astype(float)
    ds.close()
    return vals
