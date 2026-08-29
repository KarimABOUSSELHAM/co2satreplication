"""Static features for the satellite paper replication.

Sources per paper Table 3: EPA attributes (Phase 1), satellite zenith angle
(paper eqs 1-3), altitude (Mapzen->EPQS substitution), EDGAR v8.0
surroundings, Hu et al. 2022 consumption surroundings.
"""

from __future__ import annotations

from typing import Union
from pandas.api.extensions import ExtensionArray
from numpy.typing import ArrayLike
import numpy as np
import pandas as pd

# Paper's constants (section 2.1.3) — keep verbatim for replication
SAT_LON_DEG = -75.2
R_EARTH_KM = 6370.0
R_SAT_KM = 42156.0


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
