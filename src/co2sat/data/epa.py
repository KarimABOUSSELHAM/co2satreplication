"""
Ingesting raw input data from EPA CAMPD as the paper suggested in section 2.1.1.
"""

from loguru import logger

# import os
import time
from pathlib import Path
import math
import pandas as pd
import re
from typing import Iterable

import httpx
from tqdm import tqdm

NON_CONUS = frozenset({"AK", "HI", "PR", "VI", "GU", "MP", "AS"})

# Conversion: EPA reports CO2 in short tons; paper uses metric tons
SHORT_TON_TO_METRIC = 0.90718474


def list_bulk_files(api_key: str, listing_url: str) -> list[dict]:
    """Fetch the full bulk-files listing from EPA CAMPD."""
    headers = {"x-api-key": api_key}
    resp = httpx.get(listing_url, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.json()["items"]


def select_daily_emissions(
    items: list[dict],
    years: Iterable[int] = (2021, 2022),
    exclude_states: Iterable[str] = NON_CONUS,
) -> list[dict]:
    """Filter the listing to daily-emissions files for given years, CONUS only."""
    years_set = {str(y) for y in years}
    exclude = set(exclude_states)
    return [
        it
        for it in items
        if it["metadata"].get("dataType") == "Emissions"
        and "daily" in it["filename"].lower()
        and str(it["metadata"].get("year")) in years_set
        and it["metadata"].get("stateCode") not in exclude
    ]


def download_file(
    s3_path: str,
    out_path: Path,
    bulk_base: str,
    skip_if_exists: bool = True,
) -> None:
    """Download one file from the bulk-files store to local disk."""
    if skip_if_exists and out_path.exists():
        logger.debug("Skipping existing %s", out_path.name)
        return

    download_url = bulk_base.rstrip("/") + "/" + s3_path.lstrip("/")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.stream("GET", download_url, timeout=300) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                f.write(chunk)


def download_all(
    files: list[dict],
    out_dir: Path,
    bulk_base: str,
    pause_seconds: float = 0.2,
) -> list[Path]:
    """Download all selected files with a progress bar and a polite pause."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for entry in tqdm(files, desc="Downloading EPA daily emissions"):
        out_path = out_dir / entry["filename"]
        try:
            download_file(entry["s3Path"], out_path, bulk_base)
            paths.append(out_path)
        except httpx.HTTPError as e:
            logger.error("Failed to download %s: %s", entry["filename"], e)
        time.sleep(pause_seconds)  # be polite to EPA's servers
    return paths


def select_facility_files(
    items: list[dict],
    years: Iterable[int] = (2021, 2022),
) -> list[dict]:
    """Filter the listing to facility attributes files for given years."""
    years_set = {str(y) for y in years}
    return [
        it
        for it in items
        if it["metadata"].get("dataType") == "Facility"
        and str(it["metadata"].get("year")) in years_set
    ]


def parse_capacities(s) -> float:
    """Sum capacities in MW from EPA's generator-capacity string."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return 0.0
    s = str(s).strip()
    if not s:
        return 0.0
    matches = re.findall(r"\(([\d.]+)\)", s)
    return sum(float(m) for m in matches) if matches else 0.0


def is_active(status) -> bool:
    """A unit is active if its operating status starts with 'Operating'."""
    if status is None or (isinstance(status, float) and math.isnan(status)):
        return False
    return str(status).startswith("Operating")


def filter_active_conus(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only active (Operating*) facilities in the contiguous US."""
    before = len(df)
    active_mask = df["operating_status"].map(is_active)
    conus_mask = ~df["state"].isin(NON_CONUS)
    df = df[active_mask & conus_mask].copy()
    logger.info(
        "Filtered to active CONUS: %d -> %d rows (%.1f%% retained)",
        before,
        len(df),
        100 * len(df) / before if before else 0,
    )
    return df


FUEL_CATEGORIES = {
    "coal": {"Coal", "Coal Refuse", "Petroleum Coke"},
    "gas": {"Natural Gas", "Pipeline Natural Gas", "Process Gas", "Other Gas"},
    "oil": {"Diesel Oil", "Residual Oil", "Other Oil"},
}


def categorize_fuel(fuel) -> str:
    """Map an EPA primary fuel string to one of: coal, gas, oil, other."""
    if fuel is None or (isinstance(fuel, float) and math.isnan(fuel)):
        return "other"
    for category, fuels in FUEL_CATEGORIES.items():
        if fuel in fuels:
            return category
    return "other"


# ---- Facility CSV schema ----

FACILITY_COLS = {
    "Facility ID": "facility_id",
    "Facility Name": "facility_name",
    "State": "state",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Unit ID": "unit_id",
    "Unit Type": "unit_type",
    "Primary Fuel Type": "primary_fuel",
    "Secondary Fuel Type": "secondary_fuel",
    "Operating Status": "operating_status",
    "Associated Generators & Nameplate Capacity (MWe)": "capacity_str",
    "Year": "year",
}

FACILITY_DTYPES = {
    "Facility ID": "Int64",
    "Facility Name": "string",
    "State": "string",
    "Latitude": "Float64",
    "Longitude": "Float64",
    "Unit ID": "string",
    "Unit Type": "string",
    "Primary Fuel Type": "string",
    "Secondary Fuel Type": "string",
    "Operating Status": "string",
    "Associated Generators & Nameplate Capacity (MWe)": "string",
    "Year": "Int64",
}


def load_facility_csv(path: Path) -> pd.DataFrame:
    """Load one EPA facility-attributes CSV with proper typing and renaming."""
    df = pd.read_csv(
        path,
        usecols=list(FACILITY_COLS),
        dtype=FACILITY_DTYPES,
    )
    df = df.rename(columns=FACILITY_COLS)
    return df


def compute_fuel_ratios(df_units: pd.DataFrame) -> pd.DataFrame:
    """Capacity-weighted fuel ratios per facility.

    Falls back to unit-count weighting for facilities with zero parsed capacity.
    """
    df = df_units.copy()
    df["capacity_mw"] = df["capacity_str"].map(parse_capacities)
    df["fuel_category"] = df["primary_fuel"].map(categorize_fuel)

    # Primary aggregation: capacity-weighted
    cap = (
        df.groupby(["facility_id", "fuel_category"])["capacity_mw"]
        .sum()
        .unstack(fill_value=0.0)
    )
    for cat in ["coal", "gas", "oil", "other"]:
        if cat not in cap.columns:
            cap[cat] = 0.0

    # Identify facilities with zero total capacity for fallback
    totals = cap.sum(axis=1)
    zero_cap = totals == 0

    if zero_cap.any():
        # For zero-capacity facilities, use unit-count weighting
        cnt = (
            df[df["facility_id"].isin(totals[zero_cap].index)]
            .groupby(["facility_id", "fuel_category"])
            .size()
            .unstack(fill_value=0)
        )
        for cat in ["coal", "gas", "oil", "other"]:
            if cat not in cnt.columns:
                cnt[cat] = 0
        cnt_totals = cnt.sum(axis=1).replace(0, pd.NA)
        cnt_ratios = cnt.div(cnt_totals, axis=0).fillna(0.0)
        cap.loc[zero_cap] = cnt_ratios.reindex(cap.loc[zero_cap].index)
        totals = cap.sum(axis=1)

    totals_safe = totals.replace(0, pd.NA)
    ratios = cap.div(totals_safe, axis=0).fillna(0.0)
    ratios = ratios[["coal", "gas", "oil", "other"]].rename(
        columns=lambda c: f"{c}_ratio"
    )
    return ratios.reset_index()


def aggregate_facility(df_units: pd.DataFrame) -> pd.DataFrame:
    """Collapse unit-level rows to one row per facility."""

    def first_non_null(s):
        s = s.dropna()
        return s.iloc[0] if len(s) else None

    def first_mode(s):
        modes = s.mode(dropna=True)
        return modes.iloc[0] if len(modes) else None

    df = df_units.copy()
    df["capacity_mw"] = df["capacity_str"].map(parse_capacities)

    facility = df.groupby("facility_id", as_index=False).agg(
        facility_name=("facility_name", "first"),
        state=("state", "first"),
        latitude=("latitude", first_non_null),
        longitude=("longitude", first_non_null),
        capacity_mw=("capacity_mw", "sum"),
        primary_fuel=("primary_fuel", first_mode),
        operating_status=("operating_status", first_mode),
        n_units=("unit_id", "count"),
        year=("year", "max"),
    )
    return facility


def consolidate_facilities(raw_dir: Path, output_path: Path) -> pd.DataFrame:
    """Process all facility-attribute CSVs into one Parquet at facility level."""
    per_year_frames = []
    for csv_path in sorted(raw_dir.glob("*.csv")):
        logger.info("Reading %s", csv_path.name)
        df = load_facility_csv(csv_path)
        df = filter_active_conus(df)
        ratios = compute_fuel_ratios(df)
        facility = aggregate_facility(df)
        facility = facility.merge(ratios, on="facility_id", how="left")
        per_year_frames.append(facility)

    all_years = pd.concat(per_year_frames, ignore_index=True)
    # Keep most-recent year's record for each facility
    all_years = all_years.sort_values("year", ascending=True)
    facilities = all_years.drop_duplicates(
        subset="facility_id", keep="last"
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    facilities.to_parquet(output_path, index=False, compression="snappy")
    logger.info("Saved %d facilities to %s", len(facilities), output_path)
    return facilities
