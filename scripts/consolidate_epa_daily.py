import pandas as pd
from loguru import logger
from pathlib import Path

# Conversion: EPA reports CO2 in short tons; paper uses metric tons
SHORT_TON_TO_METRIC = 0.90718474

# These column names match EPA's daily emissions CSV format
# Verify by reading one file's header before relying on them
COLS_FROM_EPA = {
    "Facility ID": "facility_id",
    "State": "state",
    "Date": "date",
    "Gross Load (MWh)": "gross_load_mwh",
    "CO2 Mass (short tons)": "co2_short_tons",
    "Primary Fuel Type": "primary_fuel_type",
}


def load_daily_csv(path: Path) -> pd.DataFrame:
    """Load one EPA daily emissions CSV with proper typing and renaming."""
    df = pd.read_csv(path)
    # Only keep needed columns; renames to snake_case
    keep_cols = [c for c in COLS_FROM_EPA if c in df.columns]
    df = df[keep_cols].rename(columns={c: COLS_FROM_EPA[c] for c in keep_cols})
    df["date"] = pd.to_datetime(df["date"])
    return df


def consolidate_daily(raw_dir: Path, output_path: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(raw_dir.glob("*.csv")):
        logger.info(f"Reading {csv_path.name}")
        df = load_daily_csv(csv_path)
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full["co2_metric_tons"] = full["co2_short_tons"] * SHORT_TON_TO_METRIC

    # Aggregate to facility-day level
    # For fuel: take the mode (most common) across units; ties broken arbitrarily
    def first_mode(s: pd.Series) -> str | None:
        modes = s.mode(dropna=True)
        return modes.iloc[0] if len(modes) else None

    daily = (
        full.groupby(["facility_id", "state", "date"], as_index=False)
        .agg(
            co2_metric_tons=("co2_metric_tons", "sum"),
            gross_load_mwh=("gross_load_mwh", "sum"),
            primary_fuel_type=("primary_fuel_type", first_mode),
        )
        .sort_values(["facility_id", "date"])
        .reset_index(drop=True)
    )
    daily = daily[daily["primary_fuel_type"] != "Wood"].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_path, index=False, compression="snappy")
    logger.info(f"Saved {len(daily)} rows to {output_path}")
    return daily


def main() -> None:
    raw_dir = Path("data/raw/epa_daily")
    output_path = Path("data/processed/epa_daily.parquet")
    daily = consolidate_daily(raw_dir, output_path)
    logger.info(f"Consolidated daily emissions data has {len(daily)} rows")
    logger.info(f"Unique facilities: {daily['facility_id'].nunique()}")
    logger.info(f"Date range: {daily['date'].min()} to {daily['date'].max()}")
    df = pd.read_parquet("data/processed/epa_daily.parquet")
    PERIODS = [
        ("2021-04-01", "2021-05-20"),
        ("2021-09-01", "2021-10-01"),
        ("2022-04-01", "2022-04-30"),
        ("2022-09-01", "2022-09-29"),
    ]
    PAPER_PLANTS = [583, 590, 513, 592]
    PAPER_SAMPLES = [20306, 14464, 11213, 15243]
    # Apply paper's filter: positive load and CO2
    df_valid = df[(df["co2_metric_tons"] > 0) & (df["gross_load_mwh"] > 0)].copy()

    logger.info(
        f"{'Period':30s} {'Plants':>8s} {'(paper)':>8s} {'Δ':>6s} {'Rows':>7s} {'(paper)':>8s}"
    )
    logger.info("-" * 80)
    for (start, end), pp, ps in zip(PERIODS, PAPER_PLANTS, PAPER_SAMPLES):
        sub = df_valid[(df_valid["date"] >= start) & (df_valid["date"] < end)]
        n_p = sub["facility_id"].nunique()
        n_r = len(sub)
        delta = n_p - pp
        logger.info(
            f"  {start} to {end[:7]:8s}  {n_p:8d} {pp:8d} {delta:+5d} {n_r:7d} {ps:8d}"
        )


if __name__ == "__main__":
    main()
