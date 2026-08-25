from co2sat.utils import project_root
import pandas as pd
from loguru import logger
from src.co2sat.data.epa import categorize_fuel

NON_CONUS = frozenset({"AK", "HI", "PR", "VI", "GU", "MP", "AS"})


def main() -> None:
    root = project_root()
    daily = pd.read_parquet(root / "data" / "processed" / "epa_daily.parquet")
    facilities = pd.read_parquet(root / "data" / "processed" / "epa_facilities.parquet")
    # Drop state from facilities to avoid suffix conflict
    facilities_for_join = facilities.drop(columns=["state"])
    joined = daily.merge(facilities_for_join, on="facility_id", how="left")
    # Reproduce the filter that produced ~1,000 per period earlier
    joined["fuel_category"] = joined["primary_fuel_type"].map(categorize_fuel)
    joined = joined[
        (joined["co2_metric_tons"] > 0)
        & (joined["gross_load_mwh"] > 0)
        & (
            joined["fuel_category"].isin(["coal", "gas", "oil"])  # fossil only
            & (~joined["state"].isin(NON_CONUS))
        )
    ]
    # Diagnose join quality
    n_total = len(joined)
    n_with_coords = joined.dropna(subset=["latitude", "longitude"]).shape[0]
    n_unjoined = joined["facility_name"].isnull().sum()
    logger.info(f"Rows in joined dataset: {n_total:,}")
    logger.info(f"Rows with coordinates: {n_with_coords:,}")
    logger.info(f"Rows without facility match: {n_unjoined:,}")
    # Drop rows without coordinates — can't be matched to satellite pixels
    joined = joined.dropna(subset=["latitude", "longitude"]).copy()
    out_path = root / "data" / "processed" / "epa_daily_with_attributes.parquet"
    joined.to_parquet(out_path, index=False, compression="snappy")
    logger.info("\nFinal Phase 1 dataset:")
    logger.info(f"Rows: {len(joined):,}")
    logger.info(f"Unique facilities: {joined['facility_id'].nunique():,}")
    logger.info(f"Date range: {joined['date'].min()} to {joined['date'].max()}")
    logger.info(f"Columns: {joined.columns.tolist()}")


if __name__ == "__main__":
    main()
