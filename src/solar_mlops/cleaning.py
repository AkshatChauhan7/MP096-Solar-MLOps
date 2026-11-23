"""
Data cleaning module for solar energy prediction dataset.

This module implements detection and handling of 7 known data quality issues:
1. leaking_column - meter_reading_kwh (only filled after event, matches target almost exactly)
2. unit_mixup - check each numeric column per area group for 10x/100x/1000x differences
3. sentinel_value - dust_days has 9999 (43 occurrences per findings.yaml)
4. duplicate_rows - check df.duplicated().sum()
5. stuck_sensor - temp_c constant from 2026-05-02 to 2026-05-18
6. target_typos - energy_kwh values 10x too large (10 occurrences)
7. spelling_variants - roof_type has variants, true_levels=3

All detection logic is encapsulated in functions for viva demonstration.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from .utils import setup_logging, get_numeric_columns, ensure_dir, load_csv, save_csv

logger = setup_logging(__name__)

# Constants for data issues
LEAKING_COLUMN = "meter_reading_kwh"
TARGET_COLUMN = "energy_kwh"
SENTINEL_COLUMN = "dust_days"
SENTINEL_VALUE = 9999
STUCK_SENSOR_COLUMN = "temp_c"
STUCK_SENSOR_START = "2026-05-02"
STUCK_SENSOR_END = "2026-05-18"
STUCK_SENSOR_VALUE = 25.0
TARGET_TYPO_FACTOR = 10
TARGET_TYPO_THRESHOLD = 200  # values above this are likely 10x typos
SPELLING_COLUMN = "roof_type"
TRUE_ROOF_TYPES = ["flat", "sloped", "tin"]
NUMERIC_FEATURES = [
    "panel_area_m2",
    "sunshine_hours",
    "temp_c",
    "dust_days",
    "panel_age_yr",
    "cloud_cover_pct",
]
GROUP_COLUMN = "area"
DATE_COLUMN = "date"
ID_COLUMN = "record_id"


def load_raw_data(path: str | Path = "data/raw/train.csv") -> pd.DataFrame:
    """
    Load raw training data from CSV.

    Args:
        path: Path to raw train.csv file

    Returns:
        Raw DataFrame with all columns
    """
    logger.info(f"Loading raw data from {path}")
    df = load_csv(path)
    # Keep date as string to avoid segfault with pd.to_datetime in this environment
    # String comparison works for ISO format dates (YYYY-MM-DD)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def detect_leaking_column(df: pd.DataFrame, leaking_col: str = LEAKING_COLUMN,
                          target_col: str = TARGET_COLUMN,
                          correlation_threshold: float = 0.95,
                          close_fraction_threshold: float = 0.7,
                          close_tolerance: float = 1.0) -> Dict[str, Any]:
    """
    Detect leaking column that matches target almost exactly.

    A leaking column is one that is only filled after the event and
    matches the target variable almost exactly (data leakage).

    Uses robust detection: fraction of rows where |leaking - target| < tolerance.
    This handles cases where target has typos/outliers that drag down correlation.

    Args:
        df: Input DataFrame
        leaking_col: Suspected leaking column name
        target_col: Target column name
        correlation_threshold: Pearson correlation threshold (traditional method)
        close_fraction_threshold: Fraction of rows that must be close (robust method)
        close_tolerance: Absolute tolerance for "close" match

    Returns:
        Dictionary with detection results
    """
    logger.info(f"Detecting leaking column: {leaking_col}")

    if leaking_col not in df.columns or target_col not in df.columns:
        return {"present": False, "reason": "Column(s) not found"}

    # Calculate correlation (traditional method)
    valid_mask = df[leaking_col].notna() & df[target_col].notna()
    if valid_mask.sum() < 2:
        return {"present": False, "reason": "Insufficient valid data"}

    correlation = df.loc[valid_mask, leaking_col].corr(df.loc[valid_mask, target_col])

    # Robust method: fraction of rows where values are very close
    diff = (df.loc[valid_mask, leaking_col] - df.loc[valid_mask, target_col]).abs()
    close_fraction = (diff < close_tolerance).mean()

    # Check if leaking column has many missing values (only filled after event)
    missing_pct = df[leaking_col].isna().mean()

    # Use robust detection as primary (handles target typos)
    is_leaking = close_fraction > close_fraction_threshold

    result = {
        "present": is_leaking,
        "column": leaking_col,
        "correlation_with_target": float(correlation),
        "close_fraction": float(close_fraction),
        "close_tolerance": close_tolerance,
        "close_fraction_threshold": close_fraction_threshold,
        "missing_percentage": float(missing_pct),
    }

    if result["present"]:
        logger.warning(f"Leaking column detected: {leaking_col} "
                       f"(close_fraction={close_fraction:.2%} > {close_fraction_threshold:.0%}, "
                       f"correlation={correlation:.4f}, missing={missing_pct:.1%})")
    else:
        logger.info(f"No leaking column detected for {leaking_col} "
                    f"(close_fraction={close_fraction:.2%}, correlation={correlation:.4f})")

    return result


def detect_unit_mixup(df: pd.DataFrame,
                      numeric_cols: Optional[List[str]] = None,
                      group_col: str = GROUP_COLUMN,
                      factors: Tuple[int, ...] = (10, 100, 1000),
                      exclude_values: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
    """
    Detect unit mixup in numeric columns per group.

    Checks each numeric column within each group for values that are
    10x, 100x, or 1000x different from the group median, indicating
    potential unit inconsistencies (e.g., meters vs centimeters).

    Known sentinel values (like 9999 in dust_days) are excluded by default.

    Args:
        df: Input DataFrame
        numeric_cols: List of numeric columns to check (default: NUMERIC_FEATURES)
        group_col: Column to group by (default: "area")
        factors: Multiplication factors to check for outliers
        exclude_values: Dict of column -> list of values to exclude from check

    Returns:
        Dictionary with detection results per group and column
    """
    logger.info(f"Detecting unit mixup in columns: {numeric_cols or NUMERIC_FEATURES}")

    numeric_cols = numeric_cols or NUMERIC_FEATURES
    # Default exclude sentinel values
    if exclude_values is None:
        exclude_values = {SENTINEL_COLUMN: [SENTINEL_VALUE]}

    results = {}
    any_detected = False

    for group_val in df[group_col].unique():
        group_df = df[df[group_col] == group_val]
        results[group_val] = {}

        for col in numeric_cols:
            if col not in group_df.columns:
                continue

            # Exclude known sentinel/problematic values
            exclude_vals = exclude_values.get(col, [])
            check_df = group_df[~group_df[col].isin(exclude_vals)]

            if len(check_df) == 0:
                results[group_val][col] = {"present": False, "reason": "All values excluded"}
                continue

            median_val = check_df[col].median()
            if median_val == 0:
                results[group_val][col] = {"present": False, "reason": "Zero median"}
                continue

            col_results = {}
            for factor in factors:
                # Check for values factor times LARGER than median (unit mixup)
                outliers = check_df[check_df[col] > median_val * factor]
                if len(outliers) > 0:
                    col_results[f"{factor}x_larger"] = {
                        "count": len(outliers),
                        "values": outliers[col].tolist()[:10],
                        "median": float(median_val),
                    }
                    any_detected = True

            results[group_val][col] = col_results if col_results else {"present": False}

    result = {
        "present": any_detected,
        "by_group": results,
        "checked_factors": list(factors),
        "excluded_values": exclude_values,
    }

    if any_detected:
        logger.warning("Unit mixup detected in one or more columns/groups")
    else:
        logger.info("No unit mixup detected")

    return result


def detect_sentinel_values(df: pd.DataFrame,
                           sentinel_col: str = SENTINEL_COLUMN,
                           sentinel_val: float = SENTINEL_VALUE) -> Dict[str, Any]:
    """
    Detect sentinel values in specified column.

    Sentinel values are placeholder values (like 9999, -999) used to
    indicate missing or invalid data that should be treated as NaN.

    Args:
        df: Input DataFrame
        sentinel_col: Column to check for sentinel values
        sentinel_val: The sentinel value to detect

    Returns:
        Dictionary with detection results
    """
    logger.info(f"Detecting sentinel value {sentinel_val} in column {sentinel_col}")

    if sentinel_col not in df.columns:
        return {"present": False, "reason": f"Column {sentinel_col} not found"}

    count = (df[sentinel_col] == sentinel_val).sum()
    percentage = count / len(df) * 100

    result = {
        "present": count > 0,
        "column": sentinel_col,
        "sentinel_value": sentinel_val,
        "count": int(count),
        "percentage": float(percentage),
    }

    if result["present"]:
        logger.warning(f"Sentinel value detected: {sentinel_col}={sentinel_val} "
                       f"({count} occurrences, {percentage:.1f}%)")
    else:
        logger.info(f"No sentinel value {sentinel_val} found in {sentinel_col}")

    return result


def detect_duplicate_rows(df: pd.DataFrame,
                          subset: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Detect duplicate rows in DataFrame.

    Args:
        df: Input DataFrame
        subset: Columns to consider for duplication (default: all columns)

    Returns:
        Dictionary with detection results
    """
    logger.info("Detecting duplicate rows")

    dup_count = df.duplicated(subset=subset).sum()
    dup_percentage = dup_count / len(df) * 100

    result = {
        "present": dup_count > 0,
        "count": int(dup_count),
        "percentage": float(dup_percentage),
        "subset": subset,
    }

    if result["present"]:
        logger.warning(f"Duplicate rows detected: {dup_count} ({dup_percentage:.1f}%)")
    else:
        logger.info("No duplicate rows detected")

    return result


def detect_stuck_sensor(df: pd.DataFrame,
                        sensor_col: str = STUCK_SENSOR_COLUMN,
                        date_col: str = DATE_COLUMN,
                        start_date: str = STUCK_SENSOR_START,
                        end_date: str = STUCK_SENSOR_END,
                        stuck_value: Optional[float] = None) -> Dict[str, Any]:
    """
    Detect stuck sensor (constant values over a date range).

    A stuck sensor reports the same value continuously over a period,
    indicating sensor malfunction.

    Args:
        df: Input DataFrame
        sensor_col: Sensor column to check
        date_col: Date column for filtering (string format YYYY-MM-DD)
        start_date: Start of suspected stuck period
        end_date: End of suspected stuck period
        stuck_value: Specific value to check for (default: any constant value)

    Returns:
        Dictionary with detection results
    """
    logger.info(f"Detecting stuck sensor in {sensor_col} "
                f"from {start_date} to {end_date}")

    if sensor_col not in df.columns or date_col not in df.columns:
        return {"present": False, "reason": "Required column(s) not found"}

    # Filter to date range (string comparison works for ISO format YYYY-MM-DD)
    mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    period_df = df.loc[mask].copy()

    if len(period_df) == 0:
        return {"present": False, "reason": "No data in date range"}

    # Check if sensor values are constant in this period
    unique_values = period_df[sensor_col].nunique()
    is_constant = unique_values == 1

    if is_constant:
        detected_value = period_df[sensor_col].iloc[0]
        # If stuck_value specified, verify it matches
        if stuck_value is not None and detected_value != stuck_value:
            is_constant = False

    result = {
        "present": is_constant,
        "column": sensor_col,
        "start_date": start_date,
        "end_date": end_date,
        "stuck_value": float(period_df[sensor_col].iloc[0]) if is_constant else None,
        "unique_values_in_period": int(unique_values),
        "row_count_in_period": int(len(period_df)),
    }

    if result["present"]:
        logger.warning(f"Stuck sensor detected: {sensor_col}={result['stuck_value']} "
                       f"from {start_date} to {end_date} ({len(period_df)} rows)")
    else:
        logger.info(f"No stuck sensor detected in {sensor_col} for date range")

    return result


def detect_target_typos(df: pd.DataFrame,
                        target_col: str = TARGET_COLUMN,
                        typo_factor: float = TARGET_TYPO_FACTOR,
                        threshold: float = TARGET_TYPO_THRESHOLD) -> Dict[str, Any]:
    """
    Detect target variable typos (values typo_factor times too large).

    Common data entry error where target values are entered with
    an extra digit (e.g., 350 instead of 35.0).

    Args:
        df: Input DataFrame
        target_col: Target column name
        typo_factor: Factor by which typos are inflated (default: 10)
        threshold: Absolute threshold above which values are suspicious

    Returns:
        Dictionary with detection results
    """
    logger.info(f"Detecting target typos in {target_col} "
                f"(factor={typo_factor}, threshold={threshold})")

    if target_col not in df.columns:
        return {"present": False, "reason": f"Column {target_col} not found"}

    # Find values above threshold
    typo_mask = df[target_col] > threshold
    typo_count = typo_mask.sum()
    typo_percentage = typo_count / len(df) * 100

    # Also check relative to median
    median_val = df[target_col].median()
    relative_typos = df[df[target_col] > median_val * typo_factor]
    relative_count = len(relative_typos)

    result = {
        "present": typo_count > 0,
        "column": target_col,
        "threshold": threshold,
        "typo_factor": typo_factor,
        "absolute_count": int(typo_count),
        "absolute_percentage": float(typo_percentage),
        "relative_to_median_count": int(relative_count),
        "median_value": float(median_val),
        "suspected_typo_values": df.loc[typo_mask, target_col].tolist() if typo_count > 0 else [],
    }

    if result["present"]:
        logger.warning(f"Target typos detected: {typo_count} values > {threshold} "
                       f"({typo_percentage:.1f}%)")
    else:
        logger.info(f"No target typos detected in {target_col}")

    return result


def detect_spelling_variants(df: pd.DataFrame,
                             cat_col: str = SPELLING_COLUMN,
                             true_levels: List[str] = TRUE_ROOF_TYPES) -> Dict[str, Any]:
    """
    Detect spelling variants in categorical column.

    Identifies fuzzy matches to known true levels (e.g., "tin", "TIN", "Tin", "tin.", "fla.", "slo.")
    that should be standardized.

    Args:
        df: Input DataFrame
        cat_col: Categorical column to check
        true_levels: List of canonical/true level values

    Returns:
        Dictionary with detection results including variant mapping
    """
    logger.info(f"Detecting spelling variants in {cat_col}, "
                f"true_levels={true_levels}")

    if cat_col not in df.columns:
        return {"present": False, "reason": f"Column {cat_col} not found"}

    # Get all unique values
    unique_vals = df[cat_col].dropna().unique()

    # True levels normalized
    true_levels_norm = [str(v).strip().lower() for v in true_levels]

    # Map each unique value to closest true level
    variant_map = {}
    for val in unique_vals:
        val_str = str(val)
        val_norm = val_str.strip().lower()
        val_norm_nopunct = val_norm.rstrip('.').strip()
        matched = False
        for true_val in true_levels:
            true_norm = str(true_val).strip().lower()
            # Check various matching strategies:
            # 1. Exact match (case/space insensitive)
            # 2. Match after removing trailing punctuation
            # 3. Match common abbreviations (fla. -> flat, slo. -> sloped)
            if (val_norm == true_norm or
                val_norm_nopunct == true_norm or
                (val_norm_nopunct == "fla" and true_norm == "flat") or
                (val_norm_nopunct == "slo" and true_norm == "sloped")):
                variant_map[val_str] = true_val
                matched = True
                break
        if not matched:
            variant_map[val_str] = "UNMAPPED"

    # Count variants per true level
    variant_counts = {}
    for val, count in df[cat_col].value_counts().items():
        mapped = variant_map.get(str(val), "UNMAPPED")
        if mapped not in variant_counts:
            variant_counts[mapped] = {}
        variant_counts[mapped][str(val)] = int(count)

    # Check if any variants exist (values not exactly matching true levels)
    exact_matches = set(str(v).strip().lower() for v in true_levels)
    has_variants = any(
        str(v).strip().lower() not in exact_matches
        for v in unique_vals
    )

    result = {
        "present": has_variants,
        "column": cat_col,
        "true_levels": true_levels,
        "unique_values_found": [str(v) for v in unique_vals],
        "variant_mapping": variant_map,
        "variant_counts": variant_counts,
        "true_level_count": len(true_levels),
    }

    if result["present"]:
        logger.warning(f"Spelling variants detected in {cat_col}: "
                       f"{len(unique_vals)} unique values vs {len(true_levels)} true levels")
        for true_val, variants in variant_counts.items():
            if true_val != "UNMAPPED":
                logger.info(f"  {true_val}: {variants}")
    else:
        logger.info(f"No spelling variants detected in {cat_col}")

    return result


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to the raw DataFrame.

    This is the main cleaning pipeline that:
    1. Removes leaking column (meter_reading_kwh)
    2. Handles unit mixup (if any detected)
    3. Replaces sentinel values with NaN
    4. Removes duplicate rows
    5. Handles stuck sensor period (interpolate or remove)
    6. Fixes target typos (divide by 10)
    7. Standardizes spelling variants in roof_type

    Args:
        df: Raw input DataFrame

    Returns:
        Cleaned DataFrame
    """
    logger.info("Starting data cleaning pipeline")
    df_clean = df.copy()

    # 1. Handle leaking column - drop it entirely as it's not available at prediction time
    leak_result = detect_leaking_column(df_clean)
    if leak_result.get("present", False):
        logger.info(f"Dropping leaking column: {LEAKING_COLUMN}")
        df_clean = df_clean.drop(columns=[LEAKING_COLUMN])

    # 2. Handle unit mixup - for now, log detection but no automatic fix
    # (unit mixup detection is informational; actual fixes depend on domain knowledge)
    _ = detect_unit_mixup(df_clean)

    # 3. Handle sentinel values - replace with NaN
    sentinel_result = detect_sentinel_values(df_clean)
    if sentinel_result.get("present", False):
        logger.info(f"Replacing sentinel value {SENTINEL_VALUE} with NaN in {SENTINEL_COLUMN}")
        df_clean.loc[df_clean[SENTINEL_COLUMN] == SENTINEL_VALUE, SENTINEL_COLUMN] = np.nan

    # 4. Handle duplicate rows - drop duplicates
    dup_result = detect_duplicate_rows(df_clean)
    if dup_result.get("present", False):
        logger.info(f"Dropping {dup_result['count']} duplicate rows")
        df_clean = df_clean.drop_duplicates()

    # 5. Handle stuck sensor - interpolate temp_c values for stuck period
    stuck_result = detect_stuck_sensor(df_clean)
    if stuck_result.get("present", False):
        logger.info(f"Interpolating stuck sensor values for {STUCK_SENSOR_COLUMN}")
        mask = (df_clean[DATE_COLUMN] >= STUCK_SENSOR_START) & \
               (df_clean[DATE_COLUMN] <= STUCK_SENSOR_END)
        # Use linear interpolation based on surrounding values (method='time' requires datetime)
        df_clean.loc[mask, STUCK_SENSOR_COLUMN] = np.nan
        df_clean[STUCK_SENSOR_COLUMN] = df_clean[STUCK_SENSOR_COLUMN].interpolate(method="linear")

    # 6. Handle target typos - divide suspicious values by 10
    typo_result = detect_target_typos(df_clean)
    if typo_result.get("present", False):
        logger.info(f"Fixing {typo_result['absolute_count']} target typos in {TARGET_COLUMN}")
        typo_mask = df_clean[TARGET_COLUMN] > TARGET_TYPO_THRESHOLD
        df_clean.loc[typo_mask, TARGET_COLUMN] = df_clean.loc[typo_mask, TARGET_COLUMN] / 10

    # 7. Handle spelling variants - standardize roof_type
    spell_result = detect_spelling_variants(df_clean)
    if spell_result.get("present", False):
        logger.info(f"Standardizing spelling variants in {SPELLING_COLUMN}")
        variant_map = spell_result["variant_mapping"]
        df_clean[SPELLING_COLUMN] = df_clean[SPELLING_COLUMN].map(variant_map).fillna(df_clean[SPELLING_COLUMN])

    logger.info(f"Cleaning complete. Rows: {len(df)} -> {len(df_clean)}")
    return df_clean


def save_cleaned_data(df: pd.DataFrame,
                      path: str | Path = "data/processed/train_clean.csv") -> None:
    """
    Save cleaned DataFrame to CSV.

    Args:
        df: Cleaned DataFrame
        path: Output path
    """
    logger.info(f"Saving cleaned data to {path}")
    save_csv(df, path)
    logger.info(f"Saved {len(df)} rows to {path}")


def run_cleaning_pipeline(raw_path: str | Path = "data/raw/train.csv",
                          output_path: str | Path = "data/processed/train_clean.csv") -> pd.DataFrame:
    """
    Run the complete cleaning pipeline from raw data to saved cleaned data.

    Args:
        raw_path: Path to raw train.csv
        output_path: Path to save cleaned data

    Returns:
        Cleaned DataFrame
    """
    logger.info("Running complete cleaning pipeline")
    df_raw = load_raw_data(raw_path)
    df_clean = clean_data(df_raw)
    save_cleaned_data(df_clean, output_path)
    return df_clean


if __name__ == "__main__":
    # Allow running as script for testing
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    df = run_cleaning_pipeline()
    print(f"\nCleaned data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nRoof type distribution:\n{df['roof_type'].value_counts()}")
    print(f"\nDust days stats:\n{df['dust_days'].describe()}")
    print(f"\nEnergy kwh stats:\n{df['energy_kwh'].describe()}")