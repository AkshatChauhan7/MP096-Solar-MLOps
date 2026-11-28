
"""Reproducible Stage 1 data-quality audit for MP096 Solar MLOps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"

# Detection thresholds
LEAKAGE_CORRELATION_THRESHOLD = 0.95
SENTINEL_COLUMN = "dust_days"
SENTINEL_VALUE = 9999
STUCK_COLUMN = "temp_c"
STUCK_MIN_DAYS = 5
STUCK_MIN_ROWS_PER_DAY = 2
TARGET_COLUMN = "energy_kwh"
METER_COLUMN = "meter_reading_kwh"
TARGET_RATIO_MIN = 9.5
TARGET_RATIO_MAX = 10.5

# Canonical roof categories and known spelling variants
ROOF_MAPPING = {
    "tin": "tin",
    "tin.": "tin",
    "flat": "flat",
    "fla.": "flat",
    "sloped": "sloped",
    "slo.": "sloped",
    "slopped": "sloped",
}

# Broad physical plausibility bounds.
# These are screening limits, not proof of a unit conversion error.
PLAUSIBILITY_BOUNDS = {
    "panel_area_m2": (0, 1000),
    "sunshine_hours": (0, 24),
    "temp_c": (-50, 60),
    "cloud_cover_pct": (0, 100),
    "panel_age_yr": (0, 100),
}


def load_config() -> dict:
    """Load dataset and report paths from params.yaml."""
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config["monitoring"]


def load_datasets(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw training and later datasets."""
    train_path = ROOT / config["train_path"]
    later_path = ROOT / config["later_path"]

    for path in (train_path, later_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}. Restore the assignment data first."
            )

    return pd.read_csv(train_path), pd.read_csv(later_path)


def check_required_columns(
    train: pd.DataFrame, later: pd.DataFrame
) -> None:
    """Validate columns required by the audit."""
    required = {
        "record_id",
        "date",
        "panel_area_m2",
        "sunshine_hours",
        "temp_c",
        "dust_days",
        "panel_age_yr",
        "cloud_cover_pct",
        "roof_type",
        "meter_reading_kwh",
        "energy_kwh",
    }

    for name, frame in (("train", train), ("later", later)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name}.csv is missing required columns: {sorted(missing)}"
            )


def check_potential_leakage(
    train: pd.DataFrame, later: pd.DataFrame
) -> dict:
    """Compare meter-reading correlation with the target in each dataset."""
    results = {}

    for name, frame in (("train", train), ("later", later)):
        correlation = frame[METER_COLUMN].corr(frame[TARGET_COLUMN])

        results[name] = {
            "correlation": (
                float(correlation) if pd.notna(correlation) else None
            ),
            "threshold": LEAKAGE_CORRELATION_THRESHOLD,
            "potential_leakage": bool(
                pd.notna(correlation)
                and abs(correlation) >= LEAKAGE_CORRELATION_THRESHOLD
            ),
        }

    return {
        "column": METER_COLUMN,
        "target": TARGET_COLUMN,
        "datasets": results,
        "interpretation": (
            "High correlation is a warning requiring investigation; "
            "correlation alone does not prove target leakage."
        ),
    }


def check_unit_inconsistency(
    train: pd.DataFrame, later: pd.DataFrame
) -> dict:
    """Screen numerical columns for values outside broad physical bounds."""
    results = {}

    for name, frame in (("train", train), ("later", later)):
        violations = {}

        for column, (lower, upper) in PLAUSIBILITY_BOUNDS.items():
            values = pd.to_numeric(frame[column], errors="coerce")
            mask = values.notna() & (
                (values < lower) | (values > upper)
            )

            if mask.any():
                violations[column] = {
                    "count": int(mask.sum()),
                    "minimum_allowed": lower,
                    "maximum_allowed": upper,
                    "observed_values": [
                        float(value)
                        for value in values[mask].head(20).tolist()
                    ],
                }

        results[name] = {
            "out_of_range_values": violations,
            "screening_passed": not bool(violations),
        }

    return {
        "method": "Broad physical plausibility range screening",
        "bounds": {
            key: list(value)
            for key, value in PLAUSIBILITY_BOUNDS.items()
        },
        "datasets": results,
        "confirmed_unit_mixup": False,
        "interpretation": (
            "Range violations are candidates for investigation, not "
            "automatic proof of a unit mix-up. No unit conversion is applied."
        ),
    }


def check_sentinel_values(
    train: pd.DataFrame, later: pd.DataFrame
) -> dict:
    """Count sentinel values in both datasets."""
    results = {}

    for name, frame in (("train", train), ("later", later)):
        count = int((frame[SENTINEL_COLUMN] == SENTINEL_VALUE).sum())
        results[name] = {"count": count}

    return {
        "column": SENTINEL_COLUMN,
        "value": SENTINEL_VALUE,
        "datasets": results,
    }


def check_duplicates(
    train: pd.DataFrame, later: pd.DataFrame
) -> dict:
    """Detect complete duplicate rows and duplicate record IDs."""
    results = {}

    for name, frame in (("train", train), ("later", later)):
        results[name] = {
            "duplicate_rows": int(frame.duplicated().sum()),
            "duplicate_record_ids": int(
                frame["record_id"].duplicated().sum()
            ),
        }

    return results


def find_consecutive_dates(dates: list[str]) -> list[list[str]]:
    """Group sorted ISO dates into consecutive calendar-date sequences."""
    if not dates:
        return []

    parsed = sorted(pd.to_datetime(date) for date in dates)
    groups = [[parsed[0]]]

    for date in parsed[1:]:
        if (date - groups[-1][-1]).days == 1:
            groups[-1].append(date)
        else:
            groups.append([date])

    return [
        [date.strftime("%Y-%m-%d") for date in group]
        for group in groups
    ]


def check_stuck_sensor(train: pd.DataFrame) -> dict:
    """Find long consecutive periods of constant daily temperature."""
    daily = train.copy()
    daily["date"] = pd.to_datetime(
        daily["date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    daily = daily.dropna(subset=["date"])

    stats = daily.groupby("date")[STUCK_COLUMN].agg(
        row_count="count",
        unique_values="nunique",
        temperature="first",
    )

    suspicious_dates = stats[
        (stats["row_count"] >= STUCK_MIN_ROWS_PER_DAY)
        & (stats["unique_values"] == 1)
    ]

    sequences = find_consecutive_dates(
        suspicious_dates.index.tolist()
    )

    detected = [
        {
            "start_date": sequence[0],
            "end_date": sequence[-1],
            "consecutive_days": len(sequence),
            "temperature": float(
                suspicious_dates.loc[sequence[0], "temperature"]
            ),
            "total_rows": int(
                suspicious_dates.loc[sequence, "row_count"].sum()
            ),
        }
        for sequence in sequences
        if len(sequence) >= STUCK_MIN_DAYS
    ]

    return {
        "column": STUCK_COLUMN,
        "minimum_rows_per_day": STUCK_MIN_ROWS_PER_DAY,
        "minimum_consecutive_days": STUCK_MIN_DAYS,
        "detected_periods": detected,
    }


def check_target_anomalies(
    train: pd.DataFrame, later: pd.DataFrame
) -> dict:
    """Find energy readings approximately ten times the meter reading."""
    results = {}

    for name, frame in (("train", train), ("later", later)):
        ratio = frame[TARGET_COLUMN] / frame[METER_COLUMN]

        mask = (
            frame[TARGET_COLUMN].gt(100)
            & ratio.between(TARGET_RATIO_MIN, TARGET_RATIO_MAX)
            & np.isfinite(ratio)
        )

        records = frame.loc[
            mask,
            ["record_id", "date", METER_COLUMN, TARGET_COLUMN],
        ].copy()

        records["energy_to_meter_ratio"] = ratio[mask]

        results[name] = {
            "count": int(mask.sum()),
            "records": records.to_dict(orient="records"),
        }

    return {
        "target_column": TARGET_COLUMN,
        "comparison_column": METER_COLUMN,
        "ratio_range": [
            TARGET_RATIO_MIN,
            TARGET_RATIO_MAX,
        ],
        "datasets": results,
        "interpretation": (
            "These records are suspicious and should be reviewed "
            "before applying the Stage 2 correction."
        ),
    }


def check_roof_type_variants(
    train: pd.DataFrame, later: pd.DataFrame
) -> dict:
    """Identify raw roof labels and map recognized variants to canonical labels."""
    results = {}

    for name, frame in (("train", train), ("later", later)):
        raw_counts = (
            frame["roof_type"]
            .astype("string")
            .value_counts(dropna=False)
        )

        normalized = (
            frame["roof_type"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        mapped = normalized.map(ROOF_MAPPING)

        results[name] = {
            "raw_value_counts": {
                str(key): int(value)
                for key, value in raw_counts.items()
            },
            "canonical_value_counts": {
                str(key): int(value)
                for key, value in mapped.value_counts().items()
            },
            "unknown_values": [
                str(value)
                for value in normalized[mapped.isna()].dropna().unique()
            ],
            "canonical_levels": int(mapped.nunique()),
        }

    return {
        "column": "roof_type",
        "mapping": ROOF_MAPPING,
        "datasets": results,
    }


def run_audit(train: pd.DataFrame, later: pd.DataFrame) -> dict:
    """Run all seven independent data-quality checks."""
    check_required_columns(train, later)

    return {
        "project": "MP096 Solar MLOps",
        "group_id": "G199",
        "datasets": {
            "train": {
                "rows": int(len(train)),
                "columns": int(len(train.columns)),
            },
            "later": {
                "rows": int(len(later)),
                "columns": int(len(later.columns)),
            },
        },
        "checks": {
            "potential_target_leakage": check_potential_leakage(
                train, later
            ),
            "unit_inconsistency_screening": check_unit_inconsistency(
                train, later
            ),
            "sentinel_values": check_sentinel_values(train, later),
            "duplicates": check_duplicates(train, later),
            "stuck_sensor": check_stuck_sensor(train),
            "target_anomalies": check_target_anomalies(train, later),
            "roof_type_variants": check_roof_type_variants(
                train, later
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reproducible Stage 1 data-quality checks."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output YAML path. Defaults to reports/stage1_audit.yaml.",
    )
    args = parser.parse_args()

    config = load_config()
    train, later = load_datasets(config)
    report = run_audit(train, later)

    output_path = args.output or (
        ROOT / "reports" / "stage1_audit.yaml"
    )
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            report,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    print("Stage 1 audit completed.")
    print(f"Training records: {len(train)}")
    print(f"Later records: {len(later)}")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
