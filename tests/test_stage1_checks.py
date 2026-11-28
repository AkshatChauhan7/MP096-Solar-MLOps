
from unittest import result

import pandas as pd

from scripts.run_stage1_checks import (
    check_potential_leakage,
    check_sentinel_values,
    check_duplicates,
    check_target_anomalies,
    check_roof_type_variants,
    check_stuck_sensor,
    check_unit_inconsistency,
)


def sample_data():
    return pd.DataFrame({
        "record_id": ["R1", "R2", "R3", "R3"],
        "date": ["2026-01-01"] * 4,
        "meter_reading_kwh": [10.0, 20.0, 30.0, 30.0],
        "energy_kwh": [105.0, 20.0, 30.0, 30.0],
        "dust_days": [9999, 5, 10, 10],
        "roof_type": ["TIN", " tin ", "FLAT", "fla."],
    })


def test_sentinel_values():
    df = sample_data()
    result = check_sentinel_values(df, df)

    assert result["datasets"]["train"]["count"] == 1
    assert result["datasets"]["later"]["count"] == 1


def test_duplicates():
    df = pd.DataFrame({
        "record_id": ["R1", "R1"],
        "date": ["2026-01-01", "2026-01-01"],
        "meter_reading_kwh": [30.0, 30.0],
        "energy_kwh": [30.0, 30.0],
        "dust_days": [10, 10],
        "roof_type": ["flat", "flat"],
    })

    result = check_duplicates(df, df)

    assert result["train"]["duplicate_rows"] == 1
    assert result["train"]["duplicate_record_ids"] == 1


def test_target_anomalies():
    df = sample_data()
    result = check_target_anomalies(df, df)

    assert result["datasets"]["train"]["count"] == 1


def test_roof_type_normalization():
    df = sample_data()
    result = check_roof_type_variants(df, df)

    assert result["datasets"]["train"]["canonical_levels"] == 2
    assert result["datasets"]["train"]["unknown_values"] == []


def test_potential_leakage():
    df = pd.DataFrame({
        "meter_reading_kwh": [10, 20, 30, 40, 50],
        "energy_kwh": [10, 20, 30, 40, 50],
    })

    result = check_potential_leakage(df, df)

    assert result["datasets"]["train"]["potential_leakage"] is True
    assert result["datasets"]["later"]["potential_leakage"] is True


def test_stuck_sensor():
    dates = pd.date_range("2026-05-01", periods=7, freq="D")

    df = pd.DataFrame({
        "date": [date for date in dates for _ in range(2)],
        "temp_c": [25.0] * 14,
    })

    result = check_stuck_sensor(df)

    assert len(result["detected_periods"]) == 1

    period = result["detected_periods"][0]

    assert period["consecutive_days"] == 7
    assert period["temperature"] == 25.0
    assert period["total_rows"] == 14

def test_unit_inconsistency_screening():
    df = pd.DataFrame({
        "panel_area_m2": [20.0, 30.0, 40.0],
        "sunshine_hours": [5.0, 8.0, 12.0],
        "temp_c": [25.0, 30.0, 35.0],
        "cloud_cover_pct": [20, 40, 60],
        "panel_age_yr": [2.0, 5.0, 8.0],
    })

    invalid_df = df.copy()
    invalid_df.loc[0, "sunshine_hours"] = 30.0

    result = check_unit_inconsistency(df, invalid_df)

    assert result["datasets"]["train"]["screening_passed"] is True
    assert result["datasets"]["later"]["screening_passed"] is False
    assert (
    result["datasets"]["later"]["out_of_range_values"]["sunshine_hours"]["count"]
    == 1
)