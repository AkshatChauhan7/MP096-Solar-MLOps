"""Pytest configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def sample_raw_data():
    """Create sample raw data matching the solar dataset structure."""
    return pd.DataFrame({
        "record_id": [f"R{i:05d}" for i in range(1, 101)],
        "date": pd.date_range("2026-01-01", periods=100, freq="D").astype(str),
        "panel_area_m2": np.random.uniform(10, 60, 100),
        "sunshine_hours": np.random.uniform(2, 11, 100),
        "temp_c": np.random.uniform(15, 45, 100),
        "dust_days": np.random.randint(0, 30, 100),
        "panel_age_yr": np.random.uniform(5, 25, 100),
        "cloud_cover_pct": np.random.randint(0, 100, 100),
        "roof_type": np.random.choice(["tin", "Tin", "flat", "Flat", "sloped", "Sloped", "slopped"], 100),
        "area": np.random.choice(["Old City", "New Colony", "Industrial"], 100),
        "meter_reading_kwh": np.random.uniform(50, 500, 100),
        "energy_kwh": np.random.uniform(10, 50, 100),
    })


@pytest.fixture
def sample_raw_data_with_issues():
    """Create sample raw data with known issues from findings.yaml."""
    df = pd.DataFrame({
        "record_id": [f"R{i:05d}" for i in range(1, 21)],
        "date": [
            "2026-05-01", "2026-05-02", "2026-05-05", "2026-05-10",
            "2026-05-15", "2026-05-18", "2026-05-19", "2026-05-20",
            "2026-05-21", "2026-05-22", "2026-05-23", "2026-05-24",
            "2026-05-25", "2026-05-26", "2026-05-27", "2026-05-28",
            "2026-05-29", "2026-05-30", "2026-05-31", "2026-06-01"
        ],
        "panel_area_m2": [20] * 20,
        "sunshine_hours": [5] * 20,
        "temp_c": [25, 30, 30, 30, 30, 30, 28, 27, 26, 25, 24, 23, 22, 21, 20, 25, 26, 27, 28, 29],  # stuck at 30
        "dust_days": [1, 2, 9999, 4, 5, 9999, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],  # sentinel values
        "panel_age_yr": [5] * 20,
        "cloud_cover_pct": [10] * 20,
        "roof_type": ["tin", "Tin", "flat", "Flat", "sloped", "Sloped", "slopped", "tin", "flat", "sloped",
                      "tin", "Tin", "flat", "Flat", "sloped", "Sloped", "slopped", "tin", "flat", "sloped"],
        "area": ["Old City"] * 20,
        "meter_reading_kwh": [100] * 20,  # leaking column
        "energy_kwh": [20.0] * 19 + [233.9],
    })
    return df


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory structure."""
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture(autouse=True)
def reset_mlflow():
    """Reset MLflow state between tests."""
    import mlflow
    # End any active run
    try:
        mlflow.end_run()
    except Exception:
        pass
    yield
    try:
        mlflow.end_run()
    except Exception:
        pass


@pytest.fixture
def mock_mlflow_client():
    """Mock MLflow client for testing."""
    from unittest.mock import MagicMock
    client = MagicMock()
    return client
