# Stage 2: Cleaning, MLflow Tracking & Registry, Tests

## Overview

Stage 2 implements the core MLOps pipeline components:
1. **Data Cleaning** - Handles all data quality issues identified in `findings.yaml`
2. **MLflow Tracking & Model Registry** - Experiment tracking, model versioning, staging/production promotion
3. **Automated Tests** - Unit tests for all components

## Data Cleaning (`src/cleaning.py`)

Addresses all issues from `reports/findings.yaml`:

| Issue | Column | Treatment |
|-------|--------|-----------|
| **Leaking Column** | `meter_reading_kwh` | Removed (target leakage) |
| **Sentinel Values** | `dust_days` = 9999 (43 occurrences) | Replaced with `NaN` for imputation |
| **Stuck Sensor** | `temp_c` (2026-05-02 to 2026-05-18) | Replaced with `NaN` for imputation |
| **Target Typos** | `energy_kwh` (10 occurrences) | Values above 100 kWh are divided by 10 |
| **Spelling Variants** | `roof_type` (3 true levels) | Case, whitespace, and observed abbreviations are standardized to `tin`, `flat`, or `sloped` |

### Usage

```bash
# Run cleaning standalone
python -m src.cleaning

# Or use in code
from src.cleaning import clean_data, load_data
df = load_data("data/raw/train.csv")
clean_df = clean_data(df)
```

### Output

Cleaned data saved to `data/processed/`:
- `train_clean.csv`
- `later_clean.csv`
- `holdout_clean.csv`

## MLflow Tracking & Registry (`src/mlflow_utils.py`)

### Features

- **Experiment Tracking**: Parameters, metrics, artifacts, models
- **Model Registry**: Versioned models with stage transitions (Staging → Production)
- **Champion Alias**: Registers the lowest validation MAE model and assigns the `champion` alias
- **Model Archiving**: Archives old production versions

### Key Classes & Functions

```python
from src.mlflow_utils import (
    MLflowTracker,           # Context manager for runs
    log_params,              # Log parameters
    log_metrics,             # Log metrics
    log_model,               # Log & register model
    register_model,          # Register model version
    transition_model_stage,  # Staging/Production/Archived
    promote_model_to_production,  # Auto-promote best model
    get_model_by_stage,      # Load model from registry
)

# Usage with context manager
with MLflowTracker(run_name="experiment_1", tags={"model": "rf"}) as tracker:
    tracker.log_params({"n_estimators": 100})
    tracker.log_metrics({"mae": 0.5})
    tracker.log_model(model, signature=signature, input_example=X_train[:5])
```

### MLflow UI

```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Access at http://localhost:5000
```

## Training Pipeline (`src/train.py`)

### Models Supported

- **RandomForestRegressor** (default)
- **GradientBoostingRegressor**
- **Ridge** (linear baseline)

### Usage

```bash
# Train all models and compare
python -m src.train

# Or in code
from src.train import train_and_compare_models
results = train_and_compare_models()
```

### MLflow Integration

Each of three candidate training runs logs:
- Dataset info (paths, shapes, features)
- SHA-256 of the original `data/raw/train.csv`
- Model hyperparameters
- Validation metrics (MAE, MSE, RMSE, R²), including an `mae` metric
- Cross-validation scores
- Model artifact with signature

The best candidate is registered once as v1 and assigned the `champion` alias.

## Tests (`tests/`)

### Test Coverage

| Test File | Coverage |
|-----------|----------|
| `test_cleaning.py` | All cleaning functions + integration |
| `test_train.py` | Preprocessing, pipelines, evaluation |
| `test_mlflow_utils.py` | All MLflow utilities (mocked) |

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_cleaning.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Test Fixtures (`conftest.py`)

- `sample_raw_data` - Synthetic solar data
- `sample_raw_data_with_issues` - Data with known issues from findings
- `temp_data_dir` - Temporary directory for test data
- `reset_mlflow` - Auto-cleanup MLflow runs
- `mock_mlflow_client` - Mocked MLflow client

## Project Structure

```
MP096-Solar-MLOps/
├── src/
│   ├── __init__.py
│   ├── cleaning.py          # Data cleaning pipeline
│   ├── mlflow_utils.py      # MLflow tracking & registry
│   └── train.py             # Training pipeline
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Pytest fixtures
│   ├── test_cleaning.py     # Cleaning tests
│   ├── test_train.py        # Training tests
│   └── test_mlflow_utils.py # MLflow tests
├── data/
│   ├── raw/                 # Raw data (DVC tracked)
│   └── processed/           # Cleaned data (generated)
├── reports/
│   └── findings.yaml        # Data quality findings
├── requirements.txt         # Dependencies
├── pyproject.toml           # Project config
├── run_stage2.py            # Full pipeline runner
└── STAGE2_README.md         # This file
```

## Running the Full Pipeline

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .[dev]

# Run complete Stage 2 pipeline
python run_stage2.py
```

## Configuration

Key constants in `src/cleaning.py` (matching `findings.yaml`):

```python
SENTINEL_VALUE = 9999
SENTINEL_COLUMN = "dust_days"
STUCK_SENSOR_COLUMN = "temp_c"
STUCK_START_DATE = "2026-05-02"
STUCK_END_DATE = "2026-05-18"
LEAKING_COLUMN = "meter_reading_kwh"
ROOF_TYPE_MAPPING = {...}  # 3 true levels
```

MLflow config in `src/mlflow_utils.py`:

```python
EXPERIMENT_NAME = "solar-mlops-mp096"
REGISTERED_MODEL_NAME = "solar-power-predictor"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
```

## Next Steps (Stage 3)

- [ ] Model Monitoring (PSI drift detection)
- [ ] Automated Retraining Pipeline
- [ ] Deployment (API/Container)
- [ ] A/B Testing Framework
- [ ] Data Quality Monitoring
