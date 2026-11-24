#!/usr/bin/env python
"""Stage 2 Pipeline Runner for Solar MLOps Project.

This script orchestrates the complete Stage 2 workflow:
1. Data Cleaning
2. Model Training with MLflow Tracking
3. Model Registration
4. Running Tests
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0:
        print(f"\n[OK] {description} completed successfully")
        return True
    else:
        print(f"\n[FAIL] {description} failed with return code {result.returncode}")
        return False


def main():
    """Run the complete Stage 2 pipeline."""
    project_root = Path(__file__).parent
    src_dir = project_root / "src"

    # Ensure we're in the project directory
    import os
    os.chdir(project_root)

    print("="*60)
    print("SOLAR MLOPS - STAGE 2 PIPELINE")
    print("="*60)
    print("Steps:")
    print("  1. Data Cleaning")
    print("  2. Model Training with MLflow Tracking")
    print("  3. Run Tests")

    all_success = True

    # Step 1: Data Cleaning
    print("\n" + "="*60)
    print("STEP 1: DATA CLEANING")
    print("="*60)
    print("This will:")
    print("  - Remove leaking column (meter_reading_kwh)")
    print("  - Fix sentinel values (dust_days=9999 -> NaN)")
    print("  - Fix stuck sensor (temp_c 2026-05-02 to 2026-05-18)")
    print("  - Standardize roof_type spelling variants")
    print("  - Impute missing values")
    print("  - Save cleaned data to data/processed/")

    success = run_command(
        [sys.executable, "-m", "src.cleaning"],
        "Data Cleaning"
    )
    all_success = all_success and success

    # Step 2: Model Training
    print("\n" + "="*60)
    print("STEP 2: MODEL TRAINING WITH MLFLOW")
    print("="*60)
    print("This will:")
    print("  - Train multiple models (RandomForest, GradientBoosting, Ridge)")
    print("  - Track experiments with MLflow")
    print("  - Log parameters, metrics, and models")
    print("  - Register best model to MLflow Model Registry")
    print("  - Transition best model to Staging")

    success = run_command(
        [sys.executable, "-m", "src.train"],
        "Model Training"
    )
    all_success = all_success and success

    # Step 3: Run Tests
    print("\n" + "="*60)
    print("STEP 3: RUN TESTS")
    print("="*60)
    print("Running unit tests for:")
    print("  - Data cleaning (test_cleaning.py)")
    print("  - Training pipeline (test_train.py)")
    print("  - MLflow utilities (test_mlflow_utils.py)")

    success = run_command(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        "Unit Tests"
    )
    all_success = all_success and success

    # Summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    if all_success:
        print("[OK] All Stage 2 steps completed successfully!")
        print("\nNext steps:")
        print("  - Check MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db")
        print("  - Review model in MLflow Model Registry")
        print("  - Proceed to Stage 3: Monitoring & Deployment")
    else:
        print("[FAIL] Some steps failed. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()