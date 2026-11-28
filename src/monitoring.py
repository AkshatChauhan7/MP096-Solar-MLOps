"""Phase 1: reference-based PSI and registered version 1 area MAE.

PSI = sum((actual_fraction - reference_fraction) * log(actual/reference)).
Ten equal-width bins are fitted ONLY to reference values; the outer edges
extend to infinity so later observations are never discarded. Zero fractions
are replaced by epsilon (not added to nonzero fractions). Missing values use
a separate bucket when present. A constant reference uses below/equal/above
buckets, since ten distinct equal-width bins cannot be fitted to a point.
All-missing references, empty inputs and infinite observations are rejected.
"""

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

from .cleaning import clean_data, load_data
from .mlflow_utils import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME
from .train import NUMERICAL_FEATURES, prepare_data

AREAS = ("Old City", "New Colony", "Industrial")


def population_stability_index(reference, actual, bins: int = 10, epsilon: float = 0.0001) -> float:
    """Return PSI using shared reference bins and explicit missingness."""
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    if not np.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must be finite and between 0 and 1")
    arrays = [
        pd.Series(values, dtype="Float64").to_numpy(dtype=float, na_value=np.nan)
        for values in (reference, actual)
    ]
    if any(len(values) == 0 for values in arrays):
        raise ValueError("PSI requires nonempty samples")
    if any(np.isinf(values).any() for values in arrays):
        raise ValueError("PSI observations must be finite or missing")
    observed = arrays[0][~np.isnan(arrays[0])]
    if not len(observed):
        raise ValueError("PSI reference must contain an observed value")
    low, high = observed.min(), observed.max()
    if low == high:
        edges = np.array([-np.inf, low, np.nextafter(low, np.inf), np.inf])
    else:
        edges = np.linspace(low, high, bins + 1)
        edges[0], edges[-1] = -np.inf, np.inf
    missing_bucket = any(np.isnan(values).any() for values in arrays)
    fractions = []
    for values in arrays:
        missing = np.isnan(values)
        counts = np.histogram(values[~missing], bins=edges)[0]
        if missing_bucket:
            counts = np.append(counts, missing.sum())
        fractions.append(np.where(counts == 0, epsilon, counts / len(values)))
    expected, observed = fractions
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def psi_by_feature(
    train: pd.DataFrame, later: pd.DataFrame, bins: int = 10, epsilon: float = 0.0001
) -> dict[str, float]:
    """Monitor the six numeric prediction features, excluding target/leakage."""
    return {
        feature: population_stability_index(train[feature], later[feature], bins, epsilon)
        for feature in NUMERICAL_FEATURES
    }


def group_mae(target, predictions, groups) -> dict[str, float]:
    """Compute positional group MAE, rejecting absent groups and invalid values."""
    y, pred = np.asarray(target, dtype=float), np.asarray(predictions, dtype=float)
    area = np.asarray(groups)
    if any(values.ndim != 1 for values in (y, pred, area)):
        raise ValueError("Targets, predictions and groups must be one-dimensional")
    if not len(y) or not (len(y) == len(pred) == len(area)):
        raise ValueError("Targets, predictions and groups need equal nonzero lengths")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Targets and predictions must be finite")
    if pd.isna(area).any() or set(area) != set(AREAS):
        raise ValueError(f"Expected all and only these areas: {AREAS}")
    errors = np.abs(y - pred)
    return {name: float(errors[area == name].mean()) for name in AREAS}


def load_v1_model(config: dict | None = None):
    """Load immutable registry version 1, regardless of the champion alias.

    Failure propagates: never substitute a freshly fitted model or champion.
    """
    config = config or {}
    tracking_uri = config.get("tracking_uri", MLFLOW_TRACKING_URI)
    name = config.get("name", REGISTERED_MODEL_NAME)
    mlflow.set_tracking_uri(tracking_uri)
    artifact_dir = os.environ.get("STAGE4_V1_ARTIFACT_DIR") or config.get("v1_artifact_dir")
    if artifact_dir:
        # A relocated MLflow artifact must identify the original v1 run. A bare
        # pickle cannot prove provenance and may omit its preprocessing pipeline.
        manifest = yaml.safe_load((Path(artifact_dir) / "MLmodel").read_text(encoding="utf-8"))
        version = MlflowClient(tracking_uri=tracking_uri).get_model_version(name, "1")
        if not manifest.get("run_id") or manifest["run_id"] != version.run_id:
            raise ValueError("Relocated artifact does not belong to registered v1's run")
        return mlflow.sklearn.load_model(str(artifact_dir))
    return mlflow.sklearn.load_model(f"models:/{name}/1")


def update_findings(path: Path, values: dict) -> None:
    """Update computed monitoring fields while preserving comments/other stages."""
    content = path.read_text(encoding="utf-8")
    document = yaml.safe_load(content)
    monitoring = document["monitoring"]
    for key, value in values.items():
        if key not in monitoring:
            raise ValueError(f"Unknown monitoring field: {key}")
        monitoring[key] = value
    # Only replace the monitoring block. Preserve the assignment's comments by
    # changing individual scalar values rather than reserializing the document.
    prefix, block = content.split("monitoring:\n", 1)
    section = None
    lines = block.splitlines(keepends=True)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key = stripped.split(":", 1)[0].strip("'\"")
        indent = len(line) - len(line.lstrip())
        if indent == 2:
            section = key
            replacement = values.get(key)
            if isinstance(replacement, dict) or key not in values:
                continue
        elif indent == 4 and section in values and isinstance(values[section], dict):
            replacement = values[section][key]
        else:
            continue
        before, after = line.split(":", 1)
        comment = "  #" + after.split("#", 1)[1].rstrip() if "#" in after else ""
        scalar = yaml.safe_dump(replacement, default_flow_style=True).splitlines()[0]
        lines[index] = f"{before}: {scalar}{comment}\n"
    updated = prefix + "monitoring:\n" + "".join(lines)
    if yaml.safe_load(updated) != document:
        raise ValueError("Findings update could not preserve the document structure")
    path.write_text(updated, encoding="utf-8")


def run_monitoring(config: dict, *, psi_only: bool = False) -> dict:
    """Run on original CSVs; only train goes through the Stage 2 cleaner.

    later.csv is already clean: never repair its target using training rules.
    Full execution updates findings only after the original v1 loads/predicts.
    Explicit psi-only execution can record drift when v1 artifacts are absent.
    """
    train = clean_data(load_data(config["train_path"]))
    later = load_data(config["later_path"])
    psi = psi_by_feature(train, later, config["bins"], config["epsilon"])
    results = {"psi": psi, "most_drifted_feature": max(psi, key=psi.get)}
    if not psi_only:
        features, target = prepare_data(later)
        predictions = load_v1_model(config.get("model")).predict(features)
        maes = group_mae(target, predictions, later["area"])
        results.update(mae_by_group=maes, worst_served_group=max(maes, key=maes.get))
        output = Path(config["predictions_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"record_id": later["record_id"], "prediction": predictions}).to_csv(
            output, index=False
        )
    update_findings(Path(config["findings_path"]), results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument(
        "--psi-only",
        action="store_true",
        help="Explicit partial run; leave all MAE fields untouched",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.params.read_text(encoding="utf-8"))["monitoring"]
    print(yaml.safe_dump(run_monitoring(config, psi_only=args.psi_only), sort_keys=False))


if __name__ == "__main__":
    main()
