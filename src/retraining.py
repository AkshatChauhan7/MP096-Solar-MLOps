"""Stage 4 chronological retraining, comparison, registry and final predictions.

Run --retrain-only while the original v1 artifact is unavailable. Candidate
training never substitutes a new model for v1 or changes a registry alias.
The default run requires original v1 before publishing any final results.
"""

import argparse
import hashlib
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

from .cleaning import clean_data, load_data
from .monitoring import group_mae, load_v1_model, psi_by_feature, update_findings
from .train import FEATURE_COLUMNS, create_model_pipeline, prepare_data


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_id_digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame["record_id"].to_json(orient="values").encode()).hexdigest()


def chronological_split(later: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stable date order, first floor(70% * n) for fitting; remaining rows check.

    Stable sorting preserves original row order for ties. This follows the
    assignment's row split, even when the boundary falls within a single date.
    """
    dates = pd.to_datetime(later["date"], format="%Y-%m-%d", errors="raise")
    if dates.isna().any():
        raise ValueError("All later rows need valid dates")
    split = int(len(later) * 0.7)
    if split == 0 or split == len(later):
        raise ValueError("Need nonempty chronological training and check sets")
    order = np.argsort(dates.to_numpy(), kind="stable")
    ordered = later.iloc[order]
    return ordered.iloc[:split].copy(), ordered.iloc[split:].copy()


def promotion_decision(mae_v1: float, mae_v2: float) -> str:
    """Promote only for at least 12% lower check MAE; zero baseline keeps v1."""
    if not all(np.isfinite(value) and value >= 0 for value in (mae_v1, mae_v2)):
        raise ValueError("MAEs must be finite and nonnegative")
    return "promote" if mae_v1 > 0 and mae_v2 <= 0.88 * mae_v1 else "keep"


def predict_checked(model, features: pd.DataFrame) -> np.ndarray:
    values = np.asarray(model.predict(features), dtype=float)
    if values.shape != (len(features),) or not np.isfinite(values).all():
        raise ValueError("Model must produce one finite prediction per row")
    return values


def validate_records(frame: pd.DataFrame, label: str) -> None:
    if frame.empty or frame["record_id"].isna().any() or frame["record_id"].duplicated().any():
        raise ValueError(f"{label} requires nonempty, unique, nonmissing record_id values")


def load_datasets(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = clean_data(load_data(config["train_path"]))
    later = load_data(config["later_path"])
    for name, frame in (("train", train), ("later", later)):
        validate_records(frame, name)
        features, target = prepare_data(frame)
        if features.isna().any().any() or not np.isfinite(target.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} contains missing features or invalid targets")
    return train, later


def fit_candidate(train: pd.DataFrame, later: pd.DataFrame, model_config: dict):
    """Reuse Stage 2 preprocessing/model factory, fitting on eligible rows only."""
    early, check = chronological_split(later)
    combined = pd.concat([train, early], ignore_index=True)
    features, target = prepare_data(combined)
    model = create_model_pipeline(model_config["type"], **model_config["params"])
    model.fit(features, target)
    check_features, check_target = prepare_data(check)
    mae = float(mean_absolute_error(check_target, predict_checked(model, check_features)))
    return model, early, check, mae


def stage4_experiment(client: MlflowClient, settings: dict) -> str:
    name = settings["experiment_name"]
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    root = Path(settings["artifact_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return client.create_experiment(name, artifact_location=root.as_uri())


def log_pipeline(model, features, metrics: dict, params: dict, settings: dict, role: str) -> dict:
    """Track a model without automatic registration or touching old versions."""
    mlflow.set_tracking_uri(settings["tracking_uri"])
    mlflow.set_registry_uri(settings["tracking_uri"])
    client = MlflowClient(tracking_uri=settings["tracking_uri"])
    experiment = stage4_experiment(client, settings)
    with mlflow.start_run(experiment_id=experiment, run_name=f"stage4_{role}") as run:
        mlflow.log_params(params)
        mlflow.set_tags({"stage": "4", "stage4_role": role})
        mlflow.log_metrics(metrics)
        info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            serialization_format="cloudpickle",
            signature=infer_signature(features, predict_checked(model, features)),
            input_example=features.iloc[:3],
        )
        return {"run_id": run.info.run_id, "model_uri": info.model_uri}


def train_candidate(config: dict) -> dict:
    """Train/log actual v2 even without v1; never register/promote it here."""
    settings = config["retraining"]
    train, later = load_datasets(config)
    model, early, check, mae = fit_candidate(train, later, settings["model"])
    fingerprints = {
        "data_sha256": sha256_file(config["train_path"]),
        "later_sha256": sha256_file(config["later_path"]),
    }
    params = {
        **fingerprints,
        "train_rows": len(train),
        "later_fit_rows": len(early),
        "later_check_rows": len(check),
        "model_type": settings["model"]["type"],
        **settings["model"]["params"],
        "split": "date_sorted_first_70_percent",
    }
    tracked = log_pipeline(
        model, early[FEATURE_COLUMNS], {"mae_v2_on_check": mae}, params, settings, "v2_candidate"
    )
    candidate_path = Path(settings["candidate_path"])
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, candidate_path)
    report = {
        "status": "candidate_trained_comparison_pending",
        **tracked,
        **fingerprints,
        "model_config": settings["model"],
        "candidate_path": str(candidate_path),
        "candidate_sha256": sha256_file(candidate_path),
        "train_rows": len(train),
        "later_fit_rows": len(early),
        "later_check_rows": len(check),
        "fit_record_ids_sha256": row_id_digest(early),
        "check_record_ids_sha256": row_id_digest(check),
        "mae_v2_on_check": mae,
    }
    report_path = Path(settings["candidate_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    update_findings(Path(config["findings_path"]), {"mae_v2_on_check": mae})
    return report


def load_candidate(config: dict, later: pd.DataFrame):
    """Reject stale or modified candidates before comparing/publishing."""
    settings = config["retraining"]
    report = yaml.safe_load(Path(settings["candidate_report"]).read_text(encoding="utf-8"))
    for key, path in (
        ("data_sha256", config["train_path"]),
        ("later_sha256", config["later_path"]),
        ("candidate_sha256", settings["candidate_path"]),
    ):
        if report[key] != sha256_file(path):
            raise ValueError(f"Candidate fingerprint mismatch: {key}; rerun --retrain-only")
    early, check = chronological_split(later)
    if (
        report["model_config"] != settings["model"]
        or report["fit_record_ids_sha256"] != row_id_digest(early)
        or report["check_record_ids_sha256"] != row_id_digest(check)
    ):
        raise ValueError("Candidate configuration/split changed; rerun --retrain-only")
    mlflow.set_tracking_uri(settings["tracking_uri"])
    mlflow.set_registry_uri(settings["tracking_uri"])
    client = MlflowClient(tracking_uri=settings["tracking_uri"])
    run = client.get_run(report["run_id"])
    if run.data.params.get("data_sha256") != report["data_sha256"]:
        raise ValueError("Candidate run does not match the original training data")
    # Load the logged model used by registry publication, not just the local
    # backup, so evaluation and the eventual champion use the same artifact.
    model = mlflow.sklearn.load_model(report["model_uri"])
    return model, report, check


def register_pair(v1, report: dict, features: pd.DataFrame, maes: dict, config: dict) -> tuple:
    """Use a separate registry name, preserving all existing source versions.

    The assignment requires retrained v2 to be version 2. Existing source v2
    predates Stage 4, so a new namespace holds a provenance-tagged v1 copy and
    the retrained v2. Reruns accept only the exact same candidate run.
    """
    settings = config["retraining"]
    source = config["model"]["name"]
    name = settings["registered_model_name"]
    if name == source:
        raise ValueError("Stage 4 registry name must differ from the existing source model")
    client = MlflowClient(tracking_uri=settings["tracking_uri"])
    versions = {
        int(version.version): version for version in client.search_model_versions(f"name='{name}'")
    }
    source_client = MlflowClient(tracking_uri=config["model"]["tracking_uri"])
    original = source_client.get_model_version(source, "1")
    if versions:
        if set(versions) != {1, 2} or versions[2].run_id != report["run_id"]:
            raise ValueError(
                "Stage 4 registry already contains a different/incomplete pair; use a new name"
            )
        baseline_run = client.get_run(versions[1].run_id)
        if baseline_run.data.params.get("original_v1_run_id") != original.run_id:
            raise ValueError("Stage 4 baseline does not match the original v1")
        return name, versions[1].run_id
    baseline = log_pipeline(
        v1,
        features,
        {"mae_v1_on_check": maes["mae_v1_on_check"]},
        {"data_sha256": report["data_sha256"], "original_v1_run_id": original.run_id},
        settings,
        "v1_import",
    )
    first = mlflow.register_model(baseline["model_uri"], name)
    second = mlflow.register_model(report["model_uri"], name)
    if int(first.version) != 1 or int(second.version) != 2:
        raise RuntimeError("Unexpected registry versions; champion was not changed")
    return name, baseline["run_id"]


def write_holdout_predictions(model, holdout: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Preserve all original holdout rows/order and the exact required schema."""
    validate_records(holdout, "holdout")
    features = holdout[FEATURE_COLUMNS]
    result = pd.DataFrame(
        {
            "record_id": holdout["record_id"].to_numpy(),
            "prediction": predict_checked(model, features),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def update_tracking(path: Path, values: dict) -> None:
    """Change only requested tracking scalars, preserving the assignment form."""
    content = path.read_text(encoding="utf-8")
    document = yaml.safe_load(content)
    if not set(values).issubset(document["tracking"]):
        raise ValueError("Unknown tracking field")
    document["tracking"].update(values)
    prefix, rest = content.split("tracking:\n", 1)
    block, suffix = rest.split("\nmonitoring:\n", 1)
    lines = block.splitlines()
    for index, line in enumerate(lines):
        key = line.strip().split(":", 1)[0]
        if key in values:
            scalar = yaml.safe_dump(values[key], default_flow_style=True).splitlines()[0]
            lines[index] = f"  {key}: {scalar}"
    updated = prefix + "tracking:\n" + "\n".join(lines) + "\nmonitoring:\n" + suffix
    if yaml.safe_load(updated) != document:
        raise ValueError("Tracking update could not preserve document structure")
    path.write_text(updated, encoding="utf-8")


def finalize(config: dict) -> dict:
    """Compare actual models, apply rule, publish champion and predict holdout."""
    v1 = load_v1_model(config["model"])  # Fail before any publication if unavailable.
    train, later = load_datasets(config)
    v2, report, check = load_candidate(config, later)
    features, target = prepare_data(check)
    maes = {
        "mae_v1_on_check": float(mean_absolute_error(target, predict_checked(v1, features))),
        "mae_v2_on_check": float(mean_absolute_error(target, predict_checked(v2, features))),
    }
    decision = promotion_decision(maes["mae_v1_on_check"], maes["mae_v2_on_check"])
    later_predictions = predict_checked(v1, later[FEATURE_COLUMNS])
    groups = group_mae(later["energy_kwh"], later_predictions, later["area"])
    psi = psi_by_feature(train, later, config["bins"], config["epsilon"])
    holdout = load_data(config["holdout_path"])
    validate_records(holdout, "holdout")
    chosen = v2 if decision == "promote" else v1
    predict_checked(chosen, holdout[FEATURE_COLUMNS])  # Validate before changing aliases.
    name, baseline_run = register_pair(v1, report, features, maes, config)
    settings = config["retraining"]
    client = MlflowClient(tracking_uri=settings["tracking_uri"])
    champion_version = 2 if decision == "promote" else 1
    client.set_registered_model_alias(name, "champion", str(champion_version))
    champion = mlflow.sklearn.load_model(f"models:/{name}@champion")
    write_holdout_predictions(champion, holdout, Path(settings["holdout_predictions_path"]))
    champion_path = Path(settings["champion_path"])
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, champion_path)
    output = Path(config["predictions_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"record_id": later["record_id"], "prediction": later_predictions}).to_csv(
        output, index=False
    )
    results = {
        "psi": psi,
        "most_drifted_feature": max(psi, key=psi.get),
        "mae_by_group": groups,
        "worst_served_group": max(groups, key=groups.get),
        **maes,
        "decision": decision,
    }
    update_findings(Path(config["findings_path"]), results)
    update_tracking(
        Path(config["findings_path"]),
        {
            "experiment_name": settings["experiment_name"],
            "best_run_id": baseline_run,
            "registered_model_name": name,
            "champion_version": champion_version,
        },
    )
    report_path = Path(settings["final_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        yaml.safe_dump(
            {
                "status": "complete",
                **results,
                "registered_model_name": name,
                "champion_version": champion_version,
                "candidate_run_id": report["run_id"],
                "holdout_rows": len(holdout),
                "improvement_pct": (
                    100
                    * (maes["mae_v1_on_check"] - maes["mae_v2_on_check"])
                    / maes["mae_v1_on_check"]
                    if maes["mae_v1_on_check"] > 0
                    else None
                ),
                "champion_sha256": sha256_file(champion_path),
                "runtime": {
                    "python": platform.python_version(),
                    **{
                        package: version(package)
                        for package in (
                            "mlflow",
                            "scikit-learn",
                            "numpy",
                            "pandas",
                            "scipy",
                            "joblib",
                        )
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report["status"] = "comparison_complete"
    Path(settings["candidate_report"]).write_text(
        yaml.safe_dump(report, sort_keys=False), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--retrain-only", action="store_true")
    mode.add_argument("--finalize", action="store_true", help="Use existing verified candidate")
    args = parser.parse_args()
    config = yaml.safe_load(args.params.read_text(encoding="utf-8"))["monitoring"]
    if args.retrain_only:
        result = train_candidate(config)
    else:
        if not args.finalize:
            load_v1_model(config["model"])
            train_candidate(config)
        result = finalize(config)
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
