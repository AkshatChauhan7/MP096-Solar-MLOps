"""Portable artifact relocation never rewrites models or unrelated URIs."""

import sqlite3

import pytest

from src.artifact_paths import portable_location, relocate_artifacts


@pytest.mark.parametrize("prefix", ["file:C:/old/project/", "file:///D:/other/", "", "C:\\old\\"])
def test_local_directory_becomes_portable(tmp_path, prefix):
    (tmp_path / "mlruns" / "1").mkdir(parents=True)
    assert portable_location(prefix + "mlruns/1", tmp_path) == (
        None if prefix == "" else "mlruns/1"
    )


@pytest.mark.parametrize(
    "value",
    [
        "https://host/mlruns/1",
        "s3://bucket/mlruns/1",
        "models:/model/1",
        "runs:/abc/model",
        "mlruns/../../outside",
        "file:C:/old/mlruns/missing",
    ],
)
def test_unrelated_missing_or_escaping_locations_are_untouched(tmp_path, value):
    assert portable_location(value, tmp_path) is None


def test_registry_history_and_model_bytes_are_preserved(tmp_path):
    artifacts = tmp_path / "mlruns" / "1" / "artifacts"
    artifacts.mkdir(parents=True)
    model = artifacts / "model.pkl"
    model.write_bytes(b"original-model-bytes")
    database = tmp_path / "mlflow.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE model_versions (name TEXT, version INTEGER, run_id TEXT, storage_location TEXT)"
        )
        connection.execute(
            "INSERT INTO model_versions VALUES (?,?,?,?)",
            ("original", 1, "original-run", "file:C:/old/mlruns/1/artifacts"),
        )
        connection.execute(
            "CREATE TABLE registered_model_aliases (name TEXT, alias TEXT, version INTEGER)"
        )
        connection.execute("INSERT INTO registered_model_aliases VALUES ('original','champion',1)")
    changes = relocate_artifacts(database, tmp_path)
    assert len(changes) == 1
    assert relocate_artifacts(database, tmp_path) == []
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT * FROM model_versions").fetchone() == (
            "original",
            1,
            "original-run",
            "mlruns/1/artifacts",
        )
        assert connection.execute("SELECT * FROM registered_model_aliases").fetchone() == (
            "original",
            "champion",
            1,
        )
    assert model.read_bytes() == b"original-model-bytes"


def test_traversal_cannot_relocate_into_an_unrelated_repository_directory(tmp_path):
    (tmp_path / "secret").mkdir()
    assert portable_location("file:C:/old/mlruns/../secret", tmp_path) is None
