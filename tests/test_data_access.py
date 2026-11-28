"""Restore original bytes safely using the repository's DVC pointers."""

import hashlib

import pytest
import yaml

from src.data_access import restore_assignment_data


@pytest.fixture
def data_paths(tmp_path):
    assignment = tmp_path / "assignment"
    source = assignment / "data"
    source.mkdir(parents=True)
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("train.csv", "later.csv", "holdout_features.csv"):
        content = b"example\r\n1\r\n"
        (source / name).write_bytes(content)
        if name != "holdout_features.csv":
            (raw / (name + ".dvc")).write_text(
                yaml.safe_dump({"outs": [{"md5": hashlib.md5(content).hexdigest(), "path": name}]})
            )
    return assignment, raw


def test_restore_preserves_bytes_and_is_repeatable(data_paths):
    assignment, raw = data_paths
    first = restore_assignment_data(assignment, raw)
    assert restore_assignment_data(assignment, raw) == first
    for name, digest in first.items():
        assert (raw / name).read_bytes() == (assignment / "data" / name).read_bytes()
        assert hashlib.sha256((raw / name).read_bytes()).hexdigest() == digest


def test_bad_source_hash_prevents_all_copies(data_paths):
    assignment, raw = data_paths
    (assignment / "data" / "later.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="DVC fingerprint"):
        restore_assignment_data(assignment, raw)
    assert not (raw / "train.csv").exists()


def test_different_existing_file_is_never_overwritten(data_paths):
    assignment, raw = data_paths
    (raw / "holdout_features.csv").write_bytes(b"different")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        restore_assignment_data(assignment, raw)
    assert (raw / "holdout_features.csv").read_bytes() == b"different"
    assert not (raw / "train.csv").exists()


def test_holdout_git_newline_conversion_is_allowed(data_paths):
    assignment, raw = data_paths
    (raw / "holdout_features.csv").write_bytes(b"example\n1\n")
    restore_assignment_data(assignment, raw)
    assert (raw / "holdout_features.csv").read_bytes() == b"example\r\n1\r\n"
