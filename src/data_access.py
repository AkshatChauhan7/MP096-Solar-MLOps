"""Restore assignment data without changing the existing DVC fingerprints."""

import argparse
import hashlib
import shutil
from pathlib import Path

import yaml


def restore_assignment_data(assignment_dir: Path, raw_dir: Path) -> dict[str, str]:
    """Verify all sources before copying; never overwrite different local data.

    Train/later must match their committed DVC MD5s. Holdout is already tracked
    by this repository; Git's LF/CRLF conversion is allowed for that file only.
    """
    sources = {}
    for name in ("train.csv", "later.csv", "holdout_features.csv"):
        source = assignment_dir / "data" / name
        content = source.read_bytes()
        destination = raw_dir / name
        if name != "holdout_features.csv":
            pointer = yaml.safe_load(destination.with_suffix(".csv.dvc").read_text())
            expected = pointer["outs"][0]["md5"]
            if hashlib.md5(content).hexdigest() != expected:
                raise ValueError(f"{source} does not match the committed DVC fingerprint")
        if destination.exists() and destination.read_bytes() != content:
            newline_only = name == "holdout_features.csv" and destination.read_bytes().replace(
                b"\r\n", b"\n"
            ) == content.replace(b"\r\n", b"\n")
            if not newline_only:
                raise ValueError(f"Refusing to overwrite different data: {destination}")
        sources[name] = source
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        if not (raw_dir / name).exists() or (raw_dir / name).read_bytes() != source.read_bytes():
            shutil.copyfile(source, raw_dir / name)
    return {
        name: hashlib.sha256(source.read_bytes()).hexdigest() for name, source in sources.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assignment_dir", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    print(yaml.safe_dump(restore_assignment_data(args.assignment_dir, args.raw_dir)))


if __name__ == "__main__":
    main()