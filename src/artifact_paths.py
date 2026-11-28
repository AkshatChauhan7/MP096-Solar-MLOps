"""Relocate existing local MLflow artifact references without changing history.

Only local mlruns/mlartifacts directories that exist inside this checkout are
eligible. SQLite has no MLflow API for updating all artifact URI fields, so a
single transaction changes these fields only and records old/new locations.
Run all project commands from the repository root for relative URI resolution.
"""

import argparse
import re
import sqlite3
from pathlib import Path
from urllib.parse import unquote

import yaml

ARTIFACT_COLUMNS = {
    "experiments": "artifact_location",
    "runs": "artifact_uri",
    "logged_models": "artifact_location",
    "model_versions": "storage_location",
}


def portable_location(location: str | None, root: Path) -> str | None:
    """Map a local artifact suffix to a checked, repository-relative directory."""
    if not location:
        return None
    normalized = unquote(location).replace("\\", "/")
    if "://" in normalized and not normalized.startswith("file:"):
        return None
    if normalized.startswith("models:") or normalized.startswith("runs:"):
        return None
    match = re.search(r"(?:^|/)(mlruns|mlartifacts)(?:/|$)", normalized)
    if match is None:
        return None
    relative = normalized[match.start(1) :]
    candidate = (root / relative).resolve()
    try:
        result = candidate.relative_to(root.resolve()).as_posix()
    except ValueError:
        return None
    if result.split("/")[0] not in {"mlruns", "mlartifacts"}:
        return None
    if not candidate.is_dir() or result == location:
        return None
    return result


def relocate_artifacts(database: Path, root: Path) -> list[dict]:
    """Transactionally update artifact locations; no model content/history edits."""
    if not database.is_file():
        raise FileNotFoundError(database)
    changes = []
    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for table, column in ARTIFACT_COLUMNS.items():
            if table not in tables:
                continue
            # All SQL identifiers come from the fixed allowlist above.
            for rowid, old in connection.execute(f"SELECT rowid, {column} FROM {table}").fetchall():
                new = portable_location(old, root)
                if new is not None:
                    connection.execute(f"UPDATE {table} SET {column}=? WHERE rowid=?", (new, rowid))
                    changes.append(
                        {
                            "table": table,
                            "rowid": rowid,
                            "column": column,
                            "original": old,
                            "portable": new,
                        }
                    )
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("mlflow.db"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, default=Path("reports/artifact_relocation.yaml"))
    args = parser.parse_args()
    changes = relocate_artifacts(args.database, args.root)
    previous = (
        yaml.safe_load(args.report.read_text(encoding="utf-8")) if args.report.exists() else []
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(yaml.safe_dump(previous + changes, sort_keys=False), encoding="utf-8")
    print(f"Updated {len(changes)} local artifact references; registry history preserved.")


if __name__ == "__main__":
    main()
