# SCSE3040 MLOps Mini Project

- Group ID: G199
- Problem ID: MP096

## Group Members

- S24CSEU1804
- S24CSEU1814
- S24CSEU1786
- S24CSEU1803

## Stage 4 monitoring

Run commands from the repository root using the installed project environment:

For reproducible evaluation, create an isolated environment and install
`requirements-stage4.txt`. It matches the recovered artifact's scikit-learn,
NumPy and SciPy versions; pandas 3.1.1 replaces the yanked pandas 3.0.4 recorded
in the original manifest. The existing environment need not be changed.

```powershell
python -m venv .venv-stage4
.venv-stage4/Scripts/python.exe -m pip install -r requirements-stage4.txt
.venv-stage4/Scripts/Activate.ps1
```

```powershell
python -m src.data_access D:\G199_MP096
python -m src.monitoring --psi-only
python -m pytest
```

Data restoration verifies the existing DVC fingerprints without committing raw
train/later CSVs. `params.yaml` controls monitoring paths and model configuration.
Only training data goes through the existing Stage 2 cleaner; later data is clean.
PSI uses ten equal-width training bins, open outer tails and epsilon 0.0001 for
zero frequencies. The actual P13 lab was not included in the assignment folder.

The recovered artifact directory is configured with a repository-relative path
in `params.yaml`. Restore the shared `mlruns` directory at the repository root.
To relocate registry file paths and run original-v1 monitoring:

```powershell
python -m src.artifact_paths
python -m src.monitoring
```

The loader checks the artifact's run ID against registered version 1. Alternatively,
set `STAGE4_V1_ARTIFACT_DIR` to override `monitoring.model.v1_artifact_dir` in
`params.yaml`. Registry relocation updates only artifact locations for directories
present inside the checkout, retaining original versions, runs, aliases and model
bytes. Its reversible audit is `reports/artifact_relocation.yaml`. There is no
fallback to champion, a new fit, or dummy
predictions. See `reports/stage4_audit.yaml` for prerequisites and existing gaps.

## Stage 4 retraining and finalization

```powershell
# Train/log a real candidate without requiring v1 or changing registry aliases.
python -m src.retraining --retrain-only

# After configuring the original artifact directory, compare and finish.
python -m src.retraining --finalize

# Or retrain and finalize together, once v1 is available.
python -m src.retraining
```

The candidate uses the existing Ridge pipeline with alpha 1.0, fitted on cleaned
train plus the first 70% of later rows after stable date sorting. The final 30%
is reserved for evaluation. Candidate data, model and split fingerprints prevent
using stale results. Later targets are never passed through training typo fixes.
`reports/stage4_candidate.yaml` records the actual run and candidate check MAE.
The candidate artifact is local/ignored; on a fresh checkout, rerun training.

Existing `solar-power-predictor` versions 1 and 2 and its champion alias are
preserved. Finalization creates a separate `solar-power-predictor-stage4`
registry entry: version 1 is a provenance-linked copy of the recovered original
v1 pipeline, and version 2 is the retrained candidate. Only a check MAE at least
12% lower promotes v2; a zero-MAE baseline is kept. Rerunning finalization with
the same candidate reuses these versions. A different candidate requires a new
Stage 4 registry name, so existing versions cannot be silently replaced.

Finalization loads the selected `@champion`, exports `models/model.joblib`, and
writes `predictions/holdout_predictions.csv` with exactly `record_id,prediction`
in original holdout order. It updates the monitoring and tracking fields in the
assignment form. No final output or promotion result is generated while v1 is
missing. Stage 3 has not been changed to serve this exported champion yet.

The normal test suite uses synthetic fixtures, including real isolated MLflow
registration and both promotion outcomes. Actual-data integration is opt-in and
uses a temporary copy of the registry and temporary output files:

```powershell
$env:STAGE4_V1_ARTIFACT_DIR = 'D:\path\to\original-v1\artifacts'
$env:RUN_STAGE4_INTEGRATION = '1'
python -m pytest tests/test_stage4_integration.py -v
```

On Windows, if DVC cannot create its system cache, point that cache inside the
ignored repository cache before using DVC:

```powershell
$env:DVC_SITE_CACHE_DIR = Join-Path $PWD '.dvc/cache/site'
$env:ITERATIVE_DO_NOT_TRACK = '1'
python -m dvc status
```
