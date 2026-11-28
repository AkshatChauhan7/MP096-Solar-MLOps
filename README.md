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

To run v1 monitoring after recovering the original MLflow artifact directory
(including `MLmodel` and its serialized pipeline):

```powershell
$env:STAGE4_V1_ARTIFACT_DIR = 'D:\path\to\original-v1\artifacts'
python -m src.monitoring
```

The loader checks the artifact's run ID against registered version 1. Alternatively,
set `monitoring.model.v1_artifact_dir` in `params.yaml`, or restore the registry's
original artifact location. There is no fallback to champion, a new fit, or dummy
predictions. See `reports/stage4_audit.yaml` for prerequisites and existing gaps.
