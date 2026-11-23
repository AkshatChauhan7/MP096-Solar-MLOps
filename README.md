# SCSE3040 MLOps Mini Project

- **Group ID:** G199
- **Problem ID:** MP096

## Project Overview

This project implements a complete MLOps pipeline for solar panel power output prediction (MP096). The pipeline covers:

1. **Data Engineering** - Cleaning, feature engineering, and DVC versioning
2. **Model Training** - Experiment tracking with MLflow, model selection
3. **Model Serving** - FastAPI service with Docker and Kubernetes deployment
4. **Monitoring & Retraining** - PSI drift detection, automated retraining, champion/challenger promotion

## Quick Start

### Prerequisites
- Python 3.12+
- DVC
- Docker
- Kubernetes cluster (or minikube/kind)

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Pull DVC data
dvc pull
```

### Training
```bash
# Run training pipeline (Stage 2)
python -m src.solar_ml.train
```

### Serving (Local)
```bash
# Start API server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
# Build image
docker build -t mp096-api:1.0 .

# Run container
docker run -p 8000:8000 mp096-api:1.0
```

### Kubernetes
```bash
# Apply manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## API Endpoints

- `POST /predict` - Predict solar panel output power
- `GET /health` - Health check

## Project Structure (Stage 2)

```
MP096-Solar-MLOps/
├── data/                # Data files (tracked by DVC)
│   ├── raw/
│   │   ├── train.csv.dvc
│   │   ├── later.csv.dvc
│   │   └── holdout_features.csv
│   └── processed/       # Cleaned data (local only)
├── models/              # Trained model artifacts
│   └── model.joblib
├── reports/             # Metrics, PSI, findings
│   ├── findings.yaml
│   └── metrics.json
├── src/
│   └── solar_ml/        # Core ML pipeline code
│       ├── __init__.py
│       ├── data.py      # Data loading, auditing, cleaning
│       └── train.py     # MLflow training pipeline
├── tests/               # Pytest tests
│   ├── test_data.py
│   └── test_train.py
├── mlflow.db            # MLflow tracking database
├── params.yaml          # Configuration parameters
├── requirements.txt     # Pinned dependencies
└── README.md
```

## Monitoring & Retraining

The pipeline implements:
- **Population Stability Index (PSI)** with 10 bins for drift detection
- **Per-area MAE tracking** (Old City, New Colony, Industrial)
- **Champion/Challenger promotion** requiring ≥12% MAE improvement
- **MLflow model alias** management for production model

