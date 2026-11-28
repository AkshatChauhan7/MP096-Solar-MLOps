import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_entry.update(record.extra_fields)
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())

logger = logging.getLogger("solar_api")
logger.setLevel(logging.INFO)
logger.handlers = [handler]
logger.propagate = False

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
EXPECTED_FEATURES = [
    "panel_area_m2",
    "sunshine_hours",
    "temp_c",
    "dust_days",
    "panel_age_yr",
    "cloud_cover_pct",
    "roof_type",
    "area",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


model = load_model()


class SolarRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    temp: Optional[float] = Field(default=None, ge=-50, le=80)
    temp_c: Optional[float] = Field(default=None, ge=-50, le=80)
    panel_area_m2: float = Field(default=20.0, gt=0)
    sunshine_hours: float = Field(default=6.0, ge=0, le=24)
    dust_days: float = Field(default=2.0, ge=0)
    panel_age_yr: float = Field(default=1.0, ge=0)
    cloud_cover_pct: float = Field(default=20.0, ge=0, le=100)
    roof_type: str = Field(default="flat", min_length=1)
    area: str = Field(default="urban", min_length=1)

    @model_validator(mode="after")
    def sync_temperature(self):
        if self.temp_c is None and self.temp is not None:
            self.temp_c = self.temp
        elif self.temp_c is None:
            self.temp_c = 25.0
        return self

    def to_model_dict(self) -> dict:
        return {
            "panel_area_m2": self.panel_area_m2,
            "sunshine_hours": self.sunshine_hours,
            "temp_c": self.temp_c,
            "dust_days": self.dust_days,
            "panel_age_yr": self.panel_age_yr,
            "cloud_cover_pct": self.cloud_cover_pct,
            "roof_type": self.roof_type,
            "area": self.area,
        }


class BatchPredictRequest(BaseModel):
    records: list[SolarRecord] = Field(..., min_length=1)


PredictPayload = Union[BatchPredictRequest, SolarRecord]

app = FastAPI(title="Solar MLOps API")


@app.get("/health")
def health():
    logger.info("Health check called", extra={"extra_fields": {"endpoint": "/health", "status": "ok"}})
    return {"status": "ok"}


@app.post("/predict")
async def predict(payload: PredictPayload, request: Request):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start = time.perf_counter()

    records = payload.records if isinstance(payload, BatchPredictRequest) else [payload]
    feature_order = (
        list(getattr(model, "feature_names_in_", EXPECTED_FEATURES))
        if hasattr(model, "feature_names_in_")
        else EXPECTED_FEATURES
    )

    try:
        df = pd.DataFrame([rec.to_model_dict() for rec in records])
        for col in feature_order:
            if col not in df.columns:
                raise ValueError(f"Missing required feature: {col}")
        df = df[feature_order]
        preds = model.predict(df)
        predictions = [float(val) for val in preds]
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.error(
            "Prediction failed",
            extra={
                "extra_fields": {
                    "request_id": req_id,
                    "rows": len(records),
                    "latency_ms": elapsed_ms,
                    "error": str(exc),
                }
            },
        )
        raise HTTPException(status_code=400, detail=f"Inference error: {exc}") from exc

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "Prediction completed",
        extra={
            "extra_fields": {
                "request_id": req_id,
                "rows": len(predictions),
                "latency_ms": elapsed_ms,
            }
        },
    )
    return {
        "request_id": req_id,
        "count": len(predictions),
        "predictions": predictions,
        "latency_ms": elapsed_ms,
    }
