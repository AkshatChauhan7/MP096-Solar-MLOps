"""FastAPI application for solar energy prediction."""

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from pydantic_core import PydanticCustomError


# JSON logging setup
class JSONFormatter(logging.Formatter):
    """Format log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "name", "pathname", "process", "processName",
                "relativeCreated", "thread", "threadName", "exc_info",
                "exc_text", "stack_info", "asctime"
            }:
                log_obj[key] = value
        return json.dumps(log_obj)


# Configure logger
logger = logging.getLogger("solar_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logger.handlers = [handler]


# Global model variable
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    model_path = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
    model = joblib.load(model_path)
    logger.info("Model loaded successfully", extra={"model_path": str(model_path)})
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Solar Energy Prediction API",
    description="Predict solar panel energy output",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    """Request model for /predict endpoint with validation."""
    panel_area_m2: float = Field(..., description="Panel area in square meters")
    sunshine_hours: float = Field(..., description="Hours of sunshine")
    temp_c: float = Field(..., description="Temperature in Celsius")
    dust_days: int = Field(..., ge=0, description="Days since last cleaning (>= 0)")
    panel_age_yr: int = Field(..., le=30, description="Panel age in years (<= 30)")
    cloud_cover_pct: float = Field(..., description="Cloud cover percentage")
    roof_type: Literal["flat", "sloped", "tin"] = Field(..., description="Roof type")
    area: Literal["Old City", "New Colony", "Industrial"] = Field(..., description="Area group")

    @field_validator("panel_age_yr")
    @classmethod
    def validate_panel_age(cls, v: int) -> int:
        if v > 30:
            raise PydanticCustomError(
                "value_error",
                "panel_age_yr must be <= 30",
                {"input": v}
            )
        return v

    @field_validator("dust_days")
    @classmethod
    def validate_dust_days(cls, v: int) -> int:
        if v < 0:
            raise PydanticCustomError(
                "value_error",
                "dust_days must be >= 0",
                {"input": v}
            )
        return v


class PredictionResponse(BaseModel):
    """Response model for /predict endpoint."""
    prediction: float


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict energy output for a solar panel.

    Accepts exactly 8 fields:
    - panel_area_m2: Panel area in square meters
    - sunshine_hours: Hours of sunshine
    - temp_c: Temperature in Celsius
    - dust_days: Days since last cleaning (>= 0)
    - panel_age_yr: Panel age in years (<= 30)
    - cloud_cover_pct: Cloud cover percentage
    - roof_type: One of ["flat", "sloped", "tin"]
    - area: One of ["Old City", "New Colony", "Industrial"]
    """
    # Convert request to DataFrame with correct column order
    input_df = pd.DataFrame([{
        "panel_area_m2": request.panel_area_m2,
        "sunshine_hours": request.sunshine_hours,
        "temp_c": request.temp_c,
        "dust_days": request.dust_days,
        "panel_age_yr": request.panel_age_yr,
        "cloud_cover_pct": request.cloud_cover_pct,
        "roof_type": request.roof_type,
        "area": request.area,
    }])

    # Make prediction
    try:
        prediction = float(model.predict(input_df)[0])
    except Exception as e:
        logger.error("Prediction failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Prediction failed")

    # Log prediction (no PII)
    logger.info(
        "Prediction made",
        extra={
            "panel_area_m2": request.panel_area_m2,
            "sunshine_hours": request.sunshine_hours,
            "temp_c": request.temp_c,
            "dust_days": request.dust_days,
            "panel_age_yr": request.panel_age_yr,
            "cloud_cover_pct": request.cloud_cover_pct,
            "roof_type": request.roof_type,
            "area": request.area,
            "prediction": prediction,
        }
    )

    return PredictionResponse(prediction=prediction)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)