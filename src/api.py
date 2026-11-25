import logging
import time
import uuid
from fastapi import FastAPI, Request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("solar_api")

app = FastAPI(title="Solar MLOps Stage 3 API")


@app.get("/health")
def health():
    logger.info("Health check called")
    return {"status": "ok"}


@app.post("/predict")
async def predict(payload: dict, request: Request):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()

    rows = payload.get("records", [payload] if isinstance(payload, dict) else [])
    predictions = [0.0 for _ in rows]

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(f"req_id={req_id} | rows={len(rows)} | latency_ms={elapsed_ms}")
    return {
        "request_id": req_id,
        "count": len(predictions),
        "predictions": predictions,
        "latency_ms": elapsed_ms,
    }
