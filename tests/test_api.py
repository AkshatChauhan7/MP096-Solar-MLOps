from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    response = client.post("/predict", json={"records": [{"temp": 30.0}]})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert len(data["predictions"]) == 1
