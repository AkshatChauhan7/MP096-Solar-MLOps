"""Tests for the FastAPI solar prediction API."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with lifespan support."""
    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Test that /health returns HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_correct_response(self, client):
        """Test that /health returns {'status': 'ok'}."""
        response = client.get("/health")
        assert response.json() == {"status": "ok"}


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_valid_input_returns_prediction(self, client):
        """Test /predict with valid input returns a prediction."""
        valid_input = {
            "panel_area_m2": 10.5,
            "sunshine_hours": 8.0,
            "temp_c": 25.0,
            "dust_days": 5,
            "panel_age_yr": 5,
            "cloud_cover_pct": 20.0,
            "roof_type": "flat",
            "area": "Old City",
        }
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], (int, float))

    def test_predict_panel_age_gt_30_returns_422(self, client):
        """Test /predict with panel_age_yr > 30 returns 422."""
        invalid_input = {
            "panel_area_m2": 10.5,
            "sunshine_hours": 8.0,
            "temp_c": 25.0,
            "dust_days": 5,
            "panel_age_yr": 35,  # > 30
            "cloud_cover_pct": 20.0,
            "roof_type": "flat",
            "area": "Old City",
        }
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422

    def test_predict_dust_days_negative_returns_422(self, client):
        """Test /predict with dust_days < 0 returns 422."""
        invalid_input = {
            "panel_area_m2": 10.5,
            "sunshine_hours": 8.0,
            "temp_c": 25.0,
            "dust_days": -1,  # < 0
            "panel_age_yr": 5,
            "cloud_cover_pct": 20.0,
            "roof_type": "flat",
            "area": "Old City",
        }
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422

    def test_predict_invalid_roof_type_returns_422(self, client):
        """Test /predict with invalid roof_type returns 422."""
        invalid_input = {
            "panel_area_m2": 10.5,
            "sunshine_hours": 8.0,
            "temp_c": 25.0,
            "dust_days": 5,
            "panel_age_yr": 5,
            "cloud_cover_pct": 20.0,
            "roof_type": "invalid_type",
            "area": "Old City",
        }
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422

    def test_predict_invalid_area_returns_422(self, client):
        """Test /predict with invalid area returns 422."""
        invalid_input = {
            "panel_area_m2": 10.5,
            "sunshine_hours": 8.0,
            "temp_c": 25.0,
            "dust_days": 5,
            "panel_age_yr": 5,
            "cloud_cover_pct": 20.0,
            "roof_type": "flat",
            "area": "Invalid Area",
        }
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422

    def test_predict_missing_field_returns_422(self, client):
        """Test /predict with missing required field returns 422."""
        invalid_input = {
            "panel_area_m2": 10.5,
            "sunshine_hours": 8.0,
            "temp_c": 25.0,
            "dust_days": 5,
            "panel_age_yr": 5,
            "cloud_cover_pct": 20.0,
            "roof_type": "flat",
            # missing "area"
        }
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])