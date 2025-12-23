# ============================================================
# FILE: tests/unit/test_api.py
# ============================================================
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_endpoint(self):
        """Health endpoint should return healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_ready_endpoint(self):
        """Ready endpoint should return ready status."""
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    def test_predict_valid_input(self):
        """Predict endpoint should accept valid ECG input."""
        ecg_signal = [[0.1] * 5000 for _ in range(12)]  # 12 leads, 5000 samples

        response = client.post(
            "/api/v1/predict",
            json={
                "ecg_signal": ecg_signal,
                "patient_age": 65,
                "patient_sex": "M",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0
        assert "ecg_id" in data
        assert "model_version" in data

    def test_predict_invalid_leads(self):
        """Predict endpoint should reject ECG with wrong number of leads."""
        ecg_signal = [[0.1] * 5000 for _ in range(10)]  # Only 10 leads

        response = client.post(
            "/api/v1/predict",
            json={"ecg_signal": ecg_signal},
        )

        assert response.status_code == 422  # Validation error

    def test_predict_without_demographics(self):
        """Predict endpoint should work without demographics."""
        ecg_signal = [[0.1] * 5000 for _ in range(12)]

        response = client.post(
            "/api/v1/predict",
            json={"ecg_signal": ecg_signal},
        )

        assert response.status_code == 200
