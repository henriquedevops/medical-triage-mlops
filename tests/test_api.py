from fastapi.testclient import TestClient

from triage.api.main import app


def test_health_reports_loaded_model():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_backend"] == "sklearn"


def test_triage_returns_valid_urgency_and_scores():
    with TestClient(app) as client:
        response = client.post(
            "/triage",
            json={"text": "Patient presents with acute chest pain and diaphoresis."},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["urgency"] in {"normal", "atencao", "urgente"}
        assert set(body["scores"].keys()) == {"normal", "atencao", "urgente"}
        assert abs(sum(body["scores"].values()) - 1.0) < 1e-3


def test_triage_rejects_empty_text():
    with TestClient(app) as client:
        response = client.post("/triage", json={"text": ""})
        assert response.status_code == 422


def test_metrics_endpoint_exposes_prometheus_format():
    with TestClient(app) as client:
        client.post("/triage", json={"text": "Routine follow-up, no acute findings."})
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "triage_requests_total" in response.text
        assert "triage_predictions_total" in response.text
