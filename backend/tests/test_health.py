from tests.conftest import CLASS_NAMES


def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert set(data["classes"]) == set(CLASS_NAMES)


def test_classes_endpoint(client):
    resp = client.get("/classes")
    assert resp.status_code == 200
    assert set(resp.json()) == set(CLASS_NAMES)
