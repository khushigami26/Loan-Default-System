from app import app


def test_health_endpoint():
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("status") == "healthy"
    assert "mongodb" in data


def test_health_full_endpoint():
    client = app.test_client()
    resp = client.get("/health-full")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("status") == "online"
    assert "mongodb" in data
    assert "model_loaded" in data


def test_home_page_redirects_or_renders():
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
