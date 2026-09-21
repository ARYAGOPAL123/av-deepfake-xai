import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.main import app


client = TestClient(app)


def test_status_is_explicit_demo_mode_without_checkpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] in {"DEMO MODE", "LIVE MODE"}
    assert "tau" in payload and "u_max" in payload and "device" in payload


def test_frontend_and_fixtures_are_available():
    assert client.get("/").status_code == 200
    fixtures = client.get("/api/fixtures").json()
    assert {item["name"] for item in fixtures} == {"real", "fake", "uncertain"}


def test_fixture_job_is_labelled_and_has_no_live_claim():
    response = client.post("/api/analyse?fixture_name=real")
    assert response.status_code == 200
    job = client.get(f"/api/jobs/{response.json()['job_id']}").json()
    assert job["stage"] == "Done"
    assert job["result"]["demo_fixture"] is True
    assert "not a model prediction" in job["result"]["demo_badge"]


def test_invalid_upload_is_rejected():
    response = client.post("/api/analyse", files={"file": ("notes.txt", b"not video", "text/plain")})
    assert response.status_code == 400


def test_metrics_empty_state_is_data_driven():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    assert "items" in response.json() and "figures" in response.json()


def test_review_requires_a_reviewer_and_note():
    response = client.post("/api/review/1", json={"status": "approved", "reviewer": "", "note": ""})
    assert response.status_code == 400


def test_audit_export_and_pdf_endpoints():
    assert client.get("/api/audit/export.csv").status_code == 200
    pdf = client.get("/api/report/fixture-real.pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF")
