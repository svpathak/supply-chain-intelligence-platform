from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import data_quality


def test_data_quality_endpoint_serves_existing_report(tmp_path, monkeypatch):
    report_path = tmp_path / "data_quality_report.json"
    expected = {
        "report_version": "1.0",
        "summary": {"checks_executed": 1},
        "findings": [],
    }
    report_path.write_text(json.dumps(expected), encoding="utf-8")
    monkeypatch.setattr(data_quality, "REPORT_PATH", report_path)

    app = FastAPI()
    app.include_router(data_quality.router)
    response = TestClient(app).get("/data-quality/report")

    assert response.status_code == 200
    assert response.json() == expected


def test_data_quality_endpoint_generates_missing_report(tmp_path, monkeypatch):
    expected = {
        "report_version": "1.0",
        "summary": {"checks_executed": 1},
        "findings": [],
    }
    report_path = tmp_path / "nested" / "data_quality_report.json"
    monkeypatch.setattr(data_quality, "REPORT_PATH", report_path)
    monkeypatch.setattr(data_quality, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(data_quality, "run_quality_checks", lambda _path: expected)

    app = FastAPI()
    app.include_router(data_quality.router)
    response = TestClient(app).get("/data-quality/report")

    assert response.status_code == 200
    assert response.json() == expected
    assert json.loads(report_path.read_text(encoding="utf-8")) == expected
