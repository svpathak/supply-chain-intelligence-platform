"""Real HTTP integration tests against a temporary SQLite data product."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.core import config


@pytest.fixture()
def live_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "integration.sqlite3"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("""
            CREATE TABLE staging_ports_current (
                port_code TEXT, port_name TEXT, region TEXT,
                avg_congestion_score REAL, is_current INTEGER
            );
            INSERT INTO staging_ports_current VALUES
                ('SHA', 'Shanghai', 'ASIA', 0.6, 1),
                ('RTM', 'Rotterdam', 'EUROPE', 0.4, 1);
            CREATE TABLE integration_shipments (
                shipment_id TEXT, booking_date TEXT, planned_departure TEXT,
                planned_arrival TEXT, container_count INTEGER, weight_tons REAL,
                cargo_type TEXT, origin_port TEXT, destination_port TEXT,
                actual_departure TEXT, actual_arrival TEXT, status TEXT,
                actual_delay_hours REAL, on_time_flag INTEGER, route_key TEXT,
                transit_days_planned REAL, transit_days_actual REAL
            );
            INSERT INTO integration_shipments VALUES (
                'SHP-IT-001', '2025-01-01 10:00:00', '2025-01-03 10:00:00',
                '2025-01-13 10:00:00', 2, 12.5, 'GENERAL', 'SHA', 'RTM',
                '2025-01-03 11:00:00', '2025-01-14 12:00:00', 'DELIVERED',
                26.0, 0, 'SHA → RTM', 10.0, 11.08
            );
            CREATE VIEW serving_shipments AS SELECT * FROM integration_shipments;
            CREATE VIEW serving_route_stats AS
            SELECT origin_port, destination_port,
                   AVG(actual_delay_hours) AS average_delay_hours,
                   AVG(CASE WHEN on_time_flag IS NOT NULL THEN CAST(on_time_flag AS REAL) END) AS on_time_rate,
                   COUNT(*) AS shipment_count
            FROM serving_shipments GROUP BY origin_port, destination_port;
        """)
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(config.settings, "database_path", db_path)
    monkeypatch.setattr(config.settings, "model_artifact_dir", Path(__file__).resolve().parents[2] / "artifacts")
    app = create_app()
    with TestClient(app) as client:
        yield client


def test_real_shipment_lookup_returns_curated_record(live_api):
    response = live_api.get("/shipments/SHP-IT-001")
    assert response.status_code == 200
    payload = response.json()
    assert payload["shipment_id"] == "SHP-IT-001"
    assert payload["origin_port"] == "SHA" and payload["destination_port"] == "RTM"


def test_unknown_shipment_returns_structured_404(live_api):
    response = live_api.get("/shipments/DOES-NOT-EXIST")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "SHIPMENT_NOT_FOUND"


def test_data_quality_report_endpoint_returns_json(live_api):
    response = live_api.get("/data-quality/report")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict) and "findings" in payload and "summary" in payload


def test_predict_delay_uses_loaded_artifact_and_real_repository(live_api):
    response = live_api.post("/predict-delay", json={"shipment_id": "SHP-IT-001"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["shipment_id"] == "SHP-IT-001"
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["model_status"] == "experimental_not_for_operational_decisions"
