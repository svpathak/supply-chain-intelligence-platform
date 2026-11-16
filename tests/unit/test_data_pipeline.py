from __future__ import annotations

import pandas as pd

from app.data.database import connect
from app.data.ingestion import run_ingestion
from app.data.transformations import build_integration_and_serving


def _write_inputs(folder, *, shipment_arrival="2025-01-04 00:00:00", conflicting_duplicate=True):
    folder.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([
        {
            "port_code": "AAA", "port_name": "Alpha", "country": "A",
            "region": "R", "timezone": "UTC+0", "avg_congestion_score": 0.2,
        },
        {
            "port_code": "BBB", "port_name": "Beta", "country": "B",
            "region": "R", "timezone": "UTC+0", "avg_congestion_score": 0.3,
        },
    ]).to_csv(folder / "ports.csv", index=False)

    shipments = pd.DataFrame([
        {
            "shipment_id": "SHP-2", "customer_id": "C2",
            "origin_port": "BBB", "destination_port": "AAA",
            "vessel_id": "V2", "booking_date": "2024-12-02 00:00:00",
            "planned_departure": "2025-01-02 00:00:00",
            "actual_departure": None, "planned_arrival": "2025-01-05 00:00:00",
            "actual_arrival": None, "container_count": 2,
            "cargo_type": "General", "weight_tons": 20.0, "status": "IN_TRANSIT",
        },
    ])
    shipment_one = {
        "shipment_id": "SHP-1", "customer_id": "C1",
        "origin_port": "AAA", "destination_port": "BBB",
        "vessel_id": "V1", "booking_date": "2024-12-01 00:00:00",
        "planned_departure": "2025-01-01 00:00:00",
        "actual_departure": "2025-01-01 00:00:00",
        "planned_arrival": "2025-01-03 00:00:00",
        "actual_arrival": shipment_arrival,
        "container_count": 1, "cargo_type": "General",
        "weight_tons": 10.0, "status": "DELIVERED",
    }
    shipment_three = {**shipment_one, "shipment_id": "SHP-3", "actual_arrival": "2025-01-04 00:00:00"}
    shipments = pd.concat([
        shipments,
        pd.DataFrame([shipment_three, shipment_one]),
    ], ignore_index=True)
    if conflicting_duplicate:
        conflicting = {**shipment_one, "actual_departure": "2025-01-01 12:00:00", "actual_arrival": "2025-01-03 00:00:00"}
        shipments = pd.concat([shipments, pd.DataFrame([conflicting])], ignore_index=True)
    shipments.to_csv(folder / "shipments.csv", index=False)

    # Same event_id, different event content: both are distinct feed events.
    events = pd.DataFrame([
        {
            "event_id": "EV-1", "event_timestamp": "2025-01-01 01:00:00",
            "port_code": "AAA", "vessel_id": "V1", "event_type": "ARRIVAL",
            "delay_minutes": 0, "notes": "first",
        },
        {
            "event_id": "EV-1", "event_timestamp": "2025-01-01 02:00:00",
            "port_code": "BBB", "vessel_id": "V2", "event_type": "DEPARTURE",
            "delay_minutes": 5, "notes": "second",
        },
        {
            "event_id": "EV-2", "event_timestamp": "2025-01-01 03:00:00",
            "port_code": "AAA", "vessel_id": "V1", "event_type": "BERTHED",
            "delay_minutes": 0, "notes": "repeat",
        },
        {
            "event_id": "EV-2", "event_timestamp": "2025-01-01 03:00:00",
            "port_code": "AAA", "vessel_id": "V1", "event_type": "BERTHED",
            "delay_minutes": 0, "notes": "repeat",
        },
    ])
    events.to_csv(folder / "port_events.csv", index=False)


def _run_full_pipeline(data_dir, db_path):
    summary = run_ingestion(data_dir, db_path)
    conn = connect(db_path)
    try:
        build_integration_and_serving(conn)
        return summary, conn
    except Exception:
        conn.close()
        raise


def test_pipeline_derives_required_fields_and_nulls(tmp_path):
    data_dir = tmp_path / "raw"
    db_path = tmp_path / "warehouse.sqlite3"
    _write_inputs(data_dir)

    summary = run_ingestion(data_dir, db_path)
    conn = connect(db_path)
    try:
        build_integration_and_serving(conn)
        row = conn.execute("""
            SELECT actual_delay_hours, on_time_flag, route_key,
                   transit_days_planned, transit_days_actual
            FROM integration_shipments
            WHERE shipment_id = 'SHP-3'
        """).fetchone()

        assert row == (24.0, True, "AAA → BBB", 2.0, 3.0)

        null_row = conn.execute("""
            SELECT actual_delay_hours, on_time_flag, transit_days_actual
            FROM integration_shipments
            WHERE shipment_id = 'SHP-2'
        """).fetchone()
        assert null_row == (None, None, None)
        assert summary["staging"]["shipments"]["conflicting_business_key_groups_quarantined"] == 1
        assert summary["staging"]["shipments"]["conflicting_business_key_rows_quarantined"] == 2
        assert conn.execute("SELECT COUNT(*) FROM staging_shipments_quarantine").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM raw_shipments").fetchone()[0] == 4
        assert conn.execute("SELECT COUNT(*) FROM integration_shipments").fetchone()[0] == 2
        # Two different payloads sharing EV-1 are retained; the exact EV-2
        # replay inside the same input batch is stored once.
        assert conn.execute("SELECT COUNT(*) FROM staging_port_events").fetchone()[0] == 3
    finally:
        conn.close()


def test_pipeline_is_idempotent_for_repeated_snapshot_and_events(tmp_path):
    data_dir = tmp_path / "raw"
    db_path = tmp_path / "warehouse.sqlite3"
    _write_inputs(data_dir)

    first = run_ingestion(data_dir, db_path)
    conn = connect(db_path)
    try:
        build_integration_and_serving(conn)
        first_counts = {
            "current_shipments": conn.execute(
                "SELECT COUNT(*) FROM staging_shipments_current WHERE is_current"
            ).fetchone()[0],
            "shipment_history": conn.execute(
                "SELECT COUNT(*) FROM staging_shipments_history"
            ).fetchone()[0],
            "events": conn.execute("SELECT COUNT(*) FROM staging_port_events").fetchone()[0],
            "integrated": conn.execute("SELECT COUNT(*) FROM integration_shipments").fetchone()[0],
        }
    finally:
        conn.close()

    second = run_ingestion(data_dir, db_path)
    conn = connect(db_path)
    try:
        build_integration_and_serving(conn)
        second_counts = {
            "current_shipments": conn.execute(
                "SELECT COUNT(*) FROM staging_shipments_current WHERE is_current"
            ).fetchone()[0],
            "shipment_history": conn.execute(
                "SELECT COUNT(*) FROM staging_shipments_history"
            ).fetchone()[0],
            "events": conn.execute("SELECT COUNT(*) FROM staging_port_events").fetchone()[0],
            "integrated": conn.execute("SELECT COUNT(*) FROM integration_shipments").fetchone()[0],
        }
    finally:
        conn.close()

    assert first_counts == second_counts
    assert second["staging"]["shipments"]["inserted"] == 0
    assert second["staging"]["shipments"]["updated"] == 0
    assert second["staging"]["shipments"]["unchanged"] == 2
    assert second["staging"]["port_events"]["appended"] == 0


def test_changed_business_record_moves_previous_version_to_history(tmp_path):
    data_dir = tmp_path / "raw"
    db_path = tmp_path / "warehouse.sqlite3"
    _write_inputs(data_dir, shipment_arrival="2025-01-04 00:00:00", conflicting_duplicate=False)
    run_ingestion(data_dir, db_path)

    _write_inputs(data_dir, shipment_arrival="2025-01-05 00:00:00", conflicting_duplicate=False)
    second = run_ingestion(data_dir, db_path)

    conn = connect(db_path)
    try:
        history_count = conn.execute(
            "SELECT COUNT(*) FROM staging_shipments_history WHERE shipment_id='SHP-1'"
        ).fetchone()[0]
        current_arrival = conn.execute("""
            SELECT actual_arrival FROM staging_shipments_current
            WHERE shipment_id='SHP-1' AND is_current=TRUE
        """).fetchone()[0]
    finally:
        conn.close()

    assert second["staging"]["shipments"]["updated"] == 1
    assert history_count == 1
    assert str(current_arrival).startswith("2025-01-05")


def test_new_conflict_quarantines_key_and_archives_previous_current_version(tmp_path):
    data_dir = tmp_path / "raw"
    db_path = tmp_path / "warehouse.sqlite3"
    _write_inputs(data_dir, conflicting_duplicate=False)
    run_ingestion(data_dir, db_path)

    _write_inputs(data_dir, conflicting_duplicate=True)
    run_ingestion(data_dir, db_path)

    conn = connect(db_path)
    try:
        current_count = conn.execute(
            "SELECT COUNT(*) FROM staging_shipments_current WHERE shipment_id='SHP-1' AND is_current=1"
        ).fetchone()[0]
        history_count = conn.execute(
            "SELECT COUNT(*) FROM staging_shipments_history WHERE shipment_id='SHP-1'"
        ).fetchone()[0]
        quarantine_count = conn.execute(
            "SELECT COUNT(*) FROM staging_shipments_quarantine WHERE shipment_id='SHP-1'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert current_count == 0
    assert history_count == 1
    assert quarantine_count == 2
