from __future__ import annotations

import pandas as pd

from app.quality import run_quality_checks


def _write_inputs(tmp_path):
    pd.DataFrame([
        {"port_code": "AAA", "port_name": "A", "country": "X", "region": "R",
         "timezone": "UTC+0", "avg_congestion_score": 0.5},
        {"port_code": "BBB", "port_name": "B", "country": "Y", "region": "R",
         "timezone": "UTC+1", "avg_congestion_score": 0.6},
    ]).to_csv(tmp_path / "ports.csv", index=False)
    pd.DataFrame([
        {"shipment_id": "S1", "customer_id": "C1", "origin_port": "AAA",
         "destination_port": "BBB", "vessel_id": "V1",
         "booking_date": "2024-01-01 00:00", "planned_departure": "2024-01-02 00:00",
         "actual_departure": "2024-01-03 00:00", "planned_arrival": "2024-01-04 00:00",
         "actual_arrival": "2024-01-05 00:00", "container_count": 1,
         "cargo_type": "General", "weight_tons": 10, "status": "DELIVERED"},
        {"shipment_id": "S2", "customer_id": "C2", "origin_port": "UNKNOWN",
         "destination_port": "BBB", "vessel_id": "V2",
         "booking_date": "2024-01-01 00:00", "planned_departure": "2024-01-02 00:00",
         "actual_departure": "2023-12-31 12:00", "planned_arrival": "2024-01-04 00:00",
         "actual_arrival": "2024-01-05 00:00", "container_count": 0,
         "cargo_type": " general ", "weight_tons": -2, "status": "delivered"},
    ]).to_csv(tmp_path / "shipments.csv", index=False)
    pd.DataFrame([
        {"event_id": "E1", "event_timestamp": "2024-01-02 00:00", "port_code": "AAA",
         "vessel_id": "V1", "event_type": "ARRIVAL", "delay_minutes": 0, "notes": ""},
        {"event_id": "E1", "event_timestamp": "2024-01-02 01:00", "port_code": "AAA",
         "vessel_id": "V1", "event_type": "DEPARTURE", "delay_minutes": 5, "notes": ""},
    ]).to_csv(tmp_path / "port_events.csv", index=False)


def test_standalone_checks_detect_cross_table_and_domain_issues(tmp_path):
    _write_inputs(tmp_path)
    report = run_quality_checks(tmp_path)
    by_name = {f["check_name"]: f for f in report["findings"]}

    assert by_name["shipments.origin_port_not_in_ports_reference"]["rows_affected"] == 1
    assert by_name["shipments.negative_weight"]["rows_affected"] == 1
    assert by_name["shipments.non_positive_container_count"]["rows_affected"] == 1
    assert by_name["shipments.actual_departure_before_booking"]["rows_affected"] == 1
    assert by_name["port_events.repeated_event_id"]["rows_affected"] == 2
    assert by_name["port_events.repeated_event_id"]["conflicting_key_groups"] == 1


def test_clean_fixture_reports_zero_for_key_integrity_checks(tmp_path):
    _write_inputs(tmp_path)
    report = run_quality_checks(tmp_path)
    by_name = {f["check_name"]: f for f in report["findings"]}

    assert by_name["ports.duplicate_port_code"]["rows_affected"] == 0
    assert by_name["port_events.port_code_not_in_ports_reference"]["rows_affected"] == 0
    assert report["summary"]["checks_executed"] == len(report["findings"])


def test_percentages_use_dataset_row_count(tmp_path):
    _write_inputs(tmp_path)
    report = run_quality_checks(tmp_path)
    by_name = {f["check_name"]: f for f in report["findings"]}

    assert by_name["shipments.negative_weight"]["percentage_of_total"] == 50.0
