from pathlib import Path

import pandas as pd

from app.ml.port_event_features import (
    EVENT_FEATURE_COLUMNS,
    add_point_in_time_port_event_features,
)


def test_port_event_features_exclude_same_time_and_future_events(tmp_path: Path):
    events = pd.DataFrame([
        {"event_id": "E1", "event_timestamp": "2024-01-01 12:00:00", "port_code": "AAA", "vessel_id": "V1", "event_type": "DELAYED", "delay_minutes": 60, "notes": "before window end"},
        {"event_id": "E2", "event_timestamp": "2024-01-08 11:59:00", "port_code": "AAA", "vessel_id": "V2", "event_type": "ARRIVAL", "delay_minutes": 0, "notes": "before booking"},
        {"event_id": "E3", "event_timestamp": "2024-01-08 12:00:00", "port_code": "AAA", "vessel_id": "V3", "event_type": "DELAYED", "delay_minutes": 300, "notes": "same timestamp"},
        {"event_id": "E4", "event_timestamp": "2024-01-08 12:01:00", "port_code": "AAA", "vessel_id": "V4", "event_type": "DELAYED", "delay_minutes": 300, "notes": "future"},
    ])
    events.to_csv(tmp_path / "port_events.csv", index=False)
    frame = pd.DataFrame([{
        "shipment_id": "S1",
        "booking_date": pd.Timestamp("2024-01-08 12:00:00"),
        "origin_port": "AAA",
        "destination_port": "BBB",
    }])

    enriched, audit = add_point_in_time_port_event_features(frame, tmp_path)

    assert enriched.loc[0, "origin_events_7d_count"] == 2
    assert enriched.loc[0, "origin_events_7d_delayed_count"] == 1
    assert enriched.loc[0, "origin_events_7d_mean_delay_minutes"] == 30
    assert enriched.loc[0, "destination_events_7d_count"] == 0
    assert audit["strict_booking_cutoff"].startswith("event_timestamp < booking_date")
    assert set(EVENT_FEATURE_COLUMNS).issubset(enriched.columns)


def test_conflicting_duplicate_event_ids_are_quarantined(tmp_path: Path):
    events = pd.DataFrame([
        {"event_id": "DUP", "event_timestamp": "2024-01-01 10:00:00", "port_code": "AAA", "vessel_id": "V1", "event_type": "DELAYED", "delay_minutes": 60, "notes": "one"},
        {"event_id": "DUP", "event_timestamp": "2024-01-02 10:00:00", "port_code": "AAA", "vessel_id": "V2", "event_type": "ARRIVAL", "delay_minutes": 0, "notes": "two"},
        {"event_id": "KEEP", "event_timestamp": "2024-01-03 10:00:00", "port_code": "AAA", "vessel_id": "V3", "event_type": "ARRIVAL", "delay_minutes": 5, "notes": "keep"},
    ])
    events.to_csv(tmp_path / "port_events.csv", index=False)
    frame = pd.DataFrame([{
        "shipment_id": "S1", "booking_date": pd.Timestamp("2024-01-04 10:00:00"),
        "origin_port": "AAA", "destination_port": "BBB",
    }])

    enriched, audit = add_point_in_time_port_event_features(frame, tmp_path)

    assert audit["duplicate_event_id_groups_quarantined"] == 1
    assert audit["duplicate_event_id_rows_quarantined"] == 2
    assert enriched.loc[0, "origin_events_7d_count"] == 1
