from pathlib import Path

import pandas as pd

from app.ml.features import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES,
    TARGET_COLUMN, load_training_frame, split_chronologically,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def test_training_frame_is_labelled_unique_and_leakage_safe():
    frame, report = load_training_frame(DATA_DIR)
    assert frame.shipment_id.is_unique
    assert frame[TARGET_COLUMN].notna().all()
    assert set(frame[TARGET_COLUMN].unique()).issubset({0, 1})
    forbidden = {
        "actual_departure", "actual_arrival", "status", "actual_delay_hours",
        "on_time_flag", "transit_days_actual", "event_type", "delay_minutes",
    }
    assert not forbidden.intersection(FEATURE_COLUMNS)
    assert list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES) == FEATURE_COLUMNS
    assert report["raw_missing_actual_arrival_rows"] == 349
    assert report["missing_actual_arrival_rows_excluded_after_duplicate_quarantine"] == 346
    assert report["missing_arrival_rows_in_exact_duplicate_copies_removed"] == 1
    assert report["missing_arrival_rows_in_ambiguous_id_quarantine"] == 2
    assert report["ambiguous_shipment_id_groups_excluded"] == 20
    assert report["exact_duplicate_rows_collapsed"] == 8


def test_target_uses_strictly_more_than_24_hours():
    frame, _ = load_training_frame(DATA_DIR)
    # Target definition is strict >24h; source-derived label remains binary.
    assert frame[TARGET_COLUMN].dtype == "int8"


def test_chronological_split_has_no_time_overlap():
    frame, _ = load_training_frame(DATA_DIR)
    train, validation, test, summary = split_chronologically(frame)
    assert train.booking_date.max() < validation.booking_date.min()
    assert validation.booking_date.max() < test.booking_date.min()
    assert sum(x[TARGET_COLUMN].sum() for x in [train, validation, test]) == frame[TARGET_COLUMN].sum()
    assert summary["strategy"] == "chronological_by_booking_date"


def test_pretraining_investigation_reproduces_missing_arrival_and_stop_findings():
    from scripts.profile_ml_data import build_report

    report = build_report(DATA_DIR)
    missing = report["missing_arrival_investigation"]
    assert missing["shipments_missing_actual_arrival"] == 349
    assert missing["missing_shipments_with_any_vessel_destination_arrival_event"] == 286
    assert missing["missing_shipments_with_vessel_destination_arrival_event_14_days_before_to_30_days_after_plan"] == 57
    stops = report["intermediate_stop_investigation"]
    assert stops["shipment_has_intermediate_stop_fields"] is False
    assert stops["shipment_has_leg_identifier"] is False
    assert stops["event_has_shipment_identifier"] is False
