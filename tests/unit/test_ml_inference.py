from pathlib import Path

import numpy as np

from app.ml.features import FEATURE_COLUMNS
from app.ml.inference import DelayPredictor


class FakeModel:
    def __init__(self):
        self.seen = None

    def predict_proba(self, frame):
        self.seen = frame.copy()
        return np.array([[0.7, 0.3]])


def test_predictor_builds_booking_time_features_only_and_scores():
    metadata = {"model_version": "test-v1", "feature_columns": FEATURE_COLUMNS}
    model = FakeModel()
    predictor = DelayPredictor(model, metadata)
    result = predictor.predict({
        "shipment_id": "SHP-TEST",
        "booking_date": "2025-01-01 00:00:00",
        "planned_departure": "2025-01-03 00:00:00",
        "planned_arrival": "2025-01-13 00:00:00",
        "container_count": 2,
        "weight_tons": 12.5,
        "cargo_type": " general ",
        "origin_port": "cnsha",
        "destination_port": "nlrtm",
        "origin_region": "asia",
        "destination_region": "europe",
        "origin_congestion_score": 0.4,
        "destination_congestion_score": 0.2,
        # Deliberately supplied to prove the predictor ignores outcome fields.
        "actual_arrival": "2025-01-20 00:00:00",
        "actual_delay_hours": 168,
        "status": "DELAYED",
    })
    assert result["shipment_id"] == "SHP-TEST"
    assert result["probability"] == 0.3
    assert result["delay_risk"] == "low"
    assert result["model_version"] == "test-v1"
    assert result["model_status"] == "experimental_not_for_operational_decisions"
    assert list(model.seen.columns) == FEATURE_COLUMNS
    assert model.seen.loc[0, "planned_transit_days"] == 10
    assert model.seen.loc[0, "booking_lead_days"] == 2
    assert model.seen.loc[0, "cargo_type"] == "GENERAL"
    assert "actual_arrival" not in model.seen.columns
    assert "status" not in model.seen.columns


def test_predictor_loads_committed_artifact():
    project_root = Path(__file__).resolve().parents[2]
    predictor = DelayPredictor.load(project_root / "artifacts")
    assert predictor.metadata["model_name"] == "logistic_regression"
    assert predictor.model_version
