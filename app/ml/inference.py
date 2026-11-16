"""Startup-loaded, leakage-safe inference for the persisted delay model."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.ml.features import FEATURE_COLUMNS, NUMERIC_FEATURES, CATEGORICAL_FEATURES


class ModelArtifactError(RuntimeError):
    """The persisted model or metadata cannot be loaded or used."""


class DelayPredictor:
    def __init__(self, model: Any, metadata: dict[str, Any]) -> None:
        self.model = model
        self.metadata = metadata
        self.model_version = str(metadata.get("model_version", "unknown"))
        expected = metadata.get("feature_columns", [])
        if expected != FEATURE_COLUMNS:
            raise ModelArtifactError(
                "Model feature contract does not match the application feature allow-list."
            )

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "DelayPredictor":
        artifact_dir = Path(artifact_dir)
        model_path = artifact_dir / "delay_final.joblib"
        metadata_path = artifact_dir / "delay_model_metadata.joblib"
        if not model_path.is_file() or not metadata_path.is_file():
            raise ModelArtifactError(
                f"Required model artifacts are missing from {artifact_dir}. Run scripts/train_delay_model.py."
            )
        try:
            model = joblib.load(model_path)
            metadata = joblib.load(metadata_path)
            if not isinstance(metadata, dict):
                raise TypeError("Model metadata must be a dictionary")
            return cls(model, metadata)
        except Exception as exc:
            raise ModelArtifactError("Could not load the saved delay model artifacts.") from exc

    @staticmethod
    def _as_datetime(value: Any) -> datetime | None:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if isinstance(value, datetime):
            return value
        parsed = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(parsed) else parsed.to_pydatetime()

    def build_feature_row(self, source: dict[str, Any]) -> pd.DataFrame:
        """Build exactly the booking-time allow-list; never read outcome fields."""
        booking_date = self._as_datetime(source.get("booking_date"))
        planned_departure = self._as_datetime(source.get("planned_departure"))
        planned_arrival = self._as_datetime(source.get("planned_arrival"))

        def day_delta(end: datetime | None, start: datetime | None) -> float:
            if end is None or start is None:
                return np.nan
            return (end - start).total_seconds() / 86400.0

        row: dict[str, Any] = {
            "container_count": source.get("container_count"),
            "weight_tons": source.get("weight_tons"),
            "planned_transit_days": day_delta(planned_arrival, planned_departure),
            "booking_lead_days": day_delta(planned_departure, booking_date),
            "booking_month": booking_date.month if booking_date else np.nan,
            "booking_day_of_week": booking_date.weekday() if booking_date else np.nan,
            "planned_departure_month": planned_departure.month if planned_departure else np.nan,
            "planned_departure_day_of_week": planned_departure.weekday() if planned_departure else np.nan,
            "origin_congestion_score": source.get("origin_congestion_score"),
            "destination_congestion_score": source.get("destination_congestion_score"),
            "origin_port": source.get("origin_port"),
            "destination_port": source.get("destination_port"),
            "cargo_type": source.get("cargo_type"),
            "origin_region": source.get("origin_region"),
            "destination_region": source.get("destination_region"),
        }
        frame = pd.DataFrame([{name: row.get(name) for name in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)
        for column in NUMERIC_FEATURES:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if pd.notna(frame.loc[0, "weight_tons"]) and frame.loc[0, "weight_tons"] < 0:
            frame.loc[0, "weight_tons"] = np.nan
        if pd.notna(frame.loc[0, "container_count"]) and frame.loc[0, "container_count"] <= 0:
            frame.loc[0, "container_count"] = np.nan
        for column in CATEGORICAL_FEATURES:
            value = frame.loc[0, column]
            frame.loc[0, column] = str(value).strip().upper() if pd.notna(value) else "UNKNOWN"
            if frame.loc[0, column] == "":
                frame.loc[0, column] = "UNKNOWN"
        return frame

    def predict(self, source: dict[str, Any]) -> dict[str, Any]:
        features = self.build_feature_row(source)
        try:
            probability = float(self.model.predict_proba(features)[0, 1])
        except Exception as exc:
            raise ModelArtifactError("The saved model could not score the booking-time features.") from exc
        # These are display bands only. The model is not calibrated or approved
        # for operational decisions; see README ML limitations.
        risk = "low" if probability < 0.33 else "medium" if probability < 0.66 else "high"
        return {
            "shipment_id": str(source["shipment_id"]),
            "delay_risk": risk,
            "probability": probability,
            "model_version": self.model_version,
            "model_status": "experimental_not_for_operational_decisions",
        }
