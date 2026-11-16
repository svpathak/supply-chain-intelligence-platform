"""Leakage-safe booking-time feature construction for shipment delay prediction."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TARGET_COLUMN = "is_late_over_24h"

# This allow-list is intentional: outcome and post-booking columns cannot enter
# the model matrix by accident when the source schema grows.
NUMERIC_FEATURES = [
    "container_count",
    "weight_tons",
    "planned_transit_days",
    "booking_lead_days",
    "booking_month",
    "booking_day_of_week",
    "planned_departure_month",
    "planned_departure_day_of_week",
    "origin_congestion_score",
    "destination_congestion_score",
]
CATEGORICAL_FEATURES = [
    "origin_port",
    "destination_port",
    "cargo_type",
    "origin_region",
    "destination_region",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_training_frame(data_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load sources and prepare one unambiguous labelled shipment per row.

    Exclusions are conservative: exact duplicates are collapsed; all remaining
    repeated shipment IDs are quarantined; missing outcomes and unknown port
    references are not imputed. Missing *predictor* values are handled later by
    a train-fitted sklearn preprocessing pipeline.
    """
    data_dir = Path(data_dir)
    shipments = pd.read_csv(data_dir / "shipments.csv")
    ports = pd.read_csv(data_dir / "ports.csv")

    original_rows = len(shipments)
    raw_missing_arrival_rows = int(pd.to_datetime(shipments["actual_arrival"], errors="coerce").isna().sum())
    exact_duplicate_mask = shipments.duplicated()
    exact_duplicate_rows = int(exact_duplicate_mask.sum())
    missing_arrival_in_exact_duplicate_copies = int((exact_duplicate_mask & shipments["actual_arrival"].isna()).sum())
    shipments = shipments.drop_duplicates().copy()

    # Parse timestamps explicitly; invalid/missing timestamps remain NaT.
    timestamp_columns = [
        "booking_date", "planned_departure", "planned_arrival", "actual_arrival"
    ]
    for column in timestamp_columns:
        shipments[column] = pd.to_datetime(shipments[column], errors="coerce")

    # After exact duplicates are collapsed, repeated IDs still represent
    # conflicting records. Exclude every row in those groups rather than
    # arbitrarily selecting a version.
    duplicate_id_mask = shipments["shipment_id"].duplicated(keep=False)
    ambiguous_id_rows = int(duplicate_id_mask.sum())
    missing_arrival_in_ambiguous_id_rows = int((duplicate_id_mask & shipments["actual_arrival"].isna()).sum())
    ambiguous_id_groups = int(shipments.loc[duplicate_id_mask, "shipment_id"].nunique())
    shipments = shipments.loc[~duplicate_id_mask].copy()

    missing_arrival_rows = int(shipments["actual_arrival"].isna().sum())
    shipments = shipments.loc[shipments["actual_arrival"].notna()].copy()

    required_schedule = ["booking_date", "planned_departure", "planned_arrival"]
    missing_schedule_rows = int(shipments[required_schedule].isna().any(axis=1).sum())
    shipments = shipments.dropna(subset=required_schedule).copy()

    known_ports = set(ports["port_code"].dropna().astype(str))
    valid_port_mask = (
        shipments["origin_port"].astype("string").isin(known_ports)
        & shipments["destination_port"].astype("string").isin(known_ports)
    )
    invalid_port_rows = int((~valid_port_mask).sum())
    shipments = shipments.loc[valid_port_mask].copy()

    # Target is defined only after eligibility filters. Do not use status:
    # source profiling showed it reveals the outcome almost perfectly.
    delay_hours = (
        shipments["actual_arrival"] - shipments["planned_arrival"]
    ).dt.total_seconds() / 3600.0
    shipments[TARGET_COLUMN] = (delay_hours > 24.0).astype("int8")

    port_features = ports[[
        "port_code", "region", "avg_congestion_score"
    ]].copy()
    port_features["port_code"] = port_features["port_code"].astype(str)
    port_features = port_features.drop_duplicates("port_code")

    shipments = shipments.merge(
        port_features.rename(columns={
            "port_code": "origin_port",
            "region": "origin_region",
            "avg_congestion_score": "origin_congestion_score",
        }), on="origin_port", how="left", validate="many_to_one"
    )
    shipments = shipments.merge(
        port_features.rename(columns={
            "port_code": "destination_port",
            "region": "destination_region",
            "avg_congestion_score": "destination_congestion_score",
        }), on="destination_port", how="left", validate="many_to_one"
    )

    # Every feature below is computable from booking-time schedule/booking
    # metadata or static port reference data.
    shipments["planned_transit_days"] = (
        shipments["planned_arrival"] - shipments["planned_departure"]
    ).dt.total_seconds() / 86400.0
    shipments["booking_lead_days"] = (
        shipments["planned_departure"] - shipments["booking_date"]
    ).dt.total_seconds() / 86400.0
    shipments["booking_month"] = shipments["booking_date"].dt.month
    shipments["booking_day_of_week"] = shipments["booking_date"].dt.dayofweek
    shipments["planned_departure_month"] = shipments["planned_departure"].dt.month
    shipments["planned_departure_day_of_week"] = shipments["planned_departure"].dt.dayofweek

    # Treat domain-invalid numeric predictors as missing rather than teaching
    # the model that negative cargo weight or zero containers are meaningful.
    negative_weight_rows = int((pd.to_numeric(shipments["weight_tons"], errors="coerce") < 0).sum())
    non_positive_container_rows = int((pd.to_numeric(shipments["container_count"], errors="coerce") <= 0).sum())
    shipments.loc[pd.to_numeric(shipments["weight_tons"], errors="coerce") < 0, "weight_tons"] = np.nan
    shipments.loc[pd.to_numeric(shipments["container_count"], errors="coerce") <= 0, "container_count"] = np.nan

    # Canonicalise only superficial category formatting. Missing cargo remains
    # missing for explicit UNKNOWN imputation in the fitted pipeline.
    shipments["cargo_type"] = shipments["cargo_type"].astype("string").str.strip().str.upper()
    shipments.loc[shipments["cargo_type"].eq(""), "cargo_type"] = pd.NA
    for col in ["origin_port", "destination_port", "origin_region", "destination_region"]:
        shipments[col] = shipments[col].astype("string").str.strip().str.upper()

    # Retain only the feature allow-list, target, and metadata needed for split.
    result = shipments[["shipment_id", "booking_date", TARGET_COLUMN, *FEATURE_COLUMNS]].copy()
    result = result.replace([np.inf, -np.inf], np.nan)

    report = {
        "source_shipment_rows": original_rows,
        "exact_duplicate_rows_collapsed": exact_duplicate_rows,
        "ambiguous_shipment_id_groups_excluded": ambiguous_id_groups,
        "ambiguous_shipment_id_rows_excluded": ambiguous_id_rows,
        "raw_missing_actual_arrival_rows": raw_missing_arrival_rows,
        "missing_actual_arrival_rows_excluded_after_duplicate_quarantine": missing_arrival_rows,
        "missing_arrival_rows_in_exact_duplicate_copies_removed": missing_arrival_in_exact_duplicate_copies,
        "missing_arrival_rows_in_ambiguous_id_quarantine": missing_arrival_in_ambiguous_id_rows,
        "missing_booking_or_planned_schedule_rows_excluded": missing_schedule_rows,
        "invalid_origin_or_destination_port_rows_excluded": invalid_port_rows,
        "negative_weight_predictors_set_missing": negative_weight_rows,
        "non_positive_container_count_predictors_set_missing": non_positive_container_rows,
        "eligible_labelled_rows": int(len(result)),
        "positive_rows": int(result[TARGET_COLUMN].sum()),
        "negative_rows": int((result[TARGET_COLUMN] == 0).sum()),
        "positive_rate": float(result[TARGET_COLUMN].mean()) if len(result) else None,
        "booking_date_min": result["booking_date"].min().isoformat() if len(result) else None,
        "booking_date_max": result["booking_date"].max().isoformat() if len(result) else None,
    }
    return result, report


def split_chronologically(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Split by booking date, keeping all equal timestamps in one partition."""
    ordered = frame.sort_values(["booking_date", "shipment_id"]).reset_index(drop=True)
    dates = ordered["booking_date"].sort_values().drop_duplicates().to_numpy()
    if len(dates) < 3:
        raise ValueError("At least three distinct booking dates are required")
    train_cut = dates[max(0, int(len(dates) * train_fraction) - 1)]
    validation_cut = dates[max(0, int(len(dates) * (train_fraction + validation_fraction)) - 1)]
    train = ordered.loc[ordered["booking_date"] <= train_cut].copy()
    validation = ordered.loc[
        (ordered["booking_date"] > train_cut) & (ordered["booking_date"] <= validation_cut)
    ].copy()
    test = ordered.loc[ordered["booking_date"] > validation_cut].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological split produced an empty partition")
    summary = {
        "strategy": "chronological_by_booking_date",
        "fractions_requested": {"train": train_fraction, "validation": validation_fraction, "test": 1 - train_fraction - validation_fraction},
        "partitions": {},
    }
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        summary["partitions"][name] = {
            "rows": int(len(part)),
            "positive_rows": int(part[TARGET_COLUMN].sum()),
            "positive_rate": float(part[TARGET_COLUMN].mean()),
            "booking_date_min": part.booking_date.min().isoformat(),
            "booking_date_max": part.booking_date.max().isoformat(),
        }
    return train, validation, test, summary
