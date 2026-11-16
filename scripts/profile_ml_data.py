"""Reproducible source investigation performed before ML training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build_report(data_dir: Path) -> dict:
    shipments = pd.read_csv(data_dir / "shipments.csv")
    ports = pd.read_csv(data_dir / "ports.csv")
    events = pd.read_csv(data_dir / "port_events.csv")
    for column in ["booking_date", "planned_departure", "planned_arrival", "actual_arrival"]:
        shipments[column] = pd.to_datetime(shipments[column], errors="coerce")
    events["event_timestamp"] = pd.to_datetime(events["event_timestamp"], errors="coerce")

    missing = shipments.loc[shipments.actual_arrival.isna()].copy()
    arrivals = events.loc[events.event_type.eq("ARRIVAL"), ["vessel_id", "port_code", "event_timestamp"]]
    event_matches = missing[["shipment_id", "vessel_id", "destination_port", "planned_arrival"]].merge(
        arrivals,
        left_on=["vessel_id", "destination_port"],
        right_on=["vessel_id", "port_code"],
        how="left",
    )
    any_match = event_matches.groupby("shipment_id").event_timestamp.apply(lambda values: values.notna().any())
    event_matches["delta_days_from_plan"] = (
        event_matches.event_timestamp - event_matches.planned_arrival
    ).dt.total_seconds() / 86400
    in_window = event_matches.loc[event_matches.delta_days_from_plan.between(-14, 30)]
    window_match = in_window.groupby("shipment_id").event_timestamp.apply(lambda values: values.notna().any())

    # Dataset structure is assessed independently of event-to-shipment matches.
    vessel_port_coverage = events.groupby("vessel_id").port_code.nunique()
    event_matches_per_missing_shipment = event_matches.groupby("shipment_id").event_timestamp.count()

    # The target/status association is diagnostic only. Status is never used as
    # a model feature; the association is evidence of leakage risk.
    labelled = shipments.loc[shipments.actual_arrival.notna()].copy()
    delay_hours = (labelled.actual_arrival - labelled.planned_arrival).dt.total_seconds() / 3600
    labelled["target_late_over_24h"] = (delay_hours > 24).astype(int)
    status_target = pd.crosstab(labelled.status.fillna("<NULL>"), labelled.target_late_over_24h)
    status_target = {
        str(status): {str(int(k)): int(v) for k, v in row.items()}
        for status, row in status_target.iterrows()
    }

    return {
        "source_rows": {
            "shipments": int(len(shipments)),
            "ports": int(len(ports)),
            "port_events": int(len(events)),
        },
        "missing_arrival_investigation": {
            "shipments_missing_actual_arrival": int(len(missing)),
            "missing_shipments_with_any_vessel_destination_arrival_event": int(any_match.sum()),
            "missing_shipments_with_vessel_destination_arrival_event_14_days_before_to_30_days_after_plan": int(window_match.sum()),
            "matching_event_count_per_shipment_median": float(event_matches_per_missing_shipment.median()),
            "matching_event_count_per_shipment_max": int(event_matches_per_missing_shipment.max()),
            "reason_not_imputed": "port_events has no shipment_id or leg identifier; vessel + destination matches are ambiguous and cannot establish which shipment arrived at a given timestamp",
        },
        "port_event_grain_investigation": {
            "distinct_event_vessels": int(events.vessel_id.nunique()),
            "distinct_event_ports": int(events.port_code.nunique()),
            "vessel_count_with_events_at_all_reference_ports": int((vessel_port_coverage == ports.port_code.nunique()).sum()),
            "total_vessels_with_events": int(len(vessel_port_coverage)),
            "interpretation": "events are port-level and vessel-linked, not shipment- or leg-linked; all event vessels appear at all reference ports, so vessel/port coincidence is not a reliable shipment join",
        },
        "intermediate_stop_investigation": {
            "shipment_columns": shipments.columns.tolist(),
            "shipment_has_intermediate_stop_fields": False,
            "shipment_has_leg_identifier": False,
            "event_has_shipment_identifier": "shipment_id" in events.columns,
            "event_has_leg_identifier": any("leg" in c.lower() for c in events.columns),
            "assumption": "predict destination arrival delay only; intermediate stop schedules and outcomes are not represented at shipment grain",
        },
        "status_target_diagnostic": {
            "cross_tab_status_by_target": status_target,
            "modeling_decision": "exclude status: observed source values reveal the target almost perfectly and are not legitimate booking-time predictors",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/ml_data_investigation.json")
    args = parser.parse_args()
    report = build_report(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
