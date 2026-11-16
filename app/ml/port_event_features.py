"""Point-in-time port-level event features for booking-time prediction.

Events are aggregated by port only. They are not attributed to individual
shipments because the source has no shipment or voyage-leg identifier.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS_DAYS = (7, 30)
EVENT_FEATURE_COLUMNS = [
    f"{side}_events_{days}d_{metric}"
    for side in ("origin", "destination")
    for days in WINDOWS_DAYS
    for metric in ("count", "mean_delay_minutes", "delayed_count", "delayed_rate")
] + ["event_observation_7d_coverage", "event_observation_30d_coverage"]


def _port_history_features(
    queries: pd.DataFrame,
    events: pd.DataFrame,
    side: str,
) -> pd.DataFrame:
    """Compute trailing-window features using events strictly before booking."""
    output = pd.DataFrame(index=queries.index)
    for days in WINDOWS_DAYS:
        output[f"{side}_events_{days}d_count"] = 0.0
        output[f"{side}_events_{days}d_mean_delay_minutes"] = np.nan
        output[f"{side}_events_{days}d_delayed_count"] = 0.0
        output[f"{side}_events_{days}d_delayed_rate"] = np.nan

    # Work port-by-port; searchsorted makes each booking-time lookup logarithmic.
    for port_code, query_rows in queries.groupby(f"{side}_port", sort=False):
        port_events = events.loc[events["port_code"].eq(port_code)].sort_values("event_timestamp")
        if port_events.empty:
            continue
        event_times = port_events["event_timestamp"].to_numpy(dtype="datetime64[ns]")
        delay_values = port_events["delay_minutes"].to_numpy(dtype=float)
        delayed_values = port_events["is_delayed_event"].to_numpy(dtype=float)
        delay_cumulative = np.concatenate(([0.0], np.cumsum(delay_values)))
        delayed_cumulative = np.concatenate(([0.0], np.cumsum(delayed_values)))

        booking_times = query_rows["booking_date"].to_numpy(dtype="datetime64[ns]")
        # side='left' excludes events at exactly booking_date: strict point-in-time cutoff.
        ends = np.searchsorted(event_times, booking_times, side="left")
        for days in WINDOWS_DAYS:
            starts = np.searchsorted(
                event_times,
                booking_times - np.timedelta64(days, "D"),
                side="left",
            )
            counts = ends - starts
            delay_sums = delay_cumulative[ends] - delay_cumulative[starts]
            delayed_counts = delayed_cumulative[ends] - delayed_cumulative[starts]
            means = np.divide(
                delay_sums, counts,
                out=np.full(len(counts), np.nan, dtype=float),
                where=counts > 0,
            )
            rates = np.divide(
                delayed_counts, counts,
                out=np.full(len(counts), np.nan, dtype=float),
                where=counts > 0,
            )
            idx = query_rows.index
            output.loc[idx, f"{side}_events_{days}d_count"] = counts.astype(float)
            output.loc[idx, f"{side}_events_{days}d_mean_delay_minutes"] = means
            output.loc[idx, f"{side}_events_{days}d_delayed_count"] = delayed_counts
            output.loc[idx, f"{side}_events_{days}d_delayed_rate"] = rates
    return output


def add_point_in_time_port_event_features(
    frame: pd.DataFrame,
    data_dir: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Add trailing port-level aggregates using only events before each booking.

    All records belonging to duplicated event_id groups are quarantined because
    event_id is not unique and those repeated IDs have conflicting payloads.
    Event rows without an event_type remain eligible for generic counts and
    delay aggregates; they cannot increment the DELAYED-event count.
    """
    data_dir = Path(data_dir)
    events = pd.read_csv(data_dir / "port_events.csv")
    original_event_rows = len(events)
    duplicate_id_mask = events["event_id"].duplicated(keep=False)
    duplicate_id_groups = int(events.loc[duplicate_id_mask, "event_id"].nunique())
    duplicate_id_rows = int(duplicate_id_mask.sum())
    events = events.loc[~duplicate_id_mask].copy()

    events["event_timestamp"] = pd.to_datetime(events["event_timestamp"], errors="coerce")
    events["delay_minutes"] = pd.to_numeric(events["delay_minutes"], errors="coerce")
    events["port_code"] = events["port_code"].astype("string").str.strip().str.upper()
    events["event_type"] = events["event_type"].astype("string").str.strip().str.upper()
    events = events.dropna(subset=["event_timestamp", "port_code", "delay_minutes"]).copy()
    events["is_delayed_event"] = events["event_type"].eq("DELAYED").fillna(False).astype(float)
    events = events.sort_values(["port_code", "event_timestamp"]).reset_index(drop=True)

    result = frame.copy()
    result["booking_date"] = pd.to_datetime(result["booking_date"], errors="coerce")
    result["origin_port"] = result["origin_port"].astype("string").str.strip().str.upper()
    result["destination_port"] = result["destination_port"].astype("string").str.strip().str.upper()

    query_base = result[["booking_date"]].copy()
    origin_queries = query_base.assign(origin_port=result["origin_port"])
    destination_queries = query_base.assign(destination_port=result["destination_port"])
    origin_features = _port_history_features(origin_queries, events, "origin")
    destination_features = _port_history_features(destination_queries, events, "destination")
    result = pd.concat([result, origin_features, destination_features], axis=1)

    # The event file begins during the shipment observation period. These
    # fractions describe how much of each trailing window is observable from
    # this file's start; they do not assert that the source is historically complete.
    event_start = events["event_timestamp"].min()
    elapsed_days = (result["booking_date"] - event_start).dt.total_seconds() / 86400.0
    for days in WINDOWS_DAYS:
        result[f"event_observation_{days}d_coverage"] = (
            elapsed_days.clip(lower=0, upper=days) / days
        ).astype(float)

    # Auditable point-in-time guarantee: each query's end index is the first
    # event timestamp >= booking_date. The feature calculation uses [start, end).
    report = {
        "source_event_rows": int(original_event_rows),
        "duplicate_event_id_groups_quarantined": duplicate_id_groups,
        "duplicate_event_id_rows_quarantined": duplicate_id_rows,
        "event_rows_used": int(len(events)),
        "event_type_missing_rows_retained_for_generic_aggregates": int(events["event_type"].isna().sum()),
        "event_timestamp_min": event_start.isoformat() if pd.notna(event_start) else None,
        "event_timestamp_max": events["event_timestamp"].max().isoformat() if len(events) else None,
        "strict_booking_cutoff": "event_timestamp < booking_date; same-time and future events excluded",
        "windows_days": list(WINDOWS_DAYS),
        "aggregation_grain": "port-level only; no shipment/voyage-leg attribution",
        "feature_columns": EVENT_FEATURE_COLUMNS,
        "timestamp_limitation": "Source timestamps are naive and contain no timezone offset; comparisons use source-local naive timestamps without timezone conversion.",
        "history_coverage_limitation": "Coverage fractions measure elapsed time since the first retained event in this file, not verified source completeness before that date.",
    }
    return result, report
