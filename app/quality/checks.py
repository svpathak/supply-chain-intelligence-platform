"""Standalone data-quality checks over the original CSV inputs.

This module deliberately has no dependency on a database or the ingestion pipeline.
It reports observations; it does not mutate, repair, or drop source records.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "ports": {
        "port_code", "port_name", "country", "region", "timezone",
        "avg_congestion_score",
    },
    "shipments": {
        "shipment_id", "customer_id", "origin_port", "destination_port",
        "vessel_id", "booking_date", "planned_departure", "actual_departure",
        "planned_arrival", "actual_arrival", "container_count", "cargo_type",
        "weight_tons", "status",
    },
    "port_events": {
        "event_id", "event_timestamp", "port_code", "vessel_id", "event_type",
        "delay_minutes", "notes",
    },
}

TIMESTAMP_COLUMNS = {
    "shipments": (
        "booking_date", "planned_departure", "actual_departure",
        "planned_arrival", "actual_arrival",
    ),
    "port_events": ("event_timestamp",),
}


def _as_text(series: pd.Series) -> pd.Series:
    """String view that preserves nulls instead of turning them into 'nan'."""
    return series.astype("string")


def _add(
    findings: list[dict[str, Any]],
    *,
    name: str,
    dataset: str,
    affected_rows: int,
    total_rows: int,
    severity: str,
    action: str,
    detection: str,
    disposition: str = "flag",
) -> None:
    pct = (100.0 * affected_rows / total_rows) if total_rows else 0.0
    findings.append({
        "check_name": name,
        "dataset": dataset,
        "rows_affected": int(affected_rows),
        "total_rows": int(total_rows),
        "percentage_of_total": round(pct, 2),
        "severity": severity,
        "recommended_action": action,
        "detection_rule": detection,
        "disposition": disposition,
    })


def _missing_count(df: pd.DataFrame, column: str) -> int:
    if column not in df:
        return 0
    s = df[column]
    return int((s.isna() | (s.astype("string").str.strip() == "").fillna(False)).sum())


def _parse_dates(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    return pd.to_datetime(df[column], errors="coerce")


def run_quality_checks(input_dir: str | Path) -> dict[str, Any]:
    """Run all checks against ports.csv, shipments.csv and port_events.csv."""
    base = Path(input_dir)
    frames: dict[str, pd.DataFrame] = {}
    findings: list[dict[str, Any]] = []
    file_names = {"ports": "ports.csv", "shipments": "shipments.csv",
                  "port_events": "port_events.csv"}

    for dataset, filename in file_names.items():
        path = base / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")
        frames[dataset] = pd.read_csv(path, keep_default_na=True)

    for dataset, df in frames.items():
        n = len(df)
        missing_columns = sorted(REQUIRED_COLUMNS[dataset] - set(df.columns))
        _add(
            findings, name=f"{dataset}.required_columns_present",
            dataset=dataset, affected_rows=(n if missing_columns else 0), total_rows=n,
            severity="critical" if missing_columns else "info",
            action=("Restore the missing source columns or update the declared input contract; "
                    "do not continue ingestion with an incompatible schema."
                    if missing_columns else "No action; required columns are present."),
            detection=f"Compare CSV headers with the required {dataset} schema.",
            disposition="quarantine" if missing_columns else "none",
        )
        if missing_columns:
            findings[-1]["missing_columns"] = missing_columns
            # Add absent columns as null placeholders so the report can continue
            # and enumerate all detectable issues instead of crashing on drift.
            for missing_column in missing_columns:
                df[missing_column] = pd.NA

        exact_duplicates = int(df.duplicated(keep="first").sum())
        _add(
            findings, name=f"{dataset}.exact_duplicate_rows", dataset=dataset,
            affected_rows=exact_duplicates, total_rows=n,
            severity="warning" if exact_duplicates else "info",
            action=("Retain an audit copy, then deduplicate exact repeated records if the "
                    "dataset grain confirms they are redundant."
                    if exact_duplicates else "No action; no exact duplicate rows found."),
            detection="Count rows identical across every supplied column.",
            disposition="deduplicate_with_audit" if exact_duplicates else "none",
        )

        for col in sorted(REQUIRED_COLUMNS[dataset]):
            if col not in df:
                continue
            # Null checks are emitted selectively below with business-aware severity.
            if col in TIMESTAMP_COLUMNS.get(dataset, ()):
                parsed = _parse_dates(df, col)
                bad = int((df[col].notna() & parsed.isna()).sum())
                _add(
                    findings, name=f"{dataset}.{col}.timestamp_parse_failures",
                    dataset=dataset, affected_rows=bad, total_rows=n,
                    severity="critical" if bad else "info",
                    action=("Quarantine malformed timestamp values and inspect source format."
                            if bad else "No action; all populated values parse as timestamps."),
                    detection=f"Parse {col} with pandas.to_datetime(errors='coerce'); count non-null source values that become null.",
                    disposition="quarantine" if bad else "none",
                )

    ports = frames["ports"]
    shipments = frames["shipments"]
    events = frames["port_events"]
    np, ns, ne = len(ports), len(shipments), len(events)

    # Reference data checks.
    for col, label, severity, action in [
        ("country", "missing_country", "warning",
         "Flag for reference-data enrichment; do not infer a country from the port code."),
    ]:
        count = _missing_count(ports, col)
        _add(findings, name=f"ports.{label}", dataset="ports",
             affected_rows=count, total_rows=np, severity=severity,
             action=action, detection=f"Count null or blank values in {col}.")

    if "port_code" in ports:
        dup_codes = int(ports["port_code"].duplicated(keep=False).sum())
        _add(findings, name="ports.duplicate_port_code", dataset="ports",
             affected_rows=dup_codes, total_rows=np,
             severity="critical" if dup_codes else "info",
             action=("Quarantine conflicting reference keys and restore one authoritative row per port_code."
                     if dup_codes else "No action; port_code is unique."),
             detection="Count every row whose port_code occurs more than once.",
             disposition="quarantine" if dup_codes else "none")

    # Shipment completeness: timestamp nulls are independent checks because their
    # paired occurrence does not establish why an actual milestone is absent.
    for col, severity, action in [
        ("actual_departure", "warning", "Flag as an unobserved milestone; resolve against shipment lifecycle/status before imputing."),
        ("actual_arrival", "warning", "Flag as an unobserved milestone; resolve against shipment lifecycle/status before imputing."),
        ("cargo_type", "warning", "Flag and request source correction; do not invent a cargo classification."),
        ("weight_tons", "warning", "Flag for source correction; do not impute a physical weight without a documented rule."),
        ("status", "warning", "Flag for lifecycle reconciliation; preserve the raw status value."),
    ]:
        count = _missing_count(shipments, col)
        _add(findings, name=f"shipments.missing_{col}", dataset="shipments",
             affected_rows=count, total_rows=ns, severity=severity, action=action,
             detection=f"Count null or blank values in {col}.")

    # Shipment business-key uniqueness and conflict detection.
    if "shipment_id" in shipments:
        dup_mask = shipments["shipment_id"].notna() & shipments["shipment_id"].duplicated(keep=False)
        dup_rows = shipments.loc[dup_mask]
        dup_groups = int(dup_rows["shipment_id"].nunique())
        conflict_groups = 0
        if not dup_rows.empty:
            payload_cols = [c for c in shipments.columns if c != "shipment_id"]
            conflict_groups = int(
                dup_rows.groupby("shipment_id", dropna=True)[payload_cols]
                .apply(lambda g: len(g.drop_duplicates()) > 1)
                .sum()
            )
        _add(findings, name="shipments.repeated_shipment_id", dataset="shipments",
             affected_rows=int(dup_mask.sum()), total_rows=ns,
             severity="critical" if conflict_groups else ("warning" if dup_groups else "info"),
             action=("Quarantine or reconcile conflicting versions using an authoritative update timestamp/source rule; "
                     "do not silently select a record."
                     if conflict_groups else ("Deduplicate repeated keys only after confirming records are equivalent."
                     if dup_groups else "No action; shipment_id is unique.")),
             detection="Count all rows in repeated shipment_id groups; classify a group as conflicting when non-key payloads differ.",
             disposition="quarantine" if conflict_groups else ("deduplicate_with_audit" if dup_groups else "none"))
        findings[-1]["duplicate_key_groups"] = dup_groups
        findings[-1]["conflicting_key_groups"] = conflict_groups

    if "weight_tons" in shipments:
        weights = pd.to_numeric(shipments["weight_tons"], errors="coerce")
        negative = int((weights < 0).sum())
        _add(findings, name="shipments.negative_weight", dataset="shipments",
             affected_rows=negative, total_rows=ns, severity="warning" if negative else "info",
             action=("Quarantine or correct from the source system; physical shipment weight cannot be negative."
                     if negative else "No action; no negative weights found."),
             detection="Numeric weight_tons < 0.", disposition="quarantine" if negative else "none")

    if "container_count" in shipments:
        containers = pd.to_numeric(shipments["container_count"], errors="coerce")
        nonpositive = int((containers <= 0).sum())
        _add(findings, name="shipments.non_positive_container_count", dataset="shipments",
             affected_rows=nonpositive, total_rows=ns,
             severity="warning" if nonpositive else "info",
             action=("Flag for source review; confirm whether zero-container shipments are valid before rejecting them."
                     if nonpositive else "No action; all container counts are positive."),
             detection="Numeric container_count <= 0.", disposition="flag" if nonpositive else "none")

    if "cargo_type" in shipments:
        cargo = _as_text(shipments["cargo_type"])
        cargo_bad = int((cargo.notna() & (cargo != cargo.str.strip().str.title())).sum())
        _add(findings, name="shipments.cargo_type_noncanonical_format", dataset="shipments",
             affected_rows=cargo_bad, total_rows=ns, severity="warning" if cargo_bad else "info",
             action=("Normalize only through an explicit approved mapping; preserve the original value for audit."
                     if cargo_bad else "No action; populated cargo types follow the chosen title-case convention."),
             detection="Compare populated cargo_type values with strip()+title(); this is a formatting heuristic, not an accepted-value taxonomy.",
             disposition="normalize_with_mapping" if cargo_bad else "none")

    if "status" in shipments:
        status = _as_text(shipments["status"])
        status_bad = int((status.notna() & (status != status.str.strip().str.upper())).sum())
        _add(findings, name="shipments.status_noncanonical_format", dataset="shipments",
             affected_rows=status_bad, total_rows=ns, severity="warning" if status_bad else "info",
             action=("Preserve raw values and define an explicit lifecycle mapping with the source owner; do not merge COMPLETE and COMPLETED without confirmation."
                     if status_bad else "No action; populated statuses follow trimmed uppercase formatting."),
             detection="Compare populated status values with strip()+upper(); semantic label mapping is intentionally not inferred.",
             disposition="normalize_with_mapping" if status_bad else "none")

    # Shipment chronology.
    date_cols = {c: _parse_dates(shipments, c) for c in TIMESTAMP_COLUMNS["shipments"]}
    temporal_rules = [
        ("booking_after_planned_departure", "booking_date", "planned_departure",
         lambda a, b: a > b, "Booking should not occur after planned departure."),
        ("planned_arrival_before_planned_departure", "planned_arrival", "planned_departure",
         lambda a, b: a < b, "Planned arrival should not precede planned departure."),
        ("actual_departure_before_booking", "actual_departure", "booking_date",
         lambda a, b: a < b, "Actual departure should not precede booking time."),
        ("actual_arrival_before_actual_departure", "actual_arrival", "actual_departure",
         lambda a, b: a < b, "Actual arrival should not precede actual departure."),
    ]
    for rule, left, right, predicate, description in temporal_rules:
        if left in date_cols and right in date_cols:
            mask = date_cols[left].notna() & date_cols[right].notna() & predicate(date_cols[left], date_cols[right])
            count = int(mask.sum())
            _add(findings, name=f"shipments.{rule}", dataset="shipments",
                 affected_rows=count, total_rows=ns, severity="warning" if count else "info",
                 action=("Flag affected records and verify timestamp semantics/timezones with the source; do not auto-shift timestamps."
                         if count else "No action; no chronology violations found."),
                 detection=description, disposition="flag" if count else "none")

    # Cross-table shipment -> port referential integrity.
    if "port_code" in ports:
        valid_ports = set(_as_text(ports["port_code"]).dropna().str.strip())
        invalid_masks = {}
        for col in ("origin_port", "destination_port"):
            if col in shipments:
                values = _as_text(shipments[col]).str.strip()
                invalid_masks[col] = values.isna() | ~values.isin(valid_ports)
                count = int(invalid_masks[col].sum())
                _add(findings, name=f"shipments.{col}_not_in_ports_reference",
                     dataset="shipments", affected_rows=count, total_rows=ns,
                     severity="critical" if count else "info",
                     action=("Quarantine or correct unknown/missing port references against the authoritative port master; do not fabricate port mappings."
                             if count else "No action; all populated references resolve to ports."),
                     detection=f"Compare {col} values with ports.port_code; null/blank references are also unresolved.",
                     disposition="quarantine" if count else "none")
        if len(invalid_masks) == 2:
            union = invalid_masks["origin_port"] | invalid_masks["destination_port"]
            count = int(union.sum())
            _add(findings, name="shipments.any_port_reference_unresolved",
                 dataset="shipments", affected_rows=count, total_rows=ns,
                 severity="critical" if count else "info",
                 action=("Use this union count for shipment-level impact; do not add origin and destination counts because a row can fail both."
                         if count else "No action; every shipment resolves both port references."),
                 detection="Count unique shipment rows failing either origin or destination port lookup.",
                 disposition="quarantine" if count else "none")

    # Port event checks.
    for col, severity, action in [
        ("event_type", "warning", "Flag for source correction; retain the event row because the event payload may still be useful."),
        ("port_code", "critical", "Quarantine or correct unresolved event port references using the port master."),
    ]:
        count = _missing_count(events, col)
        _add(findings, name=f"port_events.missing_{col}", dataset="port_events",
             affected_rows=count, total_rows=ne, severity=severity, action=action,
             detection=f"Count null or blank values in {col}.")

    if "event_id" in events:
        event_dup_mask = events["event_id"].notna() & events["event_id"].duplicated(keep=False)
        repeated = events.loc[event_dup_mask]
        groups = int(repeated["event_id"].nunique())
        payload_cols = [c for c in events.columns if c != "event_id"]
        conflict_groups = int(
            repeated.groupby("event_id", dropna=True)[payload_cols]
            .apply(lambda g: len(g.drop_duplicates()) > 1).sum()
        ) if not repeated.empty else 0
        _add(findings, name="port_events.repeated_event_id", dataset="port_events",
             affected_rows=int(event_dup_mask.sum()), total_rows=ne,
             severity="critical" if conflict_groups else ("warning" if groups else "info"),
             action=("event_id is not a safe deduplication key in this feed; retain distinct payloads and establish a composite/source event key."
                     if conflict_groups else ("Review repeated identifiers and confirm event grain."
                     if groups else "No action; event_id is unique.")),
             detection="Count all rows in repeated event_id groups; compare non-key payloads to identify conflicting groups.",
             disposition="retain_and_reconcile_key" if conflict_groups else ("review" if groups else "none"))
        findings[-1]["duplicate_key_groups"] = groups
        findings[-1]["conflicting_key_groups"] = conflict_groups

    if "port_code" in events and "port_code" in ports:
        valid_ports = set(_as_text(ports["port_code"]).dropna().str.strip())
        event_ports = _as_text(events["port_code"]).str.strip()
        bad = event_ports.isna() | ~event_ports.isin(valid_ports)
        count = int(bad.sum())
        _add(findings, name="port_events.port_code_not_in_ports_reference",
             dataset="port_events", affected_rows=count, total_rows=ne,
             severity="critical" if count else "info",
             action=("Quarantine or correct unresolved event port references against the port master."
                     if count else "No action; all populated event port references resolve."),
             detection="Compare port_events.port_code with ports.port_code.",
             disposition="quarantine" if count else "none")

    if "delay_minutes" in events:
        delay = pd.to_numeric(events["delay_minutes"], errors="coerce")
        negative = int((delay < 0).sum())
        _add(findings, name="port_events.negative_delay_minutes", dataset="port_events",
             affected_rows=negative, total_rows=ne,
             severity="warning" if negative else "info",
             action=("Flag negative delay values and confirm whether they encode early arrival or invalid data."
                     if negative else "No action; no negative delay values found."),
             detection="Numeric delay_minutes < 0.", disposition="flag" if negative else "none")

    notes_missing = _missing_count(events, "notes")
    _add(findings, name="port_events.missing_notes", dataset="port_events",
         affected_rows=notes_missing, total_rows=ne, severity="info",
         action="Treat notes as optional descriptive text unless a downstream use case makes them mandatory.",
         detection="Count null or blank notes; reported for completeness visibility, not classified as a defect.",
         disposition="informational_only")

    # Timestamp timezone metadata is absent in the CSV string representation.
    naive_timestamp_rows = 0
    for dataset, columns in TIMESTAMP_COLUMNS.items():
        frame = frames[dataset]
        for col in columns:
            if col in frame:
                naive_timestamp_rows += int(frame[col].notna().sum())
    timestamp_total = sum(
        int(frames[d][c].notna().sum())
        for d, cols in TIMESTAMP_COLUMNS.items() for c in cols if c in frames[d]
    )
    _add(findings, name="timestamps.timezone_metadata_not_declared",
         dataset="shipments+port_events", affected_rows=naive_timestamp_rows,
         total_rows=timestamp_total, severity="warning",
         action="Document whether source timestamps are UTC or local wall time and add explicit offsets at the source boundary before cross-zone instant comparisons.",
         detection="CSV timestamp values contain no explicit UTC offset/timezone metadata; all populated values are treated as naive.",
         disposition="document_and_resolve_semantics")

    # Summarize input files and findings without embedding raw row-level data.
    return {
        "report_version": "1.0",
        "scope": "Standalone checks against raw CSV files; no ingestion or database required.",
        "input_directory": str(base),
        "datasets": {
            name: {"file": file_names[name], "rows": int(len(df)),
                   "columns": [str(c) for c in df.columns]}
            for name, df in frames.items()
        },
        "summary": {
            "checks_executed": len(findings),
            "checks_with_findings": sum(1 for f in findings if f["rows_affected"] > 0),
            "critical_checks_with_findings": sum(
                1 for f in findings if f["rows_affected"] > 0 and f["severity"] == "critical"
            ),
            "warning_checks_with_findings": sum(
                1 for f in findings if f["rows_affected"] > 0 and f["severity"] == "warning"
            ),
            "info_checks_with_findings": sum(
                1 for f in findings if f["rows_affected"] > 0 and f["severity"] == "info"
            ),
        },
        "findings": findings,
        "interpretation_notes": [
            "Rows affected is measured against the dataset named by each check; timezone metadata uses the count of populated timestamp values as its denominator.",
            "Checks can overlap. For example, exact duplicate shipment rows also participate in repeated shipment_id groups; do not sum finding counts to estimate unique bad rows.",
            "Formatting checks are heuristics, not semantic mappings. Raw category values are not modified.",
            "A finding is a detection, not an automatic cleaning operation. Recommended actions are advisory and source data is never changed.",
            "The assignment states there are twelve intentional issues but does not enumerate them; this report records reproducible observations and additional diagnostics without claiming every finding maps one-to-one to an intentional issue.",
        ],
    }
