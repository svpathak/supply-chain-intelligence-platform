from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sqlite3
import uuid

import pandas as pd

from app.data.database import connect, table_name

TABLES = ("ports", "shipments", "port_events")
TIMESTAMP_COLUMNS = {
    "shipments": ["booking_date", "planned_departure", "actual_departure", "planned_arrival", "actual_arrival"],
    "port_events": ["event_timestamp"],
}
BUSINESS_KEYS = {"ports": "port_code", "shipments": "shipment_id"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def read_source_csv(path: str | Path, table: str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")
    frame = pd.read_csv(path)
    for column in TIMESTAMP_COLUMNS.get(table, []):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _normalise(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().isoformat(sep=" ")
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _stable_hash(record: dict) -> str:
    normalized = {key: _normalise(value) for key, value in record.items()}
    payload = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sqlite_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series.dtype) or pd.api.types.is_integer_dtype(series.dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series.dtype):
        return "REAL"
    return "TEXT"


def _create_table_from_frame(conn: sqlite3.Connection, name: str, frame: pd.DataFrame, extra: str = "") -> None:
    definitions = [f"{_quote(col)} {_sqlite_type(frame[col])}" for col in frame.columns]
    if extra:
        definitions.extend(part.strip() for part in extra.split(",") if part.strip())
    conn.execute(f"CREATE TABLE IF NOT EXISTS {_quote(name)} ({', '.join(definitions)})")


def _insert_frame(conn: sqlite3.Connection, name: str, frame: pd.DataFrame) -> None:
    columns = list(frame.columns)
    placeholders = ",".join("?" for _ in columns)
    sql = f"INSERT INTO {_quote(name)} ({','.join(_quote(c) for c in columns)}) VALUES ({placeholders})"
    rows = [tuple(_normalise(v) for v in row) for row in frame.itertuples(index=False, name=None)]
    if rows:
        conn.executemany(sql, rows)


def _load_raw_snapshot(conn: sqlite3.Connection, table: str, frame: pd.DataFrame, batch_id: str, loaded_at: datetime) -> None:
    landed = frame.copy()
    landed["ingestion_timestamp"] = loaded_at.isoformat(sep=" ")
    landed["batch_id"] = batch_id
    target = table_name("raw", table)
    conn.execute(f"DROP TABLE IF EXISTS {_quote(target)}")
    _create_table_from_frame(conn, target, landed)
    _insert_frame(conn, target, landed)


def _ensure_version_tables(conn: sqlite3.Connection, table: str, frame: pd.DataFrame) -> list[str]:
    columns = list(frame.columns)
    extra = ', "record_hash" TEXT, "valid_from" TEXT, "valid_to" TEXT, "is_current" INTEGER'
    current = table_name("staging", f"{table}_current")
    history = table_name("staging", f"{table}_history")
    _create_table_from_frame(conn, current, frame, extra)
    conn.execute(f"CREATE TABLE IF NOT EXISTS {_quote(history)} AS SELECT * FROM {_quote(current)} WHERE 0")
    return columns


def _upsert_snapshot(
    conn: sqlite3.Connection,
    table: str,
    frame: pd.DataFrame,
    business_key: str,
    loaded_at: datetime,
    batch_id: str,
    *,
    quarantine_conflicts: bool = False,
) -> dict[str, int]:
    columns = _ensure_version_tables(conn, table, frame)
    if frame[business_key].isna().any():
        raise ValueError(f"{table}.{business_key} contains null business keys")

    exact_deduplicated = frame.drop_duplicates(subset=columns, keep="first").copy()
    exact_duplicate_rows = len(frame) - len(exact_deduplicated)
    conflict_keys: set[str] = set()
    quarantined_rows = exact_deduplicated.iloc[0:0].copy()

    if quarantine_conflicts:
        exact_deduplicated["_record_hash"] = exact_deduplicated.apply(
            lambda row: _stable_hash({key: row[key] for key in columns}), axis=1
        )
        distinct_versions = exact_deduplicated.groupby(business_key)["_record_hash"].nunique()
        conflict_keys = {str(key) for key in distinct_versions[distinct_versions > 1].index}
        quarantined_rows = exact_deduplicated[
            exact_deduplicated[business_key].astype(str).isin(conflict_keys)
        ].drop(columns=["_record_hash"])
        exact_deduplicated = exact_deduplicated[
            ~exact_deduplicated[business_key].astype(str).isin(conflict_keys)
        ].drop(columns=["_record_hash"])

        quarantine_table = table_name("staging", f"{table}_quarantine")
        quarantine_snapshot = quarantined_rows.copy()
        quarantine_snapshot["quarantine_reason"] = "conflicting_duplicate_business_key"
        quarantine_snapshot["quarantined_at"] = loaded_at.isoformat(sep=" ")
        quarantine_snapshot["batch_id"] = batch_id
        conn.execute(f"DROP TABLE IF EXISTS {_quote(quarantine_table)}")
        _create_table_from_frame(conn, quarantine_table, quarantine_snapshot)
        _insert_frame(conn, quarantine_table, quarantine_snapshot)

    current = table_name("staging", f"{table}_current")
    history = table_name("staging", f"{table}_history")
    if conflict_keys:
        placeholders = ",".join("?" for _ in conflict_keys)
        key_values = sorted(conflict_keys)
        conn.execute(
            f"INSERT INTO {_quote(history)} ({','.join(_quote(c) for c in columns + ['record_hash', 'valid_from', 'valid_to', 'is_current'])}) "
            f"SELECT {','.join(_quote(c) for c in columns)}, record_hash, valid_from, ?, 0 "
            f"FROM {_quote(current)} WHERE is_current = 1 AND CAST({_quote(business_key)} AS TEXT) IN ({placeholders})",
            [loaded_at.isoformat(sep=" "), *key_values],
        )
        conn.execute(
            f"DELETE FROM {_quote(current)} WHERE is_current = 1 AND CAST({_quote(business_key)} AS TEXT) IN ({placeholders})",
            key_values,
        )

    rows = conn.execute(
        f"SELECT {_quote(business_key)}, record_hash FROM {_quote(current)} WHERE is_current = 1"
    ).fetchall()
    hashes = {str(key): row_hash for key, row_hash in rows}
    inserted = updated = unchanged = 0
    insert_columns = columns + ["record_hash", "valid_from", "valid_to", "is_current"]
    insert_sql = (
        f"INSERT INTO {_quote(current)} ({','.join(_quote(c) for c in insert_columns)}) "
        f"VALUES ({','.join('?' for _ in insert_columns)})"
    )
    for record in exact_deduplicated.to_dict(orient="records"):
        key = str(record[business_key])
        row_hash = _stable_hash(record)
        old_hash = hashes.get(key)
        if old_hash == row_hash:
            unchanged += 1
            continue
        if old_hash is not None:
            conn.execute(
                f"INSERT INTO {_quote(history)} ({','.join(_quote(c) for c in columns + ['record_hash', 'valid_from', 'valid_to', 'is_current'])}) "
                f"SELECT {','.join(_quote(c) for c in columns)}, record_hash, valid_from, ?, 0 "
                f"FROM {_quote(current)} WHERE {_quote(business_key)} = ? AND is_current = 1",
                (loaded_at.isoformat(sep=" "), record[business_key]),
            )
            conn.execute(
                f"DELETE FROM {_quote(current)} WHERE {_quote(business_key)} = ? AND is_current = 1",
                (record[business_key],),
            )
            updated += 1
        else:
            inserted += 1
        values = [_normalise(record[col]) for col in columns] + [row_hash, loaded_at.isoformat(sep=" "), None, 1]
        conn.execute(insert_sql, values)

    return {
        "inserted": inserted,
        "updated": updated,
        "unchanged": unchanged,
        "exact_duplicate_rows_collapsed": exact_duplicate_rows,
        "conflicting_business_key_groups_quarantined": len(conflict_keys),
        "conflicting_business_key_rows_quarantined": len(quarantined_rows),
    }


def _append_event_feed(conn: sqlite3.Connection, frame: pd.DataFrame) -> dict[str, int]:
    if "event_id" not in frame.columns:
        raise ValueError("port_events is missing required event_id column")
    incoming = frame.copy()
    incoming["event_hash"] = incoming.apply(lambda row: _stable_hash(row.to_dict()), axis=1)
    incoming = incoming.drop_duplicates(subset=["event_hash"], keep="first")
    target = table_name("staging", "port_events")
    _create_table_from_frame(conn, target, incoming)
    before = conn.execute(f"SELECT COUNT(*) FROM {_quote(target)}").fetchone()[0]
    existing = {row[0] for row in conn.execute(f"SELECT event_hash FROM {_quote(target)}")}
    new_rows = incoming[~incoming["event_hash"].isin(existing)]
    _insert_frame(conn, target, new_rows)
    appended = len(new_rows)
    return {"appended": appended, "duplicate_exact_rows_in_batch": len(frame) - len(incoming), "already_ingested_rows": len(incoming) - appended}


def run_ingestion(data_dir: str | Path, db_path: str | Path) -> dict:
    """Atomically refresh raw snapshots and idempotently update staging tables."""
    data_dir = Path(data_dir)
    frames = {table: read_source_csv(data_dir / f"{table}.csv", table) for table in TABLES}
    batch_id, loaded_at = str(uuid.uuid4()), _utc_now()
    conn = connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            for table, frame in frames.items():
                _load_raw_snapshot(conn, table, frame, batch_id, loaded_at)
            stats = {
                table: _upsert_snapshot(
                    conn, table, frames[table], key, loaded_at, batch_id,
                    quarantine_conflicts=(table == "shipments"),
                )
                for table, key in BUSINESS_KEYS.items()
            }
            stats["port_events"] = _append_event_feed(conn, frames["port_events"])
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return {"batch_id": batch_id, "loaded_at_utc": loaded_at.isoformat() + "Z", "raw_rows": {k: len(v) for k, v in frames.items()}, "staging": stats}
    finally:
        conn.close()
