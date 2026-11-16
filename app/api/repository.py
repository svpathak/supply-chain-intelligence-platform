"""SQL repository over the curated SQLite serving data product."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any


class DataStoreUnavailable(RuntimeError):
    """Raised when the serving data product cannot be queried."""


class ShipmentRepository:
    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def _execute(self, sql: str, params: list[Any] | None = None):
        try:
            return self.connection.execute(sql, params or [])
        except Exception as exc:
            raise DataStoreUnavailable("The serving data product could not be queried.") from exc

    @staticmethod
    def _rows(cursor) -> list[dict[str, Any]]:
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_shipment(self, shipment_id: str) -> dict[str, Any] | None:
        cursor = self._execute("SELECT * FROM serving_shipments WHERE shipment_id = ? LIMIT 1", [shipment_id])
        rows = self._rows(cursor)
        return rows[0] if rows else None

    def list_shipments(self, *, origin=None, destination=None, status=None, date_from: date | None = None, date_to: date | None = None, limit: int = 50, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        clauses, params = [], []
        if origin is not None:
            clauses.append("origin_port = ?"); params.append(origin)
        if destination is not None:
            clauses.append("destination_port = ?"); params.append(destination)
        if status is not None:
            clauses.append("UPPER(TRIM(status)) = UPPER(TRIM(?))"); params.append(status)
        if date_from is not None:
            clauses.append("date(booking_date) >= ?"); params.append(date_from.isoformat())
        if date_to is not None:
            clauses.append("date(booking_date) <= ?"); params.append(date_to.isoformat())
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        total = int(self._execute(f"SELECT COUNT(*) FROM serving_shipments {where_sql}", params).fetchone()[0])
        cursor = self._execute(f"""SELECT * FROM serving_shipments {where_sql}
            ORDER BY (booking_date IS NULL), booking_date DESC, shipment_id LIMIT ? OFFSET ?""", [*params, limit, offset])
        return self._rows(cursor), total

    def resolve_port_code(self, port_identifier: str) -> str | None:
        row = self._execute("""SELECT port_code FROM staging_ports_current
            WHERE is_current = 1 AND (UPPER(TRIM(port_code)) = UPPER(TRIM(?))
            OR UPPER(TRIM(port_name)) = UPPER(TRIM(?)))
            ORDER BY CASE WHEN UPPER(TRIM(port_code)) = UPPER(TRIM(?)) THEN 0 ELSE 1 END LIMIT 1""",
            [port_identifier, port_identifier, port_identifier]).fetchone()
        return str(row[0]) if row else None

    def get_route_stats(self, origin: str, destination: str) -> dict[str, Any] | None:
        cursor = self._execute("""SELECT origin_port, destination_port, average_delay_hours, on_time_rate, shipment_count
            FROM serving_route_stats WHERE origin_port = ? AND destination_port = ? LIMIT 1""", [origin, destination])
        rows = self._rows(cursor)
        return rows[0] if rows else None

    def get_booking_features(self, shipment_id: str) -> dict[str, Any] | None:
        cursor = self._execute("""SELECT s.shipment_id, s.booking_date, s.planned_departure, s.planned_arrival,
            s.container_count, s.weight_tons, s.cargo_type, s.origin_port, s.destination_port,
            origin.region AS origin_region, origin.avg_congestion_score AS origin_congestion_score,
            destination.region AS destination_region, destination.avg_congestion_score AS destination_congestion_score
            FROM integration_shipments s
            LEFT JOIN staging_ports_current origin ON s.origin_port = origin.port_code AND origin.is_current = 1
            LEFT JOIN staging_ports_current destination ON s.destination_port = destination.port_code AND destination.is_current = 1
            WHERE s.shipment_id = ? LIMIT 1""", [shipment_id])
        rows = self._rows(cursor)
        return rows[0] if rows else None

    def get_route_stats_filtered(self, *, date_from=None, date_to=None, origin=None, destination=None, status=None, limit: int = 10) -> list[dict[str, Any]]:
        clauses, params = [], []
        if date_from is not None:
            clauses.append("date(booking_date) >= ?"); params.append(date_from.isoformat())
        if date_to is not None:
            clauses.append("date(booking_date) <= ?"); params.append(date_to.isoformat())
        if origin is not None:
            clauses.append("origin_port = ?"); params.append(origin)
        if destination is not None:
            clauses.append("destination_port = ?"); params.append(destination)
        if status is not None:
            clauses.append("UPPER(TRIM(status)) = UPPER(TRIM(?))"); params.append(status)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = self._execute(f"""SELECT origin_port, destination_port,
            AVG(actual_delay_hours) AS average_delay_hours,
            AVG(CASE WHEN on_time_flag IS NOT NULL THEN CAST(on_time_flag AS REAL) END) AS on_time_rate,
            COUNT(*) AS shipment_count FROM serving_shipments {where_sql}
            GROUP BY origin_port, destination_port HAVING COUNT(actual_delay_hours) > 0
            ORDER BY average_delay_hours DESC LIMIT ?""", [*params, limit])
        return self._rows(cursor)

    def serving_ready(self) -> bool:
        row = self._execute("""SELECT COUNT(*) FROM sqlite_master
            WHERE type = 'view' AND name IN ('serving_shipments', 'serving_route_stats')""").fetchone()
        return bool(row and row[0] == 2)
