from __future__ import annotations

from datetime import date

from app.api.repository import ShipmentRepository


class Cursor:
    def __init__(self, rows, description):
        self.rows = rows
        self.description = [(name,) for name in description]

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class RecordingConnection:
    def __init__(self):
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params or []))
        if "COUNT(*) FROM serving_shipments" in sql:
            return Cursor([(2,)], ["count"])
        if "FROM serving_shipments" in sql and "LIMIT ? OFFSET ?" in sql:
            return Cursor(
                [("S1", "AAA", "BBB")],
                ["shipment_id", "origin_port", "destination_port"],
            )
        if "FROM serving_shipments" in sql:
            return Cursor(
                [("S1", "AAA", "BBB")],
                ["shipment_id", "origin_port", "destination_port"],
            )
        if "FROM serving_route_stats" in sql:
            return Cursor([("AAA", "BBB", 5.0, 0.9, 12)], [
                "origin_port", "destination_port", "average_delay_hours",
                "on_time_rate", "shipment_count",
            ])
        if "FROM sqlite_master" in sql:
            return Cursor([(2,)], ["count"])
        return Cursor([], [])


def test_shipment_list_uses_serving_view_bound_parameters_and_pagination():
    conn = RecordingConnection()
    repo = ShipmentRepository(conn)
    items, total = repo.list_shipments(
        origin="AAA", destination="BBB", status="DELIVERED",
        date_from=date(2025, 1, 1), date_to=date(2025, 1, 31),
        limit=10, offset=20,
    )
    assert total == 2
    assert items[0]["shipment_id"] == "S1"
    count_sql, count_params = conn.calls[0]
    page_sql, page_params = conn.calls[1]
    assert "serving_shipments" in count_sql
    assert "serving_shipments" in page_sql
    assert "origin_port = ?" in page_sql
    assert "destination_port = ?" in page_sql
    assert "date(booking_date) >= ?" in page_sql
    assert "LIMIT ? OFFSET ?" in page_sql
    assert count_params == ["AAA", "BBB", "DELIVERED", "2025-01-01", "2025-01-31"]
    assert page_params[-2:] == [10, 20]
    assert "AAA" not in page_sql


def test_route_stats_reads_preaggregated_serving_view():
    conn = RecordingConnection()
    stats = ShipmentRepository(conn).get_route_stats("AAA", "BBB")
    assert stats["average_delay_hours"] == 5.0
    assert stats["on_time_rate"] == 0.9
    assert "serving_route_stats" in conn.calls[0][0]


def test_serving_readiness_requires_both_views():
    conn = RecordingConnection()
    assert ShipmentRepository(conn).serving_ready() is True


def test_booking_feature_query_only_selects_booking_time_and_port_reference_fields():
    conn = RecordingConnection()
    repo = ShipmentRepository(conn)
    assert repo.get_booking_features("SHP-1") is None
    sql, params = conn.calls[0]
    assert "integration_shipments" in sql
    assert "staging_ports_current" in sql
    assert "s.booking_date" in sql
    assert "s.planned_departure" in sql
    assert "s.planned_arrival" in sql
    assert "s.actual_arrival" not in sql
    assert "s.actual_departure" not in sql
    assert "s.status" not in sql
    assert "actual_delay_hours" not in sql
    assert params == ["SHP-1"]


def test_filtered_route_aggregation_uses_bound_booking_date_filters():
    conn = RecordingConnection()
    ShipmentRepository(conn).get_route_stats_filtered(
        date_from=date(2025, 1, 1), date_to=date(2025, 3, 31),
        origin="CNSHA", destination="NLRTM", status="DELIVERED", limit=5,
    )
    sql, params = conn.calls[0]
    assert "GROUP BY origin_port, destination_port" in sql
    assert "date(booking_date) >= ?" in sql
    assert "date(booking_date) <= ?" in sql
    assert "origin_port = ?" in sql and "destination_port = ?" in sql
    assert params == ["2025-01-01", "2025-03-31", "CNSHA", "NLRTM", "DELIVERED", 5]
