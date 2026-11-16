from __future__ import annotations

import sqlite3

from app.data.database import table_name


def build_integration_and_serving(conn: sqlite3.Connection) -> None:
    """Rebuild curated integration table and serving views idempotently."""
    staging_shipments = table_name("staging", "shipments_current")
    integration_shipments = table_name("integration", "shipments")
    serving_shipments = table_name("serving", "shipments")
    serving_routes = table_name("serving", "route_stats")
    serving_events = table_name("serving", "port_events")
    conn.execute(f"DROP TABLE IF EXISTS {integration_shipments}")
    conn.execute(f"""
        CREATE TABLE {integration_shipments} AS
        SELECT s.shipment_id, s.customer_id, s.origin_port, s.destination_port,
               s.vessel_id, s.booking_date, s.planned_departure, s.actual_departure,
               s.planned_arrival, s.actual_arrival, s.container_count, s.cargo_type,
               s.weight_tons, s.status,
               CASE WHEN s.actual_arrival IS NOT NULL AND s.planned_arrival IS NOT NULL
                    THEN (julianday(s.actual_arrival)-julianday(s.planned_arrival))*24.0 END AS actual_delay_hours,
               CASE WHEN s.actual_arrival IS NOT NULL AND s.planned_arrival IS NOT NULL
                    THEN (julianday(s.actual_arrival)-julianday(s.planned_arrival))*24.0 <= 24 END AS on_time_flag,
               s.origin_port || ' → ' || s.destination_port AS route_key,
               CASE WHEN s.planned_departure IS NOT NULL AND s.planned_arrival IS NOT NULL
                    THEN julianday(s.planned_arrival)-julianday(s.planned_departure) END AS transit_days_planned,
               CASE WHEN s.actual_departure IS NOT NULL AND s.actual_arrival IS NOT NULL
                    THEN julianday(s.actual_arrival)-julianday(s.actual_departure) END AS transit_days_actual
        FROM {staging_shipments} s WHERE s.is_current = 1
    """)
    for view in (serving_shipments, serving_routes, serving_events):
        conn.execute(f"DROP VIEW IF EXISTS {view}")
    conn.execute(f"CREATE VIEW {serving_shipments} AS SELECT * FROM {integration_shipments}")
    conn.execute(f"""
        CREATE VIEW {serving_routes} AS
        SELECT origin_port, destination_port,
               AVG(actual_delay_hours) AS average_delay_hours,
               AVG(CASE WHEN on_time_flag IS NOT NULL THEN CAST(on_time_flag AS REAL) END) AS on_time_rate,
               COUNT(*) AS shipment_count
        FROM {integration_shipments} GROUP BY origin_port, destination_port
    """)
    conn.execute(f"CREATE VIEW {serving_events} AS SELECT * FROM {table_name('staging', 'port_events')}")
    conn.commit()
