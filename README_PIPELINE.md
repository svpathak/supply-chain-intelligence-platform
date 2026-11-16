# Data Pipeline and Data Model

## Purpose

Convert the supplied CSV snapshots and port-event feed into a repeatable local data product. The implementation uses Python, Pandas and the standard-library `sqlite3` module; SQLite requires no separately installed database package or server.

## Architecture

```text
CSV files
  |
  v
RAW TABLES (raw_ports, raw_shipments, raw_port_events)
  |  Full snapshot reload; source rows and batch metadata retained
  v
STAGING TABLES
  |-- staging_ports_current / staging_ports_history
  |-- staging_shipments_current / staging_shipments_history
  `-- staging_port_events (append-only, content-hash idempotency)
  |
  v
INTEGRATION TABLE
  `-- integration_shipments (current shipment facts + derived fields)
  |
  v
SERVING VIEWS
  |-- serving_shipments
  |-- serving_route_stats
  `-- serving_port_events
```

SQLite has no schema namespaces in this design. Logical layers are represented by stable table-name prefixes (`raw_`, `staging_`, `integration_`, `serving_`) inside one database file.

## Data model

| Table/view | Grain | Key / purpose |
|---|---|---|
| `raw_ports` | One landed source row per CSV row | Source `port_code`, plus ingestion metadata |
| `raw_shipments` | One landed source row per CSV row | Source `shipment_id`, plus ingestion metadata |
| `raw_port_events` | One landed source row per CSV row | Source `event_id`, plus ingestion metadata |
| `staging_ports_current` | One current row per port | `port_code` |
| `staging_ports_history` | One prior version per changed port | Business key and validity timestamps |
| `staging_shipments_current` | One current row per unambiguous shipment | `shipment_id` |
| `staging_shipments_history` | One prior version per changed or newly quarantined shipment | Business key and validity timestamps |
| `staging_shipments_quarantine` | Source rows for conflicting shipment-ID groups in the current snapshot | `shipment_id`, quarantine reason, batch metadata |
| `staging_port_events` | One distinct event payload | `event_hash`; source `event_id` retained |
| `integration_shipments` | One current curated row per shipment | `shipment_id`; includes assignment-derived fields |
| `serving_shipments` | Query view over integration shipments | Shipment API and assistant reads |
| `serving_route_stats` | One row per origin/destination pair | Pre-aggregated route metrics |
| `serving_port_events` | Query view over staged events | Event context |

## Idempotency and history

- **Raw:** each run transactionally replaces the raw snapshot tables and adds a batch ID and ingestion timestamp.
- **Ports and shipments:** staging upserts compare deterministic full-row hashes. Unchanged records remain untouched; changed records are copied to history before current is replaced. Exact duplicate shipment rows are collapsed. If one shipment ID has multiple distinct payloads, every distinct row in that key group is written to `staging_shipments_quarantine` and excluded from the current curated product; any previously current version is moved to history. Raw retains every source row.
- **Events:** the source `event_id` is not unique. A SHA-256 hash of the complete event payload is the replay key. Distinct payloads with a repeated event ID are retained; exact replays are ignored.
- **Integration and serving:** the curated integration table is rebuilt from current, non-quarantined staging shipment records; serving views are dropped and recreated deterministically.
- **Transactions:** ingestion uses `BEGIN IMMEDIATE` and commits all raw/staging changes together; failures roll back the batch.

## Derived fields

- `actual_delay_hours`: difference between actual and planned arrival, in hours.
- `on_time_flag`: true when `actual_delay_hours <= 24`; NULL when actual arrival or planned arrival is missing.
- `route_key`: origin and destination port codes separated by an arrow.
- `transit_days_planned`: planned arrival minus planned departure, in days.
- `transit_days_actual`: actual arrival minus actual departure, in days; NULL when either milestone is missing.

Timestamp values are stored as ISO-formatted text. SQLite's `julianday()` is used for elapsed-time calculations. This is sufficient for the supplied naive timestamps; timezone normalisation remains a production-hardening item if source timestamps later include mixed offsets.

## Run

From the repository root:

```bash
python -m scripts.run_pipeline --data-dir data/raw --db data/warehouse.sqlite3
```

The command is safe to rerun. For the supplied snapshot it retains 5,028 raw shipment rows, collapses 8 exact duplicates, quarantines 40 rows across 20 conflicting shipment-ID groups, and publishes 4,980 curated shipments. The default database path is `data/warehouse.sqlite3`; set `DATABASE_PATH` to override the API's database location.

## SQLite operating model

- The API opens a read-only connection per request; the pipeline owns writes.
- WAL mode, a bounded busy timeout and short transactions reduce reader/writer contention for this single-node workload.
- SQLite is embedded and included with supported Python builds. No database service or Python database-driver dependency is required.
- For multiple API replicas, sustained concurrent writers, high availability or larger operational workloads, migrate the serving layer to PostgreSQL and retain the same repository/service boundary. For large analytical scans, evaluate a dedicated analytical engine separately.
