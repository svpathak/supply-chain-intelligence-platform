# Supply Chain Intelligence Mini-Platform

A small, reproducible supply-chain data and intelligence platform built for the Software Engineer (Data & AI/ML) take-home assignment.

## Project status

The project includes a layered SQLite pipeline, standalone raw-file data-quality checks, six FastAPI endpoints, saved-model inference and a Groq-powered CLI assistant with three function tools. The pipeline retains 5,028 raw shipment rows, collapses 8 exact duplicate rows, quarantines 40 rows across 20 conflicting shipment-ID groups, and publishes 4,980 unambiguous shipments to the curated layer. Logistic Regression is retained as the reference model and compared with a random 50% baseline; validation ROC-AUC is 0.481, so predictions are explicitly experimental and must not drive operational decisions. The prediction artifact is loaded once at API startup. The assistant applies two explicit guardrails and records prompts, tool calls, tool responses and final answers in JSONL. The assignment does not enumerate its twelve intentional DQ issues; the report documents observed findings without claiming a one-to-one mapping.

## Architecture

```text
Raw CSVs
  |-- standalone data-quality checks --> machine-readable DQ report
  `-- ingestion
        -> raw_* (SQLite snapshot tables)
        -> staging (current + history; conflicting shipment keys quarantined; append-only events)
        -> integration (curated shipment features)
        -> serving (query views)
              |-> FastAPI
              |-> ML training (booking-time features; Logistic Regression vs random baseline)
              |      `-> persisted model artifacts
              |-> ML prediction service (startup-loaded artifact)
              `-> Groq CLI assistant
                    |-> query_shipments
                    |-> get_route_stats
                    `-> predict_delay (HTTP API)
```


## Getting started

The recommended path is Docker Compose. It builds the image, runs the idempotent data pipeline once, and starts the API after the pipeline succeeds:

```bash
docker compose up --build -d
```

Verify startup and readiness:

```bash
docker compose ps
curl http://localhost:8000/health
```

OpenAPI documentation is available at `http://localhost:8000/docs`. Compose persists the SQLite warehouse in the `warehouse_data` named volume. The raw CSVs and saved model artefacts are included in the image. The pipeline service reruns on `docker compose up` and is designed to be idempotent.

To inspect logs or stop the stack:

```bash
docker compose logs -f api
docker compose logs pipeline
docker compose down
```

`docker compose down -v` also deletes the persisted warehouse volume; use it only when intentionally resetting local state.

For a non-containerised development run:

```bash
pip install -r requirements.txt
python -m scripts.run_pipeline
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

The default database path is `data/warehouse.sqlite3`; override it with `DATABASE_PATH`. The API opens the database read-only and does not run ingestion or transformations during requests. The saved Logistic Regression artifact and metadata are loaded once at API startup; a missing/corrupt artifact fails startup rather than silently disabling prediction.

For the Groq CLI assistant, copy `.env.example` to `.env`, set `GROQ_API_KEY`, and start the API as above (or use the Compose API). Then run `python -m scripts.run_assistant` (interactive) or `python -m scripts.run_assistant --question "What is the on-time rate for Shanghai to Rotterdam?"` from an environment with project dependencies installed.

Standalone data-quality checks can be run independently of the pipeline:

```bash
python dq_check.py --input ./data/raw
```

The API is intended to be reachable in under ten minutes on a standard laptop, subject to the first Docker image build downloading its Python dependencies.

## API service — Part 1.1

The FastAPI application is created in `app/api/main.py`. Route handlers are intentionally thin: request validation and HTTP status mapping live at the API boundary; SQL access is delegated to `ShipmentRepository`; curated transformations and route aggregations remain in SQLite's `serving_*` views. Every request is logged as one JSON record with method, path, status code, latency and request ID.

| Endpoint | Behaviour | Main status codes |
|---|---|---|
| `GET /health` | Checks database connectivity and confirms the `serving_shipments` and `serving_route_stats` views exist; returns service version | `200`, `503` |
| `GET /shipments/{shipment_id}` | Returns all curated shipment columns for one ID | `200`, `404`, `503` |
| `GET /shipments` | Lists shipments with `origin`, `destination`, `status`, inclusive `date_from` / `date_to` booking-date filters, `limit` and `offset` | `200`, `422`, `503` |
| `GET /routes/{origin}/{destination}/stats` | Reads pre-aggregated average delay, on-time rate and shipment count from `serving_route_stats` | `200`, `404`, `503` |
| `POST /predict-delay` | Loads the startup-resident Logistic Regression artifact and scores booking-time-only features for a shipment; response marks the model experimental | `200`, `404`, `422`, `503` |
| `GET /data-quality/report` | Serves the standalone Part 2.1 JSON report | `200`, `503` |

### API design and performance decisions

- The API connects to SQLite in read-only mode, once per request, and closes the connection after the response.
- All user-supplied SQL values are bound parameters. Dynamic SQL is limited to fixed, code-authored filter clauses; user input is never interpolated into SQL.
- Shipment listing performs a filtered count and page query against `serving_shipments`. The route endpoint reads the already-aggregated `serving_route_stats` view; it does not join or aggregate raw tables in Python.
- Date filters apply to `booking_date`, are inclusive at both ends, and reject a `date_from` later than `date_to`. This is the documented interpretation of the assignment's unspecified date field.
- Pagination defaults to 50 rows, supports offsets, and caps `limit` at 500 to prevent unbounded responses.
- Error responses use a consistent `error.code` and `error.message` structure. Invalid request parameters return `422`; missing shipment/route resources return `404`; unavailable database/model inference returns `503`.
- `/predict-delay` loads `artifacts/delay_final.joblib` and its metadata once in the FastAPI lifespan. The repository fetches only booking date, planned schedule, shipment profile, port codes and static port reference attributes. The inference layer derives the same 15-feature allow-list used during training. Actual departure/arrival, status, delay outcomes and event fields are not selected or passed to the model.
- The response includes `model_status=experimental_not_for_operational_decisions`. The low/medium/high bands are display bands at probability cutoffs 0.33 and 0.66, not calibrated business risk thresholds. The model is integrated for assignment completeness, but its weak evaluation means it must not be used for operational decisions.

### API verification status

The focused HTTP/repository/DQ tests currently pass:

```bash
pytest tests/unit/test_api_routes.py tests/unit/test_api_repository.py \
  tests/unit/test_data_quality_api.py tests/unit/test_quality.py -q
```

The full suite passes: **44 passed, 0 skipped**. The integration module starts the FastAPI application with a temporary SQLite database and exercises real HTTP requests for shipment lookup, unknown-ID 404, the DQ report and model prediction. Pipeline tests cover derived fields, conflicting-key quarantine, history tracking and idempotent event replay. A separate smoke run against a freshly built warehouse exercised all six endpoints successfully (including the expected 404). Tests also cover parameterised repository queries, startup artifact loading, booking-time-only feature construction and the assistant's local tool-call loop, guardrails and audit logging. Python compilation and OpenAPI schema generation pass. A live Groq provider request has not been made because it requires a configured API key. Docker image build and Compose startup have not been executed because no container engine is available in the verification environment.

### API implementation map

```text
app/api/
  main.py             # FastAPI app, middleware, exception handlers
  routes.py           # HTTP endpoints; no data transformations
  schemas.py          # Pydantic request/response contracts
  dependencies.py     # Per-request read-only SQLite connection
  repository.py       # Parameterised SQL against serving views
  services.py         # Saved-model inference service
  data_quality.py     # Part 2.1 report endpoint

app/core/
  config.py           # Environment-backed settings
  logging.py          # JSON structured logging
```

## Tech Choices

| Area | Choice | Rationale |
|---|---|---|
| Language | Python | One language across data, API, ML and assistant |
| API | FastAPI | Typed REST API and generated OpenAPI docs |
| Data store | SQLite (`sqlite3`) | Embedded SQL database included with Python; no database server or third-party driver to install |
| Data processing | Pandas | Fits the supplied CSV scale |
| ML | Scikit-learn | Logistic Regression reference; bounded feature/MLP experiments were exploratory and compared with a random 50% no-skill baseline; no resampling |
| Model persistence | Joblib | Simple local model artifact |
| LLM provider | Groq `openai/gpt-oss-20b` | Current Groq production model with local function/tool calling; configurable via `GROQ_MODEL` |
| Tests | Pytest | Lightweight unit and integration testing |
| Packaging | Docker / Compose | Reproducible local startup |
| CI/CD | GitHub Actions (proposed) | Selected for push/PR checks; Option B is documented, no workflow is claimed |

### Data model and pipeline

The pipeline uses four logical layers in one SQLite database. Raw tables preserve the latest source snapshot and ingestion metadata; staging tables maintain current/history records and quarantine ambiguous shipment keys; integration contains curated shipment facts and derived fields; serving views support the API and assistant.

| Layer | Main objects | Grain / purpose |
|---|---|---|
| Raw | `raw_ports`, `raw_shipments`, `raw_port_events` | Source snapshot rows, retained with batch metadata |
| Staging | `staging_ports_current`, `staging_ports_history`, `staging_shipments_current`, `staging_shipments_history`, `staging_shipments_quarantine`, `staging_port_events` | Current and prior reference/shipment versions; conflicting shipment IDs quarantined; events deduplicated by full-payload hash |
| Integration | `integration_shipments` | One current curated row per unambiguous shipment, including delay, on-time, route and transit fields |
| Serving | `serving_shipments`, `serving_route_stats`, `serving_port_events` | Query views consumed by FastAPI and assistant tools |

On the supplied snapshot, 5,028 raw shipment rows contain 8 exact duplicate rows and 20 conflicting shipment-ID groups (40 rows). Exact duplicates are collapsed; every row in a conflicting key group is retained in `staging_shipments_quarantine` and excluded from the current curated shipment product. This prevents arbitrary record selection. The resulting serving layer contains 4,980 shipments and 671 route aggregates. The full table definitions, keys, idempotency rules and run command are documented in [`README_PIPELINE.md`](README_PIPELINE.md).

### AI assistance disclosure

AI coding assistance was used to support implementation drafting, test scaffolding, documentation and code review. Engineering decisions—including the data model, data-quality dispositions, booking-time feature allow-list, temporal evaluation approach, model selection, assistant tool contracts and guardrails—were checked against the assignment requirements and the observed data and test results.

### What changes at 100× data volume?

The supplied data is small (about 5,000 shipments and 25,000 events). At 100×, that is approximately **500,000 shipments and 2.5 million port events**. I would not switch databases based on row count alone: SQLite is more than adequate for this assignment-sized local dataset and removes the need to install a separate database engine. Its trade-off is less analytical-query performance for large scans.

For a 24/7 service at that scale, I would move the **operational serving store to PostgreSQL**. The deciding factors are workload and reliability rather than raw capacity: multiple API instances and concurrent writers, transactional updates, fine-grained row-level concurrency, connection management, mature replication/failover, backups and operational tooling. I would add indexes around shipment IDs, route/date filters and event keys, review partitioning for large time-oriented event tables, and use connection pooling and managed high availability.

I would keep analytical processing separate from the transactional workload where useful: retain immutable/raw history in object storage as partitioned Parquet, and use a dedicated analytical query/processing engine when the data and workload justify it. This is a staged evolution, not a claim that PostgreSQL is automatically faster for every analytical query. The current SQLite implementation is intentionally simple, uses Python standard-library support, and avoids operating a database server for the take-home dataset.

### What is deliberately not included

No Kubernetes, Terraform, Kafka, Spark cluster, managed database, paid vector database, or UI is being introduced for this small dataset and time-boxed assignment. These would add operational complexity without improving the required thin slice. At larger scale, the storage and processing changes above would be evaluated against measured throughput, concurrency and availability needs.

## Data quality summary

### Investigation method: two passes

The DQ investigation is intentionally separated from ingestion and follows a reusable, first-principles sequence:

1. **Pass 1 — file-local profiling:** inspect each raw CSV independently for schema/type problems, nulls and blank values, malformed values, range/domain violations, temporal coherence, duplicate rows/business keys, and inconsistent categorical representations. These checks require no database and can be reused for other datasets by adapting the schema and business rules.
2. **Pass 2 — relational and domain checks:** only after the file-local baseline, compare datasets to test referential integrity and consistency across entities (for example, shipment port codes against the port reference data, and event/shipment relationships). This catches problems that cannot be established by inspecting one file in isolation.

This order avoids conflating a local anomaly with a cross-table defect. Each finding will ultimately record the check, detection rule, affected rows and denominator, severity, recommended disposition (drop/fix/flag/quarantine), and reasoning. The standalone script will run against the original CSVs before ingestion and emit the JSON report consumed by the API.

### Industry-informed check catalogue

The investigation uses established data-testing patterns rather than a list of ad-hoc anomalies:

- **Schema / type integrity:** required columns exist, source types are parseable, and schema drift is visible. This follows Great Expectations' schema Expectations (column existence, type and table shape).
- **Completeness:** measure null and blank rates by column, but decide whether a field is required from its business meaning. A missing free-text note is not equivalent to a missing shipment key.
- **Uniqueness / grain:** test exact-row duplicates separately from business-key uniqueness. Confirm the intended grain first; a repeated event identifier may be legitimate if the source defines events at a finer grain.
- **Validity / domain constraints:** accepted categories, identifier formats, numeric ranges and controlled vocabularies. Preserve raw values before normalizing.
- **Referential integrity:** verify foreign-key-like references against the authoritative parent dataset (for example, shipment port codes against `ports.port_code`).
- **Consistency / business rules:** compare related columns in a row (date ordering, positive container counts, non-negative weights) and related entities across tables.
- **Freshness / volume / distribution:** useful in continuously refreshed production pipelines. For this static take-home snapshot, freshness is not a meaningful defect unless a delivery-time expectation is defined; row-count and distribution baselines are still useful diagnostics.

These dimensions map closely to commonly used industry tooling: dbt's built-in `unique`, `not_null`, `accepted_values`, and `relationships` data tests; Great Expectations' schema and uniqueness Expectations; and Soda's schema, validity, missingness, duplicate and freshness checks. We implement the underlying checks directly in Python/Pandas for this small standalone CSV exercise rather than adding a framework dependency. The industry tools inform the test categories; they do not determine the business rules or the severity.

**References:** [dbt data tests](https://docs.getdbt.com/docs/build/data-tests), [Great Expectations — schema validation](https://docs.greatexpectations.io/docs/reference/learn/data_quality_use_cases/schema/), [Great Expectations — uniqueness](https://docs.greatexpectations.io/docs/reference/learn/data_quality_use_cases/uniqueness/), [Soda — checks overview](https://docs.soda.io/soda-documentation/soda-v3/learning-resources/product-overview), [AWS Prescriptive Guidance — data quality checks](https://docs.aws.amazon.com/prescriptive-guidance/latest/modern-data-centric-use-cases/data-quality-checks.html).

### Standalone DQ implementation (Part 2.1)

The independent implementation is in `app/quality/checks.py`, exposed by the root-level `dq_check.py` CLI. It reads the three original CSVs directly with Pandas; it does not import or call SQLite, ingestion, or transformations. It never mutates the input files. Each finding includes:

- stable check name and dataset;
- affected row count, dataset denominator, and percentage;
- severity (`critical`, `warning`, or `info`);
- detection rule;
- recommended action and suggested disposition.

The JSON report is written to `reports/data_quality_report.json`. The FastAPI route `GET /data-quality/report` serves this persisted report and generates it from raw CSVs if the report is absent.

Run it from the repository root:

```bash
python dq_check.py --input ./data/raw --output ./reports/data_quality_report.json
```

The default paths are `data/raw` and `reports/data_quality_report.json`, so this is also valid:

```bash
python dq_check.py
```

### Executed report — supplied raw snapshot

The first run against the supplied files executed **38 checks**: **20 checks reported one or more affected rows** (5 critical, 14 warning, 1 informational). Zero-row checks are retained in JSON so reviewers can see which controls were executed and passed, rather than seeing only exceptions.

| Dataset / check | Rows affected | Percentage | Severity | Recommended handling |
|---|---:|---:|---|---|
| `ports.missing_country` | 1 / 25 | 4.00% | warning | Flag for reference-data enrichment; do not infer |
| `shipments.exact_duplicate_rows` | 8 / 5,028 | 0.16% | warning | Deduplicate only with an audit copy |
| `shipments.missing_actual_departure` | 349 / 5,028 | 6.94% | warning | Flag; resolve against lifecycle/status before imputing |
| `shipments.missing_actual_arrival` | 349 / 5,028 | 6.94% | warning | Flag; resolve against lifecycle/status before imputing |
| `shipments.missing_cargo_type` | 15 / 5,028 | 0.30% | warning | Request source correction; do not infer |
| `shipments.missing_weight_tons` | 73 / 5,028 | 1.45% | warning | Request source correction; do not impute without a rule |
| `shipments.missing_status` | 54 / 5,028 | 1.07% | warning | Reconcile lifecycle state; preserve raw value |
| `shipments.repeated_shipment_id` | 56 / 5,028 | 1.11% | critical | 28 key groups: quarantine 40 rows across 20 conflicting groups; collapse 8 exact duplicate copies; retain raw snapshot |
| `shipments.negative_weight` | 36 / 5,028 | 0.72% | warning | Quarantine or correct from source |
| `shipments.non_positive_container_count` | 23 / 5,028 | 0.46% | warning | Flag; confirm zero-count semantics |
| `shipments.cargo_type_noncanonical_format` | 261 / 5,028 | 5.19% | warning | Normalize only with an approved mapping; retain raw |
| `shipments.status_noncanonical_format` | 58 / 5,028 | 1.15% | warning | Normalize formatting only under explicit lifecycle mapping |
| `shipments.actual_departure_before_booking` | 39 / 5,028 | 0.78% | warning | Flag and verify timestamp semantics/timezones |
| `shipments.origin_port_not_in_ports_reference` | 42 / 5,028 | 0.84% | critical | Quarantine/correct unresolved origin references |
| `shipments.destination_port_not_in_ports_reference` | 55 / 5,028 | 1.09% | critical | Quarantine/correct unresolved destination references |
| `shipments.any_port_reference_unresolved` | 95 / 5,028 | 1.89% | critical | Unique shipment rows failing either port lookup |
| `port_events.missing_event_type` | 140 / 25,000 | 0.56% | warning | Flag for source correction; retain useful event payload |
| `port_events.repeated_event_id` | 206 / 25,000 | 0.82% | critical | 103 groups conflict; event_id is not a safe deduplication key |
| `port_events.missing_notes` | 1,913 / 25,000 | 7.65% | info | Treat as optional descriptive text unless business use requires it |
| `timestamps.timezone_metadata_not_declared` | 49,442 / 49,442 timestamp values | 100.00% | warning | Establish UTC vs local-time semantics at source boundary |

The JSON also records checks with zero findings: required-column presence, timestamp parsing, unique port codes, exact duplicate event rows, event-to-port reference integrity, and negative event delays, among others. The exact set and definitions are in the generated report.

**Interpretation and overlap safeguards**

- The 349 missing actual departure values and 349 missing actual arrival values occur on the same shipment rows. They are separate field-level completeness checks, not 698 unique shipments.
- The 8 exact duplicate shipment rows are included in the repeated `shipment_id` population. Of 28 repeated-key groups, 20 groups contain conflicting payloads (40 rows) and are quarantined; the other 8 groups contain exact duplicate pairs (16 rows), from which 8 extra copies are collapsed. The 56 participating rows are not an additional disjoint population.
- The origin and destination reference checks overlap; the 95-row union is the unique shipment-level impact. Do not add 42 + 55 to estimate unique shipments.
- The 206 event rows belong to 103 repeated `event_id` groups, and each repeated group has conflicting payloads. Keep distinct event payloads; do not deduplicate on `event_id`.
- `COMPLETE` and `COMPLETED` are not merged: the source lifecycle semantics are unknown. The formatting check measures case/whitespace convention only.
- Missing actual milestones are not automatically imputed or declared defects in lifecycle terms; the report flags them for status-aware review.
- Early actual arrival/departure relative to plan is not inherently a defect. Only explicit chronology inversions are flagged.
- The timestamp strings are parseable but timezone-naive. The 49,442 denominator is the number of populated timestamp values across shipment and event timestamp columns, not a count of distinct rows.
- The assignment says there are twelve intentional embedded issues but does not enumerate them. This report records all reproducible observations and additional controls found in the supplied snapshot; findings are not claimed to map one-to-one to the twelve intentional issues. Some findings are overlapping views of the same records, while others are ambiguities requiring source-owner confirmation.



## ML — Part 3.1 delay prediction

### Prediction definition and grain

The model predicts at **booking time** whether a shipment's recorded **destination arrival** will occur more than 24 hours after its planned destination arrival:

```text
is_late_over_24h = actual_arrival > planned_arrival + 24 hours
```

The positive class is 1 (more than 24 hours late); the negative class is 0 (on time or no more than 24 hours late). A missing `actual_arrival` is an unknown outcome, not a negative label. The prediction is shipment-to-destination level. The supplied shipment schema has no intermediate stop schedule, leg identifier, or stop-level actual milestone, so the model does not claim to predict delay at intermediate vessel calls.

### Source investigation before training

The raw-data investigation is reproducible with:

```bash
python scripts/profile_ml_data.py --data-dir ./data/raw --output ./reports/ml_data_investigation.json
```

Key findings and decisions:

- **349 shipments have no `actual_arrival`.** We checked `port_events.csv` for `ARRIVAL` events matching the same `vessel_id` and destination `port_code`: 286 of the 349 have at least one coincident event, but only 57 have such an event between 14 days before and 30 days after planned arrival. These matches are not shipment evidence: the event table has no `shipment_id` or leg identifier, matching vessel/port events can be repeated (up to 8 matching events per missing shipment), and all 82 event vessels have events at all 25 reference ports. We therefore cannot assign a unique event timestamp to a shipment and do **not** impute its arrival or target label.
- **Missing target handling:** all 349 raw missing-arrival rows are excluded from supervised labels. One of the 349 rows is an exact duplicate copy removed during deduplication, two are among the 40 rows quarantined for ambiguous repeated shipment IDs, and 346 remain as additional missing-target rows after those earlier steps. The report preserves these overlapping counts so exclusions are not double-counted.
- **Intermediate stops:** `shipments.csv` contains origin and destination only; it has no intermediate-stop sequence or leg key. `port_events.csv` is a port-level vessel event feed and has no shipment/leg key. Assumption: predict destination arrival delay only. We do not infer or model intermediate stops.
- **Status leakage:** among labelled rows, all 1,901 `DELAYED` records are positive and all 2,778 `DELIVERED` records are negative. `status` is therefore excluded; it reveals the outcome and is not a valid booking-time feature.
- **Duplicate and reference handling:** collapse 8 exact duplicate rows; quarantine 20 remaining repeated-ID groups (40 rows) because their shipment records conflict; exclude 89 otherwise-labelled rows whose origin or destination port is not in `ports.csv`.
- **Invalid numeric predictors:** negative weights and non-positive container counts are converted to missing values (31 and 21 rows respectively after eligibility filtering) and imputed inside the fitted training pipeline. They are not used as legitimate numeric values.

The machine-readable evidence is in `reports/ml_data_investigation.json`; the model preparation counts are repeated in `reports/model_evaluation.json`.

### Booking-time feature selection

The model uses an explicit allow-list of 15 predictors. Feature engineering is implemented in `app/ml/features.py`; inference must use the same feature order and preprocessing contract.

| Feature group | Features | Rationale |
|---|---|---|
| Route identity | `origin_port`, `destination_port` | Port and route-specific patterns available from the booking record |
| Port reference | `origin_region`, `destination_region`, `origin_congestion_score`, `destination_congestion_score` | Static reference attributes from `ports.csv`; congestion is a long-run average proxy, not live congestion |
| Planned schedule | `planned_transit_days`, `booking_lead_days` | Schedule duration and time between booking and planned departure |
| Calendar | `booking_month`, `booking_day_of_week`, `planned_departure_month`, `planned_departure_day_of_week` | Seasonality and weekday patterns available from booking/schedule timestamps |
| Shipment attributes | `container_count`, `weight_tons`, `cargo_type` | Booking-time operational/cargo attributes |

We initially exclude `customer_id` and `vessel_id`: they are high-cardinality identifiers with memorisation risk, and vessel assignment timing/reliability at booking has not been established. We exclude all actual milestones, actual transit duration, `status`, shipment ID and raw event-level records. Raw port-event rows cannot be attributed to a shipment or leg without a reliable key. A separate experiment derives only port-level aggregates strictly before booking; these are contextual features, not shipment-specific event attribution.

Preprocessing is fitted only on the training partition: numeric median imputation plus missing indicators; categorical `UNKNOWN` imputation and one-hot encoding; numeric scaling for Logistic Regression. No SMOTE, over/undersampling, focal loss or class weighting is applied in the reference Logistic Regression or experimental MLP. The cleaned labelled population remains moderately balanced (40.4% positive).

### Data summary and categorical encoding decisions

Run a reproducible profile of all three source files and the prepared model population:

```bash
python scripts/summarize_ml_data.py --data-dir ./data/raw --output ./reports/ml_data_summary.json
```

The supplied sources contain 5,028 shipment rows, 25 port reference rows, and 25,000 port-event rows. The shipment file has 28 distinct origin and destination codes before reference validation; the model's eligible population uses 25 valid port codes at both origin and destination. The cleaned model frame has 4,545 rows and 599 distinct origin/destination pairs. After formatting normalization, `cargo_type` has 10 categories.

| Categorical feature | Distinct values in eligible model frame | Encoding |
|---|---:|---|
| `origin_port` | 25 | One-hot |
| `destination_port` | 25 | One-hot |
| `cargo_type` | 10 | One-hot |
| `origin_region` | 4 | One-hot |
| `destination_region` | 4 | One-hot |

Port codes are nominal identifiers, not ordinal measurements. Label-encoding ports as integers (for example, assigning one port 1 and another 20) invents an order and distances that do not exist in the source. The current reference pipeline therefore retains `OneHotEncoder(handle_unknown="ignore")`. We separately ran the requested ordinal-encoding experiment for categories with more than three distinct values; its results and limitations are documented below. To model geographic proximity legitimately, we would need port latitude/longitude or an authoritative route-distance matrix; neither is supplied. The full per-column missingness, cardinality, value counts and numeric summaries are stored in `reports/ml_data_summary.json`.

### Model scope and baseline

The reference modelling workflow is deliberately narrow: Logistic Regression compared with a reproducible random 50% predictor. A single small MLP is retained as an isolated architecture experiment, not as a competing production model. We are not treating additional algorithms as a substitute for understanding the feature signal. The random baseline's expected ROC-AUC is 0.5; expected average precision is the positive-class prevalence (approximately 0.397 in validation). A seeded random-score realisation is included in the JSON report for reproducibility, but its single-run metrics naturally fluctuate.

We do not apply oversampling, undersampling, SMOTE, focal loss or class weighting initially. The cleaned population has approximately 40.4% positive labels, which is not sufficiently imbalanced to justify these interventions before measuring their need.

### Model-development record and final selection

The assignment asks for a booking-time model, honest precision/recall/F1/AUC reporting, leakage-safe features and a saved artifact. We treated this as a model-selection exercise rather than a search for the most complex algorithm. The chronological split is fixed at 3,181 training rows, 675 validation rows and 689 test rows. Validation has 39.70% positive labels; expected no-skill average precision is therefore 0.397, while expected ROC-AUC for random scores is 0.5.

The main experiments and what they established are summarised below. Unless explicitly stated, metrics are from the same 675-row chronological validation window. The 0.5 classification threshold is shown for predicted-class diagnostics; we did not tune it to manufacture a better result.

| Experiment | Validation ROC-AUC | Average precision | Log loss | Key observation / decision |
|---|---:|---:|---:|---|
| Logistic Regression, 15 features, one-hot | 0.481 | 0.386 | 0.689 | Reference model; predicts 74 late vs 268 actual; recall 0.104 |
| Feature groups / leave-one-group-out | Best 0.492 (port context only) | 0.403 | 0.673 | No tested group or ablation beats chance ROC-AUC |
| Cyclical calendar encoding | 0.489 (schedule-only) | 0.393 | 0.677 | No consistent gain; not adopted |
| Ordinal encoding for categories with >3 values | 0.484 | 0.397 | 0.678 | Only 6 positive predictions; not adopted |
| Mutual-information selection (3 features) | 0.475 | 0.402 | 0.674 | Predicts zero positives at threshold 0.5; rejected |
| One-hidden-layer MLP (4 units), base features | 0.490 | 0.398 | 0.763 | Recall rises to 0.272, but probability quality worsens |
| MLP (4 units) + origin-port event context, final bounded rerun | 0.493 | 0.410 | 0.762 | Best NN experiment, but ROC-AUC remains below chance and log loss is high |
| MLP (8 units) + origin-port event context | 0.472 | 0.374 | 0.918 | More positive predictions/recall, but worse ranking and log loss; rejected |

**Feature selection methodology.** Mutual information was calculated on training data only. The candidate feature count was compared using three expanding temporal folds inside the training partition. The selected three features were `planned_departure_month`, `weight_tons` and `planned_departure_day_of_week`. Their mean inner-fold ROC-AUC was 0.511, but individual folds ranged from 0.472 to 0.543 (standard deviation 0.029), and the later validation ROC-AUC was 0.475. The small inner-CV mean did not generalise to the next time period, so the selected subset was rejected.

**Encoding experiment.** The one-hot preprocessing produced 80 columns with 21.25% density on the training data. The requested mixed ordinal encoding reduced this to 17 dense columns, but ordinal values impose a false order on nominal port, cargo and region categories. Its validation ROC-AUC was only 0.484 and it predicted just 6 positive shipments. The selected-feature/ordinal combination predicted zero positives. Sparse one-hot encoding is not a meaningful bottleneck at 80 columns; Scikit-learn Logistic Regression handles sparse matrices natively. We therefore retain one-hot encoding.

**Neural-network progression.** Logistic Regression was treated as the linear/sigmoid reference. We then trained a deliberately minimal MLP with one hidden layer of four ReLU units and one sigmoid output, rather than starting with a deep network. A single bounded capacity check increased the hidden layer from four to eight units. The 8-unit network increased recall by predicting more shipments as late, but ROC-AUC, average precision and log loss all worsened. We stopped architecture exploration. SVMs were also explored earlier, but did not establish a better candidate and are not part of the retained workflow; no SVM metric is claimed in the final comparison table.

**Point-in-time event features.** Because `port_events.csv` has no shipment or voyage-leg key, we did not join events to individual shipments. We derived port-level trailing 7-day/30-day context using only events with `event_timestamp < booking_date`, after quarantining 206 rows in conflicting repeated event-ID groups. Origin-only context gave the best MLP experiment (the final bounded rerun is shown above); destination or combined context did not provide a consistent improvement. An exploratory univariate origin 30-day delayed-event count had validation ROC-AUC 0.543, but this single-feature association did not translate into robust combined-model discrimination. Event timestamps are timezone-naive and the event feed's operational meaning and completeness are not established, so these features remain experimental.

**Final model decision.** Retain `artifacts/delay_final.joblib` (Logistic Regression with train-fitted preprocessing and the 15 booking-time features) as the reference model artifact and API candidate. It is selected for simplicity, reproducibility and a stable baseline—not because it is accurate or production-ready. On validation it has precision 0.378, recall 0.104, F1 0.164 and ROC-AUC 0.481; on the previously inspected test period ROC-AUC is 0.495 and average precision 0.412. The best MLP experiment still has validation ROC-AUC 0.493 and log loss 0.762. Neither model demonstrates reliable discrimination or probability quality. **No delay model from these experiments is approved for operational decisions.** The `/predict-delay` endpoint is wired to the startup-loaded reference artifact for assignment completeness, but its response is explicitly marked experimental and must not be used for operational decisions.

The validation window has been reused during feature and architecture experiments, and the test period was inspected in earlier runs. These results are useful diagnostics for this assignment, not an unbiased final estimate of future performance. All detailed metrics, split metadata and experiment diagnostics are retained in the corresponding JSON reports listed in the README sections below.

### Feature-signal investigation

Run the validation-only feature-group analysis:

```bash
python scripts/analyze_feature_signals.py --data-dir ./data/raw --output ./reports/feature_signal_analysis.json
```

We tested the same Logistic Regression pipeline across individual feature groups, the full feature set, and leave-one-group-out ablations. Every preprocessing step (median imputation, missing indicators, scaling and one-hot encoding) is fitted on the training partition only. Feature-group selection uses the chronological validation partition; no test metrics are used for selection.

| Feature configuration | Features | Validation ROC-AUC | Average precision | Log loss |
|---|---:|---:|---:|---:|
| Port context only | 4 | 0.492 | 0.403 | 0.673 |
| Schedule/calendar, cyclical | 10 | 0.489 | 0.393 | 0.677 |
| Full set, cyclical calendar | 19 | 0.478 | 0.384 | 0.691 |
| Route identity only | 2 | 0.491 | 0.385 | 0.682 |
| Without schedule/calendar | 9 | 0.487 | 0.385 | 0.688 |
| Without shipment profile | 12 | 0.486 | 0.384 | 0.683 |
| Schedule/calendar only | 6 | 0.482 | 0.400 | 0.674 |
| Full feature set | 15 | 0.481 | 0.386 | 0.689 |
| Without port context | 11 | 0.481 | 0.385 | 0.689 |
| Without route identity | 13 | 0.478 | 0.396 | 0.680 |
| Shipment profile only | 3 | 0.477 | 0.390 | 0.678 |

**Finding:** no tested feature configuration beats the 0.5 chance ROC-AUC on the validation window. Port context is the least weak group, but 0.492 is not useful discrimination and should not be presented as predictive signal. Adding all feature groups does not improve results; the full feature set scores below several smaller groups. The random baseline has expected ROC-AUC 0.5 and expected average precision equal to validation prevalence (0.397). The feature analysis therefore does not justify promoting any current feature group as useful.

The experiment is exploratory and based on one chronological validation window. Small differences around 0.5 are noise-sensitive. The chronological test period had already been inspected in earlier experiments, so it is not described as a pristine untouched holdout. The ablation report deliberately does not use test results to select features. Full results are in `reports/feature_signal_analysis.json`.

### Feature selection and ordinal-encoding experiment

Run the controlled experiment with:

```bash
python scripts/run_feature_selection_experiment.py
```

The experiment keeps the same Logistic Regression model family and compares: (1) the current all-one-hot full feature set, (2) ordinal encoding for categorical columns with more than three distinct values, and (3) mutual-information feature selection combined with that mixed encoder. Feature ranking is fitted only on the training partition. The number of selected features is chosen by three expanding temporal folds within the training partition; the final model is refitted on all training rows and evaluated on validation. The test period is not used.

| Validation configuration | ROC-AUC | Average precision | Log loss | Predicted positives (of 675) |
|---|---:|---:|---:|---:|
| All one-hot, all 15 features | 0.481 | 0.386 | 0.689 | 74 |
| Ordinal for categories with >3 values, all 15 features | 0.484 | 0.397 | 0.678 | 6 |
| Mutual-information selected 3 features + mixed encoding | 0.475 | 0.402 | 0.674 | 0 |
| Random 50% baseline (seed 42; one noisy realization) | 0.454 | 0.363 | 1.063 | 344 |

Validation contains 268 positive and 407 negative examples. The random baseline row is one seeded realization; its expected ROC-AUC is 0.5 and expected average precision is 0.397 (the validation prevalence), so its single-run 0.454 / 0.363 values should not be over-interpreted. The no-skill references are ROC-AUC 0.5 and average precision 0.397 (the validation prevalence). The ordinal full-feature run has a slightly higher ROC-AUC and average precision than all-one-hot, but both ROC-AUCs remain below chance and the ordinal run predicts only six positives. The selected-feature run has average precision only marginally above prevalence, ROC-AUC below chance, and predicts **zero positive cases** at threshold 0.5. It is not a viable model.

The inner temporal CV selected k=3, with features `planned_departure_month`, `weight_tons` and `planned_departure_day_of_week`. Its mean ROC-AUC was 0.511 across folds, but fold scores ranged from 0.472 to 0.543 (standard deviation 0.029); the later validation period then scored 0.475. This is a clear example of why feature selection must be checked on a later period and why a slightly-above-chance CV average is not enough evidence of signal. The selected features contain no categorical columns, so ordinal encoding does not contribute to that selected-feature result.

**Does one-hot sparsity explain the weak model?** Not convincingly for this dataset. On the 3,181-row training set, the one-hot transformed matrix is only 80 columns wide (density 21.25%); the mixed ordinal matrix is 17 columns wide (density 100%). Scikit-learn Logistic Regression handles sparse matrices directly, and 80 columns is not a high-dimensional problem at this data scale. Ordinal encoding reduces the number of columns, but imposes an artificial order on nominal port, cargo and region categories. In this run, it also caused the model to predict almost no late shipments. The results do not justify replacing the one-hot reference pipeline.

**Decision:** keep the existing all-one-hot Logistic Regression as the reproducible reference artifact. Keep the feature-selection/ordinal experiment report and artifact separate for review, but do not promote either experimental pipeline to the API or claim improved predictive performance. Full results, including per-fold scores, MI rankings, class counts and matrix profiles, are in `reports/feature_selection_experiment.json`.


### Minimal neural-network experiment: one step beyond Logistic Regression

The user-requested architecture experiment treats Logistic Regression as the linear baseline: an affine combination of inputs followed by a logistic probability output. Rather than jumping to a deep network, the experiment adds exactly **one hidden layer with four ReLU units**, then retains a single binary probability output. The four hidden activations are learned nonlinear combinations of the 80 transformed input dimensions (15 source features after train-fitted imputation, scaling and one-hot encoding).

```text
15 booking-time source features
          |
train-fitted preprocessing
          |
80 transformed dimensions
          |
Dense(4, ReLU)       # the only added representation layer
          |
Dense(1, sigmoid)    # binary probability output
```

Configuration: Adam, initial learning rate 0.001, L2 penalty (`alpha=0.01`), random seed 42, maximum 2,000 epochs, no random early-stopping split. The model converged in 1,271 iterations without a convergence warning. Preprocessing was fitted on the chronological training partition only. The existing Logistic Regression artifact was not overwritten; the MLP is saved separately as `artifacts/delay_mlp_experiment.joblib`.

| Validation metric | Logistic Regression | 1-layer MLP (4 units) |
|---|---:|---:|
| ROC-AUC | 0.481 | 0.490 |
| Average precision | 0.386 | 0.398 |
| Log loss | 0.689 | 0.763 |
| Predicted positives at 0.5 | 74 | 204 |
| Actual positives | 268 | 268 |
| Recall at 0.5 | 0.104 | 0.272 |

The MLP makes more positive predictions and improves validation recall, but ROC-AUC remains below the 0.5 no-skill reference, average precision is essentially equal to the 0.397 positive prevalence, and log loss is materially worse. Its higher recall is accompanied by 131 false positives versus 46 for Logistic Regression. On the already-inspected test period, MLP ROC-AUC is 0.486 and AP is 0.396, compared with Logistic Regression's 0.495 and 0.412; test is reported only for transparency and not used for selection.

**Conclusion:** the one-hidden-layer network is a valid minimal nonlinear extension, but this experiment does not show that nonlinear representation learning recovers predictive signal from the current features. Keep Logistic Regression as the reference artifact; retain the MLP only as a documented experiment. Do not increase layer count or width merely to chase metrics. The next useful improvement is likely new booking-time information (especially carefully constructed pre-booking port-event context), not a larger network.

Run it with:

```bash
python scripts/run_neural_network_experiment.py
```

The complete architecture, training diagnostics, validation/test metrics, class distributions and random baseline are recorded in `reports/neural_network_experiment.json`.

### Point-in-time port-event context experiment

The shipment and event files overlap in time, and the event feed contains 25,000 records. Because the event table has no `shipment_id` or voyage-leg key, these records cannot be joined to an individual shipment. We tested only **port-level contextual aggregates**, separately for the shipment origin and destination ports. Every aggregate is calculated from events strictly earlier than that shipment's `booking_date`; events at the exact booking timestamp and all later events are excluded.

Feature windows are trailing 7 and 30 days. For each side/window we derive event count, mean `delay_minutes`, count of `event_type == DELAYED`, and delayed-event rate. Two additional fields describe the fraction of the 7/30-day window elapsed since the first retained event in the file. Those fractions measure observable file history only; they do not prove source completeness before the file begins.

**Event-source handling:** 103 repeated `event_id` groups (206 rows) have conflicting payloads, so every row in those groups is quarantined rather than choosing one arbitrarily. The remaining 24,794 rows are used. The 138 retained rows with missing `event_type` still contribute to generic event counts and delay aggregates, but not to the `DELAYED` event count. Event timestamps range from 2024-06-01 to 2025-10-14 and are timezone-naive; comparisons use the source timestamps as supplied, without assuming UTC or converting time zones.

We compared the same 4-neuron, one-hidden-layer MLP with base features, origin-only event context, destination-only event context and all event context. Logistic Regression is included as the linear reference. All preprocessing is fitted on the chronological training partition; results below are validation-only (675 rows, 268 positives). The random baseline's expected ROC-AUC is 0.5 and expected average precision is validation prevalence (0.397).

| Validation configuration | ROC-AUC | Average precision | Log loss | Recall | Predicted positives |
|---|---:|---:|---:|---:|---:|
| Logistic Regression, base | 0.481 | 0.386 | 0.689 | 0.104 | 74 |
| Logistic Regression + origin events | 0.466 | 0.387 | 0.696 | 0.127 | 97 |
| Logistic Regression + origin and destination events | 0.459 | 0.382 | 0.698 | 0.123 | 97 |
| MLP (4 units), base | 0.490 | 0.398 | 0.763 | 0.272 | 204 |
| MLP (4 units) + origin events | **0.497** | **0.413** | **0.758** | 0.261 | 171 |
| MLP (4 units) + destination events | 0.477 | 0.389 | 0.776 | 0.213 | 164 |
| MLP (4 units) + origin and destination events | 0.492 | 0.391 | 0.774 | 0.325 | 222 |

The best result is the MLP with **origin-only** event context: average precision rises from 0.398 to 0.413, and precision at threshold 0.5 is 0.409. However, ROC-AUC is still 0.497 (below the 0.5 no-skill reference), and log loss remains high at 0.758. Adding destination context does not help; adding all port events increases recall but reduces average precision and worsens log loss. Logistic Regression does not benefit from the new event features in this run.

As an exploratory diagnostic (not feature selection), individual origin-event features have validation ROC-AUCs around 0.512–0.543; the strongest are 30-day delayed-event count (0.543), 30-day delayed-event rate (0.541) and 7-day delayed-event count (0.541). This suggests a weak origin-port context association, but the combined model does not yet turn it into reliable ranking performance. A single chronological validation window is not sufficient to claim a robust effect.

**Decision:** retain the origin-event MLP only as a separate experiment (`artifacts/delay_mlp_4_units_plus_origin_events_experiment.joblib`). Do not replace `artifacts/delay_final.joblib`, wire event features into API inference, or claim production readiness. The full report, including feature definitions, point-in-time audit, univariate diagnostics, model metrics and predicted-class distributions, is `reports/port_event_feature_experiment.json`.

Run the experiment with:

```bash
python scripts/run_port_event_feature_experiment.py
```

### What the feature results tell us

- **Schedule/calendar:** weak alone (ROC-AUC 0.482). Replacing integer month/weekday values with sine/cosine pairs raises schedule-only validation ROC-AUC slightly to 0.489, still below chance. Applying cyclical encoding to the full feature set gives 0.478, so this is not a useful improvement yet.
- **Route identity:** origin/destination one-hot categories alone are weak (0.491). There are 599 distinct route pairs across 4,545 eligible rows, so route-level patterns are sparse. One-hot encoding individual origin and destination ports is preferable to one-hot encoding all 599 route combinations.
- **Port context:** the strongest single group (0.492), but still below chance. Static average congestion and broad region may be too coarse to represent actual conditions at booking.
- **Shipment profile:** weakest standalone group (0.477). Container count, weight and cargo type alone do not discriminate late arrivals in this validation window.
- **Full model:** combining every feature group did not help (0.481), suggesting noise/redundancy rather than complementary signal in this sample.

Port codes remain nominal. The >3-cardinality ordinal-encoding experiment did not demonstrate a useful improvement; keep one-hot encoding as the reference until defensible geographic or route features are sourced.

### Current single-model evaluation

Run the summary, feature analysis and single-model training:

```bash
python scripts/summarize_ml_data.py
python scripts/analyze_feature_signals.py
python scripts/run_feature_selection_experiment.py
python scripts/run_neural_network_experiment.py
python scripts/run_port_event_feature_experiment.py
python -m app.ml.train
pytest -q
```

The training script saves the Logistic Regression reference model as `artifacts/delay_final.joblib` with `artifacts/delay_model_metadata.joblib`. The feature-selection and MLP experiment artifacts are separate and are not the API model. `reports/model_evaluation.json` contains actual and predicted class counts, classification metrics and the random baseline comparison.

| Partition | Rows | Positive rate | Booking-date range |
|---|---:|---:|---|
| Train | 3,181 | 40.71% | 2024-06-01 to 2025-05-18 |
| Validation | 675 | 39.70% | 2025-05-18 to 2025-08-03 |
| Test | 689 | 39.62% | 2025-08-03 to 2025-10-14 |

At threshold 0.5, the full-feature Logistic Regression validation result is precision 0.378, recall 0.104, F1 0.164, ROC-AUC 0.481 and average precision 0.386. It predicts 74 positives among 675 validation rows, compared with 268 actual positives. The test result is ROC-AUC 0.495 and average precision 0.412. Neither partition demonstrates useful ranking performance. The random baseline's expected ROC-AUC is 0.5; the model does not beat it.

**Decision:** retain Logistic Regression as the sole model implementation for reproducibility and future feature work, but mark it not production-ready. Do not tune the threshold to disguise weak ranking performance. A threshold changes the precision/recall operating point; it cannot create missing ranking signal.

### Next feature investigations

The next experiments should add or improve information, not add more algorithms:

1. Validate the weak origin-port event association on a new, untouched future cohort or additional historical data; the current test window has already been inspected.
2. Investigate event-feed semantics and timestamp timezone provenance with the data owner before treating port aggregates as operational signals.
3. Test defensible route features only if port coordinates or reliable route-distance data can be sourced; do not invent port proximity from port-code integers.
4. Inspect feature distributions and target rates by booking-time cohort for temporal drift, data-generation artefacts or label construction issues. Do not deepen the network without evidence of additional signal.

### Production monitoring plan

If the model were deployed after obtaining a stronger signal and business approval, I would monitor:

- **Input/data quality:** missingness, unknown categories, invalid numeric values, feature ranges, port-reference coverage and schema drift.
- **Prediction behaviour:** score distribution, predicted high-risk rate, feature drift (for example PSI), and changes by route/port/cargo segment.
- **Outcome performance:** once arrivals mature, precision, recall, F1, ROC-AUC and PR-AUC by booking cohort; calibration and false-negative rate are important for operational use.
- **Operational health:** inference latency, error rate, artifact version, request volume and model load failures.

Trigger investigation/retraining when confirmed outcome metrics degrade beyond an agreed baseline (for example ROC-AUC drops by 0.05 or recall breaches an agreed service threshold across a sufficiently large cohort), or when sustained feature drift/data-contract changes are detected. Retraining must use a fresh chronological holdout and must not be triggered by drift alone without checking label quality. The present artefact is for assignment demonstration only and should not drive shipment decisions.

### Final bounded neural-network capacity check: 4 vs 8 hidden units

To close the neural-network architecture question without an open-ended model search, we ran one deliberately bounded capacity step: the same one-hidden-layer MLP with 4 versus 8 ReLU units, both using the same 25 booking-time inputs (the original 15 features plus 10 strictly pre-booking origin-port event context features). The network still has one output probability. Preprocessing is fitted on the chronological training partition only. Both models use Adam (0.001), L2 alpha 0.01, seed 42, and no random early-stopping split. We did not inspect the already-used test partition again.

| Validation metric (n=675) | MLP, 4 units + origin events | MLP, 8 units + origin events |
|---|---:|---:|
| ROC-AUC | **0.493** | 0.472 |
| Average precision | **0.410** | 0.374 |
| Log loss | **0.762** | 0.918 |
| Predicted positives at 0.5 | 171 | 263 |
| Recall | 0.257 | **0.366** |

The current reproducible rerun's 4-unit figures differ slightly from the earlier saved port-event experiment report; this final decision report records the exact current run. Both architectures converged without a convergence warning. The 8-unit model predicts more positives and raises recall, but its ranking metrics and probability loss are worse. It does not beat the 0.5 ROC-AUC no-skill reference or expected random average precision (validation prevalence 0.397).

**Final decision:** do not increase network capacity. Close neural-network architecture exploration at one hidden layer with 4 units as the best small MLP experiment. It remains experimental, not production-ready. Keep Logistic Regression as the reference/API candidate because neither MLP configuration establishes reliable discrimination or probability quality. Do not connect the event-augmented MLP to API inference. The 8-unit artifact is preserved only for reproducibility. The validation period has been reused across experiments, and the chronological test period was previously inspected, so neither is a pristine final estimate.

Reproduce the bounded check with:

```bash
python scripts/finalize_neural_network.py
```

Machine-readable decision, both model metrics, training diagnostics, event audit and promotion rule: `reports/final_neural_network_decision.json`. Experimental 8-unit artifact: `artifacts/delay_mlp_8_units_plus_origin_events_experiment.joblib`.


## GenAI Assistant — Part 3.2

The assistant is a lightweight command-line application using **Groq Chat Completions with local function calling**. It is intentionally a small Python tool dispatcher, not an agent framework or separate MCP service. The model chooses among the three required tools; the application executes the requested function and returns its result to Groq for the final response.

**Provider and model.** Groq model `openai/gpt-oss-20b` is the default (`GROQ_MODEL` can override it). Groq's current published developer pricing lists $0.075 per million input tokens and $0.30 per million output tokens for this model. A short query using roughly 2,500–4,000 input tokens across the tool-call/final-answer turns and 300–500 output tokens is approximately **$0.0003–$0.0005 per query**. This is an estimate, not a fixed fee; actual token usage is captured in the audit log and prices/availability can change. See [Groq supported models and pricing](https://console.groq.com/docs/models) and [Groq local tool calling](https://console.groq.com/docs/tool-use/local-tool-calling).

### Tools exposed to the model

| Tool | Behaviour | Data path |
|---|---|---|
| `query_shipments(filters)` | Lists curated shipments with origin, destination, status and booking-date filters. `aggregate_by_route=true` returns route aggregates for a date window, supporting ranked-route questions. | Read-only SQLite `serving_shipments` |
| `get_route_stats(origin, destination)` | Returns average delay hours, on-time rate and shipment count for one route. Accepts port code or port name. | Read-only SQLite `serving_route_stats` |
| `predict_delay(shipment_id)` | Requests a booking-time delay-risk score for a shipment. | HTTP `POST /predict-delay`; API uses the saved artifact |

The assistant uses the tools only when the model requests them. It does not calculate business metrics from free-form model reasoning; values come from SQLite or the prediction endpoint. For “this quarter” questions, the model is instructed to calculate the calendar-quarter date window from the current date and pass that window to `query_shipments` with `aggregate_by_route=true`. If the requested time window has no source data, the assistant should report that rather than inventing results.

### Two explicit guardrails

1. **Off-domain refusal:** before making any LLM request, a deterministic domain-keyword gate checks whether the question concerns shipments, routes, ports, delays, on-time performance, congestion, cargo, shipping or delay risk. Questions outside that scope are refused locally.
2. **Tool provenance:** factual answers append a `Sources:` footer listing the exact tool names that were executed. The system prompt also instructs the model to cite factual claims inline as `[tool: tool_name]`. If no data tool ran, the response says that no shipment-specific facts were retrieved.

### LLM/tool observability

Every Groq completion is written to `logs/genai_assistant.jsonl` with timestamp, model, full prompt/message history sent, tool definitions made available, returned content/tool-call requests and token usage. Each local tool execution is logged with its name, arguments and response. The final user-facing answer and tool provenance are logged separately. This is deliberately transparent for the assignment; production deployment should restrict access to these logs, apply retention/redaction rules and avoid logging unnecessary personal or commercially sensitive fields.

### Run the assistant

```bash
cp .env.example .env
# Add your Groq API key to .env (GROQ_API_KEY=...)
python -m scripts.run_pipeline
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
# In another terminal:
python -m scripts.run_assistant
# Or one question:
python -m scripts.run_assistant --question "What is the on-time rate for Shanghai to Rotterdam?"
```

The CLI uses the configured `DATABASE_PATH` for read-only shipment/route queries and `ASSISTANT_API_BASE_URL` for the prediction tool. `GROQ_API_KEY`, `GROQ_MODEL`, `GENAI_AUDIT_LOG`, `DATABASE_PATH` and `ASSISTANT_API_BASE_URL` are configurable through environment variables; `.env` is ignored by Git. No live Groq call is made by the unit tests, so tests do not require an API key or incur token charges.

**Why no MCP layer?** Three stable, application-owned functions do not justify another protocol server, deployment unit or discovery layer in a time-boxed take-home. Direct local function calling is simpler to inspect, test and debug, while keeping the tool contracts explicit. If this grew into a production assistant with many independently owned tools, multiple versions, shared integrations or third-party systems, I would evaluate MCP for standardised tool discovery, schemas, versioning and reuse. MCP is a future architectural option, not a dependency needed for this three-tool thin slice.


## If I had more time

For the delay model, I would prioritise better evidence and better booking-time data over trying more algorithms. First, I would obtain a fresh future booking cohort and preserve it as an untouched temporal holdout; the current validation period has been reused and the test period has already been inspected. Second, I would clarify the port-event feed's timestamp timezone, event definitions, `delay_minutes` semantics and source completeness with the data owner, and request a shipment/voyage-leg identifier so event observations can be attributed correctly. Third, I would add trusted booking-time information that is currently absent, such as scheduled vessel/voyage identity with known assignment time, historical route transit distributions, port congestion snapshots, weather, and carrier/service reliability. I would then evaluate calibration and precision-recall trade-offs against business costs on multiple chronological cohorts. I would only revisit model families after these inputs show measurable signal. Until then, Logistic Regression remains the transparent reference artifact, not a production-ready predictor.


## CI/CD Plan

**Selected assignment option: Option B — documented plan.** A working CI workflow is intentionally not claimed or represented by a green badge. The repository uses GitHub Actions as the proposed CI platform because the expected submission target is GitHub and Actions integrates pull-request checks, build logs and release permissions without introducing a separate CI service.

### Pipeline stages

1. **Checkout and environment:** check out the pull-request commit, set up the supported Python version, enable pip caching, and install the pinned/ranged project dependencies plus the lint tool.
2. **Static quality:** run Ruff for linting and formatting checks. Fail on syntax/import issues and agreed lint violations rather than automatically rewriting code in CI.
3. **Unit tests:** run fast tests for feature engineering, data-quality checks, repository query construction, API validation, inference contracts and GenAI tool orchestration. External Groq calls remain mocked; CI must not require a paid API key.
4. **Integration tests:** run the real FastAPI application against a temporary SQLite database and the committed model artefact. Exercise successful shipment lookup, unknown-ID 404, data-quality report and `/predict-delay`. SQLite is provided by Python’s standard library, and these required tests run without a database service or third-party driver.
5. **Container build:** build the Docker image from the same commit. A later extension can run the Compose smoke test against the built image; this is a deployment gate before promotion.

### Quality gates

- **Before merge:** lint passes; all unit and integration tests pass; the Docker image builds; no secrets are present in the diff; changes to the data/model contract include corresponding tests and README updates.
- **Before deployment:** only a merged, immutable commit/image tag is eligible; the image build and complete test suite must be green; the release candidate must pass `/health` and smoke requests for shipment lookup, DQ report and prediction in staging.
- The Groq key is not needed for CI. A separate, explicitly approved smoke test may validate the live provider integration in a protected environment; it must not be a merge requirement because it is network- and credential-dependent.

### Deployment strategy

For this assignment, deployment is a simple container release: build once from the merged commit, tag the image with the commit SHA, publish it to a private container registry, and deploy that immutable tag to the target environment. Apply database/pipeline changes as a controlled one-shot migration/job before switching API traffic. Start the new API container, wait for its health check, then run the endpoint smoke suite. Keep the previous image tag available until the new release is accepted. Do not rebuild an image independently in each environment.

### Rollback mechanism

If readiness or smoke checks fail, stop routing traffic to the new container and redeploy the previous known-good image tag. Preserve the SQLite volume and raw data; do not delete persistent data as part of an application rollback. If a release includes an incompatible data-model change, use backward-compatible expand/migrate/contract changes or restore a verified warehouse backup rather than assuming an application image rollback can reverse data changes. Record the failed commit, image digest, observed errors and rollback time.

### Why Option B for this submission

The assignment explicitly accepts either a working pipeline or a detailed plan. Option B keeps the time-boxed implementation focused on the working service, data product, ML and GenAI thin slice. It is more honest than adding an unverified workflow and claiming CI is green. The plan is concrete enough to implement as a GitHub Actions workflow when repository permissions and deployment target are available.

## Production Support Runbook

### Verify health after deployment

1. Check container state and startup logs: `docker compose ps` and `docker compose logs --tail=100 api`.
2. Call `GET /health`. A healthy response must report `status: "ok"`, the expected API version and `database: "ready"`.
3. Run smoke requests: `GET /shipments/{known-id}`, `GET /data-quality/report`, and `POST /predict-delay` for a known shipment. Confirm expected HTTP status and structured response fields.
4. Check request logs for method, path, status code, latency and request ID. The health check is a readiness check for SQLite serving views, not a complete external dependency or business-quality monitor.

### Likely failure modes and diagnosis

| Failure mode | Symptoms | Diagnosis / action |
|---|---|---|
| Pipeline or SQLite warehouse unavailable/incomplete | API startup may succeed but `/health` returns 503; shipment and route queries fail | Inspect `docker compose logs pipeline` and API logs; verify raw CSVs and `DATABASE_PATH`; rerun the idempotent pipeline; confirm `serving_shipments` and `serving_route_stats` exist. |
| Missing or incompatible model artefact | API fails during startup, or prediction returns 503 if predictor is unavailable | Check that `artifacts/delay_final.joblib` and `artifacts/delay_model_metadata.joblib` are present and belong to the same model version; restore both from the same release. Do not retrain inside the API process. |
| Groq assistant cannot answer | CLI reports authentication, rate-limit or provider/network errors; API endpoints may remain healthy | Check `GROQ_API_KEY`, configured `GROQ_MODEL`, provider status, network access and `logs/genai_assistant.jsonl`. The assistant is a separate CLI capability; its failure should not take down the core API. Never print the API key in logs. |

### Roll back a bad release

- Redeploy the previous immutable image tag and verify `/health` plus the smoke endpoints.
- Keep the persistent warehouse volume intact. Application rollback and data rollback are separate operations.
- If the failure was caused by a data migration, restore a verified backup or apply the documented forward-fix; do not delete the warehouse volume as a quick fix.
- Preserve logs and the failing image digest for diagnosis. For the local Compose environment, use the previous image tag/configuration and `docker compose up -d`; `docker compose down -v` is not a rollback command.

### Production additions

Before real operational use, add centralised metrics and dashboards for request rate, latency, error rate, database/pipeline freshness, prediction volume and model input drift; alerts on readiness failures, elevated 5xx rates, stale data and failed pipeline runs; managed secret storage and rotation for Groq credentials; authenticated/authorised API access; audit-log retention and redaction; database backups and restore drills; and model performance monitoring once delayed-arrival labels mature. The current delay model is explicitly experimental and must not drive operational decisions.


## Scaling to production

The current deployment is a single API process backed by a local SQLite warehouse and a one-shot batch pipeline. This is appropriate for the supplied snapshot and a reviewer running the project locally, but it is not a highly available multi-instance production topology.

At millions of shipments and continuous 24/7 operation, the first constraints to revisit would be concurrent writes and ingestion coordination, API availability across instances, database backup/failover, and freshness/latency of serving queries. I would separate immutable raw event storage from curated serving data, use object storage with partitioned Parquet for retained history, and move the operational serving database to managed PostgreSQL when concurrent access, transactions, replication and recovery requirements justify it. A message broker and streaming processor would be introduced only if measured event freshness requirements make scheduled micro-batches insufficient.

The API would become stateless and horizontally scalable behind a load balancer, with connection pooling, managed secrets, authentication/authorisation, rate limits, distributed tracing, centralised logs/metrics, automated backups and restore drills. Pipeline runs would be orchestrated and observable, with schema/version contracts and replayable source events. Model artefacts would be versioned separately from application images, and model monitoring would track input drift, prediction distribution, calibration and delayed-label precision/recall/AUC before any retraining or promotion. The present delay model is experimental; production scale does not make its current predictive performance acceptable.
