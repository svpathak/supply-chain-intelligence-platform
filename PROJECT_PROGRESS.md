# Implementation Record

## Scope delivered

The repository implements the three assignment areas: software engineering and API service, standalone data-quality checks and a layered data product, and delay prediction plus a tool-calling GenAI assistant.

## Architecture and technology decisions

- **API:** Python 3.11, FastAPI and Pydantic; six required endpoints, structured JSON errors and per-request JSON logs.
- **Data store:** Python standard-library `sqlite3`; raw, staging, integration and serving layers use table prefixes in one local database.
- **Pipeline:** transactional raw snapshot refresh, current/history upserts, conflict quarantine, content-hash event idempotency, curated derived fields and serving views.
- **Data quality:** standalone Pandas checks read the original CSV files without requiring the database or pipeline.
- **ML:** Scikit-learn Logistic Regression reference model with train-fitted preprocessing and booking-time-only features; compared with a random 50% baseline. Experimental alternatives were evaluated but not promoted.
- **GenAI:** Groq chat completions with local Python function calling for shipment queries, route metrics and prediction requests. Two guardrails and JSONL audit logging are implemented.
- **CI/CD:** assignment Option B; the README contains the required pipeline, gates, deployment and rollback plan. No working CI workflow is claimed.

## Data processing results

- Raw inputs: 25 ports, 5,028 shipment rows and 25,000 port-event rows.
- Shipment staging: 8 exact duplicate rows collapsed; 40 rows across 20 conflicting shipment-ID groups quarantined; 4,980 unambiguous current shipments retained.
- Curated output: 4,980 shipments and 671 origin/destination route aggregates.
- Event staging: full-payload hashing retains distinct payloads with repeated event IDs and ignores exact replays.
- The standalone DQ report executes 38 checks, 20 of which report findings (5 critical, 14 warning and 1 informational). Findings may overlap and are not claimed to map one-to-one to the twelve unspecified intentional issues.

## ML decision

The target is destination arrival more than 24 hours after planned arrival. The model uses 15 features available at booking time; actual milestones, status, outcomes and shipment-level event attributes are excluded. Missing outcomes are not treated as negatives. The retained Logistic Regression reference model has validation precision 0.378, recall 0.104, F1 0.164 and ROC-AUC 0.481; the previously inspected test ROC-AUC is 0.495. Predictive performance is not better than chance, so the artifact and API response are explicitly marked experimental and not approved for operational decisions. Validation has been reused and the test period inspected; neither is represented as a pristine holdout.

## Verification

- `pytest -q`: **44 passed, 0 skipped**.
- `python -m compileall -q app scripts tests`: passed.
- Standalone DQ command executed against all supplied raw CSVs and generated the JSON report.
- Full SQLite pipeline executed against all supplied CSVs; a second run produced no duplicate shipment or event records.
- FastAPI integration tests exercised real HTTP requests against a temporary SQLite database. An additional smoke run against a freshly built warehouse returned the expected status codes for all six endpoints, including 404 for an unknown shipment.
- Local assistant tool functions were exercised against SQLite. A live Groq provider request was not made because it requires a configured API key.
- Dockerfile and Compose configuration were reviewed statically. Docker image build and Compose runtime have not been verified because a container engine was not available in the build environment.

## Remaining verification before deployment

1. Build and start the stack on a Docker-enabled machine with `docker compose up --build -d`; verify `/health` and the documented smoke endpoints.
2. Configure a Groq API key and execute a live assistant query to verify provider-side tool calling and audit output.
3. Treat the current delay model as experimental until stronger booking-time data and a fresh temporal holdout demonstrate useful performance.
