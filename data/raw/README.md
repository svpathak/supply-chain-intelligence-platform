# Raw source data

The three original CSV files are retained unchanged in this directory. The pipeline reads them directly, reloads the corresponding `raw_*` SQLite snapshot tables on each run, and adds ingestion batch metadata in the warehouse. Standalone data-quality checks continue to read these CSVs directly and do not depend on the database or ingestion pipeline.
