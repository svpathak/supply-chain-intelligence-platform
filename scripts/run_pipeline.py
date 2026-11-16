from __future__ import annotations

import argparse
import json

from app.data.database import connect
from app.data.ingestion import run_ingestion
from app.data.transformations import build_integration_and_serving


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the supply-chain data pipeline")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--db", default="data/warehouse.sqlite3")
    args = parser.parse_args()

    result = run_ingestion(args.data_dir, args.db)
    conn = connect(args.db)
    try:
        build_integration_and_serving(conn)
        result["integration"] = {
            "shipments": conn.execute(
                "SELECT COUNT(*) FROM integration_shipments"
            ).fetchone()[0]
        }
        result["serving"] = {
            "routes": conn.execute(
                "SELECT COUNT(*) FROM serving_route_stats"
            ).fetchone()[0],
            "events": conn.execute(
                "SELECT COUNT(*) FROM serving_port_events"
            ).fetchone()[0],
        }
    finally:
        conn.close()

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
