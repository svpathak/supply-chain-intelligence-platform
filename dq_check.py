"""Command-line entry point for standalone raw CSV quality checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.quality import run_quality_checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run standalone data-quality checks against raw CSV files."
    )
    parser.add_argument("--input", type=Path, default=Path("data/raw"),
                        help="Directory containing ports.csv, shipments.csv and port_events.csv")
    parser.add_argument("--output", type=Path, default=Path("reports/data_quality_report.json"),
                        help="Path for the machine-readable JSON report")
    args = parser.parse_args()

    report = run_quality_checks(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"JSON report written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
