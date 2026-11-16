from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.ml.feature_selection_experiment import run_experiment

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare label/ordinal encoding and train-only feature selection")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/feature_selection_experiment.json")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts/delay_feature_selection_experiment.joblib")
    args = parser.parse_args()
    report, model = run_experiment(args.data_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump(model, args.artifact)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
