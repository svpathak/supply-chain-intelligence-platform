"""Train the single booking-time delay model and compare with a random baseline."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.features import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN,
    load_training_frame, split_chronologically,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])


def _metrics(y_true: pd.Series, scores: np.ndarray, predictions: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "rows": int(len(y_true)),
        "threshold": 0.5,
        "actual_class_counts": {
            "0": int((y_true == 0).sum()), "1": int((y_true == 1).sum())
        },
        "predicted_class_counts": {
            "0": int((predictions == 0).sum()), "1": int((predictions == 1).sum())
        },
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "log_loss": float(log_loss(y_true, scores, labels=[0, 1])),
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
    }


def _random_baseline(y_true: pd.Series, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    scores = rng.random(len(y_true))
    predictions = (scores >= 0.5).astype("int8")
    result = _metrics(y_true, scores, predictions)
    result.update({
        "strategy": "independent uniform random score; positive if score >= 0.5",
        "seed": seed,
        "expected_roc_auc": 0.5,
    })
    return result


def train_and_evaluate(data_dir: Path, artifact_dir: Path, report_path: Path) -> dict:
    frame, preparation = load_training_frame(data_dir)
    train, validation, test, split_report = split_chronologically(frame)
    feature_columns = FEATURE_COLUMNS
    model = Pipeline([
        ("preprocess", _preprocessor()),
        ("classifier", LogisticRegression(max_iter=3000, random_state=42)),
    ])
    model.fit(train[feature_columns], train[TARGET_COLUMN].astype(int))

    results = {}
    for name, partition in [("validation", validation), ("test", test)]:
        scores = model.predict_proba(partition[feature_columns])[:, 1]
        predictions = (scores >= 0.5).astype("int8")
        results[name] = _metrics(partition[TARGET_COLUMN], scores, predictions)
        results[name]["random_50_percent_baseline"] = _random_baseline(
            partition[TARGET_COLUMN], seed=42
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    joblib.dump(model, artifact_dir / "delay_final.joblib")
    joblib.dump(model, artifact_dir / "delay_logistic_regression.joblib")
    metadata = {
        "model_name": "logistic_regression",
        "model_version": version,
        "target": "actual_arrival > planned_arrival + 24 hours",
        "positive_class": 1,
        "threshold": 0.5,
        "feature_columns": feature_columns,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "forbidden_columns": [
            "actual_departure", "actual_arrival", "actual_delay_hours",
            "on_time_flag", "status", "event_timestamp", "event_type",
            "delay_minutes", "transit_days_actual", "shipment_id",
            "customer_id", "vessel_id",
        ],
        "training_data_dir": str(data_dir),
        "split_strategy": split_report["strategy"],
        "preprocessing": (
            "median numeric imputation + missing indicators + standard scaling; "
            "UNKNOWN categorical imputation + one-hot encoding; fitted on training only"
        ),
        "class_resampling": "none",
        "production_ready": False,
    }
    joblib.dump(metadata, artifact_dir / "delay_model_metadata.joblib")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "preparation": preparation,
        "split": split_report,
        "model": {
            "name": "logistic_regression",
            "validation": results["validation"],
            "test": results["test"],
            "artifact": "artifacts/delay_final.joblib",
        },
        "selected_model": "logistic_regression",
        "baseline": "random_50_percent",
        "production_ready": False,
        "interpretation": (
            "The model is retained as an evaluated baseline candidate, not described "
            "as production-ready unless it demonstrates validation discrimination "
            "above chance and acceptable operational metrics."
        ),
        "test_set_caveat": (
            "The chronological test period has been inspected during previous model "
            "experiments; it should not be described as a pristine untouched holdout."
        ),
        "monitoring_plan": [
            "Track feature and prediction distributions by booking week and route.",
            "Once labels mature, track ROC-AUC, average precision, precision, recall, "
            "F1, calibration and late-shipment prevalence.",
            "Investigate drift or sustained metric degradation before retraining; "
            "retrain only with newly matured, quality-checked labels and temporal validation."
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the single shipment delay model")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports/model_evaluation.json")
    args = parser.parse_args()
    report = train_and_evaluate(args.data_dir, args.artifact_dir, args.report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
