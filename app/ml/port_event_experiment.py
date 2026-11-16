"""Evaluate whether strictly pre-booking port-event context adds ML signal."""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.features import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN,
    load_training_frame, split_chronologically,
)
from app.ml.port_event_features import (
    EVENT_FEATURE_COLUMNS, add_point_in_time_port_event_features,
)
from app.ml.train import _random_baseline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLP_HIDDEN_UNITS = 4


def _preprocessor(numeric_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), numeric_columns),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL_FEATURES),
    ], sparse_threshold=0)


def _metrics(y_true: pd.Series, scores: np.ndarray, predictions: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "rows": int(len(y_true)),
        "threshold": 0.5,
        "actual_class_counts": {"0": int((y_true == 0).sum()), "1": int((y_true == 1).sum())},
        "predicted_class_counts": {"0": int((predictions == 0).sum()), "1": int((predictions == 1).sum())},
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


def _evaluate(model: Pipeline, partition: pd.DataFrame, feature_columns: list[str]) -> dict:
    scores = model.predict_proba(partition[feature_columns])[:, 1]
    predictions = (scores >= 0.5).astype("int8")
    return _metrics(partition[TARGET_COLUMN], scores, predictions)


def run_experiment(
    data_dir: Path = PROJECT_ROOT / "data/raw",
    artifact_dir: Path = PROJECT_ROOT / "artifacts",
    report_path: Path = PROJECT_ROOT / "reports/port_event_feature_experiment.json",
) -> dict:
    base_frame, preparation = load_training_frame(data_dir)
    enriched_frame, event_audit = add_point_in_time_port_event_features(base_frame, data_dir)
    train, validation, test, split_report = split_chronologically(enriched_frame)

    base_features = list(FEATURE_COLUMNS)
    augmented_features = base_features + list(EVENT_FEATURE_COLUMNS)
    numeric_augmented = list(NUMERIC_FEATURES) + list(EVENT_FEATURE_COLUMNS)
    origin_event_features = [c for c in EVENT_FEATURE_COLUMNS if c.startswith("origin_")] + [
        "event_observation_7d_coverage", "event_observation_30d_coverage"
    ]
    destination_event_features = [c for c in EVENT_FEATURE_COLUMNS if c.startswith("destination_")] + [
        "event_observation_7d_coverage", "event_observation_30d_coverage"
    ]

    def small_mlp() -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=(MLP_HIDDEN_UNITS,), activation="relu", solver="adam",
            alpha=0.01, learning_rate_init=0.001, max_iter=2000, tol=1e-5,
            n_iter_no_change=40, early_stopping=False, random_state=42,
        )

    configurations = {
        "logistic_regression_base": {
            "estimator": LogisticRegression(max_iter=3000, random_state=42),
            "features": base_features,
            "numeric": list(NUMERIC_FEATURES),
        },
        "logistic_regression_plus_origin_events": {
            "estimator": LogisticRegression(max_iter=3000, random_state=42),
            "features": base_features + origin_event_features,
            "numeric": list(NUMERIC_FEATURES) + origin_event_features,
        },
        "logistic_regression_plus_port_events": {
            "estimator": LogisticRegression(max_iter=3000, random_state=42),
            "features": augmented_features,
            "numeric": numeric_augmented,
        },
        "mlp_4_units_base": {
            "estimator": small_mlp(),
            "features": base_features,
            "numeric": list(NUMERIC_FEATURES),
        },
        "mlp_4_units_plus_origin_events": {
            "estimator": small_mlp(),
            "features": base_features + origin_event_features,
            "numeric": list(NUMERIC_FEATURES) + origin_event_features,
        },
        "mlp_4_units_plus_destination_events": {
            "estimator": small_mlp(),
            "features": base_features + destination_event_features,
            "numeric": list(NUMERIC_FEATURES) + destination_event_features,
        },
        "mlp_4_units_plus_port_events": {
            "estimator": small_mlp(),
            "features": augmented_features,
            "numeric": numeric_augmented,
        },
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    artifacts: dict[str, str] = {}
    convergence: dict[str, list[str]] = {}
    for name, config in configurations.items():
        model = Pipeline([
            ("preprocess", _preprocessor(config["numeric"])),
            ("classifier", config["estimator"]),
        ])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(train[config["features"]], train[TARGET_COLUMN].astype(int))
        convergence[name] = [
            str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)
        ]
        results[name] = _evaluate(model, validation, config["features"])
        results[name]["input_feature_count"] = int(len(config["features"]))
        results[name]["transformed_input_dimensions"] = int(
            model.named_steps["preprocess"].transform(train[config["features"]].iloc[:1]).shape[1]
        )
        results[name]["random_50_percent_baseline"] = _random_baseline(
            validation[TARGET_COLUMN], seed=42
        )
        if name in {
            "logistic_regression_plus_port_events",
            "mlp_4_units_plus_origin_events",
            "mlp_4_units_plus_port_events",
        }:
            artifact_name = f"delay_{name}_experiment.joblib"
            joblib.dump(model, artifact_dir / artifact_name)
            artifacts[name] = f"artifacts/{artifact_name}"
        if name.startswith("mlp"):
            results[name]["training_iterations"] = int(model.named_steps["classifier"].n_iter_)
            results[name]["final_training_loss"] = float(model.named_steps["classifier"].loss_)

    # Exploratory one-feature-at-a-time ranking diagnostics. These are not used
    # for feature selection or model fitting and are explicitly validation-only.
    univariate_diagnostics = {}
    for column in EVENT_FEATURE_COLUMNS:
        values = validation[column]
        diagnostic = {
            "missing_rows": int(values.isna().sum()),
            "nonzero_rows": int((values.fillna(0) != 0).sum()),
            "mean": float(values.mean()) if values.notna().any() else None,
            "median": float(values.median()) if values.notna().any() else None,
        }
        if values.nunique(dropna=True) > 1:
            diagnostic["roc_auc"] = float(roc_auc_score(
                validation[TARGET_COLUMN], values.fillna(values.median())
            ))
        else:
            diagnostic["roc_auc"] = None
        univariate_diagnostics[column] = diagnostic

    # Add explicit audit checks over generated features.
    feature_nulls = {
        col: int(enriched_frame[col].isna().sum()) for col in EVENT_FEATURE_COLUMNS
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_status": "validation_only_experiment_not_promoted",
        "preparation": preparation,
        "event_source_audit": event_audit,
        "split": split_report,
        "event_feature_null_counts": feature_nulls,
        "feature_sets": {
            "base_feature_count": len(base_features),
            "augmented_feature_count": len(augmented_features),
            "added_event_feature_columns": EVENT_FEATURE_COLUMNS,
            "origin_only_feature_columns": origin_event_features,
            "destination_only_feature_columns": destination_event_features,
        },
        "validation_results": results,
        "univariate_validation_diagnostics": univariate_diagnostics,
        "artifacts": artifacts,
        "convergence_warnings": convergence,
        "selection_policy": (
            "Validation is used for the feature experiment; no test metrics are used for model or feature selection. "
            "Only point-in-time port aggregates are added, not shipment-specific event attribution."
        ),
        "test_set_policy": (
            "No test metrics are reported in this experiment because the chronological test period has already "
            "been inspected in earlier experiments and is not a pristine holdout."
        ),
        "interpretation_guardrails": [
            "A port-level aggregate is contextual and is not evidence that an event belongs to a particular shipment or vessel leg.",
            "All event timestamps used must be strictly earlier than the shipment booking timestamp.",
            "Source timestamps are naive; no timezone conversion or UTC interpretation is assumed.",
            "The event file starts during the shipment observation period; early trailing windows are only partially observable.",
            "Event ID duplicates are quarantined as conflicting records rather than arbitrarily choosing one payload.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate strictly pre-booking port-event features")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports/port_event_feature_experiment.json")
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.data_dir, args.artifact_dir, args.report), indent=2))


if __name__ == "__main__":
    main()
