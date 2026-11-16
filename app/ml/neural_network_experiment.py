"""Minimal one-hidden-layer MLP experiment against the LR reference.

The architecture deliberately adds only one small nonlinear representation
layer before the binary logistic output. It is experimental and never replaces
the Logistic Regression API artifact automatically.
"""
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
from app.ml.train import _random_baseline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HIDDEN_UNITS = 4


def _preprocessor() -> ColumnTransformer:
    """Same leakage-safe feature transformations as the LR reference."""
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), NUMERIC_FEATURES),
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


def run_experiment(
    data_dir: Path = PROJECT_ROOT / "data/raw",
    artifact_dir: Path = PROJECT_ROOT / "artifacts",
    report_path: Path = PROJECT_ROOT / "reports/neural_network_experiment.json",
) -> dict:
    frame, preparation = load_training_frame(data_dir)
    train, validation, test, split_report = split_chronologically(frame)

    # Logistic Regression is a linear map followed by a sigmoid. This MLP adds
    # just one 4-unit ReLU representation layer and retains a single sigmoid
    # probability output for the same binary target.
    model = Pipeline([
        ("preprocess", _preprocessor()),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(HIDDEN_UNITS,),
            activation="relu",
            solver="adam",
            alpha=0.01,
            batch_size="auto",
            learning_rate_init=0.001,
            max_iter=2000,
            tol=1e-5,
            n_iter_no_change=40,
            early_stopping=False,
            random_state=42,
        )),
    ])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN].astype(int))
    convergence_warnings = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]

    evaluations = {}
    for name, partition in (("validation", validation), ("test", test)):
        scores = model.predict_proba(partition[FEATURE_COLUMNS])[:, 1]
        predictions = (scores >= 0.5).astype("int8")
        evaluations[name] = _metrics(partition[TARGET_COLUMN], scores, predictions)
        evaluations[name]["random_50_percent_baseline"] = _random_baseline(partition[TARGET_COLUMN], seed=42)

    classifier = model.named_steps["classifier"]
    transformed_width = int(model.named_steps["preprocess"].transform(train[FEATURE_COLUMNS][:1]).shape[1])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "delay_mlp_experiment.joblib")

    reference_report_path = PROJECT_ROOT / "reports/model_evaluation.json"
    reference_report = json.loads(reference_report_path.read_text(encoding="utf-8"))
    reference_metrics = {
        "validation": reference_report["model"]["validation"],
        "test": reference_report["model"]["test"],
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_status": "experimental_not_promoted",
        "preparation": preparation,
        "split": split_report,
        "architecture": {
            "input_features_before_preprocessing": len(FEATURE_COLUMNS),
            "input_dimensions_after_preprocessing": transformed_width,
            "hidden_layers": 1,
            "hidden_units": HIDDEN_UNITS,
            "hidden_activation": "ReLU",
            "output": "one sigmoid probability for is_late_over_24h",
            "conceptual_progression": "linear Logistic Regression -> one small nonlinear hidden representation -> logistic probability output",
            "optimizer": "Adam",
            "learning_rate_init": 0.001,
            "l2_regularization_alpha": 0.01,
            "max_iter": 2000,
            "early_stopping": False,
            "random_state": 42,
            "training_iterations": int(classifier.n_iter_),
            "final_training_loss": float(classifier.loss_),
            "convergence_warnings": convergence_warnings,
        },
        "preprocessing": "train-fitted median imputation + missing indicators + standard scaling; UNKNOWN imputation + one-hot encoding",
        "model": {
            "name": "one_hidden_layer_mlp_4_units",
            "validation": evaluations["validation"],
            "test": evaluations["test"],
            "artifact": "artifacts/delay_mlp_experiment.joblib",
        },
        "reference_model": {
            "name": "logistic_regression",
            "source_report": "reports/model_evaluation.json",
            "validation": reference_metrics["validation"],
            "test": reference_metrics["test"],
        },
        "baseline": "random_50_percent",
        "test_set_caveat": (
            "The chronological test period has already been inspected in earlier experiments; "
            "test metrics are reported for transparency only and must not drive model selection."
        ),
        "decision_rule": (
            "Do not promote this MLP unless validation discrimination and probability quality "
            "meaningfully improve over Logistic Regression and the random no-skill reference, "
            "with a plausible predicted-positive distribution."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run the minimal one-hidden-layer MLP experiment")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports/neural_network_experiment.json")
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.data_dir, args.artifact_dir, args.report), indent=2))


if __name__ == "__main__":
    main()
