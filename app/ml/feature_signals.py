"""Validation-only feature-group ablation for the booking-time delay model."""
from __future__ import annotations

import json
from pathlib import Path

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

FEATURE_GROUPS = {
    "schedule_calendar": [
        "planned_transit_days", "booking_lead_days", "booking_month",
        "booking_day_of_week", "planned_departure_month",
        "planned_departure_day_of_week",
    ],
    "route_identity": ["origin_port", "destination_port"],
    "port_context": [
        "origin_region", "destination_region", "origin_congestion_score",
        "destination_congestion_score",
    ],
    "shipment_profile": ["container_count", "weight_tons", "cargo_type"],
}
FEATURE_GROUPS["full"] = [c for cols in FEATURE_GROUPS.values() for c in cols]
for group_name, group_columns in list(FEATURE_GROUPS.items()):
    if group_name == "full":
        continue
    FEATURE_GROUPS[f"without_{group_name}"] = [
        c for c in FEATURE_GROUPS["full"] if c not in group_columns
    ]


def _make_pipeline(columns: list[str]) -> Pipeline:
    numeric = [c for c in columns if c not in CATEGORICAL_FEATURES]
    categorical = [c for c in columns if c in CATEGORICAL_FEATURES]
    transformers = []
    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers)),
        ("classifier", LogisticRegression(max_iter=3000, random_state=42)),
    ])


def _metrics(y_true: pd.Series, scores: np.ndarray) -> dict:
    labels = (scores >= 0.5).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, labels, labels=[0, 1]).ravel()
    return {
        "rows": int(len(y_true)),
        "actual_positive_rate": float(y_true.mean()),
        "predicted_class_counts": {
            "0": int((labels == 0).sum()), "1": int((labels == 1).sum())
        },
        "accuracy": float(accuracy_score(y_true, labels)),
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "recall": float(recall_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "log_loss": float(log_loss(y_true, scores, labels=[0, 1])),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
    }


def run_feature_signal_analysis(data_dir: Path) -> dict:
    """Fit one LR model per feature group; use validation only for comparison."""
    frame, preparation = load_training_frame(data_dir)
    train, validation, test, split = split_chronologically(frame)
    y_train = train[TARGET_COLUMN].astype(int)
    y_validation = validation[TARGET_COLUMN].astype(int)

    results = {}
    for name, columns in FEATURE_GROUPS.items():
        model = _make_pipeline(columns)
        model.fit(train[columns], y_train)
        scores = model.predict_proba(validation[columns])[:, 1]
        results[name] = {
            "feature_count": len(columns),
            "features": columns,
            "validation": _metrics(y_validation, scores),
        }

    # Calendar values are periodic, not ordinal: Dec and Jan are adjacent,
    # as are Sunday and Monday. Compare sine/cosine encodings on validation.
    cyclical_columns = {
        "booking_month_sin": lambda d: np.sin(2 * np.pi * (d["booking_month"] - 1) / 12),
        "booking_month_cos": lambda d: np.cos(2 * np.pi * (d["booking_month"] - 1) / 12),
        "booking_dow_sin": lambda d: np.sin(2 * np.pi * d["booking_day_of_week"] / 7),
        "booking_dow_cos": lambda d: np.cos(2 * np.pi * d["booking_day_of_week"] / 7),
        "planned_departure_month_sin": lambda d: np.sin(2 * np.pi * (d["planned_departure_month"] - 1) / 12),
        "planned_departure_month_cos": lambda d: np.cos(2 * np.pi * (d["planned_departure_month"] - 1) / 12),
        "planned_departure_dow_sin": lambda d: np.sin(2 * np.pi * d["planned_departure_day_of_week"] / 7),
        "planned_departure_dow_cos": lambda d: np.cos(2 * np.pi * d["planned_departure_day_of_week"] / 7),
    }
    train_cyc = train.copy()
    validation_cyc = validation.copy()
    for name, fn in cyclical_columns.items():
        train_cyc[name] = fn(train_cyc)
        validation_cyc[name] = fn(validation_cyc)
    raw_calendar = [
        "booking_month", "booking_day_of_week",
        "planned_departure_month", "planned_departure_day_of_week",
    ]
    cyc_calendar = list(cyclical_columns)
    full_cyclical_columns = [
        c for c in FEATURE_GROUPS["full"] if c not in raw_calendar
    ] + cyc_calendar
    cyclical_model = _make_pipeline(full_cyclical_columns)
    cyclical_model.fit(train_cyc[full_cyclical_columns], y_train)
    cyclical_scores = cyclical_model.predict_proba(validation_cyc[full_cyclical_columns])[:, 1]
    results["full_cyclical_calendar"] = {
        "feature_count": len(full_cyclical_columns),
        "features": full_cyclical_columns,
        "validation": _metrics(y_validation, cyclical_scores),
        "description": "Full feature set with month and weekday integers replaced by sine/cosine pairs",
    }

    cyc_schedule_columns = [
        c for c in FEATURE_GROUPS["schedule_calendar"] if c not in raw_calendar
    ] + cyc_calendar
    cyc_schedule_model = _make_pipeline(cyc_schedule_columns)
    cyc_schedule_model.fit(train_cyc[cyc_schedule_columns], y_train)
    cyc_schedule_scores = cyc_schedule_model.predict_proba(validation_cyc[cyc_schedule_columns])[:, 1]
    results["schedule_calendar_cyclical"] = {
        "feature_count": len(cyc_schedule_columns),
        "features": cyc_schedule_columns,
        "validation": _metrics(y_validation, cyc_schedule_scores),
        "description": "Schedule/calendar-only features with cyclical month and weekday encoding",
    }

    # A reproducible random-50% classifier. ROC-AUC for its independent random
    # scores is a single noisy realization; expected no-skill ROC-AUC is 0.5.
    rng = np.random.default_rng(42)
    random_scores = rng.random(len(validation))
    random_labels = (random_scores >= 0.5).astype("int8")
    baseline = _metrics(y_validation, random_scores)
    baseline["predicted_class_counts"] = {
        "0": int((random_labels == 0).sum()),
        "1": int((random_labels == 1).sum()),
    }
    baseline["expected_roc_auc"] = 0.5
    baseline["strategy"] = "independent uniform random score; classify positive at score >= 0.5; seed=42"

    ranked = sorted(
        (
            {"configuration": name, "roc_auc": values["validation"]["roc_auc"],
             "average_precision": values["validation"]["average_precision"],
             "log_loss": values["validation"]["log_loss"]}
            for name, values in results.items()
        ),
        key=lambda item: item["roc_auc"],
        reverse=True,
    )
    return {
        "methodology": (
            "One model family only: Logistic Regression. Compare feature groups "
            "on chronological validation data. Preprocessing is fitted only on "
            "the training partition. No test metrics are used for feature selection."
        ),
        "preparation": preparation,
        "split": split,
        "feature_groups": FEATURE_GROUPS,
        "validation_results": results,
        "random_50_percent_baseline_validation": baseline,
        "validation_ranking": ranked,
        "best_validation_configuration": ranked[0]["configuration"],
        "best_validation_roc_auc": ranked[0]["roc_auc"],
        "chance_level_roc_auc": 0.5,
        "any_configuration_beats_chance_on_validation": any(
            item["roc_auc"] > 0.5 for item in ranked
        ),
        "interpretation": (
            "Feature-group results are exploratory and based on one chronological "
            "validation window. A small difference around 0.5 is not evidence of "
            "robust predictive signal. Do not select features using the test set."
        ),
        "test_set_note": (
            "The chronological test period was already inspected during earlier "
            "model experiments; it is not a pristine holdout. This analysis does "
            "not use test results for feature selection."
        ),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("reports/feature_signal_analysis.json"))
    args = parser.parse_args()
    report = run_feature_signal_analysis(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
