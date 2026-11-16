"""Compare categorical encoding and train-only mutual-information feature selection.

This is an experiment, not the production training path. Feature selection is
fit within temporal CV folds of the training partition; the validation period is
used once for the final comparison. The previously inspected test period is not
used for selection or reported here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from app.ml.features import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN,
    load_training_frame, split_chronologically,
)


def _categorical_cardinality(train: pd.DataFrame) -> dict[str, int]:
    return {c: int(train[c].nunique(dropna=True)) for c in CATEGORICAL_FEATURES}


def _mixed_encoder(train: pd.DataFrame) -> tuple[ColumnTransformer, dict]:
    cardinality = _categorical_cardinality(train)
    ordinal_cols = [c for c in CATEGORICAL_FEATURES if cardinality[c] > 3]
    onehot_cols = [c for c in CATEGORICAL_FEATURES if cardinality[c] <= 3]
    transformers = [(
        "numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), NUMERIC_FEATURES,
    )]
    if ordinal_cols:
        transformers.append((
            "ordinal_categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ("scaler", StandardScaler()),
            ]), ordinal_cols,
        ))
    if onehot_cols:
        transformers.append((
            "onehot_categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), onehot_cols,
        ))
    return ColumnTransformer(transformers, remainder="drop"), {
        "cardinality_on_fit_data": cardinality,
        "ordinal_encoded_gt_3": ordinal_cols,
        "one_hot_encoded_le_3": onehot_cols,
    }


def _model(train: pd.DataFrame, columns: list[str]) -> Pipeline:
    numeric = [c for c in columns if c in NUMERIC_FEATURES]
    categorical = [c for c in columns if c in CATEGORICAL_FEATURES]
    cardinality = {c: int(train[c].nunique(dropna=True)) for c in categorical}
    ordinal_cols = [c for c in categorical if cardinality[c] > 3]
    onehot_cols = [c for c in categorical if cardinality[c] <= 3]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), numeric))
    if ordinal_cols:
        transformers.append(("ordinal_categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scaler", StandardScaler()),
        ]), ordinal_cols))
    if onehot_cols:
        transformers.append(("onehot_categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), onehot_cols))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers)),
        ("classifier", LogisticRegression(max_iter=3000, random_state=42)),
    ])


def _mi_ranking(train: pd.DataFrame, y: pd.Series) -> list[dict]:
    """Rank raw columns using MI; categorical codes are treated as discrete."""
    encoded = pd.DataFrame(index=train.index)
    discrete_mask: list[bool] = []
    for col in FEATURE_COLUMNS:
        if col in CATEGORICAL_FEATURES:
            values = train[[col]].astype("string").fillna("UNKNOWN")
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            encoded[col] = enc.fit_transform(values).ravel()
            discrete_mask.append(True)
        else:
            values = pd.to_numeric(train[col], errors="coerce")
            encoded[col] = values.fillna(values.median() if values.notna().any() else 0)
            discrete_mask.append(False)
    scores = mutual_info_classif(
        encoded[FEATURE_COLUMNS], y.astype(int),
        discrete_features=np.asarray(discrete_mask), random_state=42,
    )
    return sorted(
        [{"feature": c, "mutual_information": float(s)} for c, s in zip(FEATURE_COLUMNS, scores)],
        key=lambda x: (-x["mutual_information"], x["feature"]),
    )


def _matrix_profile(matrix) -> dict:
    shape = [int(matrix.shape[0]), int(matrix.shape[1])]
    nonzero = int(matrix.nnz) if hasattr(matrix, "nnz") else int(np.count_nonzero(matrix))
    total = int(matrix.shape[0] * matrix.shape[1])
    return {"shape": shape, "nonzero_values": nonzero, "density": float(nonzero / total) if total else 0.0}


def _random_baseline(y: pd.Series, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    scores = rng.random(len(y))
    result = _scores(y, scores)
    result["expected_roc_auc"] = 0.5
    result["expected_average_precision"] = float(y.mean())
    result["seed"] = seed
    result["strategy"] = "independent uniform random score; positive if score >= 0.5"
    return result


def _scores(y: pd.Series, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype("int8")
    return {
        "rows": int(len(y)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "predicted_class_counts": {
            "0": int((predictions == 0).sum()),
            "1": int((predictions == 1).sum()),
        },
        "actual_class_counts": {
            "0": int((y == 0).sum()), "1": int((y == 1).sum()),
        },
    }


def run_experiment(data_dir: Path) -> tuple[dict, Pipeline]:
    frame, preparation = load_training_frame(data_dir)
    train, validation, _test, split = split_chronologically(frame)
    y_train = train[TARGET_COLUMN].astype(int)
    y_validation = validation[TARGET_COLUMN].astype(int)

    # Baseline: current all-one-hot categorical preprocessing.
    baseline = Pipeline([
        ("preprocess", ColumnTransformer([
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]), NUMERIC_FEATURES),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL_FEATURES),
        ])),
        ("classifier", LogisticRegression(max_iter=3000, random_state=42)),
    ])
    baseline.fit(train[FEATURE_COLUMNS], y_train)
    baseline_scores = _scores(y_validation, baseline.predict_proba(validation[FEATURE_COLUMNS])[:, 1])
    onehot_matrix_profile = _matrix_profile(baseline.named_steps["preprocess"].transform(train[FEATURE_COLUMNS]))

    # Requested encoding: ordinal encode categorical columns with >3 train values.
    mixed = _model(train, FEATURE_COLUMNS)
    mixed.fit(train[FEATURE_COLUMNS], y_train)
    mixed_scores = _scores(y_validation, mixed.predict_proba(validation[FEATURE_COLUMNS])[:, 1])
    mixed_matrix_profile = _matrix_profile(mixed.named_steps["preprocess"].transform(train[FEATURE_COLUMNS]))
    _, encoding_policy = _mixed_encoder(train)

    # Feature selection: rank only inside each temporal training fold. Select k
    # using mean inner-fold ROC-AUC, then refit selector/model on all train rows.
    candidates = [3, 5, 8, 10, 12, len(FEATURE_COLUMNS)]
    candidates = sorted(set(k for k in candidates if k <= len(FEATURE_COLUMNS)))
    cv = TimeSeriesSplit(n_splits=3)
    cv_results = []
    for k in candidates:
        fold_aucs = []
        for fit_idx, hold_idx in cv.split(train):
            fold_train = train.iloc[fit_idx]
            fold_hold = train.iloc[hold_idx]
            ranking = _mi_ranking(fold_train[FEATURE_COLUMNS], fold_train[TARGET_COLUMN])
            selected = [x["feature"] for x in ranking[:k]]
            fold_model = _model(fold_train, selected)
            fold_model.fit(fold_train[selected], fold_train[TARGET_COLUMN].astype(int))
            fold_prob = fold_model.predict_proba(fold_hold[selected])[:, 1]
            fold_aucs.append(float(roc_auc_score(fold_hold[TARGET_COLUMN], fold_prob)))
        cv_results.append({"k": k, "fold_roc_auc": fold_aucs, "mean_roc_auc": float(np.mean(fold_aucs)),
                           "std_roc_auc": float(np.std(fold_aucs, ddof=0))})

    # Deterministic tie-break: fewer features when mean CV AUC is equal.
    best_cv = sorted(cv_results, key=lambda x: (-x["mean_roc_auc"], x["k"]))[0]
    final_ranking = _mi_ranking(train[FEATURE_COLUMNS], y_train)
    selected_features = [x["feature"] for x in final_ranking[:best_cv["k"]]]
    selected_model = _model(train, selected_features)
    selected_model.fit(train[selected_features], y_train)
    selected_scores = _scores(
        y_validation, selected_model.predict_proba(validation[selected_features])[:, 1]
    )

    report = {
        "methodology": {
            "model_family": "Logistic Regression only",
            "validation": "Chronological validation period evaluated after inner temporal-CV selection; no test data used.",
            "feature_selection": "Univariate mutual information ranking fitted separately within each TimeSeriesSplit training fold; k selected by mean inner-fold ROC-AUC; final ranking refit on all training rows.",
            "encoding_experiment": "OrdinalEncoder for categorical features with >3 distinct non-null values in the relevant training fit; OneHotEncoder for <=3. Unknown categories map to -1 for ordinal features or are ignored for one-hot features.",
            "selection_limitations": "Mutual information is univariate and may miss interactions; ordinal encoding of nominal categories imposes artificial order and equal spacing. This is an experiment, not an assumed improvement.",
            "test_set": "Not used. Previously inspected chronological test window remains non-pristine and is deliberately excluded.",
        },
        "preparation": preparation,
        "split": split,
        "encoding_policy_full_train": encoding_policy,
        "transformed_training_matrix_profiles": {
            "all_one_hot": onehot_matrix_profile,
            "mixed_ordinal_gt3": mixed_matrix_profile,
        },
        "mutual_information_ranking_train_only": final_ranking,
        "feature_selection_inner_temporal_cv": cv_results,
        "selected_k": int(best_cv["k"]),
        "selected_features": selected_features,
        "validation_results": {
            "all_one_hot_full_features": baseline_scores,
            "mixed_ordinal_gt3_full_features": mixed_scores,
            "mixed_ordinal_gt3_mutual_information_selected": selected_scores,
        },
        "random_50_percent_baseline_validation": _random_baseline(y_validation),
        "chance_references": {
            "roc_auc": 0.5,
            "average_precision_validation_prevalence": float(y_validation.mean()),
        },
        "interpretation": "Compare validation results to the all-one-hot reference and no-skill references. Do not promote the experiment unless the gain is material and stable; all comparisons share the same single Logistic Regression family.",
    }
    return report, selected_model
