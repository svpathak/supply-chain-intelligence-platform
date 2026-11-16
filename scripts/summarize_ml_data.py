"""Create a reproducible source and modelling-data profile for Part 3.1."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.ml.features import load_training_frame, split_chronologically, FEATURE_COLUMNS, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def source_profile(path: Path) -> dict:
    df = pd.read_csv(path)
    columns = {}
    for col in df.columns:
        item = {
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": round(float(df[col].isna().mean() * 100), 3),
            "distinct_non_null": int(df[col].nunique(dropna=True)),
        }
        if df[col].dtype == "object":
            item["top_values"] = {str(k): int(v) for k, v in df[col].value_counts(dropna=False).head(10).items()}
        elif pd.api.types.is_numeric_dtype(df[col]):
            item["describe"] = {str(k): (None if pd.isna(v) else float(v)) for k, v in df[col].describe().items()}
        columns[col] = item
    return {"rows": int(len(df)), "columns": int(len(df.columns)), "column_profile": columns}

def build_report(data_dir: Path) -> dict:
    frame, prep = load_training_frame(data_dir)
    train, validation, test, split = split_chronologically(frame)
    ports = pd.read_csv(data_dir / "ports.csv")
    route_pairs = frame[["origin_port", "destination_port"]].drop_duplicates()
    categories = {}
    for col in CATEGORICAL_FEATURES:
        categories[col] = {
            "distinct_values": int(frame[col].nunique(dropna=True)),
            "values": sorted(frame[col].dropna().astype(str).unique().tolist()),
        }
    numeric = {}
    for col in NUMERIC_FEATURES:
        numeric[col] = {str(k): (None if pd.isna(v) else float(v)) for k, v in frame[col].describe().items()}
    categorical_target_rates = {}
    for col in CATEGORICAL_FEATURES:
        grouped = frame.groupby(col, dropna=False).is_late_over_24h.agg(["count", "mean"]).sort_values("count", ascending=False)
        categorical_target_rates[col] = [
            {"category": str(category), "rows": int(row["count"]), "late_rate": float(row["mean"])}
            for category, row in grouped.iterrows()
        ]
    route_groups = frame.groupby(["origin_port", "destination_port"]).is_late_over_24h.agg(["count", "mean"]).sort_values("count", ascending=False)
    top_routes = [
        {"origin_port": str(route[0]), "destination_port": str(route[1]), "rows": int(row["count"]), "late_rate": float(row["mean"])}
        for route, row in route_groups.head(20).iterrows()
    ]
    numeric_target_correlations = {
        col: (None if frame[col].nunique(dropna=True) < 2 else float(frame[[col, "is_late_over_24h"]].corr(method="spearman").iloc[0, 1]))
        for col in NUMERIC_FEATURES
    }
    return {
        "source_data": {
            "shipments": source_profile(data_dir / "shipments.csv"),
            "ports": source_profile(data_dir / "ports.csv"),
            "port_events": source_profile(data_dir / "port_events.csv"),
        },
        "prepared_modelling_population": {
            "rows": int(len(frame)),
            "target_counts": {"0_on_time_or_within_24h": int((frame.is_late_over_24h == 0).sum()), "1_more_than_24h_late": int((frame.is_late_over_24h == 1).sum())},
            "positive_rate": float(frame.is_late_over_24h.mean()),
            "unique_origin_destination_pairs": int(len(route_pairs)),
            "known_port_codes": int(ports.port_code.nunique()),
            "booking_date_min": frame.booking_date.min().isoformat(),
            "booking_date_max": frame.booking_date.max().isoformat(),
            "preparation_counts": prep,
            "chronological_split": split,
        },
        "feature_contract": {
            "feature_count": len(FEATURE_COLUMNS),
            "features": FEATURE_COLUMNS,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "categorical_cardinality_after_cleaning": categories,
            "numeric_summary": numeric,
            "numeric_feature_target_spearman_correlations_descriptive_only": numeric_target_correlations,
            "categorical_feature_target_rates_descriptive_only": categorical_target_rates,
            "top_20_routes_by_sample_count_descriptive_only": top_routes,
            "encoding": "OneHotEncoder(handle_unknown='ignore') for categorical fields; numeric fields imputed and scaled for Logistic Regression/SVM. Port codes are nominal identifiers, not ordinal or geographic coordinates.",
            "port_encoding_decision": "One-hot encode origin_port and destination_port. Do not label-encode port codes as ordered integers: no meaningful distance or ordering is encoded by the alphanumeric code. Geographic distance would require coordinates or authoritative route-distance data, which are not supplied.",
        },
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/raw')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/ml_data_summary.json')
    args = parser.parse_args()
    report = build_report(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
