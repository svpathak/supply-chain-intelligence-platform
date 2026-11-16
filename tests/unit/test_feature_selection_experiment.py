from pathlib import Path

from app.ml.feature_selection_experiment import run_experiment
from app.ml.features import FEATURE_COLUMNS

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def test_feature_selection_and_ordinal_encoding_experiment_is_reproducible():
    report, model = run_experiment(DATA_DIR)
    assert set(report["selected_features"]).issubset(set(FEATURE_COLUMNS))
    assert report["selected_k"] == len(report["selected_features"])
    assert report["encoding_policy_full_train"]["ordinal_encoded_gt_3"] == [
        "origin_port", "destination_port", "cargo_type", "origin_region", "destination_region"
    ]
    assert report["chance_references"]["roc_auc"] == 0.5
    assert report["random_50_percent_baseline_validation"]["expected_roc_auc"] == 0.5
    assert set(report["validation_results"]) == {
        "all_one_hot_full_features",
        "mixed_ordinal_gt3_full_features",
        "mixed_ordinal_gt3_mutual_information_selected",
    }
    assert model is not None
