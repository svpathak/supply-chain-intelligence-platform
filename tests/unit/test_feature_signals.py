from pathlib import Path

from app.ml.feature_signals import FEATURE_GROUPS, run_feature_signal_analysis
from app.ml.features import FEATURE_COLUMNS

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def test_feature_groups_cover_allow_list_without_duplicates():
    assert set(FEATURE_GROUPS["full"]) == set(FEATURE_COLUMNS)
    assert len(FEATURE_GROUPS["full"]) == len(set(FEATURE_GROUPS["full"]))


def test_feature_signal_analysis_uses_validation_and_reports_random_baseline():
    report = run_feature_signal_analysis(DATA_DIR)
    assert "validation_results" in report
    assert "test" not in str(report["validation_results"]).lower()
    assert report["random_50_percent_baseline_validation"]["expected_roc_auc"] == 0.5
    assert set(report["validation_results"]) >= {
        "schedule_calendar", "route_identity", "port_context",
        "shipment_profile", "full"
    }
