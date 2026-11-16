from pathlib import Path

import joblib

from app.ml.train import train_and_evaluate

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def test_training_writes_single_model_and_random_baseline(tmp_path):
    report = train_and_evaluate(DATA_DIR, tmp_path / "artifacts", tmp_path / "report.json")
    assert report["selected_model"] == "logistic_regression"
    assert report["production_ready"] is False
    for name in ["delay_final.joblib", "delay_logistic_regression.joblib", "delay_model_metadata.joblib"]:
        path = tmp_path / "artifacts" / name
        assert path.exists()
        joblib.load(path)
    assert "random_50_percent_baseline" in report["model"]["validation"]
    assert report["model"]["validation"]["random_50_percent_baseline"]["expected_roc_auc"] == 0.5
    assert report["model"]["validation"]["roc_auc"] >= 0
    assert report["model"]["test"]["roc_auc"] <= 1
