from pathlib import Path

import joblib

from app.ml.neural_network_experiment import HIDDEN_UNITS, run_experiment

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def test_minimal_mlp_is_one_hidden_layer_and_remains_experimental(tmp_path):
    report = run_experiment(
        data_dir=DATA_DIR,
        artifact_dir=tmp_path / "artifacts",
        report_path=tmp_path / "reports" / "mlp.json",
    )
    assert HIDDEN_UNITS == 4
    assert report["architecture"]["hidden_layers"] == 1
    assert report["architecture"]["hidden_units"] == 4
    assert report["architecture"]["input_dimensions_after_preprocessing"] == 80
    assert report["experiment_status"] == "experimental_not_promoted"
    assert report["model"]["validation"]["rows"] == 675
    assert "predicted_class_counts" in report["model"]["validation"]
    assert report["reference_model"]["name"] == "logistic_regression"
    artifact = tmp_path / "artifacts" / "delay_mlp_experiment.joblib"
    assert artifact.exists()
    loaded = joblib.load(artifact)
    assert loaded.named_steps["classifier"].hidden_layer_sizes == (4,)
