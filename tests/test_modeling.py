from smartbuild.data import generate_demo_data
from smartbuild.modeling import (
    anomaly_classification_metrics,
    calibrate_residual_threshold,
    load_model,
    train_model,
)


def test_training_creates_valid_artifact(tmp_path):
    model_path = tmp_path / "model.joblib"
    result = train_model(generate_demo_data(days=45), model_path)
    artifact = load_model(model_path)
    assert model_path.exists()
    assert result.metrics["mae"] < result.metrics["baseline_mae"]
    assert result.metrics["r2"] > 0.5
    assert artifact["residual_threshold"] > 0
    assert "holdout_start" in artifact
    assert 0 <= result.metrics["anomaly_precision"] <= 1
    assert 0 <= result.metrics["anomaly_recall"] <= 1
    assert 0 <= result.metrics["anomaly_f1"] <= 1


def test_anomaly_metrics_match_known_confusion_counts():
    metrics = anomaly_classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert metrics["anomaly_precision"] == 0.5
    assert metrics["anomaly_recall"] == 0.5
    assert metrics["anomaly_f1"] == 0.5
    assert metrics["anomaly_true_positives"] == 1
    assert metrics["anomaly_false_positives"] == 1


def test_threshold_calibration_supports_labelled_and_unlabelled_data():
    residuals = [0.1, 0.2, 0.3, 3.0, 4.0]
    labelled, method = calibrate_residual_threshold(residuals, [0, 0, 0, 1, 1])
    percentile, fallback_method = calibrate_residual_threshold(residuals)
    assert 0.3 <= labelled < 3.0
    assert method == "calibration_f1_with_injected_labels"
    assert percentile > 3.0
    assert fallback_method == "calibration_97_5_percentile"
