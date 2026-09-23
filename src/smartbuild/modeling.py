from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from .config import FEATURE_COLUMNS, MODEL_PATH, RANDOM_STATE, TARGET
from .features import build_features


@dataclass(frozen=True)
class TrainingResult:
    metrics: dict[str, float]
    test_results: pd.DataFrame
    feature_importance: pd.DataFrame
    split_timestamp: pd.Timestamp


def anomaly_classification_metrics(
    labels: pd.Series | np.ndarray,
    predictions: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Return transparent binary event-detection metrics."""
    y_true = np.asarray(labels, dtype=int)
    y_pred = np.asarray(predictions, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError("labels and predictions must have the same shape")
    return {
        "anomaly_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "anomaly_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "anomaly_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "anomaly_true_positives": float(np.sum((y_true == 1) & (y_pred == 1))),
        "anomaly_false_positives": float(np.sum((y_true == 0) & (y_pred == 1))),
        "anomaly_false_negatives": float(np.sum((y_true == 1) & (y_pred == 0))),
        "anomaly_true_negatives": float(np.sum((y_true == 0) & (y_pred == 0))),
    }


def calibrate_residual_threshold(
    absolute_residuals: pd.Series | np.ndarray,
    labels: pd.Series | np.ndarray | None = None,
) -> tuple[float, str]:
    """Calibrate a residual threshold without touching final holdout data.

    When synthetic labels are available, select the calibration threshold that
    maximizes F1. For unlabeled uploads, use a robust high residual percentile.
    """
    residuals = np.asarray(absolute_residuals, dtype=float)
    if residuals.size == 0:
        raise ValueError("absolute_residuals must not be empty")
    if labels is None:
        return float(np.quantile(residuals, 0.975)), "calibration_97_5_percentile"

    y_true = np.asarray(labels, dtype=int)
    if y_true.shape != residuals.shape:
        raise ValueError("labels and residuals must have the same shape")
    ordered = np.unique(np.sort(residuals))
    candidates = (ordered[:-1] + ordered[1:]) / 2
    if candidates.size == 0:
        candidates = ordered
    scored = [
        (f1_score(y_true, residuals > threshold, zero_division=0), threshold) for threshold in candidates
    ]
    _, best_threshold = max(scored, key=lambda item: (item[0], item[1]))
    return float(best_threshold), "calibration_f1_with_injected_labels"


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_model(
    raw_data: pd.DataFrame,
    model_path: Path | str = MODEL_PATH,
    test_fraction: float = 0.2,
    calibration_fraction: float = 0.1,
) -> TrainingResult:
    """Train, calibrate the anomaly threshold, and test chronologically."""
    if not 0.1 <= test_fraction <= 0.4:
        raise ValueError("test_fraction must be between 0.1 and 0.4")
    if not 0.05 <= calibration_fraction <= 0.2:
        raise ValueError("calibration_fraction must be between 0.05 and 0.2")
    if test_fraction + calibration_fraction >= 0.5:
        raise ValueError("test_fraction and calibration_fraction leave too little training data")

    data = build_features(raw_data)
    holdout_index = int(len(data) * (1 - test_fraction))
    calibration_index = int(len(data) * (1 - test_fraction - calibration_fraction))
    if calibration_index < 100 or holdout_index - calibration_index < 24 or len(data) - holdout_index < 24:
        raise ValueError("Not enough observations for a reliable chronological split")

    train = data.iloc[:calibration_index]
    calibration = data.iloc[calibration_index:holdout_index]
    test = data.iloc[holdout_index:]
    x_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
    x_calibration, y_calibration = calibration[FEATURE_COLUMNS], calibration[TARGET]
    x_test, y_test = test[FEATURE_COLUMNS], test[TARGET]

    baseline = DummyRegressor(strategy="mean").fit(x_train, y_train)
    model = RandomForestRegressor(
        n_estimators=240,
        min_samples_leaf=3,
        max_features=0.8,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ).fit(x_train, y_train)

    prediction = model.predict(x_test)
    baseline_prediction = baseline.predict(x_test)
    metrics = regression_metrics(y_test, prediction)
    baseline_metrics = regression_metrics(y_test, baseline_prediction)
    metrics.update({f"baseline_{key}": value for key, value in baseline_metrics.items()})
    metrics["mae_improvement_pct"] = (
        100 * (baseline_metrics["mae"] - metrics["mae"]) / baseline_metrics["mae"]
    )

    calibration_prediction = model.predict(x_calibration)
    calibration_residual = y_calibration.to_numpy() - calibration_prediction
    calibration_labels = (
        calibration["injected_anomaly"] if "injected_anomaly" in calibration.columns else None
    )
    residual_threshold, threshold_method = calibrate_residual_threshold(
        np.abs(calibration_residual), calibration_labels
    )
    residual = y_test.to_numpy() - prediction
    result_frame = test.copy()
    result_frame["predicted_energy_kwh"] = prediction
    result_frame["residual_kwh"] = residual
    result_frame["is_anomaly"] = (np.abs(residual) > residual_threshold).astype(int)
    if "injected_anomaly" in result_frame.columns:
        metrics.update(
            anomaly_classification_metrics(result_frame["injected_anomaly"], result_frame["is_anomaly"])
        )

    importance = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_frame = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importance.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "residual_threshold": residual_threshold,
        "threshold_method": threshold_method,
        "feature_importance": importance_frame,
        "trained_through": str(train["timestamp"].iloc[-1]),
        "calibrated_through": str(calibration["timestamp"].iloc[-1]),
        "holdout_start": str(test["timestamp"].iloc[0]),
        "split_counts": {
            "training": len(train),
            "calibration": len(calibration),
            "holdout": len(test),
        },
    }
    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    return TrainingResult(metrics, result_frame, importance_frame, test["timestamp"].iloc[0])


def load_model(model_path: Path | str = MODEL_PATH) -> dict:
    artifact = joblib.load(model_path)
    expected = {"model", "feature_columns", "metrics", "residual_threshold"}
    missing = expected.difference(artifact)
    if missing:
        raise ValueError(f"Invalid model artifact; missing: {sorted(missing)}")
    return artifact
