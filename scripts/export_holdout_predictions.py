"""Export out-of-sample actual-versus-predicted values for inspection."""

import json
from pathlib import Path

import pandas as pd

from smartbuild.config import DATA_PATH, FEATURE_COLUMNS, MODEL_PATH, TARGET
from smartbuild.features import build_features
from smartbuild.modeling import load_model

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    data = build_features(raw)
    artifact = load_model(MODEL_PATH)
    holdout_start = pd.Timestamp(artifact["holdout_start"])
    holdout = data.loc[data["timestamp"] >= holdout_start].copy()

    holdout["predicted_energy_kwh"] = artifact["model"].predict(holdout[FEATURE_COLUMNS])
    holdout["residual_kwh"] = holdout[TARGET] - holdout["predicted_energy_kwh"]
    holdout["absolute_error_kwh"] = holdout["residual_kwh"].abs()
    holdout["anomaly_score"] = holdout["absolute_error_kwh"] / artifact["residual_threshold"]
    holdout["is_anomaly"] = (holdout["anomaly_score"] > 1).astype(int)

    columns = [
        "timestamp",
        "outdoor_temp_c",
        "humidity_pct",
        "occupancy",
        TARGET,
        "predicted_energy_kwh",
        "residual_kwh",
        "absolute_error_kwh",
        "anomaly_score",
        "is_anomaly",
        "injected_anomaly",
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = OUTPUT_DIR / "holdout_predictions.csv"
    metrics_path = OUTPUT_DIR / "holdout_metrics.json"
    holdout[columns].to_csv(predictions_path, index=False)
    metrics_path.write_text(
        json.dumps(artifact["metrics"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(holdout):,} holdout predictions to {predictions_path}")
    print(f"Wrote evaluation metrics to {metrics_path}")


if __name__ == "__main__":
    main()
