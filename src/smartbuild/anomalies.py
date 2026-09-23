from __future__ import annotations

import numpy as np
import pandas as pd


def score_anomalies(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("actual and predicted must have the same shape")
    residual = actual_values - predicted_values
    severity = np.abs(residual) / max(float(threshold), 1e-9)
    return pd.DataFrame(
        {
            "actual_energy_kwh": actual_values,
            "predicted_energy_kwh": predicted_values,
            "residual_kwh": residual,
            "anomaly_score": severity,
            "is_anomaly": (severity > 1).astype(int),
        }
    )


def operational_explanation(row: pd.Series) -> str:
    """Translate a flagged residual into a cautious, domain-readable message."""
    residual = float(row.get("residual_kwh", 0))
    if residual <= 0:
        return "Consumption is below the model expectation; verify schedules before treating this as a fault."
    temperature = float(row.get("outdoor_temp_c", 20))
    occupancy = float(row.get("occupancy", 0))
    hour = pd.Timestamp(row.get("timestamp")).hour if "timestamp" in row else 12
    if hour < 6 or hour > 21:
        return "Unexpected after-hours demand; review HVAC, lighting, and plug-load schedules."
    if temperature < 5 or temperature > 28:
        return (
            "Elevated demand coincides with extreme outdoor temperature; "
            "inspect HVAC efficiency and setpoints."
        )
    if occupancy < 10:
        return (
            "Demand is high relative to low occupancy; check equipment left operating or schedule overrides."
        )
    return (
        "Demand exceeds the expected operating range; inspect HVAC and plug-load "
        "subsystems for abnormal operation."
    )
