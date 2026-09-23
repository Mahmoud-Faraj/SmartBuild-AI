from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_STATE, RAW_COLUMNS


def generate_demo_data(
    days: int = 180,
    start: str = "2026-01-01",
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate deterministic, realistic hourly commercial-building telemetry.

    The included dataset is synthetic and is intended for a reproducible demo.
    Injected anomalies emulate schedule overruns and abnormal HVAC/plug loads.
    """
    if days < 14:
        raise ValueError("days must be at least 14 to support lagged features")

    rng = np.random.default_rng(random_state)
    ts = pd.date_range(start=start, periods=days * 24, freq="h")
    frame = pd.DataFrame({"timestamp": ts})
    day_of_year = frame["timestamp"].dt.dayofyear.to_numpy()
    hour = frame["timestamp"].dt.hour.to_numpy()
    weekday = frame["timestamp"].dt.dayofweek.to_numpy()

    seasonal = -2.0 + 16.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    daily = 3.5 * np.sin(2 * np.pi * (hour - 8) / 24)
    frame["outdoor_temp_c"] = seasonal + daily + rng.normal(0, 1.7, len(frame))
    frame["humidity_pct"] = np.clip(
        63 - 0.75 * frame["outdoor_temp_c"] + rng.normal(0, 6, len(frame)), 25, 95
    )

    open_hours = (hour >= 7) & (hour <= 19) & (weekday < 5)
    arrival_curve = np.clip(np.sin(np.pi * (hour - 7) / 12), 0, 1)
    occupancy = np.where(open_hours, 18 + 102 * arrival_curve, 2)
    frame["occupancy"] = np.maximum(0, np.rint(occupancy + rng.normal(0, 6, len(frame)))).astype(int)

    heating = np.maximum(18 - frame["outdoor_temp_c"], 0)
    cooling = np.maximum(frame["outdoor_temp_c"] - 22, 0)
    frame["hvac_kwh"] = np.maximum(
        3.2 + 0.58 * heating + 0.72 * cooling + 0.035 * frame["occupancy"] + rng.normal(0, 0.8, len(frame)),
        1.0,
    )
    frame["lighting_kwh"] = np.maximum(
        1.1 + 4.8 * open_hours.astype(float) + 0.018 * frame["occupancy"] + rng.normal(0, 0.35, len(frame)),
        0.6,
    )
    frame["plug_load_kwh"] = np.maximum(
        1.8 + 0.055 * frame["occupancy"] + rng.normal(0, 0.45, len(frame)), 0.8
    )
    frame["critical_load_kwh"] = np.maximum(
        3.6 + 0.004 * frame["occupancy"] + rng.normal(0, 0.15, len(frame)), 2.8
    )

    anomaly = np.zeros(len(frame), dtype=bool)
    eligible = np.arange(48, len(frame) - 24)
    starts = rng.choice(eligible, size=max(12, days // 8), replace=False)
    for start_index in starts:
        duration = int(rng.integers(2, 7))
        end_index = min(start_index + duration, len(frame))
        anomaly[start_index:end_index] = True
        if rng.random() < 0.65:
            frame.loc[start_index : end_index - 1, "hvac_kwh"] *= rng.uniform(1.45, 1.85)
        else:
            frame.loc[start_index : end_index - 1, "plug_load_kwh"] += rng.uniform(4.0, 7.0)

    frame["total_energy_kwh"] = frame[["hvac_kwh", "lighting_kwh", "plug_load_kwh", "critical_load_kwh"]].sum(
        axis=1
    ) + rng.normal(0, 0.25, len(frame))
    frame["injected_anomaly"] = anomaly.astype(int)
    return frame[RAW_COLUMNS]


def validate_input(frame: pd.DataFrame) -> None:
    required = {
        "timestamp",
        "outdoor_temp_c",
        "humidity_pct",
        "occupancy",
        "total_energy_kwh",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Input data is empty")


def clean_input(frame: pd.DataFrame) -> pd.DataFrame:
    validate_input(frame)
    cleaned = frame.copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="raise")
    cleaned = cleaned.sort_values("timestamp").drop_duplicates("timestamp")
    numeric = ["outdoor_temp_c", "humidity_pct", "occupancy", "total_energy_kwh"]
    cleaned[numeric] = cleaned[numeric].apply(pd.to_numeric, errors="coerce")
    if cleaned[numeric].isna().any().any():
        raise ValueError("Required numeric columns contain missing or invalid values")
    return cleaned.reset_index(drop=True)
