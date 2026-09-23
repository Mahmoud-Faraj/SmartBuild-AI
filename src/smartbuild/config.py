from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "building_energy.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "energy_model.joblib"
RANDOM_STATE = 42
TARGET = "total_energy_kwh"

RAW_COLUMNS = [
    "timestamp",
    "outdoor_temp_c",
    "humidity_pct",
    "occupancy",
    "hvac_kwh",
    "lighting_kwh",
    "plug_load_kwh",
    "critical_load_kwh",
    TARGET,
    "injected_anomaly",
]

FEATURE_COLUMNS = [
    "outdoor_temp_c",
    "humidity_pct",
    "occupancy",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "lag_1h",
    "lag_24h",
    "rolling_24h",
]
