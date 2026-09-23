from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, TARGET
from .data import clean_input


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create time and historical features without using future observations."""
    data = clean_input(frame)
    ts = data["timestamp"]
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    month = ts.dt.month

    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    data["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    data["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    data["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    data["is_weekend"] = (dow >= 5).astype(int)
    data["lag_1h"] = data[TARGET].shift(1)
    data["lag_24h"] = data[TARGET].shift(24)
    data["rolling_24h"] = data[TARGET].shift(1).rolling(24, min_periods=24).mean()
    return data.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
