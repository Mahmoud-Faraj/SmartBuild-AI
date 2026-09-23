from smartbuild.config import FEATURE_COLUMNS
from smartbuild.data import generate_demo_data
from smartbuild.features import build_features


def test_features_use_only_historical_target_values():
    raw = generate_demo_data(days=30)
    featured = build_features(raw)
    assert set(FEATURE_COLUMNS).issubset(featured.columns)
    first = featured.iloc[0]
    source_index = raw.index[raw["timestamp"] == first["timestamp"]][0]
    assert first["lag_1h"] == raw.loc[source_index - 1, "total_energy_kwh"]
    assert first["lag_24h"] == raw.loc[source_index - 24, "total_energy_kwh"]
