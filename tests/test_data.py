import pandas as pd
import pytest

from smartbuild.data import clean_input, generate_demo_data


def test_demo_data_is_reproducible_and_complete():
    first = generate_demo_data(days=30, random_state=7)
    second = generate_demo_data(days=30, random_state=7)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 30 * 24
    assert first["injected_anomaly"].sum() > 0
    assert (first["total_energy_kwh"] > 0).all()


def test_clean_input_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_input(pd.DataFrame({"timestamp": ["2026-01-01"]}))
