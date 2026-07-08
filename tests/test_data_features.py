import numpy as np
import pandas as pd

from solar_forecast import data, features
from solar_forecast.config import DEFAULT_SITE
from solar_forecast.features import FEATURE_COLUMNS


def _synth(n=72):
    idx = pd.date_range("2023-06-01T00:00", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "ghi": np.abs(np.sin(np.arange(n) / 6.0)) * 600,
            "temp_c": 25 + 5 * np.sin(np.arange(n) / 12.0),
            "cloud_pct": (np.arange(n) * 7 % 100).astype(float),
            "wind_ms": 2.0,
        },
        index=idx,
    )


def test_features_build_columns_and_no_nan():
    feats = features.build(_synth(), DEFAULT_SITE)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert not feats.isna().any().any()
    assert len(feats) == 72 - 24  # 24h lag drops the first day


def test_parse_open_meteo_schema():
    payload = {
        "hourly": {
            "time": ["2023-06-01T00:00", "2023-06-01T01:00"],
            "shortwave_radiation": [0.0, 10.0],
            "direct_radiation": [0.0, 5.0],
            "diffuse_radiation": [0.0, 5.0],
            "temperature_2m": [24.0, 23.5],
            "cloud_cover": [10.0, 20.0],
            "wind_speed_10m": [1.0, 1.2],
        }
    }
    df = data.parse_open_meteo(payload)
    assert list(df.columns) == ["ghi", "direct", "dhi", "temp_c", "cloud_pct", "wind_ms"]
    assert str(df.index.tz) == "UTC"
    assert len(df) == 2
    assert not df[["ghi", "temp_c"]].isna().any().any()
