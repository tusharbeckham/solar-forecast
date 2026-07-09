import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

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


def test_features_include_poa_aoi_and_rolling():
    feats = features.build(_synth(96), DEFAULT_SITE)
    # POA-era features present
    for col in ("cos_aoi", "csi_roll24", "clear_sky_index", "ghi_cs"):
        assert col in feats.columns
    # cos(AOI) is clipped into [0, 1]; clear-sky index is non-negative
    assert feats["cos_aoi"].between(0.0, 1.0).all()
    assert (feats["clear_sky_index"] >= 0.0).all()


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


def test_get_forecast_hits_forecast_api_and_parses():
    payload = {
        "hourly": {
            "time": ["2026-01-01T00:00", "2026-01-01T01:00"],
            "shortwave_radiation": [0.0, 12.0],
            "direct_radiation": [0.0, 6.0],
            "diffuse_radiation": [0.0, 6.0],
            "temperature_2m": [20.0, 20.5],
            "cloud_cover": [5.0, 8.0],
            "wind_speed_10m": [1.0, 1.1],
        }
    }
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    with patch.object(data.requests, "get", return_value=resp) as mock_get:
        df = data.get_forecast(DEFAULT_SITE, past_days=30, forecast_days=3)

    _, kwargs = mock_get.call_args
    assert mock_get.call_args[0][0] == data.FORECAST_URL      # forecast endpoint, not archive
    assert kwargs["params"]["past_days"] == 30
    assert kwargs["params"]["forecast_days"] == 3
    assert list(df.columns) == ["ghi", "direct", "dhi", "temp_c", "cloud_pct", "wind_ms"]
    assert len(df) == 2
