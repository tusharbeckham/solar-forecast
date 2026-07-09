import numpy as np
import pandas as pd

from solar_forecast import evaluate, features, models, physics
from solar_forecast.config import DEFAULT_SITE


def _synth_cloudy(days=120):
    """Actual GHI = clear-sky GHI reduced by random clouds; the residual model should recover this."""
    idx = pd.date_range("2023-03-01T00:00", periods=days * 24, freq="h", tz="UTC")
    ghi_cs = physics.clear_sky_ghi(DEFAULT_SITE, idx).to_numpy()
    rng = np.random.default_rng(0)
    cloud = rng.uniform(0.0, 1.0, len(idx))
    ghi = ghi_cs * (1.0 - 0.7 * cloud)
    return pd.DataFrame(
        {"ghi": ghi, "temp_c": 25.0, "cloud_pct": cloud * 100.0, "wind_ms": 2.0}, index=idx
    )


def test_residual_model_beats_clear_sky():
    result = evaluate.backtest(DEFAULT_SITE, _synth_cloudy(), n_splits=4)
    # Clear-sky ignores clouds; the residual model uses them -> positive skill score.
    assert result["skill_vs_clear_sky"] > 0.0
    assert result["rmse"] >= 0.0


def test_residual_forecaster_fit_predict_nonneg():
    df = _synth_cloudy(60)
    feats = features.build(df, DEFAULT_SITE)
    y = physics.observed_power(DEFAULT_SITE, df).reindex(feats.index)
    clear = physics.clear_sky_power(DEFAULT_SITE, feats.index, temp_c=feats["temp_c"]).reindex(feats.index)

    model = models.ResidualForecaster(DEFAULT_SITE).fit(feats, y - clear)
    pred = model.predict(feats, clear)

    assert len(pred) == len(feats)
    assert (pred >= 0).all()                      # forecast is clipped to >= 0
    assert model.columns == list(features.FEATURE_COLUMNS)


def test_baseline_persistence_shifts_by_lag():
    y = pd.Series(np.arange(48.0))
    p = models.baseline_persistence(y, lag=24)
    assert p.isna().sum() == 24                   # first `lag` values are unknown
    assert p.iloc[24] == y.iloc[0]                # value from 24h ago


def test_baseline_clear_sky_nonneg():
    df = _synth_cloudy(3)
    base = models.baseline_clear_sky(DEFAULT_SITE, df)
    assert len(base) == len(df)
    assert (base >= 0).all()
