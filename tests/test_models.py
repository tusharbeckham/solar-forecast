import numpy as np
import pandas as pd

from solar_forecast import evaluate, physics
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
