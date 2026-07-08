import numpy as np
import pandas as pd

from solar_forecast import physics
from solar_forecast.config import DEFAULT_SITE


def _day_index(date="2023-06-21"):
    return pd.date_range(f"{date}T00:00", f"{date}T23:00", freq="h", tz="UTC")


def test_clear_sky_ghi_has_day_and_night():
    ghi = physics.clear_sky_ghi(DEFAULT_SITE, _day_index())
    assert ghi.max() > 100.0            # strong midday sun
    assert (ghi < 1.0).sum() >= 5       # several night hours ~ 0
    assert float(ghi.min()) >= 0.0


def test_clear_sky_power_nonneg_with_daytime_peak():
    p = physics.clear_sky_power(DEFAULT_SITE, _day_index())
    assert (p >= 0).all()
    assert p.max() > 0.0


def test_pv_power_increases_with_irradiance():
    lo = physics.pv_power(np.array([200.0]), np.array([25.0]), DEFAULT_SITE)[0]
    hi = physics.pv_power(np.array([800.0]), np.array([25.0]), DEFAULT_SITE)[0]
    assert hi > lo > 0.0
    assert physics.pv_power(np.array([0.0]), np.array([25.0]), DEFAULT_SITE)[0] == 0.0
