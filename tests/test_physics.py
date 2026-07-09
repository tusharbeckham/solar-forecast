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


def test_cos_zenith_day_positive_night_negative():
    cz = physics.cos_zenith(DEFAULT_SITE, _day_index())
    assert cz.max() > 0.5               # sun high near solar noon
    assert cz.min() < 0.0               # sun below the horizon at night
    assert cz.max() <= 1.0 and cz.min() >= -1.0


def test_cos_aoi_in_range_and_faces_sun_midday():
    ca = physics.cos_aoi(DEFAULT_SITE, _day_index())
    assert ca.max() <= 1.0 and ca.min() >= -1.0
    assert ca.max() > 0.0               # panel faces the sun around midday


def test_poa_global_nonneg_scales_and_zero_at_night():
    idx = _day_index()
    ghi = physics.clear_sky_ghi(DEFAULT_SITE, idx).to_numpy()
    dhi = 0.12 * ghi
    direct_h = ghi - dhi
    poa = physics.poa_global(DEFAULT_SITE, idx, ghi, direct_h, dhi)
    assert (poa >= 0).all()
    # POA scales up with the horizontal components
    poa2 = physics.poa_global(DEFAULT_SITE, idx, 2 * ghi, 2 * direct_h, 2 * dhi)
    assert poa2.sum() >= poa.sum() > 0.0
    # no sun (deep night: clear-sky GHI is exactly 0) -> zero plane-of-array irradiance
    deep_night = ghi <= 1e-9
    assert np.all(poa[deep_night] == 0.0)


def test_clear_sky_power_nonneg_with_daytime_peak():
    p = physics.clear_sky_power(DEFAULT_SITE, _day_index())
    assert (p >= 0).all()
    assert p.max() > 0.0


def test_observed_power_from_components_nonneg():
    idx = _day_index()
    ghi = physics.clear_sky_ghi(DEFAULT_SITE, idx).to_numpy()
    df = pd.DataFrame(
        {"ghi": ghi, "direct": 0.88 * ghi, "dhi": 0.12 * ghi, "temp_c": 25.0}, index=idx
    )
    p = physics.observed_power(DEFAULT_SITE, df)
    assert (p >= 0).all()
    assert p.max() > 0.0


def test_pv_power_increases_with_irradiance():
    lo = physics.pv_power(np.array([200.0]), np.array([25.0]), DEFAULT_SITE)[0]
    hi = physics.pv_power(np.array([800.0]), np.array([25.0]), DEFAULT_SITE)[0]
    assert hi > lo > 0.0
    assert physics.pv_power(np.array([0.0]), np.array([25.0]), DEFAULT_SITE)[0] == 0.0
