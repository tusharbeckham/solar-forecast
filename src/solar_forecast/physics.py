"""Physics: clear-sky irradiance prior + a simple PV power model.

Self-contained: uses a numpy solar-geometry + Haurwitz clear-sky model by default so the package
works with no extra deps. If `pvlib` is installed, the more accurate Ineichen clear-sky model is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Site

try:  # optional, more accurate clear-sky
    import pvlib  # noqa: F401
    _HAS_PVLIB = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_PVLIB = False


def _ensure_utc(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def cos_zenith(site: Site, index) -> np.ndarray:
    """Cosine of the solar zenith angle for each UTC timestamp, clipped to [-1, 1]."""
    idx = _ensure_utc(index)
    doy = idx.dayofyear.to_numpy()
    hour_utc = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    decl = np.radians(23.45) * np.sin(np.radians(360.0 / 365.0 * (doy - 81)))
    solar_time = hour_utc + site.longitude / 15.0          # approx local solar time
    hour_angle = np.radians(15.0 * (solar_time - 12.0))
    lat = np.radians(site.latitude)
    cz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return np.clip(cz, -1.0, 1.0)


def clear_sky_ghi(site: Site, index) -> pd.Series:
    """Clear-sky global horizontal irradiance (W/m^2)."""
    idx = _ensure_utc(index)
    if _HAS_PVLIB:
        loc = pvlib.location.Location(site.latitude, site.longitude, tz="UTC", altitude=site.altitude_m)
        return loc.get_clearsky(idx, model="ineichen")["ghi"].astype(float)
    cz = cos_zenith(site, idx)
    ghi = np.where(cz > 0, 1098.0 * cz * np.exp(-0.059 / np.maximum(cz, 1e-6)), 0.0)
    return pd.Series(ghi, index=idx, name="ghi_cs")


def pv_power(ghi, temp_c, site: Site) -> np.ndarray:
    """PV power (W) from GHI: P = area * eff * GHI * PR, with temperature derating.

    Cell temperature is approximated as ambient plus an irradiance-driven rise.
    """
    ghi = np.asarray(ghi, dtype=float)
    temp_c = np.asarray(temp_c, dtype=float)
    t_cell = temp_c + ghi * 0.025                       # ~+20 C at 800 W/m^2
    derate = 1.0 + site.temp_coeff_per_c * (t_cell - 25.0)
    p = site.panel_area_m2 * site.efficiency * ghi * site.performance_ratio * derate
    return np.clip(p, 0.0, None)


def clear_sky_power(site: Site, index, temp_c=25.0) -> pd.Series:
    """Expected PV power under clear skies (the physical prior)."""
    idx = _ensure_utc(index)
    ghi_cs = clear_sky_ghi(site, idx)
    tc = np.full(len(idx), float(temp_c)) if np.isscalar(temp_c) else np.asarray(temp_c, dtype=float)
    return pd.Series(pv_power(ghi_cs.to_numpy(), tc, site), index=ghi_cs.index, name="clear_sky_power")


def observed_power(site: Site, df: pd.DataFrame) -> pd.Series:
    """Proxy 'actual' PV power from OBSERVED ghi + temperature.

    Stand-in for real PV telemetry: applies the PV model to the measured (cloud-affected) GHI.
    Replace with real inverter output when available.
    """
    idx = _ensure_utc(df.index)
    ghi = df["ghi"].to_numpy(dtype=float)
    temp = df["temp_c"].to_numpy(dtype=float) if "temp_c" in df.columns else np.full(len(df), 25.0)
    return pd.Series(pv_power(ghi, temp, site), index=idx, name="observed_power")
