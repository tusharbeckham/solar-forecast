"""Physics: clear-sky irradiance prior + PV power model, with plane-of-array (POA) transposition.

Self-contained numpy solar geometry + Haurwitz clear-sky + isotropic POA by default. Uses pvlib's
Ineichen clear-sky model if pvlib is installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Site

try:  # optional, more accurate clear-sky
    import pvlib  # noqa: F401
    _HAS_PVLIB = True
except Exception:  # pragma: no cover
    _HAS_PVLIB = False

GROUND_ALBEDO = 0.2


def _ensure_utc(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def _solar_angles(site: Site, index):
    """Return (cos_zenith, sin_zenith, solar_azimuth_rad) as numpy arrays (azimuth clockwise from North)."""
    idx = _ensure_utc(index)
    doy = idx.dayofyear.to_numpy()
    hour_utc = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    decl = np.radians(23.45) * np.sin(np.radians(360.0 / 365.0 * (doy - 81)))
    solar_time = hour_utc + site.longitude / 15.0
    H = np.radians(15.0 * (solar_time - 12.0))  # hour angle
    lat = np.radians(site.latitude)
    cos_z = np.clip(np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(H), -1.0, 1.0)
    sin_z = np.sqrt(np.clip(1.0 - cos_z ** 2, 0.0, 1.0))
    denom = np.cos(lat) * sin_z
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    cos_az = np.clip((np.sin(decl) - np.sin(lat) * cos_z) / denom, -1.0, 1.0)
    az = np.arccos(cos_az)                       # 0..pi, from North
    az = np.where(H > 0, 2 * np.pi - az, az)     # afternoon -> western sky
    return cos_z, sin_z, az


def cos_zenith(site: Site, index) -> np.ndarray:
    """Cosine of the solar zenith angle (clipped to [-1, 1]); > 0 in daytime."""
    cz, _, _ = _solar_angles(site, index)
    return cz


def cos_aoi(site: Site, index) -> np.ndarray:
    """Cosine of the angle of incidence of beam radiation on the tilted panel."""
    cz, sz, az = _solar_angles(site, index)
    tilt = np.radians(site.tilt_deg)
    surf_az = np.radians(site.azimuth_deg)
    ca = cz * np.cos(tilt) + sz * np.sin(tilt) * np.cos(az - surf_az)
    return np.clip(ca, -1.0, 1.0)


def clear_sky_ghi(site: Site, index) -> pd.Series:
    """Clear-sky global horizontal irradiance (W/m^2)."""
    idx = _ensure_utc(index)
    if _HAS_PVLIB:
        loc = pvlib.location.Location(site.latitude, site.longitude, tz="UTC", altitude=site.altitude_m)
        return loc.get_clearsky(idx, model="ineichen")["ghi"].astype(float)
    cz, _, _ = _solar_angles(site, idx)
    ghi = np.where(cz > 0, 1098.0 * cz * np.exp(-0.059 / np.maximum(cz, 1e-6)), 0.0)
    return pd.Series(ghi, index=idx, name="ghi_cs")


def poa_global(site: Site, index, ghi, direct_h, dhi, albedo: float = GROUND_ALBEDO) -> np.ndarray:
    """Plane-of-array global irradiance (W/m^2) via an isotropic sky model.

    Inputs are HORIZONTAL components: ghi, direct_h (beam on horizontal), dhi (diffuse horizontal).
    POA = beam-on-tilt + isotropic diffuse-on-tilt + ground-reflected.
    """
    idx = _ensure_utc(index)
    cz, sz, az = _solar_angles(site, idx)
    tilt = np.radians(site.tilt_deg)
    ca = cos_aoi(site, idx)
    ghi = np.asarray(ghi, float)
    direct_h = np.asarray(direct_h, float)
    dhi = np.asarray(dhi, float)
    dni = np.where(cz > 0.02, direct_h / np.maximum(cz, 0.02), 0.0)   # beam normal
    beam_poa = dni * np.clip(ca, 0.0, None)
    diffuse_poa = dhi * (1.0 + np.cos(tilt)) / 2.0
    ground_poa = ghi * albedo * (1.0 - np.cos(tilt)) / 2.0
    return np.clip(beam_poa + diffuse_poa + ground_poa, 0.0, None)


def pv_power(irradiance, temp_c, site: Site) -> np.ndarray:
    """PV power (W) from plane-of-array (or GHI) irradiance, with temperature derating."""
    g = np.asarray(irradiance, float)
    t = np.asarray(temp_c, float)
    t_cell = t + g * 0.025
    derate = 1.0 + site.temp_coeff_per_c * (t_cell - 25.0)
    return np.clip(site.panel_area_m2 * site.efficiency * g * site.performance_ratio * derate, 0.0, None)


def clear_sky_power(site: Site, index, temp_c=25.0) -> pd.Series:
    """Expected PV power under clear skies (the physical prior), using a clear-sky POA estimate."""
    idx = _ensure_utc(index)
    ghi_cs = clear_sky_ghi(site, idx).to_numpy()
    dhi_cs = 0.12 * ghi_cs                 # clear skies: ~12% diffuse
    direct_h_cs = ghi_cs - dhi_cs
    poa_cs = poa_global(site, idx, ghi_cs, direct_h_cs, dhi_cs)
    tc = np.full(len(idx), float(temp_c)) if np.isscalar(temp_c) else np.asarray(temp_c, float)
    return pd.Series(pv_power(poa_cs, tc, site), index=idx, name="clear_sky_power")


def observed_power(site: Site, df: pd.DataFrame) -> pd.Series:
    """Proxy 'actual' PV power from OBSERVED irradiance components (uses POA). Stand-in for telemetry."""
    idx = _ensure_utc(df.index)
    ghi = df["ghi"].to_numpy(float)
    direct_h = df["direct"].to_numpy(float) if "direct" in df.columns else 0.88 * ghi
    dhi = df["dhi"].to_numpy(float) if "dhi" in df.columns else 0.12 * ghi
    temp = df["temp_c"].to_numpy(float) if "temp_c" in df.columns else np.full(len(df), 25.0)
    poa = poa_global(site, idx, ghi, direct_h, dhi)
    return pd.Series(pv_power(poa, temp, site), index=idx, name="observed_power")
