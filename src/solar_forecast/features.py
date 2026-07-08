"""Feature engineering for the residual model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import physics
from .config import Site

FEATURE_COLUMNS = [
    "ghi_cs", "clear_sky_index", "cos_aoi", "cloud_pct", "temp_c", "wind_ms",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "ghi_lag1", "ghi_lag24", "csi_roll24",
]


def _col(df: pd.DataFrame, name: str, default: float) -> np.ndarray:
    if name in df.columns:
        return df[name].to_numpy(dtype=float)
    return np.full(len(df), float(default))


def build(df: pd.DataFrame, site: Site) -> pd.DataFrame:
    """Build the feature matrix from a weather DataFrame (UTC-indexed).

    No look-ahead: lag and rolling features use only past values. Rows with NaN are dropped.
    """
    idx = pd.DatetimeIndex(df.index)
    ghi = _col(df, "ghi", 0.0)
    ghi_cs = physics.clear_sky_ghi(site, idx).to_numpy()
    csi = np.where(ghi_cs > 1.0, ghi / np.maximum(ghi_cs, 1e-6), 0.0)

    out = pd.DataFrame(index=idx)
    out["ghi_cs"] = ghi_cs
    out["clear_sky_index"] = csi
    out["cos_aoi"] = np.clip(physics.cos_aoi(site, idx), 0.0, None)
    out["cloud_pct"] = _col(df, "cloud_pct", 0.0)
    out["temp_c"] = _col(df, "temp_c", 25.0)
    out["wind_ms"] = _col(df, "wind_ms", 0.0)

    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    doy = idx.dayofyear.to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)

    ghi_series = pd.Series(ghi, index=idx)
    out["ghi_lag1"] = ghi_series.shift(1).to_numpy()
    out["ghi_lag24"] = ghi_series.shift(24).to_numpy()
    # recent cloudiness: mean clear-sky index over the previous 24 h (shifted -> past-only, no leakage)
    out["csi_roll24"] = pd.Series(csi, index=idx).rolling(24, min_periods=1).mean().shift(1).to_numpy()

    return out.dropna()
