"""Data ingest: hourly weather + irradiance from the Open-Meteo archive API (free, no key)."""
from __future__ import annotations

import pandas as pd
import requests

from .config import Site, RAW_DIR

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Open-Meteo hourly variables we request -> tidy internal names
_VARS = {
    "shortwave_radiation": "ghi",       # global horizontal irradiance (W/m^2)
    "direct_radiation": "direct",       # direct on horizontal (W/m^2)
    "diffuse_radiation": "dhi",         # diffuse horizontal (W/m^2)
    "temperature_2m": "temp_c",
    "cloud_cover": "cloud_pct",
    "wind_speed_10m": "wind_ms",
}
REQUIRED = ["ghi", "temp_c"]


def _cache_path(site: Site, start: str, end: str):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR / f"{site.name}_{start}_{end}.csv"


def parse_open_meteo(payload: dict) -> pd.DataFrame:
    """Turn an Open-Meteo archive JSON payload into a clean UTC-indexed hourly DataFrame."""
    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df = df.rename(columns={k: v for k, v in _VARS.items() if k in df.columns})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Open-Meteo response missing required columns: {missing}")
    return df


def get_weather(site: Site, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """Fetch (or load from cache) hourly weather/irradiance for a site.

    Args:
        start, end: 'YYYY-MM-DD' (inclusive) in the archive.
    Returns a UTC-indexed hourly DataFrame with columns ghi, direct, dhi, temp_c, cloud_pct, wind_ms.
    """
    cache = _cache_path(site, start, end)
    if use_cache and cache.exists():
        df = pd.read_csv(cache, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()

    params = {
        "latitude": site.latitude,
        "longitude": site.longitude,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(_VARS.keys()),
        "timezone": "UTC",
    }
    try:
        resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Open-Meteo request failed ({exc}). Check your connection, or use a cached file."
        ) from exc

    df = parse_open_meteo(resp.json())
    df.to_csv(cache)
    return df
