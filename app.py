"""Solar-Forecast dashboard: forecasts solar PV output for the visitor's own location.

- Location: asked from the visitor's browser (works on desktop + mobile); manual entry as a fallback.
- Self-updating: fetches a rolling recent window live from the global Open-Meteo archive and caches it
  for 24h, so it stays current on its own (a daily keep-alive workflow pings it even with no visitors).

Run locally:
    pip install -e ".[dashboard]"
    streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from dataclasses import replace

# Make the src-layout package importable when running from the repo root (e.g. Streamlit Cloud).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_geolocation import streamlit_geolocation  # noqa: E402

from solar_forecast import data, features, models, physics  # noqa: E402
from solar_forecast.config import DEFAULT_SITE  # noqa: E402

st.set_page_config(page_title="Solar-Forecast", page_icon="☀️", layout="centered")

st.title("☀️ Solar-Forecast")
st.write(
    "Physics-informed solar power forecast for **your location**. "
    "The data updates itself daily from the global Open-Meteo archive."
)

# --- 1. Location: ask the visitor's browser (desktop + mobile), with a manual fallback ---
st.subheader("1. Your location")
st.caption("Tap the button and allow location access to forecast for where you are.")
loc = streamlit_geolocation()

lat = lon = None
if loc and loc.get("latitude") is not None:
    lat, lon = float(loc["latitude"]), float(loc["longitude"])
    st.success(f"Using your location: {lat:.3f}, {lon:.3f}")

with st.expander("Or enter coordinates manually"):
    if st.checkbox("Enter coordinates manually"):
        lat = st.number_input("Latitude", -90.0, 90.0, float(lat) if lat is not None else 40.4168, format="%.4f")
        lon = st.number_input("Longitude", -180.0, 180.0, float(lon) if lon is not None else -3.7038, format="%.4f")

if lat is None:
    st.info("Share your location (or enter coordinates) above to see the forecast.")
    st.stop()

# --- Options (secondary; collapsed by default so mobile stays clean) ---
with st.expander("Options"):
    tilt = st.slider("Panel tilt (degrees)", 0, 60, int(DEFAULT_SITE.tilt_deg))
    months = st.select_slider("History window (months)", options=[3, 6, 12], value=12)

end_date = dt.date.today() - dt.timedelta(days=2)                      # archive lags ~1-2 days
start_date = end_date - dt.timedelta(days=int(round(months * 30.4)))


@st.cache_data(ttl=24 * 3600, show_spinner="Fetching latest data + training model...")
def load(lat: float, lon: float, tilt: float, start: str, end: str):
    """Fetch live data for the location/window and train the residual model. Cached 24h (self-refresh)."""
    azimuth = 180.0 if lat >= 0 else 0.0     # panels face the equator
    site = replace(DEFAULT_SITE, latitude=lat, longitude=lon, tilt_deg=float(tilt),
                   azimuth_deg=azimuth, altitude_m=0.0, name=f"s_{lat:.2f}_{lon:.2f}")
    df = data.get_weather(site, start, end, use_cache=False)
    feats = features.build(df, site)
    y = physics.observed_power(site, df).reindex(feats.index)
    clear = physics.clear_sky_power(site, feats.index, temp_c=feats["temp_c"]).reindex(feats.index)
    cutoff = feats.index.max() - pd.Timedelta(days=30)
    tr = feats.index <= cutoff
    model = models.ResidualForecaster(site).fit(feats[tr], (y - clear)[tr])
    return y, clear, model.predict(feats, clear)


try:
    y, clear, pred = load(lat, lon, float(tilt), start_date.isoformat(), end_date.isoformat())
except Exception as exc:  # network / API hiccup
    st.error(f"Couldn't load data right now - please try again shortly.\n\n{exc}")
    st.stop()

# --- 2. Forecast (defaults to the most recent available day) ---
st.subheader("2. Forecast")
last_day = y.index.max().date()
day = st.date_input("Day", value=last_day, min_value=y.index.min().date(), max_value=last_day)
d0 = pd.Timestamp(day, tz="UTC")
mask = (y.index >= d0) & (y.index < d0 + pd.Timedelta(days=1))

st.caption(f"Location {lat:.3f}, {lon:.3f}  -  data through {last_day}  -  showing {day}")
st.line_chart(
    pd.DataFrame({"observed (proxy)": y[mask], "clear-sky prior": clear[mask], "forecast": pred[mask]})
)
st.caption("Observed (proxy) vs clear-sky physical prior vs model forecast (watts).")
