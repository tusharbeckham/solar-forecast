"""Solar-Forecast dashboard: live, forward-looking PV forecast for the visitor's location.

- Location: from the visitor's browser (desktop + mobile); manual fallback.
- Forward-looking: uses the Open-Meteo forecast feed (recent past + next days), so it shows actuals
  up to now and the forecast into the future.
- Self-updating: fetches live and caches ~6h; a daily keep-alive workflow pings it too.

Run locally:  pip install -e ".[dashboard]"  &&  streamlit run app.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

# Make the src-layout package importable when running from the repo root (e.g. Streamlit Cloud).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_geolocation import streamlit_geolocation  # noqa: E402

from solar_forecast import data, features, models, physics  # noqa: E402
from solar_forecast.config import DEFAULT_SITE  # noqa: E402

st.set_page_config(page_title="Solar-Forecast", page_icon="☀️", layout="centered")
st.title("☀️ Solar-Forecast")
st.write(
    "Physics-informed solar power forecast for **your location** - recent days plus the days ahead. "
    "It updates itself automatically from the global Open-Meteo feed."
)

# --- 1. Location (browser geolocation; manual fallback) ---
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

with st.expander("Options"):
    tilt = st.slider("Panel tilt (degrees)", 0, 60, int(DEFAULT_SITE.tilt_deg))


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching latest data + forecast...")
def load(lat: float, lon: float, tilt: float):
    """Fetch recent-past + future data, train on the settled past, forecast the whole window.

    Cached ~6h and keyed on inputs, so it refreshes a few times a day on its own.
    """
    azimuth = 180.0 if lat >= 0 else 0.0                        # panels face the equator
    site = replace(DEFAULT_SITE, latitude=lat, longitude=lon, tilt_deg=float(tilt),
                   azimuth_deg=azimuth, altitude_m=0.0, name=f"s_{lat:.2f}_{lon:.2f}")
    df = data.get_forecast(site, past_days=92, forecast_days=7)  # ~3 months back + 7 days ahead
    feats = features.build(df, site)
    y = physics.observed_power(site, df).reindex(feats.index)
    clear = physics.clear_sky_power(site, feats.index, temp_c=feats["temp_c"]).reindex(feats.index)
    now = pd.Timestamp.now(tz="UTC")
    train_mask = feats.index <= (now - pd.Timedelta(days=2))     # train only on settled past
    model = models.ResidualForecaster(site).fit(feats[train_mask], (y - clear)[train_mask])
    pred = pd.Series(np.asarray(model.predict(feats, clear), dtype=float), index=feats.index)
    return feats.index, y, clear, pred


try:
    idx, y, clear, pred = load(lat, lon, float(tilt))
except Exception as exc:  # network / API hiccup
    st.error(f"Couldn't load data right now - please try again shortly.\n\n{exc}")
    st.stop()

now = pd.Timestamp.now(tz="UTC")
actual = y.copy()
actual[idx > now] = np.nan                                      # no actuals in the future

st.subheader("2. Forecast")

# Headline metrics for the next 24 hours
nxt = pred[(idx > now) & (idx <= now + pd.Timedelta(hours=24))]
if len(nxt):
    c1, c2 = st.columns(2)
    c1.metric("Next 24 h peak", f"{nxt.max():.0f} W")
    c2.metric("Expected at", f"{nxt.idxmax():%a %H:%M} UTC")

tab_ahead, tab_day = st.tabs(["📈 Days ahead", "📅 Pick a day"])

with tab_ahead:
    ahead = st.slider("Days ahead", 1, 7, 3)                    # view-only: reslices, no refetch
    start = now.floor("D") - pd.Timedelta(days=2)
    end = now.floor("D") + pd.Timedelta(days=ahead + 1)
    m = (idx >= start) & (idx <= end)
    st.line_chart(pd.DataFrame(
        {"actual (proxy)": actual[m], "clear-sky": clear[m], "forecast": pred[m]}, index=idx[m]))
    st.caption("The 'actual' line stops at now; the forecast continues into the future.")

with tab_day:
    dmin, dmax = idx.min().date(), idx.max().date()
    pick = st.date_input("Pick a day", value=now.date(), min_value=dmin, max_value=dmax)
    d0 = pd.Timestamp(pick, tz="UTC")
    dm = (idx >= d0) & (idx < d0 + pd.Timedelta(days=1))
    st.line_chart(pd.DataFrame(
        {"actual (proxy)": actual[dm], "clear-sky": clear[dm], "forecast": pred[dm]}, index=idx[dm]))
    st.caption(f"{pick}: hourly forecast vs actual vs clear-sky.  Available range: {dmin} to {dmax}.")

st.caption(f"Location {lat:.3f}, {lon:.3f}  -  updated automatically from Open-Meteo.")
