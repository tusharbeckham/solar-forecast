"""Streamlit dashboard for solar-forecast: pick a location + day -> forecast vs actual vs clear-sky.

Data is global (Open-Meteo ERA5 archive) and updates itself daily: the app fetches a rolling recent
window at runtime and caches it for 24h, so it stays current with no commits.

Run locally:
    pip install -e ".[dashboard]"
    streamlit run app.py

Deploy: see DEPLOY.md (Streamlit Community Cloud).
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

from solar_forecast import data, features, models, physics  # noqa: E402
from solar_forecast.config import DEFAULT_SITE  # noqa: E402

st.set_page_config(page_title="Solar-Forecast", layout="wide")
st.title("Solar-Forecast (PhysSolar)")
st.caption(
    "Physics-informed PV output forecast: a clear-sky physical prior + an ML residual model. "
    "Data is global (Open-Meteo archive) and refreshes automatically each day."
)

# --- Location (global) + panel + window ---
st.sidebar.header("Location (anywhere on Earth)")
lat = st.sidebar.number_input("Latitude", -90.0, 90.0, float(DEFAULT_SITE.latitude), format="%.4f")
lon = st.sidebar.number_input("Longitude", -180.0, 180.0, float(DEFAULT_SITE.longitude), format="%.4f")
tilt = st.sidebar.slider("Panel tilt (deg)", 0, 60, int(DEFAULT_SITE.tilt_deg))
months = st.sidebar.select_slider("History window (months)", options=[3, 6, 12, 24], value=12)

# Rolling, always-current window (archive lags ~1-2 days, so end a couple days back).
end_date = dt.date.today() - dt.timedelta(days=2)
start_date = end_date - dt.timedelta(days=int(round(months * 30.4)))


@st.cache_data(ttl=24 * 3600, show_spinner="Fetching latest data + training model...")
def load(lat: float, lon: float, tilt: float, start: str, end: str):
    """Fetch live data for the location/window, train the residual model, return series.

    Cached for 24h and keyed on the inputs, so it re-fetches fresh data at most once a day.
    """
    site = replace(DEFAULT_SITE, latitude=lat, longitude=lon, tilt_deg=float(tilt),
                   name=f"site_{lat:.2f}_{lon:.2f}")
    df = data.get_weather(site, start, end, use_cache=False)
    feats = features.build(df, site)
    y = physics.observed_power(site, df).reindex(feats.index)
    clear = physics.clear_sky_power(site, feats.index, temp_c=feats["temp_c"]).reindex(feats.index)
    cutoff = feats.index.max() - pd.Timedelta(days=30)
    train_mask = feats.index <= cutoff
    model = models.ResidualForecaster(site).fit(feats[train_mask], (y - clear)[train_mask])
    return y, clear, model.predict(feats, clear)


try:
    y, clear, pred = load(lat, lon, float(tilt), start_date.isoformat(), end_date.isoformat())
except Exception as exc:  # network / API hiccup
    st.error(f"Couldn't load data for this location/period. Try again in a moment.\n\n{exc}")
    st.stop()

# --- Day to view (defaults to the most recent available day) ---
last_day = y.index.max().date()
first_day = y.index.min().date()
day = st.sidebar.date_input("Day to view", value=last_day, min_value=first_day, max_value=last_day)
d0 = pd.Timestamp(day, tz="UTC")
mask = (y.index >= d0) & (y.index < d0 + pd.Timedelta(days=1))

st.markdown(
    f"**Location:** {lat:.3f}, {lon:.3f}  |  **data through:** {last_day}  |  **showing:** {day}"
)
chart = pd.DataFrame(
    {"observed (proxy)": y[mask], "clear-sky prior": clear[mask], "model forecast": pred[mask]}
)
st.line_chart(chart)
st.caption("Observed (proxy) vs clear-sky physical prior vs model forecast (watts).")
