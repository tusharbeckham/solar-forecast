"""Streamlit dashboard for solar-forecast: pick a day -> forecast vs actual vs clear-sky.

Run locally:
    pip install -e ".[dashboard]"
    streamlit run app.py

Deploy: see DEPLOY.md (Streamlit Community Cloud).
"""
from __future__ import annotations

import datetime as dt
import os
import sys

# Make the src-layout package importable when running from the repo root (e.g. Streamlit Cloud).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from solar_forecast import data, features, models, physics  # noqa: E402
from solar_forecast.config import DEFAULT_SITE  # noqa: E402

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")

st.set_page_config(page_title="Solar-Forecast", layout="wide")
st.title("Solar-Forecast (PhysSolar)")
st.caption("Physics-informed PV output forecast: a clear-sky physical prior + an ML residual model.")

site = DEFAULT_SITE
year = int(st.sidebar.number_input("Year", min_value=2020, max_value=2024, value=2023))


def _load_weather(year: int) -> pd.DataFrame:
    """Prefer a shipped sample (fast, offline); otherwise fetch live from Open-Meteo."""
    sample = os.path.join(SAMPLE_DIR, f"default_{year}.csv")
    if os.path.exists(sample):
        df = pd.read_csv(sample, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()
    return data.get_weather(site, f"{year}-01-01", f"{year}-12-31")


@st.cache_data(show_spinner="Preparing data + training model...")
def load(year: int):
    df = _load_weather(year)
    feats = features.build(df, site)
    y = physics.observed_power(site, df).reindex(feats.index)
    clear = physics.clear_sky_power(site, feats.index, temp_c=feats["temp_c"]).reindex(feats.index)
    cutoff = feats.index.max() - pd.Timedelta(days=30)
    train_mask = feats.index <= cutoff
    model = models.ResidualForecaster(site).fit(feats[train_mask], (y - clear)[train_mask])
    return y, clear, model.predict(feats, clear)


y, clear, pred = load(year)

day = st.sidebar.date_input("Day", value=dt.date(year, 6, 11))
d0 = pd.Timestamp(day, tz="UTC")
d1 = d0 + pd.Timedelta(days=1)
mask = (y.index >= d0) & (y.index < d1)

chart = pd.DataFrame(
    {"observed (proxy)": y[mask], "clear-sky prior": clear[mask], "model forecast": pred[mask]}
)
st.line_chart(chart)
st.caption(f"{site.name} - {day}: observed vs clear-sky prior vs model forecast (W)")
