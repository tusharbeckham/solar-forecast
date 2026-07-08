"""Streamlit dashboard for solar-forecast: pick a day -> forecast vs actual vs clear-sky.

Run:
    pip install streamlit
    streamlit run app.py
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from solar_forecast import data, features, models, physics
from solar_forecast.config import DEFAULT_SITE

st.set_page_config(page_title="Solar-Forecast", layout="wide")
st.title("Solar-Forecast (PhysSolar)")
st.caption("Physics-informed PV output forecast: a clear-sky physical prior + an ML residual model.")

site = DEFAULT_SITE
year = int(st.sidebar.number_input("Year", min_value=2020, max_value=2024, value=2023))
start, end = f"{year}-01-01", f"{year}-12-31"


@st.cache_data(show_spinner="Fetching data + training model...")
def load(start: str, end: str):
    df = data.get_weather(site, start, end)
    feats = features.build(df, site)
    y = physics.observed_power(site, df).reindex(feats.index)
    clear = physics.clear_sky_power(site, feats.index, temp_c=feats["temp_c"]).reindex(feats.index)
    cutoff = feats.index.max() - pd.Timedelta(days=30)
    train_mask = feats.index <= cutoff
    model = models.ResidualForecaster(site).fit(feats[train_mask], (y - clear)[train_mask])
    return y, clear, model.predict(feats, clear)


y, clear, pred = load(start, end)

day = st.sidebar.date_input("Day", value=dt.date(year, 6, 11))
d0 = pd.Timestamp(day, tz="UTC")
d1 = d0 + pd.Timedelta(days=1)
mask = (y.index >= d0) & (y.index < d1)

chart = pd.DataFrame(
    {"observed (proxy)": y[mask], "clear-sky prior": clear[mask], "model forecast": pred[mask]}
)
st.line_chart(chart)
st.caption(f"{site.name} — {day}: observed vs clear-sky prior vs model forecast (W)")
