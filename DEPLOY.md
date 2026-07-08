# Deploying the dashboard (Streamlit Community Cloud)

The dashboard (`app.py`) is ready to deploy for free on Streamlit Community Cloud.

## Steps (one-time, ~2 minutes)
1. Go to **https://share.streamlit.io** and sign in with your GitHub account.
2. Click **Create app** -> **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `tusharbeckham/solar-forecast`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**. The first build installs `requirements.txt` (~1-2 min), then the app goes live.

You'll get a public URL like `https://solar-forecast-<random>.streamlit.app`.

## How it's set up for deployment
- **Imports:** a small `sys.path` shim at the top of `app.py` makes the `src/`-layout package importable
  on Streamlit Cloud (no install step needed).
- **Fast cold start:** the app ships a cached 2023 dataset (`sample_data/default_2023.csv`), so the
  default view loads with **no network call** and trains in a few seconds. Other years fetch live from
  Open-Meteo.
- **Dependencies:** come from `requirements.txt` (now includes `streamlit`).

## After deploying
Add the live link to `README.md` with a badge:
```markdown
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_APP_URL)
```
Tell me the URL and I'll drop it into the README for you.
