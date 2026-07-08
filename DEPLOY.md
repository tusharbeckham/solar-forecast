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
- **Always current:** the app fetches a rolling recent window (last 12 months, ending ~2 days ago) live
  from Open-Meteo on load and caches it for 24h, so the dashboard stays up to date automatically with
  **no commits**. The first cold start takes a few seconds (fetch + train).
- **Per-visitor location:** the app asks each visitor's browser for their location (with a manual
  fallback) and forecasts for *their* spot - no fixed default. Works on desktop and mobile.
- **Auto-renew without visits:** a scheduled workflow (`.github/workflows/keep-alive.yml`) pings the
  app daily so it wakes and refreshes its data even if no one opens it.
- **Dependencies:** come from `requirements.txt` (now includes `streamlit`).

## After deploying
Add the live link to `README.md` with a badge:
```markdown
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_APP_URL)
```
Tell me the URL and I'll drop it into the README for you.
