# Architecture - Solar-Forecast (PhysSolar)

## Design idea
**Hybrid (physics + ML).** A physical clear-sky model gives the expected output under ideal skies; an
ML model corrects it using weather (clouds, temperature). More data-efficient and interpretable than
pure ML, and it degrades gracefully (falls back to the physical prior).

```
Open-Meteo API  -->  data (ingest + cache)  -->  features  ---\
                                                               >-->  models  -->  prediction  -->  evaluate  -->  reports
site config     -->  physics (clear-sky prior + POA) ---------/
```

## Package layout (`src/solar_forecast/`)
| Module | Responsibility |
|--------|----------------|
| `config.py`   | `Site` dataclass (lat, lon, tz, tilt, azimuth, panel_area, efficiency, performance_ratio, temp_coeff) + `DEFAULT_SITE` + paths. |
| `data.py`     | Fetch hourly weather+irradiance from Open-Meteo - **archive** (`get_weather`, for backtests) and **forecast** (`get_forecast`, recent + upcoming days, for the live app); cache to `data/raw/`; clean UTC-indexed DataFrame. |
| `physics.py`  | Solar geometry (zenith, azimuth, angle-of-incidence), clear-sky irradiance (pvlib Ineichen or Haurwitz fallback), **plane-of-array (POA) transposition**, PV power model, clear-sky + observed power. |
| `features.py` | Feature matrix: clear-sky index, angle-of-incidence, time encodings, cloud, temp, wind, lags + rolling clear-sky index. |
| `models.py`   | Baselines (clear-sky, persistence) + `ResidualForecaster` (predict = clear_sky + residual, clipped >= 0), save/load. |
| `evaluate.py` | Metrics (MAE/RMSE/MBE/R2/skill) + forward-chaining backtest + JSON report. |
| `plots.py`    | Matplotlib forecast-vs-actual-vs-clear-sky plot. |
| `cli.py`      | `fetch`, `train`, `predict`, `evaluate`, `plot` (argparse). |
| `app.py` *(repo root)* | Live, location-aware **Streamlit dashboard**: browser geolocation + the forecast API (recent + upcoming days), self-updating. |

## Key interfaces
- `data.get_weather(site, start, end) -> DataFrame` (archive) and `data.get_forecast(site, past_days, forecast_days) -> DataFrame` (recent + future) - both UTC-indexed (ghi, direct, dhi, temp_c, cloud_pct, wind_ms)
- `physics.clear_sky_power(site, index, temp_c) -> Series`, `physics.observed_power(site, df) -> Series`, `physics.poa_global(...)`, `physics.cos_aoi(...)`
- `features.build(df, site) -> DataFrame`  (FEATURE_COLUMNS; NaN lags dropped)
- `models.ResidualForecaster.fit(X, residual)` / `.predict(X, clear_sky) -> Series`
- `evaluate.backtest(site, df, n_splits) -> dict`

## Tech choices
- **pvlib** (optional) for clear-sky; a **self-contained fallback** (solar geometry + Haurwitz + isotropic POA) so the package runs without it.
- **pandas** (time-indexed data), **scikit-learn** (HistGradientBoostingRegressor), **matplotlib** (plots),
  **streamlit** (live dashboard), **requests** (Open-Meteo), **joblib** (model persistence). **pytorch** later (Phase 5b PINN).

## Data & leakage rules
- All splits are **time-ordered**; never shuffle across time.
- Features at time t use only information available by t (lags + rolling are shifted; no look-ahead).
- Raw API responses cached under `data/raw/` (git-ignored).

## Notes / v0 caveat
- The "actual" PV output is currently a **proxy** derived from observed irradiance via POA
  (`physics.observed_power`), standing in for real inverter telemetry. Swap in real PV data when
  available; the interfaces don't change.

## Non-goals (v1)
- Real-time serving, multi-site fleets, or a hosted API beyond the demo dashboard. Focus: one site,
  an honest backtest, and a model that beats the baselines.
