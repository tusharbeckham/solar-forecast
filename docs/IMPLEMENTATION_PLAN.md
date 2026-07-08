# Implementation Plan - Solar-Forecast (PhysSolar)

**Approach:** a physical **clear-sky model** provides the prior/baseline; an ML model learns the
**residual** (clouds, temperature, soiling). final prediction = `clear_sky_power + residual(features)`,
clipped >= 0. Ship a working baseline early, improve in small, tested increments.

**Principles:** correctness first; no data leakage (time-ordered splits, no look-ahead features);
evaluate against honest baselines (clear-sky + persistence); each phase builds on the last.

---

## Phase 0 - Scaffold  [DONE]
- Package `src/solar_forecast/{config,data,physics,features,models,evaluate,cli}.py`; editable install
  (`pyproject.toml`, src layout); `requirements.txt`; `pytest`.
- **Verified:** `pip install -e .` works; `python -m solar_forecast.cli --help` works; `pytest` green.

## Phase 1 - Data ingest  [DONE]
- `data.py` fetches from **Open-Meteo archive** (free, no key): shortwave_radiation (GHI),
  direct/diffuse_radiation, temperature_2m, cloud_cover, wind_speed_10m; caches to `data/raw/`;
  returns a clean UTC-indexed hourly DataFrame.
- **Verified:** `parse_open_meteo` fixture test -> correct schema, required cols, UTC index.

## Phase 2 - Clear-sky baseline (physics prior)  [DONE - v0]
- `physics.py`: solar geometry + clear-sky GHI (Ineichen via **pvlib** if installed, else a
  self-contained Haurwitz model), PV power `P = A*eta*GHI*PR` with temperature derating.
- **Verified:** midday output > 0, night ~ 0, increases with irradiance.
- **Next:** add plane-of-array (POA) transposition + tilt/azimuth (currently GHI-based).

## Phase 3 - Features + residual model  [DONE - v0]
- `features.py` (clear-sky index, hour/day-of-year sin-cos, cloud, temp, lags);
  `models.py` `ResidualForecaster` (sklearn HistGradientBoostingRegressor) predicting r = actual - clear_sky.
- **Verified:** residual model beats the clear-sky baseline on cloudy synthetic data (skill > 0).
- **Next:** swap the proxy target (GHI-derived) for real PV telemetry when available; tune features.

## Phase 4 - Evaluation / backtest  [DONE - v0]
- `evaluate.py`: MAE, RMSE, MBE, R2, skill score (vs clear-sky and persistence); forward-chaining CV;
  JSON report to `reports/`.
- **Verified:** metric math on known synthetic; forward-chaining folds strictly time-ordered.
- **Next:** plots (matplotlib) + a richer report; live end-to-end run on a real site-year.

## Phase 5 - Physics-informed model + dashboard  [PLANNED]
- PyTorch model, loss = data term + lambda * (physics-consistency penalty); compare vs Phase 3.
- Minimal dashboard: pick a date -> forecast vs actual vs clear-sky.

---

## Definition of done (v1)
- `cli fetch -> train -> evaluate` runs end-to-end on a real site-year.
- Residual model beats clear-sky AND persistence on a leakage-free backtest (positive skill).
- Tests green; results/plots in `reports/`; README documents reproduction.

**Current status:** Phases 0-4 implemented at v0 and unit-verified (9 tests green). Remaining: a live
site-year run, POA transposition, plots, and the Phase 5 PINN + dashboard.
