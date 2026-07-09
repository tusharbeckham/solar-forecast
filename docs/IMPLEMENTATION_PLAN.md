# Implementation Plan - Solar-Forecast (PhysSolar)

**Approach:** a physical **clear-sky model** provides the prior/baseline; an ML model learns the
**residual** (clouds, temperature, soiling). final prediction = `clear_sky_power + residual(features)`,
clipped >= 0. Ship a working baseline early, improve in small, tested increments.

**Principles:** correctness first; no data leakage (time-ordered splits, no look-ahead features);
evaluate against honest baselines (clear-sky + persistence); each phase builds on the last.

---

## Phase 0 - Scaffold  [DONE]
- Package `src/solar_forecast/{config,data,physics,features,models,evaluate,cli,plots}.py`; editable
  install (`pyproject.toml`, src layout); `requirements.txt`; `pytest`.
- **Verified:** `pip install -e .` works; `solar-forecast --help` works; `pytest` green (**20 tests**).

## Phase 1 - Data ingest  [DONE]
- `data.py` fetches from Open-Meteo: **archive** (`get_weather`, for backtests) and **forecast**
  (`get_forecast`, recent + upcoming days, for the live app) - shortwave_radiation (GHI),
  direct/diffuse_radiation, temperature_2m, cloud_cover, wind_speed_10m; caches to `data/raw/`;
  clean UTC-indexed hourly DataFrame.
- **Verified:** `parse_open_meteo` schema test; a network-free `get_forecast` test.

## Phase 2 - Clear-sky prior + POA  [DONE]
- `physics.py`: solar geometry (zenith, **azimuth, angle-of-incidence**); clear-sky GHI (Ineichen via
  **pvlib** if installed, else Haurwitz); **plane-of-array (POA) transposition** (isotropic sky model);
  PV power `P = A*eta*POA*PR` with temperature derating. Clear-sky & observed power now use POA.
- **Verified:** midday output > 0, night ~ 0, increases with irradiance; POA non-negative, scales with
  irradiance, zero at deep night; zenith/AOI in range.

## Phase 3 - Features + residual model  [DONE]
- `features.py` (clear-sky index, angle-of-incidence, hour/doy sin-cos, cloud, temp, wind, lags,
  rolling clear-sky index); `models.py` `ResidualForecaster` (sklearn HistGradientBoostingRegressor)
  predicting r = actual - clear_sky, clipped >= 0.
- **Verified:** residual model beats the clear-sky baseline (skill > 0) on cloudy synthetic + real data.
- **Next:** swap the proxy target (POA-derived) for real PV telemetry when available.

## Phase 4 - Evaluation / backtest  [DONE]
- `evaluate.py`: MAE, RMSE, MBE, R2, skill (vs clear-sky and persistence); forward-chaining CV; JSON
  report. `plots.py`: matplotlib forecast-vs-actual-vs-clear-sky.
- **Verified:** metric math on known synthetic; time-ordered folds; **live 2023 site-year run ->
  skill 0.85 vs clear-sky, 0.69 vs persistence**.

## Phase 5a - Live dashboard  [DONE]
- `app.py`: location-aware **Streamlit** app - browser geolocation + Open-Meteo forecast API
  (recent days + forward forecast), 24h self-refresh; deployed at solar-forecast.streamlit.app.

## Phase 5b - Physics-informed neural network (PINN)  [PLANNED]
- PyTorch model, loss = data term + lambda * (physics-consistency penalty: non-negativity + a clear-sky
  ceiling); compare against the Phase 3 residual model on the same backtest.

---

## Definition of done (v1)  [MET]
- `cli fetch -> train -> evaluate` runs end-to-end on a real site-year. ✓
- Residual model beats clear-sky AND persistence on a leakage-free backtest. ✓ (0.85 / 0.69)
- Tests green; results/plots in `reports/`; README documents reproduction. ✓ (20 tests)

**Current status:** Phases 0-4 + POA + the live dashboard (5a) are implemented, unit-verified
(**20 tests green**), and validated on a real 2023 site-year (skill **0.85** / **0.69**). Remaining:
the Phase 5b **PINN**, and swapping the proxy target for real inverter telemetry.
