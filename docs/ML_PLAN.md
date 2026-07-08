# ML Plan - Solar-Forecast (PhysSolar)

## Problem
Predict hourly PV output. Learn the **residual** on top of a physical clear-sky prior:
`y_hat = clear_sky_power + f(features)`, clipped >= 0. This shrinks what ML must learn to the
weather-driven deviation, which is more sample-efficient than modeling output from scratch.

## Data
- **Source:** Open-Meteo archive (hourly), one site (`DEFAULT_SITE`). Free, no key.
- **Target (v0):** `observed_power` = PV model applied to observed GHI + temperature (proxy for real
  telemetry). Replace with measured inverter output when available.
- **Split:** strictly time-ordered. Train on the earliest span, validate/test on later spans.

## Features (`features.build`)
- `ghi_cs` (clear-sky GHI), `clear_sky_index` (= GHI / GHI_cs), `cloud_pct`, `temp_c`, `wind_ms`
- Time encodings: `hour_sin/cos`, `doy_sin/cos`
- Lags: `ghi_lag1`, `ghi_lag24` (past-only; NaN rows dropped -> no leakage)

## Baselines (must beat these)
1. **Clear-sky** - the physical prior (ignores clouds).
2. **Persistence** - value 24 h earlier.

## Model
- **First model:** `HistGradientBoostingRegressor` (scikit-learn). Chosen over a linear model because
  the residual is nonlinear in clouds/temperature/time and gradient boosting handles interactions and
  monotone-ish effects well out-of-the-box, with little preprocessing. A regularized linear model is a
  reasonable interpretable fallback.
- **Config:** `max_iter=300, learning_rate=0.05, max_depth=6, random_state=42`.

## Validation
- **Forward-chaining (rolling-origin)** CV (`evaluate.forward_chaining_splits`): expanding train window,
  next block as test. No shuffling, no leakage. Report mean metrics across folds.

## Metrics + targets
- MAE, RMSE, MBE, R2, and **skill score** = `1 - RMSE_model / RMSE_reference` vs clear-sky and persistence.
- **Target:** positive skill vs BOTH baselines on the backtest. (Verified > 0 vs clear-sky on synthetic
  cloudy data; confirm on a real site-year next.)

## Experiment tracking
- Lightweight: `evaluate.write_report` dumps metrics JSON to `reports/`. Add a per-run timestamp + config
  hash as the dataset grows.

## Path to a PINN (Phase 5)
- Replace the residual regressor with a small PyTorch net; loss = data term (MSE) + `lambda` *
  physics-consistency penalty (e.g., penalize predicted output exceeding the clear-sky ceiling or going
  negative). Compare against the gradient-boosting residual model on the same backtest.
