# Solar-Forecast (PhysSolar)

Physics-informed machine learning for **solar PV output forecasting**. A physical **clear-sky model**
provides the prior; an ML model learns the **residual** (clouds, temperature). Final prediction =
`clear_sky_power + residual(features)`, clipped to >= 0.

## Why hybrid?
Pure ML must learn everything from data; a physical prior encodes what we already know (solar geometry,
clear-sky irradiance), so the model only learns the weather-driven deviation - more data-efficient, more
interpretable, and it degrades gracefully to the physical baseline.

## Install
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .            # core; add ".[physics]" for pvlib, ".[dev]" for pytest
```

## Usage
```powershell
solar-forecast fetch    --start 2023-01-01 --end 2023-12-31   # download + cache weather/irradiance
solar-forecast train    --start 2023-01-01 --end 2023-09-30   # train the residual model
solar-forecast evaluate --start 2023-01-01 --end 2023-12-31   # forward-chaining backtest vs baselines
solar-forecast predict  --start 2023-12-01 --end 2023-12-07   # predict a date range
# (equivalently: python -m solar_forecast.cli <command> ...)
```

## How it works
| Module | Role |
|--------|------|
| `data` | Open-Meteo archive fetch + local cache (GHI, temp, cloud, wind) |
| `physics` | clear-sky irradiance (pvlib or built-in fallback) + PV power model (the prior) |
| `features` | clear-sky index, time encodings, weather, lags (no look-ahead) |
| `models` | baselines (clear-sky, persistence) + `ResidualForecaster` (gradient boosting on the residual) |
| `evaluate` | MAE / RMSE / MBE / R2 + skill score; forward-chaining backtest |
| `cli` | fetch / train / predict / evaluate |

## Status
Foundation complete (Phases 0-4 at v0) and unit-tested (`pytest` green - 9 tests, incl. the residual
model beating the clear-sky baseline on cloudy data). Roadmap + design:
- [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - [docs/ML_PLAN.md](docs/ML_PLAN.md)
- [docs/PHYSICS.md](docs/PHYSICS.md) - [docs/MATH.md](docs/MATH.md) - [docs/RESEARCH.md](docs/RESEARCH.md)

## Notes
- Python 3.10+. `pvlib` is optional (a self-contained clear-sky fallback is built in).
- v0 uses a proxy PV target derived from observed irradiance; swap in real inverter telemetry when available.
