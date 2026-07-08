# Research - Solar-Forecast (PhysSolar)

## Free data sources
- **Open-Meteo Archive API** (no key). Endpoint: `https://archive-api.open-meteo.com/v1/archive`.
  Hourly vars used here: `shortwave_radiation` (GHI), `direct_radiation`, `diffuse_radiation`,
  `temperature_2m`, `cloud_cover`, `wind_speed_10m`. Global reanalysis (ERA5), ~1940-present.
  **Verified working** by this repo's `data.py` (a live request returned the expected hourly arrays).
  Docs: https://open-meteo.com/en/docs/historical-weather-api
- **PVGIS** (EU JRC) - irradiation + PV output estimates:
  https://joint-research-centre.ec.europa.eu/pvgis-online-tool_en
- **NREL NSRDB** - high-quality solar irradiance (API key required): https://nsrdb.nrel.gov/

## Forecasting approaches
- **Physical clear-sky** (Haurwitz, Ineichen-Perez): deterministic prior, no training.
- **ML regression** (gradient boosting, random forest, linear): weather -> output.
- **Hybrid / residual** (this project): ML learns the deviation from the clear-sky prior -
  data-efficient, interpretable, graceful fallback.
- **Sequence models** (LSTM, Temporal CNN): temporal dynamics; need more data + tuning.
- **PINNs**: embed physical constraints in the loss; good for consistency + low-data regimes.

## Standard input features
Irradiance (GHI/DNI/DHI), clear-sky index (GHI/GHI_cs), cloud cover, temperature, wind, humidity,
time-of-day + season encodings, and lagged values.

## Evaluation practice
- Time-ordered splits only (never random) -> avoid leakage; use forward-chaining CV.
- Report MAE/RMSE/MBE plus a **skill score** vs clear-sky and persistence baselines. A model that can't
  beat persistence adds no value.

## Key tool
- **pvlib** - the standard Python library for solar position, clear-sky, POA transposition, PV modeling:
  https://pvlib-python.readthedocs.io/
