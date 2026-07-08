# Understand this project (and run it yourself)

A plain-English guide to what Solar-Forecast does, how each piece works, and how to experiment with it
so the concepts stick. No prior ML background assumed.

---

## 1. The one-sentence idea
Predict how much power a solar panel will make in the next hours, by **starting from physics** (how much
sunlight is theoretically available on a clear day) and using **machine learning only to correct** for
what physics can't easily predict (clouds, haze, temperature).

> **Forecast = clear-sky physics prediction + ML-learned residual**

If ML has nothing useful to add, the residual is ~0 and you fall back to the physics baseline. That's the
safety net that pure "throw a neural net at it" approaches don't have.

---

## 2. Why not just use ML for everything?
You could. But the sun's position and the clear-sky maximum are *known physics* — there's no reason to
make a model rediscover them from data. Telling the model the physics up front means:
- **Less data needed** — it only learns the hard part (weather deviations).
- **More interpretable** — you can see the physical baseline and the correction separately.
- **Safer** — it degrades to a sensible physical answer when unsure.

This is the core idea of **physics-informed ML**, and it's the stepping stone to a PINN (Section 8).

---

## 3. The pipeline, step by step
```
1. DATA      Open-Meteo gives hourly weather + sunlight (irradiance) for a location and year.
2. PHYSICS   From the sun's geometry we compute the clear-sky irradiance and the clear-sky PV power
             (what the panel would make with no clouds). This is the "prior".
3. FEATURES  We turn raw weather into model inputs: clear-sky index (actual/clear-sky sunlight),
             angle of incidence, cloud %, temperature, time-of-day, and recent history (lags).
4. MODEL     A gradient-boosting model learns the RESIDUAL = (actual power - clear-sky power).
5. FORECAST  prediction = clear_sky_power + residual, clipped to >= 0.
6. EVALUATE  We backtest fairly (train on the past, predict the future) and score the result.
```

### The key physics terms (in words)
- **GHI** — global horizontal irradiance: total sunlight hitting a flat, level surface (W/m2).
- **Clear-sky irradiance** — the GHI you'd get on a perfectly cloudless day, from the sun's angle alone.
- **Clear-sky index (CSI)** — actual GHI / clear-sky GHI. ~1 means clear, near 0 means heavy cloud. This
  single number captures "how cloudy is it right now" and is the model's most important input.
- **Plane-of-array (POA)** — sunlight on the *tilted panel*, not a flat surface. Panels are angled, so we
  transpose horizontal sunlight onto the tilt using the sun's angle of incidence + an isotropic sky model.
- **Residual** — the gap between reality and the clear-sky prediction. That's the only thing ML learns.

---

## 4. How the "is it good?" question is answered honestly
Two things make the evaluation trustworthy:

1. **Forward-chaining backtest** — we never let the model see the future. We train on the first months,
   predict the next chunk, then roll forward. This mimics real forecasting.
2. **Skill score** — instead of a raw error, we ask "how much better than a dumb baseline?":
   ```
   skill = 1 - (model_RMSE / baseline_RMSE)
   ```
   - `skill_vs_clear_sky` = 0.85  ->  85% less error than the physics-only baseline.
   - `skill_vs_persistence` = 0.69  ->  69% less error than "tomorrow = yesterday".

   0 means "no better than the baseline", 1 means "perfect". Beating **both** baselines is the bar.

---

## 5. Run it yourself (copy-paste)
```powershell
# from the repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

solar-forecast fetch    --start 2023-01-01 --end 2023-12-31   # 1) get a year of real data (cached)
solar-forecast evaluate --start 2023-01-01 --end 2023-12-31   # 2) backtest -> prints MAE/RMSE/skill
solar-forecast plot     --start 2023-01-01 --end 2023-12-31   # 3) save reports/default_forecast.png
```
Then run the tests to see the guarantees the code checks:
```powershell
pytest -q
```

---

## 6. Experiments to build intuition (do these)
Each one teaches a concept. Change one thing, re-run `evaluate`, and see how the skill moves.

1. **Break the model on purpose.** In `features.py`, comment out `clear_sky_index` from `FEATURE_COLUMNS`.
   Re-run `evaluate`. Skill should *drop* — proof that CSI is the key cloud signal.
2. **Change the site.** In `config.py`, change latitude/longitude (try a cloudier or sunnier city) and the
   panel `tilt_deg`. Re-fetch and re-evaluate. Does the skill change? Why might a cloudy site be *harder*?
3. **Change the model.** In `models.py`, swap the estimator's hyperparameters (e.g. `n_estimators`,
   `max_depth`) or try a `RandomForestRegressor`. Does more complexity actually help the skill?
4. **Add a feature.** Add a new column in `features.py` (e.g. a 3-hour rolling mean of `clear_sky_index`,
   remembering to `.shift(1)` so there's no look-ahead). Re-run. Did it help?
5. **Shrink the training data.** Backtest on 3 months instead of 12. How much does less data hurt? This is
   exactly the argument for physics-informed models.

> Tip: after each change, run `pytest -q` first — if a test goes red, you changed a guarantee (e.g.
> introduced look-ahead leakage). That's the test suite doing its job.

---

## 7. Where to look in the code
| I want to understand... | Read this file |
|---|---|
| The site + panel settings | `src/solar_forecast/config.py` |
| How data is fetched/cached | `src/solar_forecast/data.py` |
| The clear-sky + POA + PV physics | `src/solar_forecast/physics.py` |
| The model inputs | `src/solar_forecast/features.py` |
| The residual model + baselines | `src/solar_forecast/models.py` |
| The backtest + metrics | `src/solar_forecast/evaluate.py` |
| The command-line entry points | `src/solar_forecast/cli.py` |
| The interactive dashboard | `app.py` |

Read them in that order — it follows the pipeline in Section 3.

---

## 8. How this becomes a PINN (the next chapter)
A **Physics-Informed Neural Network** puts the physics *inside* a neural network's training objective.
Instead of a tree model learning the residual, you'll use a small **PyTorch** network and add a
**physics-consistency penalty** to its loss, for example:
- the prediction can't exceed the clear-sky ceiling,
- the prediction can't be negative,
- energy in/out stays consistent.

The network is rewarded for fitting the data *and* obeying physics. Because this repo already has the
physics model, the data pipeline, and an honest evaluation, the PINN drops straight in and has a real
baseline (this project's numbers) to beat. That's the planned next step — no rush, and worth doing slowly.
