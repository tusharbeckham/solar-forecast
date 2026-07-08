# Math - Solar-Forecast (PhysSolar)

## Problem
Given features x_t (weather + time + clear-sky prior), predict PV output y_t as a correction on the
physical prior:
- y_t = c_t + r_t ,  where  c_t = clear_sky_power(t)  and  r_t = weather-driven residual.
- Learn  f: x_t -> r_hat_t .  Prediction:  y_hat_t = max(0, c_t + f(x_t)).

## Residual target
  r_t = y_t - c_t .   Minimizing error on r is equivalent to minimizing it on y (c_t is known/fixed).

## Loss functions (candidates)
- MSE:  mean( (r - r_hat)^2 )    - smooth, default.
- MAE:  mean( |r - r_hat| )      - robust to outliers.
- Huber: quadratic for |e| <= delta, linear beyond - robust compromise.

## Metrics
- MAE   = mean |y - y_hat|
- RMSE  = sqrt( mean( (y - y_hat)^2 ) )
- MBE   = mean( y_hat - y )                    (bias)
- R2    = 1 - SS_res / SS_tot
- Skill = 1 - RMSE_model / RMSE_reference      ( >0 beats the reference; computed vs clear-sky and persistence )

## Time-series cross-validation (no leakage)
Forward-chaining / rolling-origin: for fold i, train on [0, k*i), test on [k*i, k*(i+1)).
Train always precedes test in time; no shuffling; lag features use only past values.
(`evaluate.forward_chaining_splits`)

## PINN extension (Phase 5)
Augment the loss with a physics-consistency penalty:
  L = (1/N) * sum (y - y_hat)^2  +  lambda * P
  P = mean( relu(-y_hat)^2 )                 (never negative output)
    + mean( relu(y_hat - c_ceiling)^2 )      (never exceed the clear-sky ceiling)
lambda trades data-fit against physical plausibility.
