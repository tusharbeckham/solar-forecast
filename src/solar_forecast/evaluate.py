"""Evaluation: metrics, leakage-free forward-chaining backtest, and report writing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPORTS_DIR, Site


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


def mbe(y, yhat) -> float:
    return float(np.mean(np.asarray(yhat, float) - np.asarray(y, float)))


def r2(y, yhat) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def skill_score(y, yhat, y_ref) -> float:
    """1 - RMSE_model / RMSE_reference. > 0 means better than the reference baseline."""
    ref = rmse(y, y_ref)
    return float(1.0 - rmse(y, yhat) / ref) if ref > 0 else 0.0


def forward_chaining_splits(n: int, n_splits: int = 5):
    """Rolling-origin (expanding-window) splits as (train_idx, test_idx). Strictly time-ordered."""
    fold = n // (n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        train_end = fold * i
        test_end = min(fold * (i + 1), n)
        if fold == 0 or test_end <= train_end:
            continue
        splits.append((np.arange(0, train_end), np.arange(train_end, test_end)))
    return splits


def metrics_table(y, yhat, y_clear, y_persist=None) -> dict:
    out = {
        "mae": mae(y, yhat),
        "rmse": rmse(y, yhat),
        "mbe": mbe(y, yhat),
        "r2": r2(y, yhat),
        "skill_vs_clear_sky": skill_score(y, yhat, y_clear),
    }
    if y_persist is not None:
        out["skill_vs_persistence"] = skill_score(y, yhat, y_persist)
    return out


def backtest(site: Site, df: pd.DataFrame, n_splits: int = 5) -> dict:
    """Forward-chaining backtest of the residual model vs clear-sky & persistence baselines."""
    from . import features as _features
    from . import models as _models
    from . import physics as _physics

    feats = _features.build(df, site)
    y = _physics.observed_power(site, df).reindex(feats.index)
    clear = _physics.clear_sky_power(site, feats.index, temp_c=feats["temp_c"]).reindex(feats.index)
    persist_all = y.shift(24).reindex(feats.index)
    residual = y - clear

    agg = {"mae": [], "rmse": [], "skill_vs_clear_sky": [], "skill_vs_persistence": []}
    for tr, te in forward_chaining_splits(len(feats), n_splits):
        model = _models.ResidualForecaster(site).fit(feats.iloc[tr], residual.iloc[tr])
        yhat = model.predict(feats.iloc[te], clear.iloc[te])
        y_te, c_te = y.iloc[te], clear.iloc[te]
        p_te = persist_all.iloc[te].fillna(c_te)
        agg["mae"].append(mae(y_te, yhat))
        agg["rmse"].append(rmse(y_te, yhat))
        agg["skill_vs_clear_sky"].append(skill_score(y_te, yhat, c_te))
        agg["skill_vs_persistence"].append(skill_score(y_te, yhat, p_te))
    return {k: (float(np.mean(v)) if v else 0.0) for k, v in agg.items()}


def write_report(name: str, metrics: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{name}.json"
    path.write_text(json.dumps(metrics, indent=2))
    return path
