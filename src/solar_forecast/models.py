"""Baselines + the residual forecaster (prediction = clear_sky_power + learned residual)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import physics
from .config import MODELS_DIR, Site
from .features import FEATURE_COLUMNS


def baseline_clear_sky(site: Site, df: pd.DataFrame) -> pd.Series:
    """Physical clear-sky baseline aligned to df.index (temperature-aware)."""
    temp = df["temp_c"] if "temp_c" in df.columns else 25.0
    return physics.clear_sky_power(site, df.index, temp_c=temp)


def baseline_persistence(y: pd.Series, lag: int = 24) -> pd.Series:
    """Persistence baseline: the value from `lag` hours ago."""
    return pd.Series(y).shift(lag)


class ResidualForecaster:
    """Predicts PV output as clear_sky_power + residual(features), clipped to >= 0."""

    def __init__(self, site: Site):
        self.site = site
        self.columns = list(FEATURE_COLUMNS)
        self.model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=6, random_state=42
        )

    def fit(self, X: pd.DataFrame, residual) -> "ResidualForecaster":
        self.model.fit(X[self.columns].to_numpy(), np.asarray(residual, dtype=float))
        return self

    def predict(self, X: pd.DataFrame, clear_sky: pd.Series) -> pd.Series:
        r_hat = self.model.predict(X[self.columns].to_numpy())
        base = np.asarray(pd.Series(clear_sky).reindex(X.index), dtype=float)
        return pd.Series(np.clip(base + r_hat, 0.0, None), index=X.index, name="prediction")

    def save(self, path=None) -> Path:
        path = Path(path) if path else MODELS_DIR / f"{self.site.name}_residual.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "columns": self.columns, "site": self.site}, path)
        return path

    @classmethod
    def load(cls, path) -> "ResidualForecaster":
        blob = joblib.load(path)
        obj = cls(blob["site"])
        obj.model = blob["model"]
        obj.columns = blob["columns"]
        return obj
