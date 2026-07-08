"""Solar-Forecast (PhysSolar): physics-informed solar PV output forecasting.

Approach: a physical clear-sky model is the prior; an ML model learns the residual
(clouds, temperature). final prediction = clear_sky_power + residual(features), clipped >= 0.
"""
from . import config, data, evaluate, features, models, physics  # noqa: F401
from .config import DEFAULT_SITE, Site  # noqa: F401

__version__ = "0.0.1"
__all__ = [
    "config", "data", "physics", "features", "models", "evaluate",
    "Site", "DEFAULT_SITE", "__version__",
]
