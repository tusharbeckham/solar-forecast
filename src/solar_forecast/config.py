"""Project configuration: site definition + paths."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# repo root = .../src/solar_forecast/config.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models_store"


@dataclass(frozen=True)
class Site:
    """A PV site + its panel characteristics."""
    name: str = "default"
    latitude: float = 28.61
    longitude: float = 77.20
    timezone: str = "Asia/Kolkata"
    altitude_m: float = 216.0
    tilt_deg: float = 28.0
    azimuth_deg: float = 180.0          # 180 = due south (northern hemisphere)
    panel_area_m2: float = 1.6
    efficiency: float = 0.18            # module efficiency (fraction)
    performance_ratio: float = 0.80     # system losses (inverter, wiring, soiling)
    temp_coeff_per_c: float = -0.004    # power temperature coefficient (per degC)


DEFAULT_SITE = Site()
