"""Plotting: forecast vs observed vs clear-sky prior."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import physics  # noqa: E402
from .config import REPORTS_DIR, Site  # noqa: E402


def plot_forecast(site: Site, df: pd.DataFrame, prediction, day_start, day_end, out=None) -> Path:
    """Plot observed (proxy) vs clear-sky prior vs model forecast over a window; save a PNG."""
    idx = pd.DatetimeIndex(df.index)
    start = pd.Timestamp(day_start, tz="UTC")
    end = pd.Timestamp(day_end, tz="UTC") + pd.Timedelta(days=1)
    sub = df[(idx >= start) & (idx < end)]

    obs = physics.observed_power(site, sub)
    clear = physics.clear_sky_power(site, sub.index, temp_c=sub["temp_c"])

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(sub.index, obs, label="observed (proxy)", color="black", lw=1.5)
    ax.plot(sub.index, clear, label="clear-sky prior", color="orange", ls="--", lw=1.2)
    if prediction is not None:
        pred = pd.Series(prediction).reindex(sub.index)
        ax.plot(sub.index, pred, label="model forecast", color="tab:blue", lw=1.2)
    ax.set_ylabel("PV power (W)")
    ax.set_xlabel("time (UTC)")
    ax.set_title(f"{site.name}: forecast vs actual vs clear-sky")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(out) if out else REPORTS_DIR / f"{site.name}_forecast.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
