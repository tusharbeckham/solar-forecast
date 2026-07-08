"""Command-line interface: fetch / train / predict / evaluate / plot."""
from __future__ import annotations

import argparse

import pandas as pd

from . import data, evaluate, features, models, physics
from .config import DEFAULT_SITE, MODELS_DIR


def cmd_fetch(args):
    df = data.get_weather(DEFAULT_SITE, args.start, args.end, use_cache=not args.no_cache)
    print(f"fetched {len(df)} hourly rows [{df.index.min()} .. {df.index.max()}] -> cached")


def cmd_train(args):
    site = DEFAULT_SITE
    df = data.get_weather(site, args.start, args.end)
    feats = features.build(df, site)
    y = physics.observed_power(site, df).reindex(feats.index)
    clear = physics.clear_sky_power(site, feats.index, temp_c=feats["temp_c"]).reindex(feats.index)
    model = models.ResidualForecaster(site).fit(feats, y - clear)
    path = model.save()
    print(f"trained on {len(feats)} rows; model saved -> {path}")


def cmd_predict(args):
    site = DEFAULT_SITE
    df = data.get_weather(site, args.start, args.end)
    feats = features.build(df, site)
    clear = physics.clear_sky_power(site, feats.index, temp_c=feats["temp_c"]).reindex(feats.index)
    model = models.ResidualForecaster.load(MODELS_DIR / f"{site.name}_residual.joblib")
    print(model.predict(feats, clear).tail(12).to_string())


def cmd_evaluate(args):
    site = DEFAULT_SITE
    df = data.get_weather(site, args.start, args.end)
    result = evaluate.backtest(site, df, n_splits=args.splits)
    print("Backtest (forward-chaining):")
    for k, v in result.items():
        print(f"  {k:22s}: {v:.4f}")
    evaluate.write_report(f"{site.name}_backtest", result)


def cmd_plot(args):
    site = DEFAULT_SITE
    df = data.get_weather(site, args.start, args.end)
    feats = features.build(df, site)
    y = physics.observed_power(site, df).reindex(feats.index)
    clear = physics.clear_sky_power(site, feats.index, temp_c=feats["temp_c"]).reindex(feats.index)
    cutoff = feats.index.max() - pd.Timedelta(days=args.holdout_days)
    train_mask = feats.index <= cutoff
    model = models.ResidualForecaster(site).fit(feats[train_mask], (y - clear)[train_mask])
    pred = model.predict(feats, clear)
    day_start = args.day_start or (feats.index.max() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    day_end = args.day_end or feats.index.max().strftime("%Y-%m-%d")
    from . import plots
    out = plots.plot_forecast(site, df, pred, day_start, day_end)
    print(f"saved plot -> {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="solar-forecast",
                                description="Physics-informed solar PV output forecasting")
    sub = p.add_subparsers(dest="command", required=True)

    def _common(sp):
        sp.add_argument("--start", default="2023-01-01", help="start date YYYY-MM-DD")
        sp.add_argument("--end", default="2023-12-31", help="end date YYYY-MM-DD")

    f = sub.add_parser("fetch", help="download + cache weather/irradiance")
    _common(f)
    f.add_argument("--no-cache", action="store_true")
    f.set_defaults(func=cmd_fetch)

    t = sub.add_parser("train", help="train the residual model")
    _common(t)
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="predict output for a date range")
    _common(pr)
    pr.set_defaults(func=cmd_predict)

    e = sub.add_parser("evaluate", help="forward-chaining backtest vs baselines")
    _common(e)
    e.add_argument("--splits", type=int, default=5)
    e.set_defaults(func=cmd_evaluate)

    pl = sub.add_parser("plot", help="plot forecast vs actual vs clear-sky (saves PNG to reports/)")
    _common(pl)
    pl.add_argument("--holdout-days", type=int, default=7)
    pl.add_argument("--day-start", default=None)
    pl.add_argument("--day-end", default=None)
    pl.set_defaults(func=cmd_plot)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
