#!/usr/bin/env python3
"""Run the honest backtest and write an HTML report.

Examples
  python run_backtest.py --ticker SPY                      # all strategies on the S&P 500 ETF
  python run_backtest.py --ticker QQQ --strategy sma_crossover
  python run_backtest.py --source synthetic                # offline, fake market
  python run_backtest.py --csv my_prices.csv --ticker BTC  # your own data (date,open,high,low,close)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np

from honestquant.backtest import CostModel
from honestquant.data import load_prices
from honestquant.report import build_report
from honestquant.strategies import STRATEGIES, get_strategy
from honestquant.validation import analyze


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--strategy", default="all", help=f"'all' or one of: {', '.join(STRATEGIES)}")
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--source", default="auto", choices=["auto", "yfinance", "stooq", "synthetic"])
    ap.add_argument("--csv", default=None, help="path to your own OHLC csv")
    ap.add_argument("--commission-bps", type=float, default=1.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--vol-target", type=float, default=None, help="e.g. 0.15 to aim for 15%% yearly volatility")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--n-random", type=int, default=500, help="random-timing simulations per strategy")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--open", action="store_true", help="open the report in your browser")
    args = ap.parse_args(argv)

    t0 = time.time()
    prices = load_prices(args.ticker, args.start, args.end, source=args.source, csv_path=args.csv, seed=args.seed)
    if args.source == "synthetic":
        label, note = "SYNTHETIC", f"Synthetic market (seed {args.seed}) — not real data"
    else:
        label, note = args.ticker.upper(), f"{args.ticker.upper()} daily prices"
    note += f", {prices.index[0].date()} → {prices.index[-1].date()} ({len(prices):,} days)"
    if len(prices) < 252 * 7:
        print("Need at least ~7 years of daily data (5 to tune + time to test).", file=sys.stderr)
        return 2
    print(f"Loaded {note}")

    names = list(STRATEGIES) if args.strategy == "all" else ["buy_and_hold", get_strategy(args.strategy).name]
    names = list(dict.fromkeys(names))
    costs = CostModel(args.commission_bps, args.slippage_bps)
    bt_kwargs = {"vol_target": args.vol_target} if args.vol_target else {}

    analyses = []
    for name in names:
        t = time.time()
        a = analyze(prices, STRATEGIES[name], label, costs, args.test_frac, args.n_random, args.seed, **bt_kwargs)
        analyses.append(a)
        wf = a.walk.metrics()
        score = f"{sum(c['passed'] for c in a.checks)}/{len(a.checks)} checks" if a.checks else "benchmark"
        print(f"  {name:22s} walk-forward Sharpe {wf['sharpe']:5.2f}  CAGR {wf['cagr'] * 100:5.1f}%  "
              f"{score:10s} ({time.time() - t:.1f}s)")

    settings = {"costs": f"{args.commission_bps:g} bps commission + {args.slippage_bps:g} bps slippage",
                "execution": "next-day open", "test split": f"last {args.test_frac:.0%}",
                "random sims": args.n_random}
    if args.vol_target:
        settings["vol target"] = f"{args.vol_target:.0%}"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"{label}_report.html"
    report_path.write_text(build_report(analyses, label, note, settings), encoding="utf-8")

    def clean(v):
        if isinstance(v, (np.floating, float)):
            return None if not np.isfinite(v) else round(float(v), 6)
        if isinstance(v, np.integer):
            return int(v)
        return v

    summary = {
        a.spec.name: {
            "best_params": a.split.best_params,
            "walk_forward": {k: clean(v) for k, v in a.walk.metrics().items()},
            "out_of_sample": {k: clean(v) for k, v in a.split.out_of_sample.metrics().items()},
            "random_timing_p": clean(a.random_test.p_value) if a.random_test else None,
            "deflated_sharpe": clean(a.dsr),
            "checks": a.checks,
            "verdict": a.verdict,
        }
        for a in analyses
    }
    (out / f"{label}_results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nReport: {report_path.resolve()}  ({time.time() - t0:.0f}s)")
    if args.open:
        webbrowser.open(report_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
