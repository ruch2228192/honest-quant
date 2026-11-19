#!/usr/bin/env python3
"""Daily paper-trading bot (Alpaca paper account - fake money).

  # dry run, no account needed (uses downloaded prices and a pretend $100k):
  python paper_trade.py --symbol SPY --strategy sma_crossover --params '{"fast": 100, "slow": 200}'

  # with your Alpaca PAPER keys, still a dry run:
  export APCA_API_KEY_ID=...  APCA_API_SECRET_KEY=...
  python paper_trade.py --symbol SPY --strategy sma_crossover --alpaca

  # actually place the paper order:
  python paper_trade.py --symbol SPY --strategy sma_crossover --alpaca --submit

Run it once per trading day after the US close (22:15 Amsterdam time works).
"""
from __future__ import annotations

import argparse
import json

from honestquant.data import load_prices
from honestquant.live import AlpacaPaperBroker, SimulatedAccount, run_once
from honestquant.strategies import STRATEGIES, get_strategy


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--strategy", default="sma_crossover", choices=list(STRATEGIES))
    ap.add_argument("--params", default="{}", help='JSON, e.g. \'{"fast": 100, "slow": 200}\'')
    ap.add_argument("--allocation", type=float, default=0.95, help="fraction of equity to put to work")
    ap.add_argument("--alpaca", action="store_true", help="use your Alpaca paper account")
    ap.add_argument("--submit", action="store_true", help="really send the (paper) order")
    ap.add_argument("--source", default="auto", help="price source for dry runs without --alpaca")
    ap.add_argument("--journal", default="journal.csv")
    args = ap.parse_args(argv)

    if args.submit and not args.alpaca:
        ap.error("--submit needs --alpaca")
    spec = get_strategy(args.strategy)
    params = json.loads(args.params)

    if args.alpaca:
        broker = AlpacaPaperBroker()
    else:
        broker = SimulatedAccount(load_prices(args.symbol, "2015-01-01", source=args.source))

    d = run_once(broker, spec, args.symbol, params, args.allocation, submit=args.submit, journal=args.journal)
    state = "LONG" if d.signal > 0 else "SHORT" if d.signal < 0 else "FLAT"
    print(f"{d.bar_date}  {d.symbol} close {d.last_close:.2f}  →  strategy says {state} ({d.signal:+.2f})")
    print(f"equity ${d.equity:,.0f} · holding {d.current_qty} · target {d.target_qty}")
    if d.order_side:
        verb = "SENT" if d.submitted else "WOULD SEND (dry run)"
        print(f"{verb}: market {d.order_side} {d.order_qty} {d.symbol} for the next open"
              + (f" · order id {d.order_id}" if d.order_id else ""))
    else:
        print("No trade needed.")
    print(f"logged to {args.journal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
