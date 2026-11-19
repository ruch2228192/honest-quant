"""Paper trading with Alpaca (https://alpaca.markets) - fake money, real market.

Run once a day AFTER the US close. The bot reads the latest daily bar, asks the
strategy for tomorrow's position, and sends a market order that fills at the
next open - exactly the timing the backtester assumes.

Safety rails:
  * Only the PAPER endpoint is allowed. There is no switch for real money.
  * Dry-run is the default; nothing is sent unless you pass --submit.
  * Every decision is appended to a CSV journal so you can audit the bot later.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .data import normalise
from .strategies import StrategySpec

PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"


class AlpacaPaperBroker:
    def __init__(self, key_id: str | None = None, secret: str | None = None, base_url: str = PAPER_URL, session=None):
        if base_url.rstrip("/") != PAPER_URL:
            raise ValueError("This bot only talks to Alpaca's PAPER endpoint - no real money.")
        self.key_id = key_id or os.environ.get("APCA_API_KEY_ID")
        self.secret = secret or os.environ.get("APCA_API_SECRET_KEY")
        if not self.key_id or not self.secret:
            raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (paper keys from alpaca.markets).")
        self.base_url = PAPER_URL
        if session is None:
            import requests
            session = requests.Session()
        self.s = session
        self.s.headers.update({"APCA-API-KEY-ID": self.key_id, "APCA-API-SECRET-KEY": self.secret})

    def _get(self, url, **params):
        r = self.s.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def equity(self) -> float:
        return float(self._get(f"{self.base_url}/v2/account")["equity"])

    def position_qty(self, symbol: str) -> int:
        r = self.s.get(f"{self.base_url}/v2/positions/{symbol}", timeout=30)
        if r.status_code == 404:
            return 0
        r.raise_for_status()
        return int(float(r.json()["qty"]))

    def daily_bars(self, symbol: str, days: int = 600) -> pd.DataFrame:
        start = (datetime.now(timezone.utc) - timedelta(days=int(days * 1.5))).strftime("%Y-%m-%d")
        rows, token = [], None
        while True:
            params = {"timeframe": "1Day", "start": start, "limit": 10000, "adjustment": "all", "feed": "iex"}
            if token:
                params["page_token"] = token
            data = self._get(f"{DATA_URL}/v2/stocks/{symbol}/bars", **params)
            rows += data.get("bars") or []
            token = data.get("next_page_token")
            if not token:
                break
        df = pd.DataFrame(rows).rename(columns={"t": "date", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        return normalise(df)

    def submit_market_order(self, symbol: str, qty: int, side: str) -> dict:
        order = {"symbol": symbol, "qty": str(qty), "side": side, "type": "market", "time_in_force": "day"}
        r = self.s.post(f"{self.base_url}/v2/orders", json=order, timeout=30)
        r.raise_for_status()
        return r.json()


class SimulatedAccount:
    """Stand-in broker for dry runs without API keys."""

    def __init__(self, prices: pd.DataFrame, equity: float = 100_000.0, qty: int = 0):
        self._prices, self._equity, self._qty = prices, equity, qty

    def equity(self) -> float:
        return self._equity

    def position_qty(self, symbol: str) -> int:
        return self._qty

    def daily_bars(self, symbol: str, days: int = 600) -> pd.DataFrame:
        return self._prices

    def submit_market_order(self, symbol, qty, side):
        raise RuntimeError("SimulatedAccount cannot submit orders")


@dataclass
class Decision:
    timestamp: str
    symbol: str
    strategy: str
    params: dict
    bar_date: str
    last_close: float
    signal: float
    equity: float
    current_qty: int
    target_qty: int
    order_side: str | None
    order_qty: int
    submitted: bool
    order_id: str | None = None


def plan_order(signal: float, equity: float, price: float, current_qty: int, allocation: float = 0.95):
    """How many shares to buy/sell to move from current_qty to the target."""
    if price <= 0:
        raise ValueError("price must be positive")
    target_value = signal * allocation * equity
    target_qty = int(math.trunc(target_value / price))   # whole shares, rounded toward zero
    diff = target_qty - current_qty
    if diff == 0:
        return target_qty, None, 0
    return target_qty, ("buy" if diff > 0 else "sell"), abs(diff)


def run_once(
    broker,
    spec: StrategySpec,
    symbol: str,
    params: dict | None = None,
    allocation: float = 0.95,
    submit: bool = False,
    journal: str | Path | None = "journal.csv",
) -> Decision:
    params = {**spec.defaults, **(params or {})}
    bars = broker.daily_bars(symbol)
    if len(bars) < 30:
        raise RuntimeError("not enough price history to compute a signal")
    sig = float(spec.signal(bars, params).iloc[-1])
    last_close = float(bars["close"].iloc[-1])
    equity = broker.equity()
    current = broker.position_qty(symbol)
    target, side, qty = plan_order(sig, equity, last_close, current, allocation)

    order_id = None
    if submit and side:
        order_id = broker.submit_market_order(symbol, qty, side).get("id")

    d = Decision(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        symbol=symbol, strategy=spec.name, params=params,
        bar_date=bars.index[-1].date().isoformat(), last_close=last_close, signal=sig,
        equity=equity, current_qty=current, target_qty=target,
        order_side=side, order_qty=qty, submitted=bool(submit and side), order_id=order_id,
    )
    if journal:
        path = Path(journal)
        new = not path.exists()
        with path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(d.__dict__))
            if new:
                w.writeheader()
            w.writerow(d.__dict__)
    return d
