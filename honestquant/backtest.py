"""Event-honest vectorised backtester.

Timing model (the part most backtests get wrong):
  * A strategy's signal for day t uses data up to and including the CLOSE of t.
  * The trade happens at the OPEN of day t+1 - you can't trade at a price you
    only learn after the market closes.
  * So day t's return is: overnight gap with yesterday's position, then the
    intraday move with today's (new) position.

Costs:
  * commission + slippage, charged in basis points on every unit of turnover
  * a yearly borrow fee on short positions
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS, equity_curve, summarize


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    borrow_bps_annual: float = 50.0

    @property
    def per_unit_turnover(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 1e4


@dataclass
class BacktestResult:
    name: str
    params: dict
    signal: pd.Series        # what the strategy wanted, known at each close
    positions: pd.Series     # what we actually held during each day (after the open)
    gross_returns: pd.Series
    costs: pd.Series
    returns: pd.Series       # net daily returns
    extra: dict = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        return equity_curve(self.returns)

    def window(self, start=None, end=None) -> "BacktestResult":
        """The same backtest, looked at only between start and end (inclusive)."""
        sl = slice(start, end)
        return BacktestResult(
            self.name, self.params, self.signal.loc[sl], self.positions.loc[sl],
            self.gross_returns.loc[sl], self.costs.loc[sl], self.returns.loc[sl], self.extra,
        )

    def metrics(self) -> dict:
        m = summarize(self.returns, self.positions)
        m["cost_drag_per_year"] = float(self.costs.sum() / max(len(self.costs) / TRADING_DAYS, 1e-9))
        return m


def run_backtest(
    prices: pd.DataFrame,
    signal: pd.Series,
    costs: CostModel = CostModel(),
    name: str = "strategy",
    params: dict | None = None,
    vol_target: float | None = None,
    max_leverage: float = 1.0,
) -> BacktestResult:
    sig = signal.reindex(prices.index).astype(float).fillna(0.0)

    if vol_target:
        # Scale exposure so the position aims for a steady annual volatility.
        # Realised vol uses returns up to today's close -> no lookahead.
        logret = np.log(prices["close"]).diff()
        realised = logret.rolling(20, min_periods=20).std() * np.sqrt(TRADING_DAYS)
        scale = (vol_target / realised).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sig = sig * scale
    sig = sig.clip(-max_leverage, max_leverage)

    pos = sig.shift(1).fillna(0.0)          # executed at the next open
    prev_pos = pos.shift(1).fillna(0.0)     # held overnight into today's open

    gap = (prices["open"] / prices["close"].shift(1) - 1).fillna(0.0)
    intraday = prices["close"] / prices["open"] - 1
    gross = (1 + prev_pos * gap) * (1 + pos * intraday) - 1

    turnover = (pos - prev_pos).abs()
    trade_cost = turnover * costs.per_unit_turnover
    borrow = (-pos).clip(lower=0) * costs.borrow_bps_annual / 1e4 / TRADING_DAYS
    cost = trade_cost + borrow
    net = gross - cost

    return BacktestResult(
        name=name,
        params=dict(params or {}),
        signal=sig.rename("signal"),
        positions=pos.rename("position"),
        gross_returns=gross.rename("gross"),
        costs=cost.rename("costs"),
        returns=net.rename("returns"),
        extra={"vol_target": vol_target, "max_leverage": max_leverage, "costs": costs},
    )


def run_strategy(prices, spec, params=None, costs=CostModel(), **kwargs) -> BacktestResult:
    p = {**spec.defaults, **(params or {})}
    return run_backtest(prices, spec.signal(prices, p), costs=costs, name=spec.name, params=p, **kwargs)
