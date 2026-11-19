"""Trading strategies.

A strategy is a function ``f(prices, **params) -> pd.Series`` that returns the
*target position* for each day, using only information available at that
day's close. 1.0 = fully long, 0.0 = flat, -1.0 = fully short.

The backtester is responsible for timing: a signal computed at the close of
day t is executed at the open of day t+1. Strategies never need to shift.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


def buy_and_hold(prices: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=prices.index, name="signal")


def sma_crossover(
    prices: pd.DataFrame, fast: int = 50, slow: int = 200, allow_short: bool = False
) -> pd.Series:
    """Trend following: long when the fast moving average is above the slow one."""
    if fast >= slow:
        raise ValueError("fast must be shorter than slow")
    close = prices["close"]
    f = close.rolling(fast).mean()
    s = close.rolling(slow).mean()
    off = -1.0 if allow_short else 0.0
    sig = pd.Series(np.where(f > s, 1.0, off), index=prices.index, name="signal")
    return sig.where(s.notna(), 0.0)


def time_series_momentum(
    prices: pd.DataFrame, lookback: int = 252, skip: int = 21, allow_short: bool = False
) -> pd.Series:
    """Long if the return from t-lookback to t-skip was positive.

    Skipping the most recent month is the classic academic recipe (it avoids
    short-term reversal).
    """
    if skip >= lookback:
        raise ValueError("skip must be shorter than lookback")
    close = prices["close"]
    past = close.shift(skip) / close.shift(lookback) - 1
    off = -1.0 if allow_short else 0.0
    sig = pd.Series(np.where(past > 0, 1.0, off), index=prices.index, name="signal")
    return sig.where(past.notna(), 0.0)


def _hysteresis(enter: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    """Turn entry/exit booleans into a 0/1 position that remembers its state."""
    pos = np.zeros(len(enter))
    held = 0.0
    for i in range(len(enter)):
        if held == 0.0 and enter[i]:
            held = 1.0
        elif held == 1.0 and exit_[i]:
            held = 0.0
        pos[i] = held
    return pos


def bollinger_reversion(
    prices: pd.DataFrame, window: int = 20, entry_z: float = 2.0, exit_z: float = 0.0
) -> pd.Series:
    """Mean reversion: buy when price is ``entry_z`` std devs below its average,
    sell once it has recovered to ``exit_z``."""
    close = prices["close"]
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    z = ((close - mean) / std).to_numpy()
    with np.errstate(invalid="ignore"):
        enter = z < -entry_z
        exit_ = z > exit_z
    return pd.Series(_hysteresis(enter, exit_), index=prices.index, name="signal")


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index (0-100)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def rsi_reversion(
    prices: pd.DataFrame, period: int = 14, entry: float = 30, exit: float = 55
) -> pd.Series:
    """Mean reversion: buy when RSI is oversold, sell when it recovers."""
    if exit <= entry:
        raise ValueError("exit must be above entry")
    r = rsi(prices["close"], period).to_numpy()
    with np.errstate(invalid="ignore"):
        enter = r < entry
        exit_ = r > exit
    return pd.Series(_hysteresis(enter, exit_), index=prices.index, name="signal")


@dataclass(frozen=True)
class StrategySpec:
    name: str
    func: Callable[..., pd.Series]
    description: str
    defaults: dict = field(default_factory=dict)
    grid: dict = field(default_factory=dict)

    def signal(self, prices: pd.DataFrame, params: dict | None = None) -> pd.Series:
        return self.func(prices, **{**self.defaults, **(params or {})})

    def param_combinations(self) -> list[dict]:
        if not self.grid:
            return [dict(self.defaults)]
        keys = list(self.grid)
        combos = []
        for values in itertools.product(*(self.grid[k] for k in keys)):
            p = {**self.defaults, **dict(zip(keys, values))}
            if _valid(p):
                combos.append(p)
        return combos


def _valid(p: dict) -> bool:
    if "fast" in p and "slow" in p and p["fast"] >= p["slow"]:
        return False
    if "skip" in p and "lookback" in p and p["skip"] >= p["lookback"]:
        return False
    if "entry" in p and "exit" in p and p["exit"] <= p["entry"]:
        return False
    return True


STRATEGIES: dict[str, StrategySpec] = {
    s.name: s
    for s in [
        StrategySpec(
            "buy_and_hold",
            buy_and_hold,
            "Buy on day one and never sell. The benchmark every strategy must beat.",
        ),
        StrategySpec(
            "sma_crossover",
            sma_crossover,
            "Trend following: hold the asset while its fast moving average is above the slow one.",
            defaults={"fast": 50, "slow": 200},
            grid={"fast": [10, 20, 50, 100], "slow": [50, 100, 150, 200, 250]},
        ),
        StrategySpec(
            "time_series_momentum",
            time_series_momentum,
            "Momentum: hold the asset if it went up over the past months (skipping the latest weeks).",
            defaults={"lookback": 252, "skip": 21},
            grid={"lookback": [63, 126, 189, 252], "skip": [0, 5, 21]},
        ),
        StrategySpec(
            "bollinger_reversion",
            bollinger_reversion,
            "Mean reversion: buy sharp dips below the recent average, sell when price recovers.",
            defaults={"window": 20, "entry_z": 2.0, "exit_z": 0.0},
            grid={"window": [10, 20, 40], "entry_z": [1.0, 1.5, 2.0, 2.5], "exit_z": [0.0, 0.5]},
        ),
        StrategySpec(
            "rsi_reversion",
            rsi_reversion,
            "Mean reversion: buy when RSI says oversold, sell when it bounces back.",
            defaults={"period": 14, "entry": 30, "exit": 55},
            grid={"period": [2, 5, 14], "entry": [10, 20, 30], "exit": [50, 60, 70]},
        ),
    ]
}


def get_strategy(name: str) -> StrategySpec:
    try:
        return STRATEGIES[name]
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}") from None
