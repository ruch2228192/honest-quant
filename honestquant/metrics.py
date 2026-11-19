"""Performance metrics. Returns are simple daily returns (0.01 = +1%)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(returns: pd.Series, start: float = 1.0) -> pd.Series:
    return start * (1 + returns.fillna(0)).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Percentage below the running peak (0 = at a new high, -0.3 = 30% down)."""
    eq = equity_curve(returns)
    peak = np.maximum.accumulate(np.maximum(eq.to_numpy(), 1.0))
    return pd.Series(eq.to_numpy() / peak - 1, index=eq.index, name="drawdown")


def max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    return float(drawdown_series(returns).min())


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    r = returns.dropna()
    sd = r.std(ddof=1)
    if len(r) < 2 or not sd or not np.isfinite(sd):
        return float("nan")
    return float((r.mean() - rf / TRADING_DAYS) / sd * math.sqrt(TRADING_DAYS))


def summarize(returns: pd.Series, positions: pd.Series | None = None, rf: float = 0.0) -> dict:
    r = returns.dropna()
    n = len(r)
    years = n / TRADING_DAYS
    final = float((1 + r).prod()) if n else 1.0
    vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if n > 1 else float("nan")
    downside = float(np.sqrt(np.mean(np.minimum(r.to_numpy(), 0) ** 2)) * math.sqrt(TRADING_DAYS)) if n else 0.0
    cagr = final ** (1 / years) - 1 if years > 0 and final > 0 else float("nan")
    mdd = max_drawdown(r)
    active = r[r != 0]

    out = {
        "start": r.index[0].date().isoformat() if n else None,
        "end": r.index[-1].date().isoformat() if n else None,
        "years": round(years, 2),
        "total_return": final - 1,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe(r, rf),
        "sortino": float((r.mean() - rf / TRADING_DAYS) * TRADING_DAYS / downside) if downside > 0 else float("nan"),
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 and np.isfinite(cagr) else float("nan"),
        "hit_rate": float((active > 0).mean()) if len(active) else float("nan"),
    }
    if positions is not None:
        p = positions.reindex(r.index).fillna(0.0)
        changes = p.diff().fillna(p.iloc[0] if len(p) else 0.0)
        out["exposure"] = float((p != 0).mean()) if len(p) else 0.0
        out["trades"] = int((changes != 0).sum())
        out["turnover_per_year"] = float(changes.abs().sum() / years) if years > 0 else float("nan")
    return out


PRETTY = {
    "total_return": ("Total return", "pct"),
    "cagr": ("CAGR (yearly)", "pct"),
    "volatility": ("Volatility", "pct"),
    "sharpe": ("Sharpe ratio", "num"),
    "sortino": ("Sortino ratio", "num"),
    "max_drawdown": ("Max drawdown", "pct"),
    "calmar": ("Calmar ratio", "num"),
    "hit_rate": ("Winning days", "pct"),
    "exposure": ("Time in market", "pct"),
    "trades": ("Trades", "int"),
    "turnover_per_year": ("Turnover / year", "x"),
}


def fmt(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    if kind == "pct":
        return f"{value * 100:+.1f}%" if value < 0 else f"{value * 100:.1f}%"
    if kind == "int":
        return f"{int(value):,}"
    if kind == "x":
        return f"{value:.1f}×"
    return f"{value:.2f}"
