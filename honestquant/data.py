"""Price data loading.

Every loader returns a DataFrame indexed by date with float columns
open, high, low, close, volume - adjusted for splits and dividends so that
corporate actions don't show up as fake crashes or rallies.

Sources, tried in this order with ``source="auto"``:
  1. a local cache (data/<TICKER>.csv) from an earlier download
  2. yfinance (if installed)
  3. Stooq's free CSV endpoint (needs only ``requests``)
Use ``source="synthetic"`` for an offline, reproducible fake market.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]
CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce any OHLCV frame into the project's standard shape."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance returns (field, ticker)
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"

    missing = [c for c in ("open", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"price data is missing columns: {missing}")

    # If an adjusted close is present, scale every price by the same factor.
    if "adj_close" in df.columns:
        factor = df["adj_close"] / df["close"]
        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                df[col] = df[col] * factor
    if "high" not in df.columns:
        df["high"] = df[["open", "close"]].max(axis=1)
    if "low" not in df.columns:
        df["low"] = df[["open", "close"]].min(axis=1)
    if "volume" not in df.columns:
        df["volume"] = np.nan

    df = df[COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "close"])
    df = df[(df["open"] > 0) & (df["close"] > 0)]
    if len(df) < 2:
        raise ValueError("not enough price rows after cleaning")
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    return normalise(pd.read_csv(path))


def _from_yfinance(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf  # optional dependency

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")
    return normalise(raw)


def _from_stooq(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import requests

    symbol = ticker.lower() if "." in ticker else f"{ticker.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text.strip()
    if not text.lower().startswith("date"):
        raise ValueError(f"Stooq returned no CSV for {ticker}: {text[:80]!r}")
    df = normalise(pd.read_csv(io.StringIO(text)))
    return df.loc[start:end] if end else df.loc[start:]


def synthetic_prices(
    n_days: int = 252 * 20,
    seed: int = 7,
    start: str = "2005-01-03",
    s0: float = 100.0,
) -> pd.DataFrame:
    """A fake but realistic-ish market for offline testing.

    Two hidden regimes (calm bull / volatile bear) with Markov switching and
    fat-tailed (Student-t) shocks. Daily returns are split into an overnight
    gap and an intraday move so the open price is meaningful.
    """
    rng = np.random.default_rng(seed)
    p_stay = np.array([0.995, 0.97])
    mu = np.array([0.0006, -0.0009])
    sigma = np.array([0.008, 0.021])

    states = np.empty(n_days, dtype=int)
    s = 0
    for i in range(n_days):
        states[i] = s
        if rng.random() > p_stay[s]:
            s = 1 - s

    nu = 4.0
    shocks = rng.standard_t(nu, size=n_days) / np.sqrt(nu / (nu - 2))
    r = mu[states] + sigma[states] * shocks  # daily log return

    overnight = 0.35 * r + rng.normal(0, 0.002, n_days)
    intraday = r - overnight
    close = s0 * np.exp(np.cumsum(r))
    prev_close = np.concatenate([[s0], close[:-1]])
    open_ = prev_close * np.exp(overnight)
    wiggle = np.abs(rng.normal(0, 0.5, (2, n_days))) * sigma[states]
    high = np.maximum(open_, close) * np.exp(wiggle[0])
    low = np.minimum(open_, close) * np.exp(-wiggle[1])
    volume = rng.lognormal(mean=15, sigma=0.4, size=n_days) * (1 + states)

    idx = pd.bdate_range(start, periods=n_days, name="date")
    _ = intraday  # kept for readability: close = open * exp(intraday)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def load_prices(
    ticker: str = "SPY",
    start: str = "2005-01-01",
    end: str | None = None,
    source: str = "auto",
    csv_path: str | Path | None = None,
    use_cache: bool = True,
    seed: int = 7,
) -> pd.DataFrame:
    """Load adjusted daily OHLCV for ``ticker``.

    source: "auto" | "yfinance" | "stooq" | "csv" | "synthetic"
    """
    if source == "synthetic":
        return synthetic_prices(seed=seed)
    if source == "csv" or csv_path is not None:
        if csv_path is None:
            raise ValueError("source='csv' needs csv_path")
        df = load_csv(csv_path)
        return df.loc[start:end] if end else df.loc[start:]

    cache = CACHE_DIR / f"{ticker.upper()}.csv"
    if use_cache and source == "auto" and cache.exists():
        df = load_csv(cache)
        return df.loc[start:end] if end else df.loc[start:]

    loaders = {"yfinance": _from_yfinance, "stooq": _from_stooq}
    order = ["yfinance", "stooq"] if source == "auto" else [source]
    errors = []
    for name in order:
        if name not in loaders:
            raise ValueError(f"unknown source {name!r}")
        try:
            df = loaders[name](ticker, start, end)
        except Exception as exc:  # try the next source
            errors.append(f"{name}: {exc}")
            continue
        if use_cache:
            CACHE_DIR.mkdir(exist_ok=True)
            df.to_csv(cache)
        return df
    raise RuntimeError(
        "Could not download prices. Install yfinance (pip install yfinance), "
        "pass --csv with your own file, or use --source synthetic.\n  "
        + "\n  ".join(errors)
    )
