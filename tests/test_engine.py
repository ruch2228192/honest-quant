import numpy as np
import pandas as pd
import pytest

from honestquant.backtest import CostModel, run_backtest, run_strategy
from honestquant.data import normalise, synthetic_prices
from honestquant.metrics import max_drawdown, sharpe, summarize
from honestquant.strategies import STRATEGIES

NO_COSTS = CostModel(0, 0, 0)


@pytest.fixture(scope="module")
def prices():
    return synthetic_prices(n_days=252 * 8, seed=3)


def make_prices(opens, closes):
    idx = pd.bdate_range("2020-01-01", periods=len(opens))
    o, c = np.array(opens, float), np.array(closes, float)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 1.0}, index=idx)


# ------------------------------------------------------------------ no lookahead
@pytest.mark.parametrize("name", list(STRATEGIES))
def test_no_lookahead(prices, name):
    """Changing the future must not change the past.

    If any strategy or the engine peeked at future data, the returns up to the
    cut-off would differ between the full and the truncated history.
    """
    spec = STRATEGIES[name]
    cut = len(prices) // 2
    full = run_strategy(prices, spec).returns
    trunc = run_strategy(prices.iloc[:cut], spec).returns
    pd.testing.assert_series_equal(full.iloc[:cut], trunc, check_names=False)

    # And scrambling every future price leaves the past untouched too.
    scrambled = prices.copy()
    scrambled.iloc[cut:] = scrambled.iloc[cut:].sample(frac=1, random_state=0).to_numpy()
    again = run_strategy(scrambled, spec).returns
    pd.testing.assert_series_equal(full.iloc[:cut], again.iloc[:cut], check_names=False)


# ---------------------------------------------------------------- execution timing
def test_signal_executes_at_next_open():
    # Day 0: signal says buy at the close. We must NOT earn day 0 or day 1's overnight gap,
    # only day 1's intraday move (open 110 -> close 121 = +10%).
    p = make_prices(opens=[100, 110, 121], closes=[100, 121, 121])
    sig = pd.Series([1.0, 1.0, 1.0], index=p.index)
    res = run_backtest(p, sig, NO_COSTS)
    assert res.positions.tolist() == [0.0, 1.0, 1.0]
    assert res.returns.iloc[0] == 0
    assert res.returns.iloc[1] == pytest.approx(0.10)
    assert res.returns.iloc[2] == pytest.approx(0.0)


def test_overnight_gap_uses_yesterdays_position():
    # Held long into day 2, signal turns flat at day 1's close -> we still eat day 2's gap
    # (open 90 vs close 100 = -10%) because we can only sell at the open.
    p = make_prices(opens=[100, 100, 90], closes=[100, 100, 95])
    sig = pd.Series([1.0, 0.0, 0.0], index=p.index)
    res = run_backtest(p, sig, NO_COSTS)
    assert res.positions.tolist() == [0.0, 1.0, 0.0]
    assert res.returns.iloc[2] == pytest.approx(-0.10)


def test_short_positions_profit_when_price_falls():
    p = make_prices(opens=[100, 100, 100], closes=[100, 90, 90])
    res = run_backtest(p, pd.Series(-1.0, index=p.index), NO_COSTS, max_leverage=1)
    assert res.returns.iloc[1] == pytest.approx(0.10)


# ------------------------------------------------------------------------- costs
def test_costs_charged_per_unit_of_turnover():
    p = make_prices([100] * 6, [100] * 6)  # flat market -> only costs matter
    sig = pd.Series([1, 0, 1, 0, 1, 0], index=p.index, dtype=float)
    res = run_backtest(p, sig, CostModel(commission_bps=1, slippage_bps=4, borrow_bps_annual=0))
    # 5 position changes of size 1, 5 bps each
    assert res.costs.sum() == pytest.approx(5 * 0.0005)
    assert (1 + res.returns).prod() == pytest.approx((1 - 0.0005) ** 5)


def test_buy_and_hold_pays_costs_once(prices):
    res = run_strategy(prices, STRATEGIES["buy_and_hold"], costs=CostModel(5, 5, 0))
    assert res.costs.sum() == pytest.approx(0.001)
    assert res.metrics()["trades"] == 1


def test_borrow_fee_only_on_shorts():
    p = make_prices([100] * 253, [100] * 253)
    long = run_backtest(p, pd.Series(1.0, index=p.index), CostModel(0, 0, 100))
    short = run_backtest(p, pd.Series(-1.0, index=p.index), CostModel(0, 0, 100))
    assert long.costs.sum() == 0
    assert short.costs.sum() == pytest.approx(0.01, rel=1e-3)  # 1%/yr for a year


def test_vol_target_caps_leverage(prices):
    res = run_strategy(prices, STRATEGIES["buy_and_hold"], vol_target=0.10, max_leverage=1.5)
    assert res.positions.max() <= 1.5 + 1e-12
    assert res.positions.iloc[:21].eq(0).all()  # no position before vol can be measured


# ----------------------------------------------------------------------- metrics
def test_max_drawdown_known_value():
    r = pd.Series([0.2, -0.5, 0.5], index=pd.bdate_range("2020-01-01", periods=3))  # 1.2 -> 0.6 -> 0.9
    assert max_drawdown(r) == pytest.approx(-0.5)


def test_sharpe_scale_invariant_and_annualised():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 1000))
    assert sharpe(r) == pytest.approx(r.mean() / r.std() * np.sqrt(252))
    assert sharpe(r * 3) == pytest.approx(sharpe(r))


def test_summarize_cagr(prices):
    r = pd.Series(0.0004, index=pd.bdate_range("2020-01-01", periods=252))
    m = summarize(r)
    assert m["cagr"] == pytest.approx(1.0004**252 - 1)
    assert m["max_drawdown"] == 0


# -------------------------------------------------------------------------- data
def test_synthetic_is_deterministic():
    a, b = synthetic_prices(500, seed=1), synthetic_prices(500, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert (a["high"] >= a[["open", "close"]].max(axis=1)).all()
    assert (a["low"] <= a[["open", "close"]].min(axis=1)).all()


def test_normalise_applies_adjusted_close():
    raw = pd.DataFrame({"Date": ["2020-01-02", "2020-01-03"], "Open": [10, 10], "High": [11, 11],
                        "Low": [9, 9], "Close": [10, 10], "Adj Close": [5, 10], "Volume": [1, 1]})
    df = normalise(raw)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].tolist() == [5, 10] and df["open"].tolist() == [5, 10]


def test_normalise_handles_yfinance_multiindex():
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["SPY"]])
    raw = pd.DataFrame([[1, 2, 0.5, 1.5, 10]] * 3, columns=cols, index=pd.bdate_range("2021-01-01", periods=3))
    assert normalise(raw)["close"].tolist() == [1.5] * 3
