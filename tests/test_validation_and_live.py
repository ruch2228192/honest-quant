import numpy as np
import pandas as pd
import pytest

from honestquant.backtest import CostModel, run_backtest
from honestquant.data import synthetic_prices
from honestquant.live import AlpacaPaperBroker, SimulatedAccount, plan_order, run_once
from honestquant.strategies import STRATEGIES
from honestquant.validation import (
    analyze, expected_max_sharpe, probabilistic_sharpe, random_timing_test, train_test, walk_forward,
)


@pytest.fixture(scope="module")
def prices():
    return synthetic_prices(n_days=252 * 9, seed=11)


# ---------------------------------------------------------------- honesty tests
def test_random_timing_detects_a_real_edge(prices):
    """A cheating signal that knows tomorrow's move must look wildly significant."""
    intraday = prices["close"] / prices["open"] - 1
    cheat = (intraday.shift(-1) > 0).astype(float)  # peeks one day ahead - only for this test
    res = run_backtest(prices, cheat, CostModel())
    rt = random_timing_test(prices, res, n=200, seed=1)
    assert rt.p_value < 0.01


def test_random_timing_does_not_flag_noise(prices):
    rng = np.random.default_rng(5)
    noise = pd.Series(rng.integers(0, 2, len(prices)).astype(float), index=prices.index)
    res = run_backtest(prices, noise, CostModel())
    rt = random_timing_test(prices, res, n=200, seed=2)
    assert rt.p_value > 0.05


def test_deflation_grows_with_number_of_trials():
    rng = np.random.default_rng(0)
    few = expected_max_sharpe(rng.normal(0, 0.02, 5))
    many = expected_max_sharpe(rng.normal(0, 0.02, 500))
    assert many > few > 0


def test_probabilistic_sharpe_bounds():
    rng = np.random.default_rng(1)
    good = pd.Series(rng.normal(0.002, 0.01, 2000))
    bad = pd.Series(rng.normal(-0.002, 0.01, 2000))
    assert probabilistic_sharpe(good) > 0.99
    assert probabilistic_sharpe(bad) < 0.01


def test_train_test_tunes_only_on_the_past(prices):
    spec = STRATEGIES["sma_crossover"]
    sp = train_test(prices, spec, test_frac=0.3)
    assert sp.in_sample.returns.index.max() < sp.split_date <= sp.out_of_sample.returns.index.min()
    # The tuning result can't change if we alter prices after the split.
    future_changed = prices.copy()
    future_changed.loc[sp.split_date:, ["open", "high", "low", "close"]] *= 0.5
    assert train_test(future_changed, spec, test_frac=0.3).best_params == sp.best_params


def test_walk_forward_covers_every_day_once(prices):
    wf = walk_forward(prices, STRATEGIES["time_series_momentum"])
    assert wf.returns.index.is_unique
    assert wf.returns.index[0] == prices.index[252 * 5]
    assert wf.returns.index[-1] == prices.index[-1]
    assert len(wf.windows) == 4


def test_analyze_produces_verdict(prices):
    a = analyze(prices, STRATEGIES["rsi_reversion"], n_random=50)
    assert a.verdict and len(a.checks) == 6
    b = analyze(prices, STRATEGIES["buy_and_hold"], n_random=50)
    assert b.checks == [] and b.random_test is None


# -------------------------------------------------------------- paper trading
def test_plan_order_math():
    assert plan_order(1.0, 100_000, 500, 0, allocation=0.95) == (190, "buy", 190)
    assert plan_order(0.0, 100_000, 500, 190) == (0, "sell", 190)
    assert plan_order(1.0, 100_000, 500, 190, allocation=0.95) == (190, None, 0)
    assert plan_order(-1.0, 10_000, 300, 0, allocation=1.0) == (-33, "sell", 33)


def test_broker_refuses_live_money_endpoint():
    with pytest.raises(ValueError):
        AlpacaPaperBroker("k", "s", base_url="https://api.alpaca.markets")


class FakeBroker(SimulatedAccount):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.orders = []

    def submit_market_order(self, symbol, qty, side):
        self.orders.append((symbol, qty, side))
        return {"id": "fake-123"}


def test_run_once_dry_run_sends_nothing(prices, tmp_path):
    broker = FakeBroker(prices, equity=50_000)
    d = run_once(broker, STRATEGIES["buy_and_hold"], "SPY", submit=False, journal=tmp_path / "j.csv")
    assert broker.orders == [] and d.order_side == "buy" and not d.submitted
    assert (tmp_path / "j.csv").read_text().count("\n") == 2  # header + 1 row


def test_run_once_submit(prices, tmp_path):
    broker = FakeBroker(prices, equity=50_000)
    d = run_once(broker, STRATEGIES["buy_and_hold"], "SPY", allocation=1.0, submit=True, journal=None)
    expected = int(50_000 // prices["close"].iloc[-1])
    assert broker.orders == [("SPY", expected, "buy")] and d.order_id == "fake-123"


class FakeHTTP:
    """Minimal requests.Session stand-in to test the Alpaca client offline."""

    def __init__(self):
        self.headers, self.posted = {}, []

    class R:
        def __init__(self, data, code=200):
            self._d, self.status_code = data, code

        def json(self):
            return self._d

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

    def get(self, url, params=None, timeout=None):
        if url.endswith("/v2/account"):
            return self.R({"equity": "100000.5"})
        if "/v2/positions/" in url:
            return self.R({}, 404)
        if "/bars" in url:
            if params.get("page_token"):
                return self.R({"bars": [{"t": "2024-01-03T05:00:00Z", "o": 2, "h": 2, "l": 2, "c": 2, "v": 1}], "next_page_token": None})
            return self.R({"bars": [{"t": "2024-01-02T05:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}], "next_page_token": "p2"})
        raise AssertionError(url)

    def post(self, url, json=None, timeout=None):
        self.posted.append(json)
        return self.R({"id": "abc"})


def test_alpaca_client_parsing_and_pagination():
    http = FakeHTTP()
    b = AlpacaPaperBroker("k", "s", session=http)
    assert http.headers["APCA-API-KEY-ID"] == "k"
    assert b.equity() == 100000.5 and b.position_qty("SPY") == 0
    bars = b.daily_bars("SPY")
    assert bars["close"].tolist() == [1.0, 2.0] and str(bars.index[0].date()) == "2024-01-02"
    assert b.submit_market_order("SPY", 3, "buy")["id"] == "abc"
    assert http.posted[0] == {"symbol": "SPY", "qty": "3", "side": "buy", "type": "market", "time_in_force": "day"}
