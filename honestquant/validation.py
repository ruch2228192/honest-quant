"""The 'honest' part: tests that try hard to prove a strategy is just luck.

1. Train/test split   - tune on the past, judge on data the tuning never saw.
2. Walk-forward       - repeat that every year, stitch the unseen years together.
3. Random timing test - same trades, same time in market, random timing.
                        Does the real timing beat 1,000 random ones?
4. Deflated Sharpe    - corrects the best Sharpe for how many parameter
                        combinations we tried (Bailey & López de Prado, 2014).
5. Cost sensitivity   - how fast does the edge die as trading gets pricier?
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .backtest import BacktestResult, CostModel, run_backtest, run_strategy
from .metrics import TRADING_DAYS, sharpe, summarize
from .strategies import STRATEGIES, StrategySpec

EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- tuning
@dataclass
class Trial:
    params: dict
    score: float
    daily_sharpe: float


def grid_search(
    prices: pd.DataFrame,
    spec: StrategySpec,
    costs: CostModel,
    score_from=None,
    **bt_kwargs,
) -> list[Trial]:
    """Backtest every parameter combination on ``prices``; score on dates >= score_from.

    Returns trials sorted best-first (by annualised Sharpe).
    """
    trials = []
    for params in spec.param_combinations():
        res = run_strategy(prices, spec, params, costs, **bt_kwargs)
        r = res.returns.loc[score_from:] if score_from is not None else res.returns
        sr = sharpe(r)
        sr = sr if np.isfinite(sr) else -np.inf
        trials.append(Trial(params, sr, sr / math.sqrt(TRADING_DAYS)))
    trials.sort(key=lambda t: t.score, reverse=True)
    return trials


# ----------------------------------------------------------------- train / test split
@dataclass
class SplitResult:
    split_date: pd.Timestamp
    best_params: dict
    trials: list[Trial]
    in_sample: BacktestResult
    out_of_sample: BacktestResult
    benchmark_oos: BacktestResult
    full: BacktestResult


def train_test(prices, spec, costs=CostModel(), test_frac=0.3, **bt_kwargs) -> SplitResult:
    split_i = int(len(prices) * (1 - test_frac))
    split_date = prices.index[split_i]
    train = prices.iloc[:split_i]

    trials = grid_search(train, spec, costs, **bt_kwargs)
    best = trials[0].params

    # Signals only ever look backwards (see tests/test_no_lookahead.py), so we can
    # run once on the full history and slice - indicators get their warm-up.
    full = run_strategy(prices, spec, best, costs, **bt_kwargs)
    bench = run_strategy(prices, STRATEGIES["buy_and_hold"], None, costs)
    return SplitResult(
        split_date=split_date,
        best_params=best,
        trials=trials,
        in_sample=full.window(None, prices.index[split_i - 1]),
        out_of_sample=full.window(split_date, None),
        benchmark_oos=bench.window(split_date, None),
        full=full,
    )


# ----------------------------------------------------------------------- walk-forward
@dataclass
class WalkForwardResult:
    returns: pd.Series               # stitched out-of-sample daily returns
    positions: pd.Series
    benchmark: pd.Series             # buy & hold over the same dates
    windows: list[dict] = field(default_factory=list)

    def metrics(self) -> dict:
        return summarize(self.returns, self.positions)


def walk_forward(
    prices, spec, costs=CostModel(), train_days=TRADING_DAYS * 5, test_days=TRADING_DAYS, **bt_kwargs
) -> WalkForwardResult:
    """Re-tune every ``test_days`` on the previous ``train_days``; trade the next block blind."""
    bench_full = run_strategy(prices, STRATEGIES["buy_and_hold"], None, costs)
    rets, poss, windows = [], [], []
    start = train_days
    n = len(prices)
    while start < n:
        end = min(start + test_days, n)
        history = prices.iloc[:start]
        score_from = prices.index[start - train_days]
        trials = grid_search(history, spec, costs, score_from=score_from, **bt_kwargs)
        best = trials[0]
        res = run_strategy(prices.iloc[:end], spec, best.params, costs, **bt_kwargs)
        oos = res.window(prices.index[start], prices.index[end - 1])
        rets.append(oos.returns)
        poss.append(oos.positions)
        windows.append({
            "test_start": prices.index[start].date().isoformat(),
            "test_end": prices.index[end - 1].date().isoformat(),
            "params": best.params,
            "in_sample_sharpe": best.score,
            "out_of_sample_sharpe": sharpe(oos.returns),
        })
        start = end
    returns = pd.concat(rets) if rets else pd.Series(dtype=float)
    positions = pd.concat(poss) if poss else pd.Series(dtype=float)
    return WalkForwardResult(returns, positions, bench_full.returns.reindex(returns.index), windows)


# ------------------------------------------------------------------ random timing test
@dataclass
class RandomTestResult:
    actual_sharpe: float
    null_sharpes: np.ndarray
    p_value: float


def random_timing_test(
    prices: pd.DataFrame,
    result: BacktestResult,
    costs: CostModel = CostModel(),
    start=None,
    end=None,
    n: int = 500,
    seed: int = 0,
) -> RandomTestResult:
    """Circularly shift the strategy's signal by a random offset, many times.

    A shifted signal has exactly the same number of trades, holding periods
    and time in the market - only *when* it trades is random. If the real
    strategy's Sharpe isn't in the top few % of those, its timing skill is
    indistinguishable from luck.
    """
    rng = np.random.default_rng(seed)
    sig = result.signal.reindex(prices.index).fillna(0.0).to_numpy()
    actual = sharpe(result.returns.loc[start:end])
    m = len(sig)
    lo, hi = min(TRADING_DAYS, m // 4), max(m - TRADING_DAYS, m // 4 + 1)
    null = np.empty(n)
    for i in range(n):
        shifted = pd.Series(np.roll(sig, int(rng.integers(lo, hi))), index=prices.index)
        r = run_backtest(prices, shifted, costs).returns.loc[start:end]
        null[i] = sharpe(r)
    null = null[np.isfinite(null)]
    p = (1 + np.sum(null >= actual)) / (1 + len(null)) if len(null) else float("nan")
    return RandomTestResult(actual, null, float(p))


# --------------------------------------------------------------------- deflated sharpe
def probabilistic_sharpe(returns: pd.Series, benchmark_daily_sr: float = 0.0) -> float:
    """P(true Sharpe > benchmark), allowing for fat tails and skew (Bailey & López de Prado)."""
    r = returns.dropna().to_numpy()
    if len(r) < 3 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - benchmark_daily_sr) * math.sqrt(len(r) - 1) / math.sqrt(denom)))


def expected_max_sharpe(trial_daily_srs) -> float:
    """The best daily Sharpe you'd expect from N tries if NONE had real skill."""
    srs = np.asarray([s for s in trial_daily_srs if np.isfinite(s)])
    n = len(srs)
    if n < 2:
        return 0.0
    sd = srs.std(ddof=1)
    return float(sd * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n)
                       + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * math.e))))


def deflated_sharpe(returns: pd.Series, trial_daily_srs) -> float:
    return probabilistic_sharpe(returns, expected_max_sharpe(trial_daily_srs))


# -------------------------------------------------------------------- cost sensitivity
def cost_sensitivity(prices, spec, params, start=None, end=None, bps_levels=(0, 2, 5, 10, 20, 50), **bt_kwargs):
    rows = []
    for bps in bps_levels:
        res = run_strategy(prices, spec, params, CostModel(0.0, float(bps)), **bt_kwargs).window(start, end)
        m = summarize(res.returns)
        rows.append({"cost_bps": bps, "sharpe": m["sharpe"], "cagr": m["cagr"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ full report
@dataclass
class Analysis:
    spec: StrategySpec
    ticker: str
    split: SplitResult
    walk: WalkForwardResult
    random_test: RandomTestResult | None
    dsr: float
    psr_oos: float
    costs_table: pd.DataFrame
    checks: list[dict]
    verdict: str


def analyze(
    prices: pd.DataFrame,
    spec: StrategySpec,
    ticker: str = "",
    costs: CostModel = CostModel(),
    test_frac: float = 0.3,
    n_random: int = 500,
    seed: int = 0,
    **bt_kwargs,
) -> Analysis:
    split = train_test(prices, spec, costs, test_frac, **bt_kwargs)
    walk = walk_forward(prices, spec, costs, **bt_kwargs)
    oos = split.out_of_sample
    is_trading_strategy = spec.name != "buy_and_hold"

    rt = None
    if is_trading_strategy:
        rt = random_timing_test(prices, split.full, costs, start=split.split_date, n=n_random, seed=seed)
    # Deflate the *edge over buy & hold* (not raw Sharpe): a long-only strategy in a
    # rising market looks "significant" vs zero without any skill at all.
    bench_full = run_strategy(prices, STRATEGIES["buy_and_hold"], None, costs)
    in_idx = split.in_sample.returns.index
    trial_edges = []
    for t in split.trials:
        r = run_strategy(prices.loc[: in_idx[-1]], spec, t.params, costs, **bt_kwargs).returns
        edge = (r - bench_full.returns.reindex(r.index)).dropna()
        trial_edges.append(edge.mean() / edge.std(ddof=1) if edge.std(ddof=1) > 0 else np.nan)
    best_edge = (split.in_sample.returns - bench_full.returns.reindex(in_idx)).dropna()
    dsr = deflated_sharpe(best_edge, trial_edges) if is_trading_strategy else float("nan")
    psr_oos = probabilistic_sharpe(oos.returns)
    ct = cost_sensitivity(prices, spec, split.best_params, start=split.split_date, **bt_kwargs)

    is_m, oos_m = split.in_sample.metrics(), oos.metrics()
    bench_m = split.benchmark_oos.metrics()
    wf_m, wf_b = walk.metrics(), summarize(walk.benchmark)

    checks = []
    def check(name, passed, detail):
        if is_trading_strategy:  # the benchmark isn't graded against itself
            checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("Beats buy & hold on unseen data (train/test)",
          oos_m["sharpe"] > bench_m["sharpe"],
          f"Sharpe {oos_m['sharpe']:.2f} vs buy & hold {bench_m['sharpe']:.2f}")
    check("Beats buy & hold in walk-forward",
          wf_m["sharpe"] > wf_b["sharpe"],
          f"Sharpe {wf_m['sharpe']:.2f} vs buy & hold {wf_b['sharpe']:.2f}")
    decay = oos_m["sharpe"] / is_m["sharpe"] if is_m["sharpe"] and is_m["sharpe"] > 0 else float("nan")
    check("Keeps at least half its Sharpe out of sample",
          np.isfinite(decay) and decay >= 0.5,
          f"in-sample {is_m['sharpe']:.2f} → out-of-sample {oos_m['sharpe']:.2f}")
    if rt is not None:
        check("Timing beats random timing (p < 0.05)", rt.p_value < 0.05,
              f"p = {rt.p_value:.3f} vs {len(rt.null_sharpes)} random versions")
    check("Edge over buy & hold survives the Deflated Sharpe test (> 95%)",
          np.isfinite(dsr) and dsr > 0.95,
          f"{dsr * 100:.0f}% confidence after trying {len(split.trials)} parameter sets")
    edge_at_10 = ct.loc[ct.cost_bps == 10, "sharpe"].iloc[0]
    check("Still beats buy & hold at 10 bps per trade",
          edge_at_10 > bench_m["sharpe"],
          f"Sharpe at 10 bps: {edge_at_10:.2f}")

    passed = sum(c["passed"] for c in checks)
    if not is_trading_strategy:
        verdict = "Benchmark. This is the bar every other strategy has to clear."
    elif passed == len(checks):
        verdict = "Passed every check. Rare - now paper-trade it for months before trusting it."
    elif passed >= len(checks) - 2:
        verdict = "Promising but not proven. Some evidence of an edge, not enough to bet real money."
    else:
        verdict = "Most likely luck or overfitting. The edge doesn't survive honest testing."

    return Analysis(spec, ticker, split, walk, rt, dsr, psr_oos, ct, checks, verdict)
