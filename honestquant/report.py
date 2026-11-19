"""Self-contained HTML report (charts embedded as images, no internet needed)."""
from __future__ import annotations

import base64
import html
import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import PRETTY, drawdown_series, equity_curve, fmt, summarize  # noqa: E402
from .validation import Analysis  # noqa: E402

STRAT = "#2a78d6"      # categorical slot 1 (blue) - the strategy
BENCH = "#8a8984"      # neutral gray - buy & hold reference
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _end_labels(ax, items):
    """Direct labels at line ends, nudged apart so they never overlap."""
    items = sorted(items, key=lambda it: it[0].iloc[-1])
    lo, hi = ax.get_ylim()
    log = ax.get_yscale() == "log"
    pos = lambda v: (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) if log else (v - lo) / (hi - lo)
    fracs = [pos(it[0].iloc[-1]) for it in items]
    gap = 0.06
    for i in range(1, len(fracs)):
        fracs[i] = max(fracs[i], fracs[i - 1] + gap)
    for (series, text, bold), y in zip(items, fracs):
        ax.annotate(text, xy=(series.index[-1], series.iloc[-1]), xycoords="data",
                    xytext=(1.01, y), textcoords=("axes fraction"), va="center", fontsize=9,
                    color=INK if bold else INK2, fontweight="bold" if bold else "normal")


def chart_equity(strategy_r: pd.Series, bench_r: pd.Series, title: str) -> str:
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    s, b = equity_curve(strategy_r) * 10_000, equity_curve(bench_r) * 10_000
    ax.plot(b.index, b, color=BENCH, lw=1.6, label="Buy & hold")
    ax.plot(s.index, s, color=STRAT, lw=2, label="Strategy")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    _end_labels(ax, [(s, f"Strategy ${s.iloc[-1]:,.0f}", True), (b, f"Buy & hold ${b.iloc[-1]:,.0f}", False)])
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylabel("$10,000 grows to… (log scale)")

    ds, db = drawdown_series(strategy_r) * 100, drawdown_series(bench_r) * 100
    ax2.fill_between(db.index, db, 0, color=BENCH, alpha=0.25, lw=0)
    ax2.plot(ds.index, ds, color=STRAT, lw=1.4)
    ax2.set_ylabel("Drawdown %")
    ax2.set_title("Drawdown (how far below the last peak)", fontsize=9.5)
    fig.tight_layout()
    return _png(fig)


def chart_is_vs_oos(a: Analysis) -> str:
    sp = a.split
    labels = ["Strategy\nin-sample (tuned)", "Strategy\nout-of-sample", "Buy & hold\nout-of-sample"]
    vals = [sp.in_sample.metrics()["sharpe"], sp.out_of_sample.metrics()["sharpe"], sp.benchmark_oos.metrics()["sharpe"]]
    colors = [STRAT, STRAT, BENCH]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor=SURFACE, linewidth=2)
    bars[0].set_alpha(0.45)
    ax.axhline(0, color=INK2, lw=0.8)
    for rect, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}", (rect.get_x() + rect.get_width() / 2, v), xytext=(0, 3 if v >= 0 else -12),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_title("Sharpe: tuned vs. unseen data")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _png(fig)


def chart_random(a: Analysis) -> str | None:
    rt = a.random_test
    if rt is None or len(rt.null_sharpes) == 0:
        return None
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.hist(rt.null_sharpes, bins=30, color=BENCH, alpha=0.6, edgecolor=SURFACE, linewidth=1)
    ax.axvline(rt.actual_sharpe, color=STRAT, lw=2)
    ax.annotate(f"Real strategy\n{rt.actual_sharpe:.2f}  (p = {rt.p_value:.2f})", (rt.actual_sharpe, ax.get_ylim()[1] * 0.92),
                xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=INK, va="top")
    ax.set_title("Real timing vs. random timing")
    ax.set_xlabel("Out-of-sample Sharpe of randomly shifted trades")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _png(fig)


def chart_costs(a: Analysis) -> str:
    ct = a.costs_table
    bench = a.split.benchmark_oos.metrics()["sharpe"]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.axhline(bench, color=BENCH, lw=1.6, ls="--")
    ax.annotate("Buy & hold", (ct.cost_bps.iloc[-1], bench), xytext=(-4, 4), textcoords="offset points",
                ha="right", fontsize=8.5, color=INK2)
    ax.plot(ct.cost_bps, ct.sharpe, color=STRAT, lw=2, marker="o", ms=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xlabel("Cost per trade (basis points, 1 bp = 0.01%)")
    ax.set_ylabel("Out-of-sample Sharpe")
    ax.set_title("How trading costs eat the edge")
    fig.tight_layout()
    return _png(fig)


# --------------------------------------------------------------------------- HTML
CSS = f"""
:root {{ --ink:{INK}; --ink2:{INK2}; --grid:{GRID}; --surface:{SURFACE}; --blue:{STRAT};
        --good:#0a7d33; --bad:#c2352f; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:#f3f2ef; color:var(--ink); font:15px/1.55 -apple-system,Segoe UI,Inter,Roboto,sans-serif; }}
main {{ max-width:1040px; margin:0 auto; padding:40px 24px 80px; }}
h1 {{ font-size:30px; margin:0 0 4px; letter-spacing:-.02em; }}
h2 {{ font-size:21px; margin:48px 0 6px; letter-spacing:-.01em; }}
h3 {{ font-size:15px; margin:24px 0 8px; }}
.sub {{ color:var(--ink2); margin:0 0 20px; }}
.card {{ background:var(--surface); border:1px solid var(--grid); border-radius:12px; padding:20px; margin:16px 0; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
@media (max-width:760px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
img {{ width:100%; height:auto; display:block; }}
table {{ width:100%; border-collapse:collapse; font-size:13.5px; font-variant-numeric:tabular-nums; }}
th,td {{ text-align:right; padding:7px 10px; border-bottom:1px solid var(--grid); }}
th:first-child,td:first-child {{ text-align:left; }}
th {{ color:var(--ink2); font-weight:600; }}
.verdict {{ font-size:17px; font-weight:600; }}
.pill {{ display:inline-block; padding:2px 10px; border-radius:99px; font-size:12.5px; font-weight:600; }}
.pass {{ color:var(--good); }} .fail {{ color:var(--bad); }}
.checks li {{ list-style:none; margin:6px 0; }} .checks {{ padding:0; }}
.muted {{ color:var(--ink2); font-size:13px; }}
.warn {{ background:#fff6e0; border-color:#f0d48a; }}
details summary {{ cursor:pointer; color:var(--ink2); }}
code {{ background:#eeede9; padding:1px 5px; border-radius:4px; font-size:13px; }}
"""


def _metrics_table(cols: dict[str, dict]) -> str:
    keys = ["total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown", "exposure", "trades"]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    rows = []
    for k in keys:
        label, kind = PRETTY[k]
        cells = "".join(f"<td>{fmt(m.get(k), kind)}</td>" for m in cols.values())
        rows.append(f"<tr><td>{label}</td>{cells}</tr>")
    return f"<table><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _checks(a: Analysis) -> str:
    items = []
    for c in a.checks:
        mark = '<span class="pass">✔ PASS</span>' if c["passed"] else '<span class="fail">✘ FAIL</span>'
        items.append(f"<li>{mark} &nbsp;<b>{html.escape(c['name'])}</b> <span class='muted'>— {html.escape(c['detail'])}</span></li>")
    return f"<ul class='checks'>{''.join(items)}</ul>"


def _section(a: Analysis) -> str:
    sp, wf = a.split, a.walk
    params = ", ".join(f"{k}={v}" for k, v in sp.best_params.items()) or "none"
    wf_m = wf.metrics()
    wf_b = summarize(wf.benchmark)
    out = [f"<h2 id='{a.spec.name}'>{a.spec.name.replace('_', ' ').title()}</h2>",
           f"<p class='sub'>{html.escape(a.spec.description)}</p>"]
    if a.checks:
        passed = sum(c["passed"] for c in a.checks)
        out.append(f"<div class='card'><div class='verdict'>{passed}/{len(a.checks)} honesty checks passed — "
                   f"{html.escape(a.verdict)}</div>{_checks(a)}</div>")

    out.append("<div class='card'>" + chart_equity(wf.returns, wf.benchmark,
               "Walk-forward: re-tuned every year on the past 5 years, then traded blind").join(["<img src='", "' alt='equity curve'>"]))
    out.append(_metrics_table({"Strategy (walk-forward)": wf_m, "Buy & hold (same dates)": wf_b}) + "</div>")

    imgs = [chart_is_vs_oos(a), chart_costs(a)]
    rnd = chart_random(a)
    if rnd:
        imgs.append(rnd)
    cells = "".join(f"<div class='card'><img src='{i}' alt='chart'></div>" for i in imgs)
    out.append(f"<div class='grid2'>{cells}</div>")

    out.append(f"<div class='card'><h3>Train/test split</h3><p class='muted'>Tuned on data before "
               f"{sp.split_date.date()} ({len(sp.trials)} parameter sets tried). Best: <code>{html.escape(params)}</code>. "
               f"Then judged only on data after that date.</p>"
               + _metrics_table({"In-sample (tuned)": sp.in_sample.metrics(),
                                 "Out-of-sample": sp.out_of_sample.metrics(),
                                 "Buy & hold (out-of-sample)": sp.benchmark_oos.metrics()}) + "</div>")

    rows = "".join(
        f"<tr><td>{w['test_start']} → {w['test_end']}</td><td>{html.escape(', '.join(f'{k}={v}' for k, v in w['params'].items()) or '–')}</td>"
        f"<td>{fmt(w['in_sample_sharpe'], 'num')}</td><td>{fmt(w['out_of_sample_sharpe'], 'num')}</td></tr>"
        for w in wf.windows)
    out.append("<div class='card'><details><summary>Walk-forward windows (what the bot would have picked each year)</summary>"
               "<table><thead><tr><th>Traded period</th><th>Parameters chosen</th><th>Sharpe while tuning</th>"
               f"<th>Sharpe when traded</th></tr></thead><tbody>{rows}</tbody></table></details></div>")
    return "\n".join(out)


def build_report(analyses: list[Analysis], ticker: str, data_note: str, settings: dict) -> str:
    lb_rows = []
    ranked = sorted(analyses, key=lambda a: a.walk.metrics()["sharpe"], reverse=True)
    for a in ranked:
        m = a.walk.metrics()
        checks = f"{sum(c['passed'] for c in a.checks)}/{len(a.checks)}" if a.checks else "benchmark"
        lb_rows.append(
            f"<tr><td>{a.spec.name.replace('_', ' ') if not a.checks else f'<a href=#{a.spec.name}>' + a.spec.name.replace('_', ' ') + '</a>'}</td>"
            f"<td>{fmt(m['cagr'], 'pct')}</td><td>{fmt(m['sharpe'], 'num')}</td><td>{fmt(m['max_drawdown'], 'pct')}</td>"
            f"<td>{fmt(m.get('exposure'), 'pct')}</td><td>{checks}</td></tr>")
    leaderboard = ("<div class='card'><h3>Leaderboard — walk-forward (out-of-sample only)</h3><table><thead><tr>"
                   "<th>Strategy</th><th>CAGR</th><th>Sharpe</th><th>Max drawdown</th><th>Time in market</th>"
                   f"<th>Checks passed</th></tr></thead><tbody>{''.join(lb_rows)}</tbody></table></div>")
    settings_txt = " · ".join(f"{k}: {v}" for k, v in settings.items())
    warn = ""
    if "synthetic" in data_note.lower():
        warn = ("<div class='card warn'><b>Synthetic data.</b> These prices were generated by a random market model "
                "for offline testing — not real history. Run with a real ticker to get real results.</div>")
    # buy & hold only appears in the leaderboard and as the gray line in every chart
    sections = "\n".join(_section(a) for a in ranked if a.spec.name != "buy_and_hold")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Honest backtest — {html.escape(ticker)}</title><style>{CSS}</style></head>
<body><main>
<h1>Honest backtest: {html.escape(ticker)}</h1>
<p class="sub">{html.escape(data_note)} · generated {datetime.now():%Y-%m-%d %H:%M}<br>
<span class="muted">{html.escape(settings_txt)}</span></p>
{warn}
<div class="card"><b>How to read this.</b> Most backtests lie: they tune a strategy on the same data they grade it on,
trade at prices you couldn't actually get, and ignore costs. This report trades at the <i>next day's open</i>,
charges commission and slippage on every trade, and judges each strategy only on data it was never tuned on.
Then it asks whether the result could just be luck.</div>
{leaderboard}
{sections}
<p class="muted" style="margin-top:48px">Educational project. Past performance — especially simulated — says little about the future. Not investment advice.</p>
</main></body></html>"""
