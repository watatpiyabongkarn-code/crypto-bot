"""
charts.py — visual analytics for the v14 paper bot.

Every function reads state (via analytics.py) and returns a PNG as a BytesIO,
ready to hand to discord.File. No trading logic here. A shared dark theme keeps
all charts consistent and mobile-legible. Charts that lack enough data return
None so the caller can skip them.
"""
from __future__ import annotations
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analytics as A

# ── shared theme ──
BG, FG, GRID = '#0d1117', '#c9d1d9', '#30363d'
GREEN, RED, BLUE, AMBER = '#2ea043', '#f85149', '#388bfd', '#d29922'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
    'text.color': FG, 'axes.labelcolor': FG, 'xtick.color': FG, 'ytick.color': FG,
    'axes.edgecolor': GRID, 'grid.color': GRID, 'font.size': 10,
})


def _save(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf


def _empty_ok(series) -> bool:
    return series is None or len(series) < 2


# ───────────────────────── individual charts ─────────────────────────

def equity_curve(state):
    eq = A.equity_series(state)
    if _empty_ok(eq):
        return None
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(eq.index, eq.values, color=GREEN, lw=1.8)
    ax.fill_between(eq.index, eq.values, eq.min(), color=GREEN, alpha=0.10)
    ax.axhline(state.get('account_start', 10000), color=GRID, ls=':', lw=1)
    ax.set_title('Equity Curve', color=FG, fontweight='bold')
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    return _save(fig)


def drawdown(state):
    dd = A.drawdown_series(state)
    if _empty_ok(dd):
        return None
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.35)
    ax.plot(dd.index, dd.values, color=RED, lw=1.2)
    ax.set_title('Drawdown (%)', color=FG, fontweight='bold')
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    return _save(fig)


def returns_bars(state, freq='M', title='Monthly Returns (%)'):
    r = A.period_returns(state, freq)
    if _empty_ok(r):
        return None
    fig, ax = plt.subplots(figsize=(9, 3.6))
    colors = [GREEN if v >= 0 else RED for v in r.values]
    labels = [d.strftime('%b %d' if freq in ('D', 'W') else '%b %y') for d in r.index]
    ax.bar(range(len(r)), r.values, color=colors)
    ax.set_xticks(range(len(r)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_title(title, color=FG, fontweight='bold')
    ax.grid(alpha=0.25, axis='y')
    return _save(fig)


def allocation_pie(state, prices=None):
    alloc = {k: v for k, v in A.allocation(state, prices).items() if v > 0.5}
    if len(alloc) < 1:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    palette = [BLUE, GREEN, AMBER, RED, '#a371f7', '#f778ba', '#3fb950', '#db6d28', '#8b949e']
    ks = list(alloc.keys())
    colors = ['#8b949e' if k == 'Cash' else palette[i % len(palette)] for i, k in enumerate(ks)]
    wedges, _txt, autotxt = ax.pie(alloc.values(), labels=ks, colors=colors,
                                   autopct='%1.0f%%', startangle=90,
                                   textprops={'color': FG, 'fontsize': 9},
                                   wedgeprops={'edgecolor': BG, 'linewidth': 1.5})
    for a in autotxt:
        a.set_color('#ffffff'); a.set_fontsize(8)
    ax.set_title('Portfolio Allocation', color=FG, fontweight='bold')
    return _save(fig)


def pnl_histogram(state):
    df = A.trades_df(state)
    if df.empty or len(df) < 4:
        return None
    fig, ax = plt.subplots(figsize=(9, 3.6))
    vals = df.pnl.values
    bins = min(20, max(6, len(vals) // 2))
    n, edges, patches = ax.hist(vals, bins=bins, edgecolor=BG)
    for patch, left in zip(patches, edges[:-1]):
        patch.set_facecolor(GREEN if left >= 0 else RED)
    ax.axvline(0, color=FG, lw=1)
    ax.set_title('PnL Distribution ($ per trade)', color=FG, fontweight='bold')
    ax.grid(alpha=0.25, axis='y')
    return _save(fig)


def profit_by_coin(state):
    pc = A.per_coin(state)
    if pc.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 3.6))
    colors = [GREEN if v >= 0 else RED for v in pc.pnl.values]
    ax.bar(pc.index, pc.pnl.values, color=colors)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_title('Realized Profit by Coin ($)', color=FG, fontweight='bold')
    ax.grid(alpha=0.25, axis='y')
    return _save(fig)


def trades_per_coin(state):
    pc = A.per_coin(state)
    if pc.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(pc.index, pc.trades.values, color=BLUE)
    ax.set_title('Trades per Coin', color=FG, fontweight='bold')
    ax.grid(alpha=0.25, axis='y')
    return _save(fig)


def winrate_by_coin(state):
    pc = A.per_coin(state)
    if pc.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 3.4))
    colors = [GREEN if v >= 50 else (AMBER if v >= 33 else RED) for v in pc.win_rate.values]
    ax.bar(pc.index, pc.win_rate.values, color=colors)
    ax.axhline(50, color=GRID, ls=':', lw=1)
    ax.set_ylim(0, 100)
    ax.set_title('Win Rate by Coin (%)', color=FG, fontweight='bold')
    ax.grid(alpha=0.25, axis='y')
    return _save(fig)


def rolling_winrate(state, window=10):
    s = A.rolling_winrate(state, window)
    if _empty_ok(s):
        return None
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(range(len(s)), s.values, color=BLUE, lw=1.6)
    ax.axhline(50, color=GRID, ls=':', lw=1)
    ax.set_ylim(0, 100)
    ax.set_title(f'Rolling Win Rate (last {window} trades, %)', color=FG, fontweight='bold')
    ax.grid(alpha=0.25)
    return _save(fig)


# ───────────────────────── multi-panel dashboard image ─────────────────────────

def dashboard_grid(state, prices=None):
    """One 2x2 image summarising equity, drawdown, allocation, per-coin pnl —
    handy for a single-attachment overview in Discord."""
    eq = A.equity_series(state)
    if _empty_ok(eq):
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    (a1, a2), (a3, a4) = axes

    a1.plot(eq.index, eq.values, color=GREEN, lw=1.6)
    a1.fill_between(eq.index, eq.values, eq.min(), color=GREEN, alpha=0.10)
    a1.axhline(state.get('account_start', 10000), color=GRID, ls=':', lw=1)
    a1.set_title('Equity', fontweight='bold'); a1.grid(alpha=0.2)

    dd = A.drawdown_series(state)
    a2.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.35)
    a2.set_title('Drawdown %', fontweight='bold'); a2.grid(alpha=0.2)

    alloc = {k: v for k, v in A.allocation(state, prices).items() if v > 0.5}
    palette = [BLUE, GREEN, AMBER, RED, '#a371f7', '#f778ba', '#3fb950', '#db6d28']
    ks = list(alloc.keys())
    colors = ['#8b949e' if k == 'Cash' else palette[i % len(palette)] for i, k in enumerate(ks)]
    a3.pie(alloc.values(), labels=ks, colors=colors, autopct='%1.0f%%', startangle=90,
           textprops={'color': FG, 'fontsize': 8}, wedgeprops={'edgecolor': BG, 'linewidth': 1})
    a3.set_title('Allocation', fontweight='bold')

    pc = A.per_coin(state)
    if not pc.empty:
        cols = [GREEN if v >= 0 else RED for v in pc.pnl.values]
        a4.bar(pc.index, pc.pnl.values, color=cols)
    a4.axhline(0, color=GRID, lw=1)
    a4.set_title('Profit by Coin $', fontweight='bold'); a4.grid(alpha=0.2, axis='y')
    for ax in (a1, a2):
        for lab in ax.get_xticklabels():
            lab.set_rotation(30); lab.set_fontsize(7)
    return _save(fig)
