"""
analytics.py — read-only portfolio & trade analytics for the v14 paper bot.

Computes every metric purely from the stored state (equity_history, trades,
positions). NEVER makes trading decisions or touches strategy.py. If the data
is too short for a metric, the metric is returned as None so callers can omit
it gracefully.

All timestamps in state are ISO strings; equity_history is [[iso, equity], ...];
each trade is {book, coin, direction, entry_ts, exit_ts, entry_px, exit_px,
pnl, reason, risk_usd}. We derive R multiple, duration, and % move from these.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 52          # weekly sampling for ratio annualisation
RISK_FREE = 0.0                # crypto convention: 0


# ───────────────────────── helpers ─────────────────────────

def equity_series(state) -> pd.Series:
    """Daily-indexed equity series (deduplicated, sorted)."""
    eqh = state.get('equity_history', [])
    if not eqh:
        return pd.Series(dtype=float)
    s = pd.Series([float(e) for _, e in eqh],
                  index=pd.to_datetime([t for t, _ in eqh], utc=True, format='ISO8601'))
    s = s[~s.index.duplicated(keep='last')].sort_index()
    return s


def _dur_hours(t) -> float | None:
    try:
        a = pd.to_datetime(t['entry_ts'], utc=True)
        b = pd.to_datetime(t['exit_ts'], utc=True)
        return (b - a).total_seconds() / 3600
    except Exception:
        return None


def enrich_trade(t: dict) -> dict:
    """Return the trade dict plus derived fields (non-destructive)."""
    d = 1 if t.get('direction') == 'long' else -1
    entry, exit_ = t.get('entry_px'), t.get('exit_px')
    move_pct = (d * (exit_ / entry - 1) * 100) if (entry and exit_) else None
    risk = t.get('risk_usd') or 0
    r_mult = (t['pnl'] / risk) if risk else None
    return dict(t, move_pct=move_pct, r_multiple=r_mult, duration_h=_dur_hours(t))


def trades_df(state) -> pd.DataFrame:
    tr = state.get('trades', [])
    if not tr:
        return pd.DataFrame()
    df = pd.DataFrame([enrich_trade(t) for t in tr])
    df['exit_dt'] = pd.to_datetime(df['exit_ts'], utc=True, errors='coerce')
    df['entry_dt'] = pd.to_datetime(df['entry_ts'], utc=True, errors='coerce')
    return df


# ───────────────────────── portfolio metrics ─────────────────────────

def portfolio_metrics(state) -> dict:
    eq = equity_series(state)
    start = float(state.get('account_start', 10000))
    out = {k: None for k in (
        'equity', 'start', 'total_return', 'realized_pnl', 'unrealized_pnl',
        'daily_return', 'weekly_return', 'monthly_return', 'cagr', 'volatility',
        'sharpe', 'sortino', 'calmar', 'max_dd', 'current_dd', 'avg_dd',
        'ath', 'equity_low', 'recovery_factor', 'avg_daily_return', 'days')}
    out['start'] = start
    if eq.empty:
        out['equity'] = start
        return out

    cur = float(eq.iloc[-1])
    out['equity'] = cur
    out['total_return'] = cur / start - 1
    out['days'] = len(eq)
    out['ath'] = float(eq.cummax().iloc[-1])
    out['equity_low'] = float(eq.min())

    # realized vs unrealized
    tr = state.get('trades', [])
    out['realized_pnl'] = float(sum(t['pnl'] for t in tr))
    unreal = 0.0
    for p in state.get('positions', {}).values():
        px = p.get('last_px', p['entry'])
        unreal += p['dir'] * (px - p['entry']) * p['size'] + p.get('partial_pnl', 0.0)
    out['unrealized_pnl'] = float(unreal)

    # drawdown family
    peak = eq.cummax()
    dd = (eq - peak) / peak
    out['max_dd'] = float(dd.min())
    out['current_dd'] = float(dd.iloc[-1])
    out['avg_dd'] = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0

    # daily returns
    r = eq.pct_change().dropna()
    if len(r):
        out['avg_daily_return'] = float(r.mean())
        out['daily_return'] = float(r.iloc[-1])
        out['volatility'] = float(r.std() * np.sqrt(365))
    # trailing window returns
    out['weekly_return'] = _window_return(eq, 7)
    out['monthly_return'] = _window_return(eq, 30)

    # CAGR + ratios (need enough history)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs > 0 and cur > 0:
        out['cagr'] = (cur / start) ** (1 / yrs) - 1
    w = eq.resample('W').last().ffill().pct_change().dropna()
    if len(w) > 3 and w.std() > 0:
        out['sharpe'] = float((w.mean() - RISK_FREE) / w.std() * np.sqrt(PERIODS_PER_YEAR))
        downside = w[w < 0].std()
        out['sortino'] = float((w.mean() - RISK_FREE) / downside * np.sqrt(PERIODS_PER_YEAR)) if downside and downside > 0 else None
    if out['cagr'] is not None and out['max_dd'] and out['max_dd'] < 0:
        out['calmar'] = out['cagr'] / abs(out['max_dd'])
        out['recovery_factor'] = out['total_return'] / abs(out['max_dd'])
    return out


def _window_return(eq: pd.Series, days: int):
    if eq.empty:
        return None
    cutoff = eq.index[-1] - pd.Timedelta(days=days)
    past = eq[eq.index <= cutoff]
    base = float(past.iloc[-1]) if len(past) else float(eq.iloc[0])
    return float(eq.iloc[-1] / base - 1) if base else None


def today_return(state) -> float | None:
    """PnL over the most recent equity step (proxy for 'today')."""
    eq = equity_series(state)
    if len(eq) < 2:
        return None
    return float(eq.iloc[-1] - eq.iloc[-2])


# ───────────────────────── trading metrics ─────────────────────────

def trading_metrics(state) -> dict:
    df = trades_df(state)
    out = {k: None for k in (
        'total_closed', 'total_open', 'win_rate', 'loss_rate', 'expectancy',
        'avg_win', 'avg_loss', 'avg_r', 'largest_win', 'largest_loss',
        'avg_hold_h', 'profit_factor', 'consec_wins', 'consec_losses',
        'current_streak', 'wins', 'losses')}
    out['total_open'] = len(state.get('positions', {}))
    if df.empty:
        out['total_closed'] = 0
        return out
    out['total_closed'] = len(df)
    wins, losses = df[df.pnl > 0], df[df.pnl < 0]
    out['wins'], out['losses'] = len(wins), len(losses)
    out['win_rate'] = len(wins) / len(df)
    out['loss_rate'] = len(losses) / len(df)
    out['avg_win'] = float(wins.pnl.mean()) if len(wins) else 0.0
    out['avg_loss'] = float(losses.pnl.mean()) if len(losses) else 0.0
    gp, gl = wins.pnl.sum(), abs(losses.pnl.sum())
    out['profit_factor'] = float(gp / gl) if gl > 0 else float('inf')
    out['expectancy'] = float(df.pnl.mean())
    rs = df.r_multiple.dropna()
    out['avg_r'] = float(rs.mean()) if len(rs) else None
    out['largest_win'] = float(df.pnl.max())
    out['largest_loss'] = float(df.pnl.min())
    hold = df.duration_h.dropna()
    out['avg_hold_h'] = float(hold.mean()) if len(hold) else None
    # streaks (chronological)
    seq = (df.sort_values('exit_dt').pnl > 0).astype(int).tolist()
    out['consec_wins'] = _max_run(seq, 1)
    out['consec_losses'] = _max_run(seq, 0)
    out['current_streak'] = _current_streak(seq)
    return out


def _max_run(seq, val):
    best = run = 0
    for x in seq:
        run = run + 1 if x == val else 0
        best = max(best, run)
    return best


def _current_streak(seq):
    if not seq:
        return 0
    last = seq[-1]; n = 0
    for x in reversed(seq):
        if x == last:
            n += 1
        else:
            break
    return n if last == 1 else -n


# ───────────────────────── risk metrics ─────────────────────────

def risk_metrics(state, prices: dict | None = None) -> dict:
    prices = prices or {}
    pos = state.get('positions', {})
    eq = _live_equity(state, prices)
    out = {k: None for k in (
        'exposure_pct', 'cash_pct', 'largest_position_pct', 'avg_position_pct',
        'position_concentration', 'portfolio_risk_pct', 'risk_per_trade_pct',
        'n_positions', 'cash', 'correlation_warning')}
    out['cash'] = float(state.get('cash', 0))
    out['n_positions'] = len(pos)
    out['risk_per_trade_pct'] = float(state.get('base_risk', 0.015) * 2 * 100)
    if eq <= 0:
        return out
    notionals = []
    open_risk = 0.0
    for p in pos.values():
        px = prices.get(p['coin'], p.get('last_px', p['entry']))
        notionals.append(p['size'] * px)
        open_risk += p['size'] * abs(p['entry'] - p['stop'])
    total_notional = sum(notionals)
    out['exposure_pct'] = total_notional / eq * 100
    out['cash_pct'] = out['cash'] / eq * 100
    out['portfolio_risk_pct'] = open_risk / eq * 100
    if notionals:
        out['largest_position_pct'] = max(notionals) / eq * 100
        out['avg_position_pct'] = (total_notional / len(notionals)) / eq * 100
        # Herfindahl concentration of open exposure (0=diverse,1=single)
        w = np.array(notionals) / total_notional if total_notional else np.array([])
        out['position_concentration'] = float((w ** 2).sum()) if len(w) else None
    # correlation warning: crypto is broadly correlated; flag if >=4 same-side
    sides = [p['dir'] for p in pos.values()]
    if len(sides) >= 4 and abs(sum(sides)) >= 4:
        out['correlation_warning'] = (f"{abs(sum(sides))} positions all "
                                      f"{'long' if sum(sides) > 0 else 'short'} — "
                                      f"crypto is highly correlated, real risk is concentrated")
    return out


def _live_equity(state, prices):
    eq = float(state.get('cash', 0))
    for p in state.get('positions', {}).values():
        px = prices.get(p['coin'], p.get('last_px', p['entry']))
        eq += p['dir'] * (px - p['entry']) * p['size']
    return eq


# ───────────────────────── positions & allocation ─────────────────────────

def open_positions(state, prices: dict | None = None) -> list[dict]:
    prices = prices or {}
    rows = []
    for p in state.get('positions', {}).values():
        d = p['dir']
        px = prices.get(p['coin'], p.get('last_px', p['entry']))
        upnl = d * (px - p['entry']) * p['size'] + p.get('partial_pnl', 0.0)
        upct = d * (px / p['entry'] - 1) * 100
        risk = abs(p['entry'] - p['stop']) * p['size0']
        r_now = (d * (px - p['entry']) * p['size']) / risk if risk else None
        rows.append(dict(coin=p['coin'], side='LONG' if d > 0 else 'SHORT',
                         book='A' if p['book'] == 0 else 'B',
                         entry=p['entry'], price=px, stop=p['stop'],
                         size=p['size'], notional=p['size'] * px,
                         upnl=upnl, upct=upct, r_now=r_now,
                         partial=p.get('partial_done', False)))
    rows.sort(key=lambda x: -x['notional'])
    return rows


def allocation(state, prices: dict | None = None) -> dict:
    prices = prices or {}
    eq = _live_equity(state, prices) or float(state.get('account_start', 10000))
    alloc = {}
    for p in state.get('positions', {}).values():
        px = prices.get(p['coin'], p.get('last_px', p['entry']))
        alloc[p['coin']] = alloc.get(p['coin'], 0) + p['size'] * px / eq * 100
    alloc['Cash'] = float(state.get('cash', 0)) / eq * 100
    return alloc


# ───────────────────────── per-coin & leaders ─────────────────────────

def per_coin(state) -> pd.DataFrame:
    df = trades_df(state)
    if df.empty:
        return df
    g = df.groupby('coin').agg(
        trades=('pnl', 'size'), wins=('pnl', lambda s: (s > 0).sum()),
        pnl=('pnl', 'sum'), avg_move=('move_pct', 'mean'),
        avg_r=('r_multiple', 'mean'))
    g['win_rate'] = g['wins'] / g['trades'] * 100
    return g.sort_values('pnl', ascending=False)


def leaders(state, prices: dict | None = None) -> dict:
    out = {}
    df = trades_df(state)
    if not df.empty:
        out['best_realized'] = df.loc[df.pnl.idxmax()].to_dict()
        out['worst_realized'] = df.loc[df.pnl.idxmin()].to_dict()
        pc = per_coin(state)
        out['best_coin'] = (pc.index[0], float(pc.pnl.iloc[0])) if len(pc) else None
        out['worst_coin'] = (pc.index[-1], float(pc.pnl.iloc[-1])) if len(pc) else None
    pos = open_positions(state, prices)
    if pos:
        out['top_unrealized'] = max(pos, key=lambda x: x['upnl'])
        out['bottom_unrealized'] = min(pos, key=lambda x: x['upnl'])
    return out


# ───────────────────────── rolling series (for charts) ─────────────────────────

def drawdown_series(state) -> pd.Series:
    eq = equity_series(state)
    if eq.empty:
        return eq
    return (eq - eq.cummax()) / eq.cummax() * 100


def period_returns(state, freq='ME') -> pd.Series:
    eq = equity_series(state)
    if eq.empty:
        return eq
    return eq.resample(freq).last().ffill().pct_change().dropna() * 100


def rolling_winrate(state, window=10) -> pd.Series:
    df = trades_df(state)
    if df.empty:
        return pd.Series(dtype=float)
    s = (df.sort_values('exit_dt').set_index('exit_dt').pnl > 0).astype(float)
    return s.rolling(window, min_periods=3).mean() * 100


def rank_metrics(state) -> dict:
    """Live conviction-rank monitoring. Uses trades that carry a 'rank' field
    (momentum rank at entry, 0=top). Backfilled trades without a rank are
    ignored, so this populates going forward. R = pnl / risk_usd."""
    tr = [t for t in state.get('trades', []) if t.get('rank') is not None]
    out = dict(tracked=len(tr), by_rank=[], rolling={})
    if not tr:
        return out
    df = pd.DataFrame(tr)
    df['R'] = df['pnl'] / df['risk_usd'].replace(0, np.nan)
    df['rk'] = df['rank'].clip(upper=3)
    df['exit_dt'] = pd.to_datetime(df['exit_ts'], utc=True, errors='coerce')
    tot = df['pnl'].sum()
    cap = df.groupby('rk')['risk_usd'].sum().sum()
    open_by_rank = {}
    for p in state.get('positions', {}).values():
        rk = p.get('rank')
        if rk is None:
            continue
        rk = min(int(rk), 3)
        upnl = p['dir'] * (p.get('last_px', p['entry']) - p['entry']) * p['size'] + p.get('partial_pnl', 0.0)
        open_by_rank[rk] = open_by_rank.get(rk, 0.0) + upnl
    for rk, g in df.groupby('rk'):
        out['by_rank'].append(dict(
            rank=int(rk) + 1,
            trades=len(g),
            win_rate=round((g['pnl'] > 0).mean() * 100, 1),
            avg_R=round(float(g['R'].mean()), 3) if g['R'].notna().any() else None,
            total_pnl=round(float(g['pnl'].sum()), 2),
            pnl_share=round(float(g['pnl'].sum()) / tot * 100, 1) if tot else None,
            open_pnl=round(open_by_rank.get(int(rk), 0.0), 2),
            capital_share=round(float(g['risk_usd'].sum()) / cap * 100, 1) if cap else None))
    for win, lbl in [(90, '3m'), (180, '6m')]:
        cutoff = df['exit_dt'].max() - pd.Timedelta(days=win)
        recent = df[df['exit_dt'] >= cutoff]
        top = recent[recent['rk'] == 0]
        out['rolling'][lbl] = dict(
            top_avg_R=round(float(top['R'].mean()), 3) if len(top) and top['R'].notna().any() else None,
            top_trades=len(top), all_trades=len(recent))
    return out
