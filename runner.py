"""
runner.py — the GitHub Actions daily entry point.

Runs once a day (via .github/workflows/daily.yml):
  1. load state.json (committed in the repo)
  2. fetch market data, run strategy.daily_update()  ← unchanged trading logic
  3. save state.json  (the workflow commits it back)
  4. compute analytics, write web/data.json + web/equity.png for the dashboard
  5. push the daily report to Telegram

Every metric is computed from state — no trading decision is made here.
Env: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, DASHBOARD_URL (all optional for local test).
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

import strategy
import analytics as A
import charts
import telegram

STATE_PATH = os.environ.get('STATE_PATH', 'state.json')
WEB_DIR = os.environ.get('WEB_DIR', 'web')
DASHBOARD_URL = os.environ.get('DASHBOARD_URL', '')


import math

def _clean_json(o):
    """Recursively replace NaN/Infinity with None so JSON is valid for browsers
    (Python json writes bare NaN which JavaScript JSON.parse rejects)."""
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_json(v) for v in o]
    return o


def build_dashboard_data(state, rep=None, prices=None) -> dict:
    """Everything the static dashboard needs, precomputed server-side so the
    browser only has to render (fast, works offline, no API keys client-side)."""
    prices = prices or {p['coin']: p.get('last_px', p['entry'])
                        for p in state.get('positions', {}).values()}
    pm = A.portfolio_metrics(state)
    pm['today_pnl'] = A.today_return(state)
    tm = A.trading_metrics(state)
    rm = A.risk_metrics(state, prices)

    eq = A.equity_series(state)
    dd = A.drawdown_series(state)
    monthly = A.period_returns(state, 'ME')
    weekly = A.period_returns(state, 'W')
    roll_wr = A.rolling_winrate(state, 10)
    pc = A.per_coin(state)

    def ser(s):
        return [[d.strftime('%Y-%m-%d'), round(float(v), 4)]
                for d, v in s.items() if pd.notna(d) and pd.notna(v)] if len(s) else []

    _d = dict(
        updated=datetime.now(timezone.utc).isoformat(),
        account_start=state.get('account_start', 10000),
        portfolio=pm, trading=tm, risk=rm,
        allocation=A.allocation(state, prices),
        positions=A.open_positions(state, prices),
        trades=[A.enrich_trade(t) for t in state.get('trades', [])],
        leaders=A.leaders(state, prices),
        series=dict(equity=ser(eq), drawdown=ser(dd),
                    monthly=ser(monthly), weekly=ser(weekly),
                    rolling_winrate=ser(roll_wr)),
        per_coin=({c: {k: (None if pd.isna(v) else round(float(v), 3))
                       for k, v in row.items()}
                   for c, row in pc.iterrows()} if not pc.empty else {}),
        trends=(rep or {}).get('trends', {}),
        rank=A.rank_metrics(state),
        prices={c: round(prices.get(c), 6) for c in strategy.COINS if c in prices},
    )
    return _clean_json(_d)


def main():
    force = '--force' in sys.argv
    state = strategy.load_state(STATE_PATH)
    today = datetime.now(timezone.utc).date().isoformat()
    already = (state.get('last_update') or '')[:10] == today

    rep, prices = None, {}
    if force or not already:
        data, prices = strategy.fetch_all()
        rep = strategy.daily_update(state, data, prices, pd.Timestamp.now(tz='UTC'))
        strategy.save_state(state, STATE_PATH)
        print(f"[runner] daily_update done — equity {rep['equity']:,.0f}, "
              f"{len(rep['entries'])} entries, {len(rep['closed'])} closed")
    else:
        print('[runner] already updated today — refreshing dashboard only')
        try:
            _, prices = strategy.fetch_all()
        except Exception:
            prices = {}

    # dashboard data + equity chart
    os.makedirs(WEB_DIR, exist_ok=True)
    data = build_dashboard_data(state, rep, prices)
    with open(os.path.join(WEB_DIR, 'data.json'), 'w') as f:
        json.dump(data, f, indent=1, allow_nan=False, default=str)
    print(f"[runner] wrote {WEB_DIR}/data.json ({len(data['trades'])} trades, "
          f"{len(data['series']['equity'])} equity points)")

    png = None
    chart = charts.equity_curve(state)
    if chart:
        png = chart.getvalue()
        with open(os.path.join(WEB_DIR, 'equity.png'), 'wb') as f:
            f.write(png)

    # telegram push (only when there was a real run, or forced)
    if rep is not None:
        metrics = dict(portfolio=data['portfolio'], risk=data['risk'], trading=data['trading'])
        telegram.send_report(state, rep, metrics, png, DASHBOARD_URL)
        print('[runner] telegram report sent')


if __name__ == '__main__':
    main()
