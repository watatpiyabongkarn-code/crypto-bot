"""
backfill.py — seed the paper account with a long history so you can analyse it.

Standalone tool (does NOT change trading logic — it just replays strategy.daily_update
day by day over historical candles). Fetches full daily history from Binance with
pagination (ccxt caps a single call at 1000 candles), then replays.

Usage:
  python backfill.py            # 5 years (default)
  python backfill.py 1095       # 3 years
  python backfill.py 1826 --account 25000

Writes state.json (and refreshes web/data.json + web/equity.png via runner).
"""
from __future__ import annotations
import json
import os
import sys
import time

import pandas as pd

import strategy


def fetch_daily_history(symbol_base: str, days: int):
    """Paginated daily OHLC with exchange fallback (Binance 451-blocks GitHub IPs)."""
    import ccxt
    need = days + 220
    for name in ['binance', 'kucoin', 'okx', 'bybit', 'coinbase']:
        try:
            ex = getattr(ccxt, name)({'enableRateLimit': True})
        except Exception:
            continue
        for quote in ('USDT', 'USD'):
            sym = f'{symbol_base}/{quote}'
            try:
                since = ex.milliseconds() - need * 86_400_000
                rows, cursor, last = [], since, None
                while True:
                    batch = ex.fetch_ohlcv(sym, '1d', since=cursor, limit=1000)
                    if not batch:
                        break
                    rows += batch
                    nl = batch[-1][0]
                    if last is not None and nl <= last:
                        break
                    last = nl
                    cursor = nl + 86_400_000
                    if nl >= ex.milliseconds() - 86_400_000:
                        break
                    time.sleep(ex.rateLimit / 1000)
                if len(rows) >= 100:
                    df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
                    df = df.set_index('ts')[['open', 'high', 'low', 'close']].astype(float)
                    df = df[~df.index.duplicated(keep='last')]
                    print(f'    {symbol_base}: {len(df)} candles from {name}', flush=True)
                    return df
            except Exception:
                continue
    raise RuntimeError(f'no exchange served {symbol_base}')


def run(days: int, account: float, state_path: str = 'state.json',
        csv_dir: str | None = None):
    """csv_dir lets us replay from local CSVs (fast, offline) instead of Binance."""
    raw = {}
    if csv_dir:
        for c in strategy.COINS:
            df = pd.read_csv(os.path.join(csv_dir, f'{c}USDT_1d.csv'))
            df['ts'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            raw[c] = df.set_index('ts')[['open', 'high', 'low', 'close']].astype(float)
    else:
        for c in strategy.COINS:
            print(f'  fetching {c} …', flush=True)
            raw[c] = fetch_daily_history(f'{c}/USDT', days)

    end = min(df.index[-1] for df in raw.values())
    start = end - pd.Timedelta(days=days)
    state = strategy.default_state(account)
    n = 0
    for day in pd.date_range(start, end, freq='D'):
        dslice = {c: df[df.index < day].iloc[-300:] for c, df in raw.items()}
        # only trade coins that already exist; skip a coin whose history hasn't started.
        # 300-day trailing window == full history for this strategy (indicators look
        # back <=40 bars; ATR ewm is fully converged) but keeps replay O(n) not O(n^2).
        active = {c: d for c, d in dslice.items() if len(d) > 90}
        if not active:
            continue
        prices = {}
        for c, df in raw.items():
            nxt = df[df.index >= day]
            prices[c] = float(nxt['open'].iloc[0]) if len(nxt) else float(df['close'].iloc[-1])
        strategy.daily_update(state, active, prices, day + pd.Timedelta(minutes=10))
        n += 1
    strategy.save_state(state, state_path)
    return state, n


if __name__ == '__main__':
    days = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1826
    account = 10000.0
    if '--account' in sys.argv:
        account = float(sys.argv[sys.argv.index('--account') + 1])
    csv = os.environ.get('CSV_DIR')  # set to replay offline from local CSVs
    print(f'Backfilling {days} days (~{days/365:.1f}y), account ${account:,.0f}…')
    state, n = run(days, account, csv_dir=csv)
    s = strategy.compute_stats(state)
    print(f'Done: {n} days replayed · equity ${s["equity"]:,.0f} '
          f'({s["total_return"]*100:+.0f}%) · {s["trades"]} trades · '
          f'{len(state["positions"])} open')
    # refresh dashboard artifacts
    try:
        import runner
        wd=os.environ.get('WEB_DIR','web'); os.makedirs(wd, exist_ok=True)
        data = runner.build_dashboard_data(state)
        json.dump(runner._clean_json(data), open(os.path.join(wd,'data.json'),'w'), indent=1, allow_nan=False, default=str)
        import charts
        buf = charts.equity_curve(state)
        if buf:
            open(os.path.join(wd,'equity.png'), 'wb').write(buf.getvalue())
        print('Refreshed web/data.json + web/equity.png')
    except Exception as e:
        print('(dashboard refresh skipped:', e, ')')
