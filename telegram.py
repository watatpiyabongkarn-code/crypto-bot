"""
telegram.py — push the daily report to Telegram (send-only, no listener needed).

Uses the Bot API over plain HTTPS (urllib, no deps). The GitHub Action calls
send_report() once a day. Reads TELEGRAM_TOKEN and TELEGRAM_CHAT_ID from env
(GitHub Secrets). If either is missing it no-ops so local test runs don't fail.
"""
from __future__ import annotations
import io
import json
import os
import urllib.request

API = 'https://api.telegram.org/bot{token}/{method}'


def _post(method: str, fields: dict, files: dict | None = None):
    token = os.environ.get('TELEGRAM_TOKEN', '')
    if not token:
        print('[telegram] no token — skipping send')
        return None
    url = API.format(token=token, method=method)
    if not files:
        data = json.dumps(fields).encode()
        req = urllib.request.Request(url, data=data,
                                     headers={'Content-Type': 'application/json'})
    else:
        # multipart for photo upload
        boundary = '----v14bot'
        body = io.BytesIO()
        for k, v in fields.items():
            body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        for k, (fname, content) in files.items():
            body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{fname}"\r\n'.encode())
            body.write(b'Content-Type: image/png\r\n\r\n')
            body.write(content)
            body.write(b'\r\n')
        body.write(f'--{boundary}--\r\n'.encode())
        req = urllib.request.Request(url, data=body.getvalue(),
                                     headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f'[telegram] send failed: {e}')
        return None


def send_message(text: str):
    chat = os.environ.get('TELEGRAM_CHAT_ID', '')
    return _post('sendMessage', {'chat_id': chat, 'text': text,
                                 'parse_mode': 'HTML', 'disable_web_page_preview': True})


def send_photo(png: bytes, caption: str = ''):
    chat = os.environ.get('TELEGRAM_CHAT_ID', '')
    return _post('sendPhoto', {'chat_id': chat, 'caption': caption, 'parse_mode': 'HTML'},
                 files={'photo': ('equity.png', png)})


def format_report(state, rep: dict, metrics: dict, dashboard_url: str = '') -> str:
    """Compact HTML message for Telegram (mobile-first). Presentation only —
    every value is read from precomputed analytics; no trading logic here."""
    pm = metrics['portfolio']
    rm = metrics['risk']

    def money(v, sign=False):
        if v is None:
            return 'n/a'
        s = f"${abs(v):,.0f}"
        return (('+' if v >= 0 else '-') + s) if sign else (('-' + s) if v < 0 else s)

    def pct(v, dp=1):
        return 'n/a' if v is None else f"{'+' if v >= 0 else ''}{v*100:.{dp}f}%"

    def row(label, val):
        return f"{label:<11}{val}"

    equity = pm.get('equity')
    expo = rm.get('exposure_pct')
    invested = (equity * expo / 100) if (equity is not None and expo is not None) else None
    npos = rm.get('n_positions', 0) or 0
    dirs = [p['dir'] for p in state.get('positions', {}).values()]
    longs, shorts = sum(1 for d in dirs if d > 0), sum(1 for d in dirs if d < 0)

    port = [
        '💼 PORTFOLIO',
        row('Equity', f"{money(equity)} ({pct(pm.get('total_return'))})"),
        row('Cash', money(rm.get('cash'))),
        row('Invested', f"{money(invested)} ({(expo or 0):.1f}%)" if invested is not None else 'n/a'),
        row('Positions', f"{npos}" + (f" ({longs}L / {shorts}S)" if npos else '')),
        row('Risk Used', f"{rm['portfolio_risk_pct']:.1f}%" if rm.get('portfolio_risk_pct') is not None else 'n/a'),
    ]
    perf = [
        '',
        '📈 PERFORMANCE',
        row('Today', money(pm.get('today_pnl'), sign=True)
            + (f" ({pct(pm.get('daily_return'), 2)})" if pm.get('daily_return') is not None else '')),
        row('Week', pct(pm.get('weekly_return'))),
        row('Month', pct(pm.get('monthly_return'))),
        row('Realized', money(pm.get('realized_pnl'), sign=True)),
        row('Unrealized', money(pm.get('unrealized_pnl'), sign=True)),
    ]
    cdd = pm.get('current_dd')
    if cdd is not None and cdd < -0.001:
        perf.append(row('Drawdown', f"{cdd*100:.1f}% from ATH"))

    date = (state.get('last_update') or '')[:10]
    lines = [f"📅 <b>Daily Report</b>{(' · ' + date) if date else ''}",
             '<pre>' + '\n'.join(port + perf) + '</pre>']

    entries, closed = rep.get('entries', []), rep.get('closed', [])
    if entries or closed:
        if entries:
            lines.append('🚀 <b>Opened</b>')
            for x in entries[:6]:
                lines.append(f"  {x['coin']} {x['dir']} @ {x['px']} · pos {money(x.get('notional'))} · risk {money(x.get('risk'))}")
        if closed:
            lines.append('🏁 <b>Closed</b>')
            for t in closed[:6]:
                lines.append(f"  {t['coin']} {t['direction']} · {money(t['pnl'], sign=True)} ({t['reason']})")
    else:
        lines.append('<i>No trades today — holding.</i>')

    trends = rep.get('trends', {})
    if trends:
        bulls = sum(1 for v in trends.values() if v == 'BULL')
        regime = 'Risk On 🟢' if bulls > len(trends)/2 else ('Risk Off 🔴' if bulls < len(trends)/2 else 'Mixed 🟡')
        lines.append(f"🌐 Market: <b>{regime}</b> ({bulls}/{len(trends)} bull)")
    if dashboard_url:
        lines.append(f"📊 <a href=\"{dashboard_url}\">Open full dashboard →</a>")
    lines.append('<i>v14 paper trading · not financial advice</i>')
    return '\n'.join(lines)


def send_report(state, rep, metrics, equity_png: bytes | None, dashboard_url: str = ''):
    text = format_report(state, rep, metrics, dashboard_url)
    if equity_png:
        # caption limit is 1024 chars; if longer, send photo then text
        if len(text) <= 1000:
            send_photo(equity_png, text)
            return
        send_photo(equity_png, '📅 Daily Report')
    send_message(text)
