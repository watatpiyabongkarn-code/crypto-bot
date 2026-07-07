"""v14 paper-trading strategy module (trading logic — DO NOT MODIFY for analytics work)."""
from __future__ import annotations
import json, os, tempfile
import numpy as np, pandas as pd

COINS = ['BTC','ETH','SOL','LTC','ADA','DOGE','AVAX','DOT']
DONCHIAN, ATR_P, MOM_LB, SWING_LB = 20, 14, 30, 5
TRAIL_MULT, PARTIAL_ATR, PARTIAL_SIZE = 3.5, 2.0, 0.5
TAKER_FEE, SLIP, FUNDING_8H = 0.0005, 0.0005, 0.0001
MAX_HEAT, MAX_CONCURRENT = 0.12, 5
MAX_LEV_POS, MAX_LEV_PORT = 1.0, 2.0
DD_START, DD_FULL, DD_LOOKBACK_D = 0.10, 0.25, 180

def default_state(account=10_000.0):
    return dict(version=14, account_start=account, cash=account, base_risk=0.015,
                paused=False, positions={}, equity_history=[], trades=[], last_update=None)

def load_state(path):
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return default_state(float(os.environ.get('ACCOUNT_SIZE',10000)))

def save_state(state, path):
    d=os.path.dirname(os.path.abspath(path)) or '.'; fd,tmp=tempfile.mkstemp(dir=d,suffix='.tmp')
    with os.fdopen(fd,'w') as f: json.dump(state,f,indent=1,default=str)
    os.replace(tmp,path)

def build_2d(daily, offset):
    agg={'open':'first','high':'max','low':'min','close':'last'}
    origin=pd.Timestamp('1970-01-01',tz='UTC')+pd.Timedelta(days=offset)
    return daily.resample('2D',origin=origin).agg(agg).dropna()

def analyse(d2):
    if len(d2)<DONCHIAN+2: return None
    h,l,c=d2['high'],d2['low'],d2['close']; pc=c.shift(1)
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/ATR_P,min_periods=ATR_P,adjust=False).mean()
    dh,dl=h.rolling(DONCHIAN).max(),l.rolling(DONCHIAN).min(); l2=c>(dh+dl)/2
    mom=c.pct_change(MOM_LB)/(atr/c); i=-1; a=float(atr.iloc[i])
    if not np.isfinite(a) or a<=0: return None
    return dict(close=float(c.iloc[i]),atr=a,bull=bool(l2.iloc[i]),
        flip=bool(l2.iloc[i]!=l2.iloc[i-1]) if len(l2)>1 else False,
        brk_up=bool(c.iloc[i]>c.rolling(DONCHIAN).max().shift(1).iloc[i]),
        brk_dn=bool(c.iloc[i]<c.rolling(DONCHIAN).min().shift(1).iloc[i]),
        in_up10=bool(c.iloc[i]>c.rolling(10).max().shift(1).iloc[i]),
        in_dn10=bool(c.iloc[i]<c.rolling(10).min().shift(1).iloc[i]),
        swing_lo=float(l.iloc[-SWING_LB:].min()),swing_hi=float(h.iloc[-SWING_LB:].max()),
        mom=float(mom.iloc[i]) if np.isfinite(mom.iloc[i]) else 0.0,
        bar_high=float(h.iloc[i]),bar_low=float(l.iloc[i]),bar_open=float(d2['open'].iloc[i]))

def entry_signal(st):
    if st['bull'] and (st['brk_up'] or (st['flip'] and st['in_up10'])): return 1
    if not st['bull'] and (st['brk_dn'] or (st['flip'] and st['in_dn10'])): return -1
    return 0

def mark_equity(state, prices):
    eq=state['cash']
    for p in state['positions'].values():
        px=prices.get(p['coin'],p['last_px']); eq+=p['dir']*(px-p['entry'])*p['size']
    return eq

def _derisk(state, equity):
    hist=state['equity_history'][-DD_LOOKBACK_D:]; peak=max([e for _,e in hist]+[equity]) if hist else equity
    dd=1-equity/peak if peak>0 else 1.0
    if dd<=DD_START: return 1.0
    if dd>=DD_FULL: return 0.0
    return 1-(dd-DD_START)/(DD_FULL-DD_START)

def _close(state, key, px, reason, ts):
    p=state['positions'].pop(key); d=p['dir']
    pnl=d*(px-p['entry'])*p['size']-px*p['size']*TAKER_FEE; state['cash']+=pnl
    tr=dict(book=p['book'],coin=p['coin'],direction='long' if d>0 else 'short',
        entry_ts=p['entry_ts'],exit_ts=ts,entry_px=p['entry'],exit_px=px,
        pnl=round(pnl+p['partial_pnl'],2),reason=reason,risk_usd=round(p['risk_usd'],2),rank=p.get('rank'))
    state['trades'].append(tr); return tr

def process_book(state, book, data, prices, now_iso):
    report=dict(book='A(even)' if book==0 else 'B(odd)',closed=[],partials=[],entries=[],skipped=[],trends={},ts=now_iso)
    states={}
    for coin in COINS:
        df=data.get(coin)
        if df is None or df.empty: continue
        st=analyse(build_2d(df,book))
        if st: states[coin]=st; report['trends'][coin]='BULL' if st['bull'] else 'BEAR'
    for key in [k for k in list(state['positions']) if state['positions'][k]['book']==book]:
        p=state['positions'][key]; st=states.get(p['coin'])
        if st is None: continue
        d,px_now=p['dir'],prices.get(p['coin'],st['close']); hi,lo,op=st['bar_high'],st['bar_low'],st['bar_open']
        ptarget=p['entry']+d*p['atr0']*PARTIAL_ATR
        stop_gap=(d>0 and op<=p['stop']) or (d<0 and op>=p['stop'])
        stop_hit=(d>0 and lo<=p['stop']) or (d<0 and hi>=p['stop'])
        pt_hit=(d>0 and hi>=ptarget) or (d<0 and lo<=ptarget)
        if not p['partial_done'] and pt_hit and not stop_gap:
            qty=p['size']*PARTIAL_SIZE; pnl=d*(ptarget-p['entry'])*qty-ptarget*qty*TAKER_FEE
            state['cash']+=pnl; p['partial_pnl']+=pnl; p['size']-=qty; p['partial_done']=True
            p['stop']=max(p['stop'],p['entry']) if d>0 else min(p['stop'],p['entry'])
            report['partials'].append(dict(coin=p['coin'],px=round(ptarget,6),pnl=round(pnl,2)))
        if stop_hit:
            fill=(op if stop_gap else p['stop'])*(1-SLIP*d); report['closed'].append(_close(state,key,fill,'stop',now_iso)); continue
        state['cash']+=-d*FUNDING_8H*6*p['size']*st['close']
        if (d>0)!=st['bull']:
            report['closed'].append(_close(state,key,px_now*(1-SLIP*d),'trend_flip',now_iso)); continue
        p['best']=max(p['best'],st['close']) if d>0 else min(p['best'],st['close'])
        tr_stop=p['best']-d*TRAIL_MULT*st['atr']; p['stop']=max(p['stop'],tr_stop) if d>0 else min(p['stop'],tr_stop)
        p['last_px']=px_now; p['bars']+=1
    eq=mark_equity(state,prices); dmult=_derisk(state,eq)
    book_pos=[p for p in state['positions'].values() if p['book']==book]
    open_risk=sum(pp['size']*abs(pp['entry']-pp['stop']) for pp in book_pos)/eq if eq>0 else 1
    open_notional=sum(pp['size']*pp['last_px'] for pp in state['positions'].values())/eq if eq>0 else 9
    cands=[]
    for coin,st in states.items():
        if f'{book}:{coin}' in state['positions']: continue
        d=entry_signal(st)
        if d: cands.append((st['mom']*d,coin,d,st))
    cands.sort(key=lambda x:-x[0])
    for rank,(score,coin,d,st) in enumerate(cands):
        if state.get('paused'): report['skipped'].append(dict(coin=coin,why='paused')); continue
        if len([p for p in state['positions'].values() if p['book']==book])>=MAX_CONCURRENT:
            report['skipped'].append(dict(coin=coin,why='max positions')); continue
        px=prices.get(coin,st['close']); entry=px*(1+SLIP*d)
        stop=st['swing_lo'] if d>0 else st['swing_hi']
        if abs(entry-stop)<0.25*st['atr']: stop=entry-d*0.25*st['atr']
        stop_dist=d*(entry-stop)
        if stop_dist<=0: report['skipped'].append(dict(coin=coin,why='bad stop')); continue
        rmult=1.4 if rank==0 else (1.0 if rank==1 else 0.7)
        risk_frac=min(state['base_risk']*dmult*rmult, 8*state['base_risk']-open_risk)
        if risk_frac<=0: report['skipped'].append(dict(coin=coin,why='heat cap / breaker')); continue
        size=eq*risk_frac/stop_dist
        max_notional=min(MAX_LEV_POS/2,max(0.0,MAX_LEV_PORT/2-open_notional))*eq
        if size*entry>max_notional: size=max_notional/entry
        if size<=0: report['skipped'].append(dict(coin=coin,why='notional cap')); continue
        state['cash']-=entry*size*TAKER_FEE; open_risk+=size*stop_dist/eq; open_notional+=size*entry/eq
        state['positions'][f'{book}:{coin}']=dict(book=book,coin=coin,dir=d,entry=entry,entry_ts=now_iso,
            size=size,size0=size,stop=stop,atr0=st['atr'],best=entry,partial_done=False,partial_pnl=0.0,
            bars=0,last_px=entry,risk_usd=eq*risk_frac,rank=rank)
        report['entries'].append(dict(coin=coin,dir='LONG' if d>0 else 'SHORT',px=round(entry,6),
            stop=round(stop,6),size=round(size,6),notional=round(size*entry,2),risk=round(eq*risk_frac,2),rank=rank+1))
    eq=mark_equity(state,prices); state['equity_history'].append([now_iso,round(eq,2)]); state['last_update']=now_iso
    report['equity']=round(eq,2); return report

def daily_update(state, data, prices, now):
    book=0 if int(now.timestamp()//86400)%2==0 else 1
    return process_book(state,book,data,prices,now.isoformat())

def compute_stats(state):
    eqh=state['equity_history']
    out=dict(equity=state['account_start'],total_return=0.0,sharpe=0.0,max_dd=0.0,
             trades=len(state['trades']),win_rate=0.0,profit_factor=0.0,days=len(eqh))
    if eqh:
        eq=pd.Series([e for _,e in eqh],index=pd.to_datetime([t for t,_ in eqh],utc=True,format='ISO8601'))
        eq=eq[~eq.index.duplicated(keep='last')]; out['equity']=float(eq.iloc[-1])
        out['total_return']=float(eq.iloc[-1]/state['account_start']-1)
        if len(eq)>14:
            w=eq.resample('W').last().ffill().pct_change().dropna()
            if len(w)>3 and w.std()>0: out['sharpe']=float(w.mean()/w.std()*np.sqrt(52))
        peak=eq.cummax(); out['max_dd']=float(((eq-peak)/peak).min())
    tr=state['trades']
    if tr:
        wins=[t for t in tr if t['pnl']>0]; losses=[t for t in tr if t['pnl']<0]
        out['win_rate']=len(wins)/len(tr)
        gp=sum(t['pnl'] for t in wins); gl=abs(sum(t['pnl'] for t in losses))
        out['profit_factor']=gp/gl if gl>0 else float('inf')
    return out

def fetch_all(limit=150):
    import ccxt
    # GitHub Actions runners use US/Azure IPs, which Binance 451-blocks
    # ("restricted location"). Try exchanges in order until one serves every
    # coin. Binance stays first so local runs match the Binance-built backfill;
    # the others are fallbacks that GitHub's runners can reach.
    EXCHANGES=['binance','kucoin','okx','bybit','coinbase']
    last_err=None
    for name in EXCHANGES:
        try:
            ex=getattr(ccxt,name)({'enableRateLimit':True}); data,prices={},{}
            for coin in COINS:
                raw=None
                for quote in ('USDT','USD'):
                    try:
                        raw=ex.fetch_ohlcv(f'{coin}/{quote}','1d',limit=limit)
                        if raw and len(raw)>=100: break
                        raw=None
                    except Exception as e:
                        last_err=e; raw=None
                if not raw or len(raw)<100:
                    raise RuntimeError(f'insufficient data for {coin}')
                df=pd.DataFrame(raw,columns=['ts','open','high','low','close','volume'])
                df['ts']=pd.to_datetime(df['ts'],unit='ms',utc=True); df=df.set_index('ts')[['open','high','low','close']].astype(float)
                now=pd.Timestamp.now(tz='UTC').normalize(); prices[coin]=float(df['close'].iloc[-1]); data[coin]=df[df.index<now]
            print(f'[fetch_all] data source: {name}')
            return data,prices
        except Exception as e:
            last_err=e; print('[fetch_all] '+name+' unavailable: '+str(e)[:80]+'; trying next')