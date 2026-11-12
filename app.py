import math, time
from datetime import datetime, timezone
import ccxt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crypto Scanner", page_icon="📈", layout="centered")
CASH0 = 1000.0
TF = {"5 মিনিট": ("5m", "15m"), "15 মিনিট": ("15m", "1h"),
      "1 ঘণ্টা": ("1h", "4h"), "4 ঘণ্টা": ("4h", "1d"), "1 দিন": ("1d", "1w")}
# Hidden conservative controls
MIN_QVOL, MAX_AGE, MAX_SPREAD = 500_000, 90, 0.35
MAX_EXT_ATR, VOL_MULT, ATR_SL, RR = 0.8, 1.5, 1.6, 2.0
RISK_PCT, MAX_POS_PCT, FEE_PCT, SLIP_PCT = 0.5, 15.0, 0.10, 0.05

def pos(x):
    try: return math.isfinite(float(x)) and float(x) > 0
    except (TypeError, ValueError): return False

@st.cache_resource
def exchange_for(name):
    return getattr(ccxt, name)({"enableRateLimit": True, "timeout": 20000,
                                "options": {"defaultType": "spot"}})

def ticker_checked(ex, symbol):
    t = ex.fetch_ticker(symbol)
    bid, ask, ts = t.get("bid"), t.get("ask"), t.get("timestamp")
    if not pos(bid) or not pos(ask): raise ValueError("Bid/Ask পাওয়া যায়নি")
    bid, ask = float(bid), float(ask)
    if ask < bid: raise ValueError("Ask < Bid: অস্বাভাবিক ডেটা")
    if not pos(ts): raise ValueError("Ticker timestamp নেই; নিরাপত্তার জন্য বাদ")
    age = time.time() - float(ts)/1000
    if age < -5 or age > MAX_AGE: raise ValueError(f"Ticker fresh নয় ({age:.1f}s)")
    spread = (ask-bid)/((ask+bid)/2)*100
    if spread > MAX_SPREAD: raise ValueError(f"Spread বেশি ({spread:.3f}%)")
    return {"bid":bid, "ask":ask, "last":float(t["last"]) if pos(t.get("last")) else (bid+ask)/2,
            "spread":spread, "age":max(0,age)}

def candles(ex, symbol, tf, limit=220):
    raw = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    if not raw or len(raw)<80: raise ValueError(f"{tf} candle অপর্যাপ্ত")
    d = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    d = d.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    for c in ["open","high","low","close","volume"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna().reset_index(drop=True)
    if len(d) and int(d.iloc[-1].timestamp)+ex.parse_timeframe(tf)*1000 > ex.milliseconds():
        d=d.iloc[:-1].copy().reset_index(drop=True)
    if len(d)<70: raise ValueError(f"{tf} সম্পূর্ণ candle অপর্যাপ্ত")
    return d

def ind(d):
    x=d.copy()
    for n in (9,20,50): x[f"ema{n}"]=x.close.ewm(span=n,adjust=False).mean()
    delta=x.close.diff()
    gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    x["rsi"]=100-100/(1+gain/loss.replace(0,np.nan))
    pc=x.close.shift()
    tr=pd.concat([x.high-x.low,(x.high-pc).abs(),(x.low-pc).abs()],axis=1).max(axis=1)
    x["atr"]=tr.ewm(alpha=1/14,adjust=False).mean()
    x["vma"]=x.volume.rolling(20).mean()
    return x

def trend_ok(ex,symbol,tf):
    x=ind(candles(ex,symbol,tf,160)); a=x.iloc[-1]; b=x.iloc[-4]
    return bool(a.close>a.ema20>a.ema50 and a.ema20>b.ema20)

def signal_for(ex,symbol,entry_tf,trend_tf,q):
    if not trend_ok(ex,symbol,trend_tf): return None
    x=ind(candles(ex,symbol,entry_tf))
    a,b=x.iloc[-1],x.iloc[-2]
    atr,ema=float(a.atr),float(a.ema20)
    if not pos(atr) or not pos(ema): return None
    ext=(q["ask"]-ema)/atr
    if ext>MAX_EXT_ATR: return None
    setup=reason=None
    # Breakout then later retest, using completed candles only.
    breakout=None
    for i in range(max(20,len(x)-8),len(x)-2):
        resistance=float(x.high.iloc[i-20:i].max())
        c=x.iloc[i]
        if c.close>resistance and pos(c.vma) and c.volume>=VOL_MULT*c.vma: breakout=i
    if breakout is not None:
        level=float(x.high.iloc[breakout-20:breakout].max())
        ret=x.iloc[breakout+1:-1]
        held=any(r.low<=level*1.003 and r.close>=level for _,r in ret.iterrows())
        if held and a.close>level and a.close>b.close and a.close>a.open and 50<=a.rsi<=74:
            setup,reason="Breakout Retest","Resistance retest ধরে রেখেছে"
    if setup is None:
        recent=x.iloc[-4:-1]
        touched=(recent.low<=recent.ema20*1.004).any()
        if (touched and a.close>a.open and a.close>a.ema20 and a.close>b.close
            and a.ema9>=a.ema20 and 48<=a.rsi<=66 and pos(a.vma) and a.volume>=.7*a.vma):
            setup,reason="EMA20 Pullback","EMA20 support থেকে bullish reclaim"
    if setup is None: return None
    entry=q["ask"]; stop=min(entry-ATR_SL*atr,entry*.998)
    if not pos(entry-stop): return None
    return {"symbol":symbol,"setup":setup,"reason":reason,"entry":entry,"stop":stop,
            "target":entry+RR*(entry-stop),"rsi":float(a.rsi),"extension":ext,
            "spread":q["spread"],"age":q["age"],
            "candle_time":datetime.fromtimestamp(float(a.timestamp)/1000,timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

def top_symbols(ex,count):
    markets=ex.load_markets(); ticks=ex.fetch_tickers(); ranked=[]
    for s,m in markets.items():
        if not m.get("spot") or m.get("active") is False or m.get("quote")!="USDT": continue
        base=str(m.get("base","")).upper()
        if any(k in base for k in ("UP","DOWN","BULL","BEAR")): continue
        qv=(ticks.get(s) or {}).get("quoteVolume")
        if pos(qv) and float(qv)>=MIN_QVOL: ranked.append((s,float(qv)))
    ranked.sort(key=lambda z:z[1],reverse=True)
    return [s for s,_ in ranked[:count]]

def init():
    defaults={"cash":CASH0,"positions":{},"trades":[],"signals":[],"warnings":[],
              "last_scan":None,"last_exchange":None,"last_tf":None}
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v

def equity():
    return st.session_state.cash+sum(p["qty"]*p["entry"] for p in st.session_state.positions.values())

def paper_buy(s):
    sym=s["symbol"]
    if sym in st.session_state.positions: return "এই কয়েনে position আগে থেকেই আছে"
    entry=s["entry"]*(1+SLIP_PCT/100); risk=entry-s["stop"]
    if not pos(risk): return "Entry/Stop যাচাই ব্যর্থ"
    eq=equity(); qty=min(eq*RISK_PCT/100/risk,eq*MAX_POS_PCT/100/entry)
    cost=qty*entry*(1+FEE_PCT/100)
    if qty<=0 or cost>st.session_state.cash: return "Paper Cash পর্যাপ্ত নয়"
    st.session_state.cash-=cost
    st.session_state.positions[sym]={**s,"entry":entry,"qty":qty,"entry_fee":qty*entry*FEE_PCT/100}
    return f"{sym} Paper BUY সম্পন্ন"

def paper_close(sym,bid,reason):
    p=st.session_state.positions.get(sym)
    if not p:return
    fill=bid*(1-SLIP_PCT/100); gross=p["qty"]*fill; fee=gross*FEE_PCT/100
    proceeds=gross-fee; pnl=proceeds-p["qty"]*p["entry"]-p["entry_fee"]
    st.session_state.cash+=proceeds
    st.session_state.trades.append({"symbol":sym,"entry":p["entry"],"exit":fill,
      "qty":p["qty"],"net_pnl":pnl,"reason":reason,
      "closed_utc":datetime.now(timezone.utc).isoformat()})
    del st.session_state.positions[sym]

def run_scan(ex,count,etf,ttf):
    symbols=top_symbols(ex,count); found=[]; warns=[]
    try: btc_up=trend_ok(ex,"BTC/USDT",ttf)
    except Exception as e: btc_up=False; warns.append(f"BTC trend unavailable: {e}")
    bar=st.progress(0,text="স্ক্যান শুরু হচ্ছে")
    for i,sym in enumerate(symbols,1):
        try:
            q=ticker_checked(ex,sym)
            if sym!="BTC/USDT" and not btc_up: continue
            s=signal_for(ex,sym,etf,ttf,q)
            if s: found.append(s)
        except Exception as e: warns.append(f"{sym}: {str(e)[:140]}")
        bar.progress(i/max(1,len(symbols)),text=f"স্ক্যান হচ্ছে {i}/{len(symbols)} · {sym}")
    bar.empty()
    # SL/TP checked only during a scan, not in background.
    for sym,p in list(st.session_state.positions.items()):
        try:
            q=ticker_checked(ex,sym)
            if q["bid"]<=p["stop"]: paper_close(sym,q["bid"],"STOP LOSS")
            elif q["bid"]>=p["target"]: paper_close(sym,q["bid"],"TAKE PROFIT")
        except Exception as e: warns.append(f"{sym} exit check হয়নি: {str(e)[:100]}")
    found.sort(key=lambda s:(s["setup"]!="Breakout Retest",s["extension"]))
    return found,warns

init()
# Sidebar contains exactly the requested two controls.
with st.sidebar:
    st.subheader("⚙️ Scanner Options")
    exchange_label=st.selectbox("Exchange",["KuCoin","Binance"],index=0)
    coin_count=st.select_slider("কতটি কয়েন স্ক্যান করবে?",options=[5,10,50,100],value=10)
    st.caption("EMA, Volume ও Risk Settings স্বয়ংক্রিয়ভাবে চলবে।")

st.title("📈 Crypto Scanner")
st.caption("Spot Market · Live Data · Paper Trading Only")
st.markdown("#### Timeframe নির্বাচন করুন")
tf_label=st.radio("Entry timeframe",list(TF.keys()),horizontal=True,label_visibility="collapsed")
entry_tf,trend_tf=TF[tf_label]
st.markdown(f"""<div style="text-align:center;padding:12px;border:1px solid #666;border-radius:12px">
<div style="font-size:12px;color:gray">SELECTED TIMEFRAME</div>
<div style="font-size:25px;font-weight:700">{tf_label}</div>
<div style="font-size:12px;color:gray">Trend confirmation: {trend_tf}</div></div>""",unsafe_allow_html=True)
st.write("")
scan=st.button("🔎  START SCANNING",type="primary",use_container_width=True)
a,b,c=st.columns(3)
a.metric("Paper Cash",f"{st.session_state.cash:,.2f} USDT")
b.metric("Open",len(st.session_state.positions))
c.metric("Closed",len(st.session_state.trades))

if scan:
    exid="kucoin" if exchange_label=="KuCoin" else "binance"
    st.session_state.signals=[]; st.session_state.warnings=[]
    try:
        with st.status("Scanning market...",expanded=True) as status:
            signals,warnings=run_scan(exchange_for(exid),int(coin_count),entry_tf,trend_tf)
            st.session_state.signals=signals; st.session_state.warnings=warnings
            st.session_state.last_scan=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            st.session_state.last_exchange=exchange_label; st.session_state.last_tf=tf_label
            status.update(label="Scan complete",state="complete",expanded=False)
    except Exception as e: st.error(f"Scan শুরু করা যায়নি: {e}")

if st.session_state.last_scan:
    st.caption(f"শেষ Scan: {st.session_state.last_scan} · {st.session_state.last_exchange} · {st.session_state.last_tf}")

st.divider()
st.subheader("🎯 BUY Signals")
if st.session_state.signals:
    for s in st.session_state.signals:
        with st.container(border=True):
            l,r=st.columns([2,1]); l.markdown(f"**{s['symbol']}**"); l.caption(f"{s['setup']} · {s['reason']}")
            r.metric("RSI",f"{s['rsi']:.1f}")
            x,y,z=st.columns(3); x.metric("Entry (Ask)",f"{s['entry']:.8g}")
            y.metric("Stop",f"{s['stop']:.8g}"); z.metric("Target",f"{s['target']:.8g}")
            st.caption(f"Spread {s['spread']:.3f}% · Ticker age {s['age']:.1f}s · Extension {s['extension']:.2f} ATR")
            st.caption(f"Signal candle: {s['candle_time']}")
            if st.button(f"🧪 Paper BUY · {s['symbol']}",key=f"buy_{s['symbol']}"):
                try:
                    exid="kucoin" if st.session_state.last_exchange=="KuCoin" else "binance"
                    fresh=ticker_checked(exchange_for(exid),s["symbol"])
                    if fresh["ask"]>s["entry"]*1.003: st.warning("দাম 0.3%-এর বেশি বেড়েছে; Entry বাতিল। আবার Scan করুন।")
                    else: st.success(paper_buy(s))
                except Exception as e: st.error(f"Fresh price যাচাই হয়নি; Entry বাতিল: {e}")
else: st.info("এখনো BUY Signal নেই। শর্ত পূরণ না হলে কোনো Signal দেখানো হবে না।")

if st.session_state.warnings:
    with st.expander(f"⚠️ Scan warnings ({len(st.session_state.warnings)})"):
        for w in st.session_state.warnings[:100]: st.write("• "+w)

st.divider()
st.subheader("📂 Open Paper Positions")
if st.session_state.positions:
    for sym,p in list(st.session_state.positions.items()):
        with st.container(border=True):
            st.markdown(f"**{sym}** · {p['setup']}")
            x,y,z=st.columns(3); x.metric("Entry",f"{p['entry']:.8g}")
            y.metric("Stop",f"{p['stop']:.8g}"); z.metric("Target",f"{p['target']:.8g}")
            st.caption(f"Quantity: {p['qty']:.8g}")
            if st.button(f"Close Paper Position · {sym}",key=f"close_{sym}"):
                try:
                    exid="kucoin" if st.session_state.last_exchange=="KuCoin" else "binance"
                    q=ticker_checked(exchange_for(exid),sym); paper_close(sym,q["bid"],"MANUAL"); st.rerun()
                except Exception as e: st.error(f"Fresh Bid পাওয়া যায়নি: {e}")
else: st.caption("কোনো খোলা Paper Position নেই।")

st.subheader("📊 Closed Paper Trades")
if st.session_state.trades:
    hist=pd.DataFrame(st.session_state.trades); st.dataframe(hist,use_container_width=True,hide_index=True)
    st.metric("Total Net P/L",f"{hist.net_pnl.sum():.4f} USDT")
    st.download_button("Download CSV",hist.to_csv(index=False).encode("utf-8-sig"),
                       "paper_trade_history.csv","text/csv",use_container_width=True)
else: st.caption("এখনো কোনো Paper Trade বন্ধ হয়নি।")

st.divider()
st.caption("No live orders/API keys. SL/TP পরীক্ষা শুধু Scan চাপলে হয়—24/7 background monitoring নয়। Paper হিসাব Streamlit session-এ থাকে; server restart হলে হারাতে পারে।")
