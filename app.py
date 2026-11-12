import math, time
from datetime import datetime, timezone
import ccxt, numpy as np, pandas as pd, streamlit as st

st.set_page_config(page_title="Crypto Scanner Pro", page_icon="📈", layout="centered")

# Internal strategy settings — no indicator controls in the interface.
CASH_START = 1000.0
MIN_QVOL = 500_000
MAX_TICKER_AGE = 90
MAX_SPREAD = 0.35
MAX_EXTENSION_ATR = 0.75
MAX_5_CANDLE_RISE = 7.0
MAX_CANDLE_RANGE_ATR = 2.2
BREAKOUT_VOL = 1.5
PULLBACK_VOL = 0.70
ATR_SL = 1.6
RR = 2.0
RISK_PCT = 0.5
MAX_POSITION_PCT = 15.0
FEE_PCT = 0.10
SLIPPAGE_PCT = 0.05
SIGNAL_MAX_AGE = 90

TIMEFRAMES = {
    "5 মিনিট": ("5m", "1h"),
    "15 মিনিট": ("15m", "4h"),
    "1 ঘণ্টা": ("1h", "1d"),
    "4 ঘণ্টা": ("4h", "1d"),
    "1 দিন": ("1d", "1w"),
}

def pos(v):
    try:
        n = float(v)
        return math.isfinite(n) and n > 0
    except (TypeError, ValueError):
        return False

@st.cache_resource
def exchange_for(exchange_id):
    return getattr(ccxt, exchange_id)({
        "enableRateLimit": True, "timeout": 20000,
        "options": {"defaultType": "spot"}
    })

def checked_ticker(ex, symbol):
    t = ex.fetch_ticker(symbol)
    bid, ask, ts = t.get("bid"), t.get("ask"), t.get("timestamp")
    if not pos(bid) or not pos(ask):
        raise ValueError("বৈধ Bid/Ask পাওয়া যায়নি")
    bid, ask = float(bid), float(ask)
    if ask < bid:
        raise ValueError("Ask Bid-এর চেয়ে কম")
    if not pos(ts):
        raise ValueError("Ticker timestamp নেই")
    age = time.time() - float(ts) / 1000
    if age < -5 or age > MAX_TICKER_AGE:
        raise ValueError(f"Ticker পুরোনো/অস্বাভাবিক ({age:.1f}s)")
    spread = (ask - bid) / ((ask + bid) / 2) * 100
    if spread > MAX_SPREAD:
        raise ValueError(f"Spread বেশি ({spread:.3f}%)")
    last = float(t["last"]) if pos(t.get("last")) else (bid + ask) / 2
    return {"bid": bid, "ask": ask, "last": last,
            "spread": spread, "age": max(0, age)}

def get_candles(ex, symbol, tf, limit=220):
    raw = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    if not raw or len(raw) < 80:
        raise ValueError(f"{tf} candle অপর্যাপ্ত")
    d = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    d = d.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    for col in ["open","high","low","close","volume"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.replace([np.inf,-np.inf], np.nan).dropna().reset_index(drop=True)
    # Drop the still-forming candle.
    if len(d) and int(d.iloc[-1].timestamp) + ex.parse_timeframe(tf)*1000 > ex.milliseconds():
        d = d.iloc[:-1].copy().reset_index(drop=True)
    if len(d) < 70:
        raise ValueError(f"{tf} সম্পূর্ণ candle অপর্যাপ্ত")
    return d

def indicators(d):
    x = d.copy()
    for n in (9,20,50):
        x[f"ema{n}"] = x.close.ewm(span=n, adjust=False).mean()
    delta = x.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    x["rsi"] = 100 - 100/(1 + gain/loss.replace(0,np.nan))
    prev = x.close.shift()
    tr = pd.concat([x.high-x.low,(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1)
    x["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    x["vma"] = x.volume.rolling(20).mean()
    return x

def trend_state(ex, symbol, tf):
    x = indicators(get_candles(ex,symbol,tf,180))
    a, old = x.iloc[-1], x.iloc[-4]
    if not all(pos(a[k]) for k in ("close","ema20","ema50")):
        return "UNKNOWN"
    if a.close > a.ema20 > a.ema50 and a.ema20 > old.ema20:
        return "BULLISH"
    if a.close < a.ema20 < a.ema50 and a.ema20 < old.ema20:
        return "BEARISH"
    return "SIDEWAYS"

def top_symbols(ex, count):
    markets, tickers = ex.load_markets(), ex.fetch_tickers()
    ranked = []
    for sym, m in markets.items():
        if not m.get("spot") or m.get("active") is False or m.get("quote") != "USDT":
            continue
        base = str(m.get("base","")).upper()
        if any(tag in base for tag in ("UP","DOWN","BULL","BEAR")):
            continue
        qv = (tickers.get(sym) or {}).get("quoteVolume")
        if pos(qv) and float(qv) >= MIN_QVOL:
            ranked.append((sym,float(qv)))
    ranked.sort(key=lambda z:z[1], reverse=True)
    return [s for s,_ in ranked[:count]]

def analyze(ex, symbol, entry_tf, trend_tf, btc_state):
    q = checked_ticker(ex,symbol)
    x = indicators(get_candles(ex,symbol,entry_tf))
    a,b = x.iloc[-1],x.iloc[-2]
    atr, ema20 = float(a.atr),float(a.ema20)
    if not pos(atr) or not pos(ema20):
        raise ValueError("ATR/EMA হিসাব হয়নি")
    ext = (q["ask"]-ema20)/atr
    if ext < -0.5 or ext > MAX_EXTENSION_ATR:
        raise ValueError(f"Anti-Chase: EMA20 থেকে {ext:.2f} ATR")
    prev5 = x.iloc[-6:-1]
    if len(prev5) == 5:
        rise = (float(a.close)/float(prev5.iloc[0].close)-1)*100
        if rise > MAX_5_CANDLE_RISE:
            raise ValueError(f"Pump filter: শেষ ৫ candle-এ {rise:.1f}% বৃদ্ধি")
    if (float(a.high)-float(a.low))/atr > MAX_CANDLE_RANGE_ATR:
        raise ValueError("অস্বাভাবিক বড় candle; pump/volatility filter")
    state = trend_state(ex,symbol,trend_tf)
    if state != "BULLISH":
        raise ValueError(f"{trend_tf} Trend {state}; confirmation হয়নি")

    setup = reason = None
    # BTC is a risk-context warning, not a hard block on all altcoins.
    btc_note = ("BTC bullish" if btc_state=="BULLISH" else
                f"BTC {btc_state}: altcoin risk বেশি")

    # Breakout then retest.
    breakout = None
    for i in range(max(20,len(x)-9),len(x)-2):
        resistance = float(x.high.iloc[i-20:i].max())
        c = x.iloc[i]
        if c.close > resistance and pos(c.vma) and c.volume >= BREAKOUT_VOL*c.vma:
            breakout = i
    if breakout is not None:
        level = float(x.high.iloc[breakout-20:breakout].max())
        retests = x.iloc[breakout+1:-1]
        held = any(r.low <= level*1.003 and r.close >= level for _,r in retests.iterrows())
        if (held and a.close > level and a.close > b.close and a.close > a.open
            and 50 <= a.rsi <= 70):
            setup,reason = "Breakout Retest","Resistance retest ধরে bullish reclaim"

    # EMA20 pullback.
    if setup is None:
        recent = x.iloc[-4:-1]
        touched = (recent.low <= recent.ema20*1.004).any()
        if (touched and a.close>a.open and a.close>a.ema20 and a.close>b.close
            and a.ema9>=a.ema20 and 48<=a.rsi<=65 and pos(a.vma)
            and a.volume>=PULLBACK_VOL*a.vma):
            setup,reason = "EMA20 Pullback","EMA20 support থেকে bullish reclaim"
    if setup is None:
        raise ValueError("Trend ঠিক, কিন্তু Entry setup নেই")

    entry = q["ask"]
    stop = min(entry-ATR_SL*atr, entry*0.998)
    if not pos(entry-stop):
        raise ValueError("Stop distance ভুল")
    return {
        "symbol":symbol,"setup":setup,"reason":reason,"entry":entry,
        "stop":stop,"target":entry+RR*(entry-stop),"rsi":float(a.rsi),
        "extension":ext,"spread":q["spread"],"age":q["age"],
        "trend":state,"btc_context":btc_note,"signal_ts":time.time(),
        "candle_time":datetime.fromtimestamp(float(a.timestamp)/1000,timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    }

def init_state():
    defaults = {"cash":CASH_START,"positions":{},"trades":[],"signals":[],
                "warnings":[],"diagnostics":{},"last_scan":None,
                "last_exchange":None,"last_tf":None,"btc_state":"UNKNOWN"}
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v

def equity():
    return st.session_state.cash + sum(p["qty"]*p["entry"] for p in st.session_state.positions.values())

def paper_buy(ex, sig):
    sym=sig["symbol"]
    if sym in st.session_state.positions: return False,"এই কয়েনে Position আগে থেকেই আছে"
    if time.time()-sig["signal_ts"] > SIGNAL_MAX_AGE: return False,"Signal পুরোনো; আবার Scan করুন"
    q=checked_ticker(ex,sym)
    if q["ask"] > sig["entry"]*1.002:
        return False,"দাম Signal থেকে 0.20%-এর বেশি বেড়েছে; Entry বাতিল"
    entry=q["ask"]*(1+SLIPPAGE_PCT/100)
    risk=max(sig["entry"]-sig["stop"],entry*0.002)
    stop=entry-risk
    target=entry+RR*risk
    eq=equity()
    qty=min((eq*RISK_PCT/100)/risk,(eq*MAX_POSITION_PCT/100)/entry)
    gross=qty*entry
    fee=gross*FEE_PCT/100
    if not pos(qty) or gross+fee>st.session_state.cash:
        return False,"Paper Cash পর্যাপ্ত নয়"
    st.session_state.cash-=gross+fee
    st.session_state.positions[sym]={**sig,"entry":entry,"stop":stop,"target":target,
                                      "qty":qty,"entry_fee":fee}
    return True,f"{sym} Paper BUY সম্পন্ন"

def paper_close(sym,bid,reason):
    p=st.session_state.positions.get(sym)
    if not p:return
    fill=bid*(1-SLIPPAGE_PCT/100)
    gross=p["qty"]*fill
    fee=gross*FEE_PCT/100
    proceeds=gross-fee
    pnl=proceeds-p["qty"]*p["entry"]-p["entry_fee"]
    st.session_state.cash+=proceeds
    st.session_state.trades.append({"symbol":sym,"entry":p["entry"],"exit":fill,
        "qty":p["qty"],"net_pnl":pnl,"reason":reason,
        "closed_utc":datetime.now(timezone.utc).isoformat()})
    del st.session_state.positions[sym]

def run_scan(ex,count,etf,ttf):
    symbols=top_symbols(ex,count)
    found,warnings=[],[]
    stats={"নির্বাচিত কয়েন":len(symbols),"Ticker/Spread সমস্যা":0,
           "Pump/Anti-Chase":0,"Trend confirmation":0,"Entry setup নেই":0,
           "অন্যান্য ডেটা সমস্যা":0,"BUY Signal":0}
    try: btc=trend_state(ex,"BTC/USDT",ttf)
    except Exception as e:
        btc="UNKNOWN"; warnings.append(f"BTC trend unavailable: {str(e)[:120]}")
    bar=st.progress(0,text="মার্কেট স্ক্যান শুরু হচ্ছে...")
    for i,sym in enumerate(symbols,1):
        try:
            s=analyze(ex,sym,etf,ttf,btc)
            found.append(s); stats["BUY Signal"]+=1
        except Exception as e:
            msg=str(e)
            if any(k in msg for k in ("Spread","Bid/Ask","Ticker","timestamp")):
                stats["Ticker/Spread সমস্যা"]+=1
            elif any(k in msg for k in ("Anti-Chase","Pump filter","অস্বাভাবিক বড়")):
                stats["Pump/Anti-Chase"]+=1
            elif "Trend" in msg or "confirmation" in msg:
                stats["Trend confirmation"]+=1
            elif "Entry setup নেই" in msg:
                stats["Entry setup নেই"]+=1
            else:
                stats["অন্যান্য ডেটা সমস্যা"]+=1
            warnings.append(f"{sym}: {msg[:140]}")
        bar.progress(i/max(1,len(symbols)),text=f"স্ক্যান হচ্ছে {i}/{len(symbols)} · {sym}")
    bar.empty()
    # SL/TP checks occur only when the user presses Scan.
    for sym,p in list(st.session_state.positions.items()):
        try:
            q=checked_ticker(ex,sym)
            if q["bid"]<=p["stop"]: paper_close(sym,q["bid"],"STOP LOSS")
            elif q["bid"]>=p["target"]: paper_close(sym,q["bid"],"TAKE PROFIT")
        except Exception as e: warnings.append(f"{sym} exit check হয়নি: {str(e)[:100]}")
    found.sort(key=lambda s:(s["setup"]!="Breakout Retest",s["extension"]))
    return found,warnings,stats,btc

# -------------------- User interface --------------------
init_state()
with st.sidebar:
    st.subheader("⚙️ Scanner Options")
    exchange_label=st.selectbox("Exchange",["KuCoin","Binance"],index=0)
    coin_count=st.select_slider("কতটি কয়েন স্ক্যান করবে?",options=[5,10,50,100],value=50)
    st.caption("EMA, RSI, Volume, ATR ও Risk Settings ভিতরে স্বয়ংক্রিয়।")

st.title("📈 Crypto Scanner Pro")
st.caption("Spot Market · Live Price Validation · Paper Trading Only")
st.markdown("#### Timeframe নির্বাচন করুন")
tf_label=st.radio("Entry timeframe",list(TIMEFRAMES.keys()),horizontal=True,label_visibility="collapsed")
entry_tf,trend_tf=TIMEFRAMES[tf_label]
st.markdown(f"""<div style="text-align:center;padding:14px;border:1px solid #555;border-radius:12px">
<div style="font-size:12px;color:#999">ENTRY + TREND CONFIRMATION</div>
<div style="font-size:20px;font-weight:700">{entry_tf} Entry + {trend_tf} Trend</div></div>""",unsafe_allow_html=True)
scan=st.button("🔎 START SCANNING",type="primary",use_container_width=True)

a,b,c=st.columns(3)
a.metric("Paper Cash",f"{st.session_state.cash:,.2f} USDT")
b.metric("Open",len(st.session_state.positions))
c.metric("Closed",len(st.session_state.trades))

if scan:
    exid="kucoin" if exchange_label=="KuCoin" else "binance"
    st.session_state.signals=[]; st.session_state.warnings=[]
    try:
        with st.status("Scanning market...",expanded=True) as status:
            signals,warnings,stats,btc=run_scan(exchange_for(exid),int(coin_count),entry_tf,trend_tf)
            st.session_state.signals=signals; st.session_state.warnings=warnings
            st.session_state.diagnostics=stats; st.session_state.btc_state=btc
            st.session_state.last_scan=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            st.session_state.last_exchange=exchange_label; st.session_state.last_tf=tf_label
            status.update(label="Scan complete",state="complete",expanded=False)
    except Exception as e: st.error(f"Scan শুরু করা যায়নি: {e}")

if st.session_state.last_scan:
    st.caption(f"শেষ Scan: {st.session_state.last_scan} · {st.session_state.last_exchange} · {st.session_state.last_tf}")
    st.caption(f"BTC Trend Context: {st.session_state.btc_state}")

st.divider()
st.subheader("🎯 BUY Signals")
if st.session_state.signals:
    for s in st.session_state.signals:
        with st.container(border=True):
            l,r=st.columns([2,1]); l.markdown(f"**{s['symbol']}**"); l.caption(f"{s['setup']} · {s['reason']}")
            r.metric("RSI",f"{s['rsi']:.1f}")
            x,y,z=st.columns(3); x.metric("Entry (Ask)",f"{s['entry']:.8g}")
            y.metric("Stop",f"{s['stop']:.8g}"); z.metric("Target",f"{s['target']:.8g}")
            st.caption(f"Spread {s['spread']:.3f}% · Age {s['age']:.1f}s · Extension {s['extension']:.2f} ATR")
            st.caption(f"{s['trend']} · {s['btc_context']} · Candle: {s['candle_time']}")
            if st.button(f"🧪 Paper BUY · {s['symbol']}",key=f"buy_{s['symbol']}"):
                try:
                    exid="kucoin" if st.session_state.last_exchange=="KuCoin" else "binance"
                    ok,msg=paper_buy(exchange_for(exid),s)
                    (st.success if ok else st.warning)(msg)
                except Exception as e: st.error(f"Fresh price যাচাই হয়নি: {e}")
else:
    st.info("এখনো BUY Signal নেই। Scan Diagnostics দেখুন—কোন কারণে কয়েন বাদ পড়েছে।")

if st.session_state.diagnostics:
    with st.expander("📊 Scan Diagnostics — কয়েন কেন বাদ পড়েছে?",expanded=True):
        for label,value in st.session_state.diagnostics.items():
            st.write(f"**{label}:** {value}")
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
            if st.button(f"Close Position · {sym}",key=f"close_{sym}"):
                try:
                    exid="kucoin" if st.session_state.last_exchange=="KuCoin" else "binance"
                    q=checked_ticker(exchange_for(exid),sym)
                    paper_close(sym,q["bid"],"MANUAL"); st.rerun()
                except Exception as e: st.error(f"Fresh Bid পাওয়া যায়নি: {e}")
else: st.caption("কোনো খোলা Paper Position নেই।")

st.subheader("📊 Closed Paper Trades")
if st.session_state.trades:
    hist=pd.DataFrame(st.session_state.trades)
    st.dataframe(hist,use_container_width=True,hide_index=True)
    st.metric("Total Net P/L",f"{hist.net_pnl.sum():.4f} USDT")
    st.download_button("Download CSV",hist.to_csv(index=False).encode("utf-8-sig"),
                       "paper_trade_history.csv","text/csv",use_container_width=True)
else: st.caption("কোনো Paper Trade সম্পন্ন হয়নি।")

st.caption("সতর্কতা: Paper Trading only. Stop/Target কেবল Scan চাপলে পরীক্ষা হয়; এটি ২৪ ঘণ্টার background monitor নয়। Session/Server restart হলে session-based হিসাব মুছে যেতে পারে।")
