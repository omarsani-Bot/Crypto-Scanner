Import math
import time
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Crypto Scanner | 5m + 1h",
    page_icon="📈",
    layout="wide",
)

APP_VERSION = "2.0.0"
DEFAULT_CASH = 1000.0
QUOTE = "USDT"


def finite_number(value):
    try:
        x = float(value)
        return math.isfinite(x)
    except (TypeError, ValueError):
        return False


def positive_finite(value):
    return finite_number(value) and float(value) > 0


@st.cache_resource
def get_exchange(exchange_id):
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({
        "enableRateLimit": True,
        "timeout": 15000,
        "options": {"defaultType": "spot"},
    })


def safe_exchange_call(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def fetch_ticker_checked(exchange, symbol, max_age_seconds, max_spread_pct):
    ticker = safe_exchange_call(exchange.fetch_ticker, symbol)
    bid, ask, last = ticker.get("bid"), ticker.get("ask"), ticker.get("last")
    timestamp_ms = ticker.get("timestamp")

    if not positive_finite(bid) or not positive_finite(ask):
        raise ValueError("বৈধ Bid/Ask পাওয়া যায়নি")
    bid, ask = float(bid), float(ask)
    if ask < bid:
        raise ValueError("Ask, Bid-এর চেয়ে কম—টিকার ডেটা অস্বাভাবিক")
    if not positive_finite(timestamp_ms):
        raise ValueError("এক্সচেঞ্জ টিকারে timestamp নেই; নিরাপত্তার জন্য এন্ট্রি বন্ধ")
    age = time.time() - float(timestamp_ms) / 1000
    if age < -5:
        raise ValueError("টিকারের timestamp ভবিষ্যতের")
    if age > max_age_seconds:
        raise ValueError(f"টিকার পুরোনো: {age:.1f} সেকেন্ড")
    mid = (bid + ask) / 2
    spread = (ask - bid) / mid * 100
    if spread > max_spread_pct:
        raise ValueError(f"Spread বেশি: {spread:.3f}%")
    return {
        "bid": bid, "ask": ask,
        "last": float(last) if positive_finite(last) else (bid + ask) / 2,
        "spread_pct": spread,
        "age_seconds": max(0.0, age),
        "timestamp": datetime.fromtimestamp(float(timestamp_ms) / 1000, tz=timezone.utc),
    }


def fetch_ohlcv_frame(exchange, symbol, timeframe, limit):
    rows = safe_exchange_call(exchange.fetch_ohlcv, symbol, timeframe, None, limit)
    if not rows or len(rows) < 70:
        raise ValueError(f"{timeframe} পর্যাপ্ত candle পাওয়া যায়নি")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    # সর্বশেষ candle অসম্পূর্ণ হতে পারে; শুধু সম্পূর্ণ candle ব্যবহার করি।
    duration_ms = 5 * 60 * 1000 if timeframe == "5m" else 60 * 60 * 1000
    now_ms = exchange.milliseconds()
    if len(df) and int(df.iloc[-1]["timestamp"]) + duration_ms > now_ms:
        df = df.iloc[:-1].copy().reset_index(drop=True)
    if len(df) < 60:
        raise ValueError(f"{timeframe} সম্পূর্ণ candle ৬০টির কম")
    return df


def add_indicators(df):
    x = df.copy()
    x["ema9"] = x["close"].ewm(span=9, adjust=False).mean()
    x["ema20"] = x["close"].ewm(span=20, adjust=False).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False).mean()
    delta = x["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - (100 / (1 + rs))
    prev_close = x["close"].shift(1)
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    x["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    x["vol_ma20"] = x["volume"].rolling(20).mean()
    return x


def get_1h_trend(exchange, symbol):
    df = add_indicators(fetch_ohlcv_frame(exchange, symbol, "1h", 140))
    last = df.iloc[-1]
    bullish = (
        last["close"] > last["ema20"] > last["ema50"]
        and last["ema20"] > df.iloc[-4]["ema20"]
    )
    return bullish, float(last["close"]), df


def analyze_signal(df, trend_bull, live, settings):
    x = add_indicators(df)
    last = x.iloc[-1]
    prev = x.iloc[-2]
    atr = float(last["atr"])
    ema20 = float(last["ema20"])
    if not positive_finite(atr) or not positive_finite(ema20):
        return None
    # Live ask দিয়ে extension মাপা হয়, candle close দিয়ে নয়।
    extension_atr = (live["ask"] - ema20) / atr
    if extension_atr > settings["max_extension_atr"]:
        return None
    if not trend_bull:
        return None

    signal = None
    reason = ""

    # Breakout + retest: resistance আগের 20টি সম্পূর্ণ candle থেকে।
    lookback_high = float(x["high"].iloc[-22:-2].max())
    breakout_idx = None
    for i in range(max(1, len(x) - 7), len(x) - 2):
        candle = x.iloc[i]
        before_high = float(x["high"].iloc[i-20:i].max()) if i >= 20 else 0
        vol_ok = candle["volume"] >= settings["breakout_volume"] * x["vol_ma20"].iloc[i]
        if candle["close"] > before_high and vol_ok:
            breakout_idx = i
    if breakout_idx is not None:
        level = float(x["high"].iloc[breakout_idx-20:breakout_idx].max())
        retest = x.iloc[breakout_idx+1:-1]
        held = any(
            float(c["low"]) <= level * 1.003
            and float(c["close"]) >= level
            for _, c in retest.iterrows()
        )
        if held and last["close"] > level and last["close"] > prev["close"] and 50 <= last["rsi"] <= 76:
            signal, reason = "BREAKOUT RETEST", "আগের resistance retest করে ধরে রেখেছে"

    # Pullback: EMA20 support zone touch, তারপর bullish reclaim।
    if signal is None:
        recent = x.iloc[-4:-1]
        touched = (recent["low"] <= recent["ema20"] * 1.004).any()
        bullish_reclaim = (
            last["close"] > last["open"]
            and last["close"] > last["ema20"]
            and last["close"] > prev["close"]
            and last["ema9"] >= last["ema20"]
            and 48 <= last["rsi"] <= 66
            and last["volume"] >= 0.70 * last["vol_ma20"]
        )
        if touched and bullish_reclaim:
            signal, reason = "EMA20 PULLBACK", "EMA20 support থেকে bullish reclaim"

    if signal is None:
        return None
    # Ask entry; SL ATR-ভিত্তিক। Reward/risk ব্যবহারকারী নির্ধারণ করে।
    entry = float(live["ask"])
    stop = min(entry - settings["atr_sl"] * atr, entry * 0.999)
    risk_per_unit = entry - stop
    if not positive_finite(risk_per_unit):
        return None
    target = entry + settings["rr"] * risk_per_unit
    return {
        "symbol": live["symbol"],
        "signal": signal,
        "reason": reason,
        "entry": entry,
        "stop": stop,
        "target": target,
        "atr": atr,
        "rsi": float(last["rsi"]),
        "extension_atr": extension_atr,
        "spread_pct": live["spread_pct"],
        "ticker_age": live["age_seconds"],
        "candle_time": datetime.fromtimestamp(float(last["timestamp"]) / 1000, tz=timezone.utc),
    }


def market_symbols(exchange, quote, min_volume, max_symbols):
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()
    candidates = []
    for symbol, market in markets.items():
        if not market.get("spot") or market.get("active") is False:
            continue
        if market.get("quote") != quote or not market.get("base"):
            continue
        t = tickers.get(symbol) or {}
        qv = t.get("quoteVolume")
        if positive_finite(qv) and float(qv) >= min_volume:
            candidates.append((symbol, float(qv)))
    candidates.sort(key=lambda z: z[1], reverse=True)
    return [s for s, _ in candidates[:max_symbols]]


def init_state():
    defaults = {
        "cash": DEFAULT_CASH,
        "positions": {},
        "trades": [],
        "scan_rows": [],
        "scan_warnings": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def equity_estimate():
    total = float(st.session_state.cash)
    for p in st.session_state.positions.values():
        total += p["qty"] * p["entry"]
    return total


def paper_buy(row, risk_pct, max_position_pct, fee_pct, slippage_pct):
    symbol = row["symbol"]
    if symbol in st.session_state.positions:
        return "এই পেয়ার ইতিমধ্যে Paper Position-এ আছে"
    equity = equity_estimate()
    risk_cash = equity * risk_pct / 100
    entry = row["entry"] * (1 + slippage_pct / 100)
    stop = row["stop"]
    risk_per_unit = entry - stop
    if not positive_finite(risk_per_unit):
        return "Stop-loss ও Entry সঠিক নয়"
    qty_by_risk = risk_cash / risk_per_unit
    max_notional = equity * max_position_pct / 100
    qty_by_cap = max_notional / entry
    qty = min(qty_by_risk, qty_by_cap)
    cost = qty * entry * (1 + fee_pct / 100)
    if qty <= 0 or cost > st.session_state.cash:
        return "পর্যাপ্ত Paper Cash নেই"
    st.session_state.cash -= cost
    st.session_state.positions[symbol] = {
        **row, "entry": entry, "qty": qty,
        "entry_fee": qty * entry * fee_pct / 100,
        "opened_at": datetime.now(timezone.utc).isoformat(),
    }
    return f"Paper BUY সম্পন্ন: {symbol} | Qty {qty:.6g}"


def close_position(symbol, exit_price, fee_pct, slippage_pct, reason):
    p = st.session_state.positions.get(symbol)
    if not p:
        return
    exit_fill = exit_price * (1 - slippage_pct / 100)
    gross = p["qty"] * exit_fill
    exit_fee = gross * fee_pct / 100
    proceeds = gross - exit_fee
    pnl = proceeds - (p["qty"] * p["entry"]) - p["entry_fee"]
    st.session_state.cash += proceeds
    st.session_state.trades.append({
        "symbol": symbol, "signal": p["signal"], "entry": p["entry"],
        "exit": exit_fill, "qty": p["qty"], "net_pnl": pnl,
        "reason": reason, "closed_at": datetime.now(timezone.utc).isoformat(),
    })
    del st.session_state.positions[symbol]


def run_scan(exchange, settings):
    rows, warnings = [], []
    try:
        symbols = market_symbols(exchange, QUOTE, settings["min_volume"], settings["max_symbols"])
    except Exception as e:
        return [], [f"Market list error: {e}"]
    for symbol in symbols:
        try:
            ticker = fetch_ticker_checked(exchange, symbol, settings["max_age"], settings["max_spread"])
            ticker["symbol"] = symbol
            # BTC itself needs no BTC filter; altcoins need BTC 1h bullish.
            if symbol != "BTC/USDT":
                btc_bull, _, _ = get_1h_trend(exchange, "BTC/USDT")
                if not btc_bull:
                    continue
            trend_bull, _, _ = get_1h_trend(exchange, symbol)
            if not trend_bull:
                continue
            candles = fetch_ohlcv_frame(exchange, symbol, "5m", 220)
            row = analyze_signal(candles, trend_bull, ticker, settings)
            if row:
                rows.append(row)
        except (ccxt.NetworkError, ccxt.ExchangeError, ValueError, TypeError, KeyError) as e:
            warnings.append(f"{symbol}: {str(e)[:180]}")
        except Exception as e:
            warnings.append(f"{symbol}: unexpected: {str(e)[:180]}")
    rows.sort(key=lambda r: (r["signal"] == "BREAKOUT RETEST", -r["extension_atr"]))
    return rows, warnings


init_state()
st.title("📈 Crypto Scanner — 5m Entry + 1h Trend")
st.caption("Spot market | Paper Trading only | Live order/real-money execution নেই")
st.warning("এটি পরীক্ষামূলক টুল, লাভের নিশ্চয়তা নয়। প্রথমে Paper Trading-এ পরীক্ষা করুন।")

with st.sidebar:
    st.header("Scanner Settings")
    exchange_id = st.selectbox("Exchange", ["kucoin", "binance"], index=0)
    max_symbols = st.slider("একবারে সর্বোচ্চ কয়টি কয়েন স্ক্যান হবে", 10, 80, 30, 5)
    min_volume = st.number_input("Minimum 24h quote volume (USDT)", min_value=10000.0, value=500000.0, step=100000.0)
    max_age = st.number_input("Ticker max age (seconds)", min_value=5, max_value=300, value=45)
    max_spread = st.number_input("Maximum spread (%)", min_value=0.01, max_value=2.0, value=0.30, step=0.05)
    max_extension = st.slider("Anti-chase: EMA20 থেকে সর্বোচ্চ ATR", 0.25, 2.0, 0.75, 0.05)
    breakout_volume = st.slider("Breakout volume / 20 candle avg", 1.0, 3.0, 1.5, 0.1)
    atr_sl = st.slider("Stop-loss ATR multiplier", 1.0, 3.0, 1.5, 0.1)
    rr = st.slider("Risk : Reward target", 1.0, 4.0, 2.0, 0.1)
    st.divider()
    risk_pct = st.slider("প্রতি ট্রেডে equity risk (%)", 0.25, 1.0, 0.5, 0.05)
    max_position_pct = st.slider("সর্বোচ্চ position size (% equity)", 5, 30, 20, 1)
    fee_pct = st.number_input("Fee per side (%)", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
    slippage_pct = st.number_input("Estimated slippage (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    scan_button = st.button("🔎 এখনই Scan করুন", type="primary", use_container_width=True)
    reset_button = st.button("Paper Account Reset", use_container_width=True)

if reset_button:
    st.session_state.cash = DEFAULT_CASH
    st.session_state.positions = {}
    st.session_state.trades = []
    st.session_state.scan_rows = []
    st.session_state.scan_warnings = []
    st.success("Paper account reset হয়েছে।")

settings = {
    "max_symbols": max_symbols,
    "min_volume": min_volume,
    "max_age": int(max_age),
    "max_spread": float(max_spread),
    "max_extension_atr": float(max_extension),
    "breakout_volume": float(breakout_volume),
    "atr_sl": float(atr_sl),
    "rr": float(rr),
}

if scan_button:
    try:
        exchange = get_exchange(exchange_id)
        with st.spinner("লাইভ মার্কেট স্ক্যান হচ্ছে... API rate limit মেনে একে একে যাচাই করা হচ্ছে।"):
            found, warnings = run_scan(exchange, settings)
        st.session_state.scan_rows = found
        st.session_state.scan_warnings = warnings

        # Paper SL/TP monitoring: প্রতি Scan-এ fresh checked bid দিয়ে।
        for symbol, p in list(st.session_state.positions.items()):
            try:
                q = fetch_ticker_checked(exchange, symbol, int(max_age), float(max_spread))
                if q["bid"] <= p["stop"]:
                    close_position(symbol, q["bid"], fee_pct, slippage_pct, "STOP LOSS")
                elif q["bid"] >= p["target"]:
                    close_position(symbol, q["bid"], fee_pct, slippage_pct, "TAKE PROFIT")
            except Exception:
                pass
    except Exception as e:
        st.error(f"Exchange setup/scan error: {e}")

# Summary
c1, c2, c3 = st.columns(3)
c1.metric("Paper Cash (USDT)", f"{st.session_state.cash:,.2f}")
c2.metric("Open Positions", len(st.session_state.positions))
c3.metric("Closed Trades", len(st.session_state.trades))

st.subheader("BUY Signals")
if st.session_state.scan_rows:
    df_signals = pd.DataFrame(st.session_state.scan_rows)
    display_cols = ["symbol", "signal", "entry", "stop", "target", "rsi", "extension_atr", "spread_pct", "ticker_age"]
    st.dataframe(df_signals[display_cols], use_container_width=True, hide_index=True)
    st.caption("Signal তৈরি হয়েছে সম্পূর্ণ 5m candle, bullish 1h trend এবং fresh live Ask যাচাই করে।")
    st.markdown("**Paper entry:**")
    options = [r["symbol"] for r in st.session_state.scan_rows]
    selected = st.selectbox("যে সিগন্যালটি Paper Buy করবেন", options, key="paper_buy_symbol")
    if st.button("🧪 Selected signal Paper BUY"):
        row = next(r for r in st.session_state.scan_rows if r["symbol"] == selected)
        st.info(paper_buy(row, risk_pct, max_position_pct, fee_pct, slippage_pct))
else:
    st.info("এখনো কোনো BUY signal নেই। Anti-chase ও trend filter ইচ্ছাকৃতভাবে কঠোর রাখা হয়েছে।")

if st.session_state.scan_warnings:
    with st.expander(f"Scan warnings ({len(st.session_state.scan_warnings)})"):
        for warning in st.session_state.scan_warnings[:100]:
            st.write("• " + warning)

st.subheader("Open Paper Positions")
if st.session_state.positions:
    exchange_for_positions = get_exchange(exchange_id)
    for symbol, p in list(st.session_state.positions.items()):
        with st.container(border=True):
            st.write(f"**{symbol} — {p['signal']}**")
            a, b, c, d = st.columns(4)
            a.metric("Entry", f"{p['entry']:.8g}")
            b.metric("Stop", f"{p['stop']:.8g}")
            c.metric("Target", f"{p['target']:.8g}")
            d.metric("Quantity", f"{p['qty']:.6g}")
            try:
                quote = fetch_ticker_checked(exchange_for_positions, symbol, int(max_age), float(max_spread))
                unrealized = (quote["bid"] - p["entry"]) * p["qty"] - p["entry_fee"]
                st.write(f"Fresh Bid: `{quote['bid']:.8g}` | আনুমানিক unrealized P/L: `{unrealized:.4f} USDT`")
                if st.button(f"Close Paper Position: {symbol}", key=f"close_{symbol}"):
                    close_position(symbol, quote["bid"], fee_pct, slippage_pct, "MANUAL")
                    st.rerun()
            except Exception as e:
                st.warning(f"{symbol} live quote পাওয়া যায়নি: {e}")
else:
    st.caption("কোনো খোলা Paper Position নেই।")

st.subheader("Closed Paper Trades")
if st.session_state.trades:
    trades_df = pd.DataFrame(st.session_state.trades)
    st.dataframe(trades_df, use_container_width=True, hide_index=True)
    st.metric("মোট Net P/L (USDT)", f"{trades_df['net_pnl'].sum():.4f}")
    st.download_button(
        "Download trade history CSV",
        trades_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="paper_trade_history.csv",
        mime="text/csv",
    )
else:
    st.caption("এখনো কোনো Paper Trade বন্ধ হয়নি।")

st.divider()
st.caption(
    f"Version {APP_VERSION} | Paper state শুধু বর্তমান Streamlit session-এ থাকে; "
    "server restart/session reset হলে হারিয়ে যেতে পারে। Stop/Target monitoring শুধু Scan চাপলে হয়, "
    "এটি ২৪/৭ background monitoring নয়।"
)
