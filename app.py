MAX_TICKER_AGE_SECONDS = 30
ESTIMATED_SLIPPAGE_PCT = 0.0005  # 0.05% per side
MAX_ENTRY_EXTENSION_ATR = 0.5

def get_live_quote(symbol: str):
    ticker = exchange.fetch_ticker(symbol)

    last = ticker.get("last")
    bid = ticker.get("bid")
    ask = ticker.get("ask")
    timestamp = ticker.get("timestamp")

    if not all(is_finite_positive(x) for x in (last, bid, ask)):
        raise ValueError("Live last/bid/ask price missing or invalid.")

    last, bid, ask = float(last), float(bid), float(ask)

    if ask < bid:
        raise ValueError("Invalid crossed bid/ask.")

    if timestamp is None:
        raise ValueError("Ticker timestamp missing. Trade cancelled.")

    age_ms = exchange.milliseconds() - int(timestamp)

    if age_ms < -5000 or age_ms > MAX_TICKER_AGE_SECONDS * 1000:
        raise ValueError(
            f"Ticker stale: {age_ms / 1000:.1f} seconds."
        )

    midpoint = (bid + ask) / 2
    spread_pct = (ask - bid) / midpoint * 100

    if spread_pct > max_spread_pct:
        raise ValueError(
            f"Spread too high: {spread_pct:.3f}%."
        )

    return {
        "last": last,
        "bid": bid,
        "ask": ask,
        "spread_pct": spread_pct,
        "age_seconds": age_ms / 1000,
    }


def get_latest_price(symbol: str, side: str = "sell") -> float:
    quote = get_live_quote(symbol)
    return quote["ask"] if side == "buy" else quote["bid"]
