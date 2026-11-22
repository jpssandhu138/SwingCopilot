"""
engine.py
---------
Strategy, simulation and backtest engine for SwingCopilot AI (Intraday).

There is deliberately NO Streamlit / Google Sheets code in this file, so the
*same* logic is used by every part of the app:

    1. Pre-market scan (~8:30 AM)      -> premarket_scan()
    2. Live entry check (after 9:30)   -> evaluate_watchlist()  (partial day)
    3. Market-close square-off         -> evaluate_watchlist()  (full day)
    4. Backtest Lab (last ~60 days)    -> run_backtest()

Because the forward test and the backtest share one code path, their numbers
are directly comparable.

Strategies
~~~~~~~~~~
V1_AS_LOGGED   Your original logic, scored exactly the way the old app scored
               it (entry = previous close, stop checked before target, no costs).
V1_OPEN_ENTRY  Same V1 picks, but realistic: buy at the 9:15 open, walk the
               5-min candles in time order, square off 15:15, costs deducted.
UNIVERSE_OPEN  Baseline: buy EVERY stock at the 9:15 open with the V1 bracket.
               If a strategy can't beat this, it has no edge.
V2_ORB         New logic: daily-trend watchlist before the open, then enter only
               on an opening-range breakout confirmed by VWAP, opening RVOL and
               Nifty. Stop at the opening-range low, target = R multiple.
"""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ==========================================
# CONSTANTS
# ==========================================
IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
LAST_BAR_START = time(15, 25)
SQUARE_OFF_BAR_START = time(15, 10)   # 5-min bar 15:10-15:15 -> exit at its close (15:15)
BAR_MINUTES = 5
NIFTY_TICKER = "^NSEI"

STRATEGY_LABELS = {
    "V1_AS_LOGGED": "V1 original (prev-close entry, old scoring, no costs)",
    "V1_OPEN_ENTRY": "V1 picks, realistic (9:15 open entry, costs)",
    "UNIVERSE_OPEN": "Baseline: all stocks at 9:15 open, V1 bracket",
    "V2_ORB": "V2: ORB + VWAP + RVOL + Nifty confirmation",
}

# NOTE: Zomato was renamed Eternal (ETERNAL.NS) and Tata Motors' passenger
# vehicle business now trades as TMPV.NS after the 2025 demerger. If Yahoo
# returns nothing for a ticker it is reported as "missing" in the app, so
# you can spot stale symbols.
MASTER_UNIVERSE = {
    # --- Large Cap & Index Heavyweights ---
    "RELIANCE.NS": "Reliance Industries", "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank", "INFY.NS": "Infosys", "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India", "BHARTIARTL.NS": "Bharti Airtel",
    "LT.NS": "Larsen & Toubro", "ITC.NS": "ITC Limited", "AXISBANK.NS": "Axis Bank",
    "KOTAKBANK.NS": "Kotak Mahindra Bank", "ASIANPAINT.NS": "Asian Paints",
    "MARUTI.NS": "Maruti Suzuki", "SUNPHARMA.NS": "Sun Pharmaceutical",
    "TITAN.NS": "Titan Company", "BAJFINANCE.NS": "Bajaj Finance",
    "TMPV.NS": "Tata Motors Passenger Vehicles", "M&M.NS": "Mahindra & Mahindra",
    "HCLTECH.NS": "HCL Technologies", "ULTRACEMCO.NS": "UltraTech Cement",
    "NTPC.NS": "NTPC Limited", "POWERGRID.NS": "Power Grid Corp",
    "ONGC.NS": "Oil & Natural Gas Corp", "JSWSTEEL.NS": "JSW Steel",
    "TATASTEEL.NS": "Tata Steel",
    # --- Mid-Cap Compounders & Growth Leaders ---
    "ETERNAL.NS": "Eternal (Zomato)", "DLF.NS": "DLF Limited",
    "HAL.NS": "Hindustan Aeronautics", "BEL.NS": "Bharat Electronics",
    "TRENT.NS": "Trent Limited", "SIEMENS.NS": "Siemens India",
    "INDHOTEL.NS": "Indian Hotels Co", "CHOLAFIN.NS": "Cholamandalam Investment",
    "PIDILITIND.NS": "Pidilite Industries", "ABB.NS": "ABB India",
    "HAVELLS.NS": "Havells India", "BANKBARODA.NS": "Bank of Baroda",
    "PNB.NS": "Punjab National Bank", "TVSMOTOR.NS": "TVS Motor Company",
    "PERSISTENT.NS": "Persistent Systems", "COFORGE.NS": "Coforge Limited",
    "DIXON.NS": "Dixon Technologies", "POLYCAB.NS": "Polycab India",
    "ASHOKLEY.NS": "Ashok Leyland", "VOLTAS.NS": "Voltas Limited",
    # --- High-Beta Momentum, PSU & Breakout Stocks ---
    "SUZLON.NS": "Suzlon Energy", "CDSL.NS": "Central Depository Services",
    "BSE.NS": "BSE Limited", "MCX.NS": "Multi Commodity Exchange",
    "ANGELONE.NS": "Angel One", "SONACOMS.NS": "Sona BLW Precision",
    "KAYNES.NS": "Kaynes Technology", "DATAPATTNS.NS": "Data Patterns India",
    "IRCTC.NS": "IRCTC Limited", "RVNL.NS": "Rail Vikas Nigam",
    "MAZDOCK.NS": "Mazagon Dock Shipbuilders", "COCHINSHIP.NS": "Cochin Shipyard",
    "TITAGARH.NS": "Titagarh Rail Systems", "KALYANKJIL.NS": "Kalyan Jewellers",
    "NYKAA.NS": "FSN E-Commerce Ventures",
}

PROFILES = {
    "Conservative": {
        # V1 (original logic, unchanged so it can be compared)
        "v1_stop_loss_pct": 0.005, "v1_target_pct": 0.012,
        "v1_max_variance": 0.12, "v1_min_rvol": 1.5,
        # V2 pre-market watchlist
        "v2_watchlist_size": 12, "v2_require_nifty_uptrend": True,
        "v2_min_atr_pct": 1.2, "v2_max_atr_pct": 5.0,
        # V2 live entry rules
        "v2_orb_minutes": 15, "v2_max_gap_pct": 1.5, "v2_min_orb_rvol": 2.0,
        "v2_orb_close_pos": 0.66, "v2_max_risk_atr": 0.50, "v2_min_risk_atr": 0.20,
        "v2_reward_risk": 1.5, "v2_entry_cutoff": time(11, 0), "v2_max_trades_per_day": 3,
        # AI news veto: skip if sentiment below this or risk detected
        "min_sentiment": 0.0,
    },
    "Balanced": {
        "v1_stop_loss_pct": 0.007, "v1_target_pct": 0.018,
        "v1_max_variance": 0.20, "v1_min_rvol": 1.2,
        "v2_watchlist_size": 15, "v2_require_nifty_uptrend": False,
        "v2_min_atr_pct": 1.0, "v2_max_atr_pct": 6.0,
        "v2_orb_minutes": 15, "v2_max_gap_pct": 2.5, "v2_min_orb_rvol": 1.5,
        "v2_orb_close_pos": 0.60, "v2_max_risk_atr": 0.60, "v2_min_risk_atr": 0.20,
        "v2_reward_risk": 1.5, "v2_entry_cutoff": time(11, 30), "v2_max_trades_per_day": 5,
        "min_sentiment": -0.2,
    },
    "Aggressive Breakout": {
        "v1_stop_loss_pct": 0.010, "v1_target_pct": 0.025,
        "v1_max_variance": 0.35, "v1_min_rvol": 1.0,
        "v2_watchlist_size": 20, "v2_require_nifty_uptrend": False,
        "v2_min_atr_pct": 0.8, "v2_max_atr_pct": 8.0,
        "v2_orb_minutes": 15, "v2_max_gap_pct": 4.0, "v2_min_orb_rvol": 1.2,
        "v2_orb_close_pos": 0.50, "v2_max_risk_atr": 0.80, "v2_min_risk_atr": 0.15,
        "v2_reward_risk": 2.0, "v2_entry_cutoff": time(12, 30), "v2_max_trades_per_day": 8,
        "min_sentiment": -0.4,
    },
}

# Normalised 10-point shapes (original library, kept for V1)
TEMPLATE_PATTERNS = {
    "Bull Flag Breakout": [0.3, 0.5, 0.8, 0.9, 0.85, 0.82, 0.83, 0.84, 0.92, 1.0],
    "Opening Range Breakout (ORB)": [0.5, 0.52, 0.51, 0.53, 0.65, 0.75, 0.8, 0.88, 0.95, 1.0],
    "Cup and Handle Continuation": [0.8, 0.6, 0.4, 0.42, 0.6, 0.8, 0.82, 0.78, 0.85, 1.0],
    "V-Bottom Reversal": [1.0, 0.8, 0.5, 0.2, 0.1, 0.3, 0.6, 0.8, 0.9, 1.0],
    "Consolidation Range": [0.5, 0.52, 0.48, 0.51, 0.49, 0.5, 0.52, 0.49, 0.51, 0.5],
    "Bearish Breakdown": [0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1],
}


# ==========================================
# SMALL HELPERS
# ==========================================
def now_ist() -> datetime:
    return datetime.now(IST)


def today_ist() -> date:
    return now_ist().date()


def _add_minutes(t: time, minutes: int) -> time:
    return (datetime.combine(date(2000, 1, 1), t) + timedelta(minutes=minutes)).time()


def _f(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def no_edge_tp_probability(risk: float, reward: float) -> float:
    """P(target before stop) for a driftless random walk = risk / (risk + reward).
    This is what you'd get by coin-flipping. A strategy must beat it (after costs)."""
    if risk <= 0 or reward <= 0:
        return np.nan
    return risk / (risk + reward)


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a win rate (in %). Shows how uncertain small samples are."""
    if n == 0:
        return (np.nan, np.nan)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half) * 100, min(1.0, centre + half) * 100)


# ==========================================
# DATA LOADING (yfinance)
# ==========================================
def _ohlcv_from_batch(batch: pd.DataFrame, ticker: str) -> pd.DataFrame:
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if batch is None or batch.empty:
        return pd.DataFrame()
    cols = batch.columns
    if isinstance(cols, pd.MultiIndex):
        if ticker in cols.get_level_values(0):
            df = batch[ticker]
        elif ticker in cols.get_level_values(1):
            df = batch.xs(ticker, axis=1, level=1)
        else:
            return pd.DataFrame()
    else:
        df = batch
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    df = df[needed].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df["Volume"] = df["Volume"].fillna(0)
    return df


def download_ohlcv(tickers, period: str, interval: str) -> dict[str, pd.DataFrame]:
    """Returns {ticker: OHLCV DataFrame}.
    * Daily data: tz-naive midnight index.
    * Intraday data: IST tz-aware index, limited to 9:15-15:25 bar starts.
    auto_adjust=False so daily closes are on the same (unadjusted) price scale as
    intraday candles. The old code used adjusted closes as the entry price."""
    import yfinance as yf  # imported lazily so the engine can be unit-tested offline

    tickers = list(dict.fromkeys(tickers))
    batch = yf.download(
        tickers, period=period, interval=interval, group_by="ticker",
        auto_adjust=False, progress=False, threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = _ohlcv_from_batch(batch, t)
        if df.empty:
            continue
        idx = pd.DatetimeIndex(df.index)
        if interval.endswith("d") or interval.endswith("wk") or interval.endswith("mo"):
            if idx.tz is not None:
                idx = idx.tz_convert(IST).tz_localize(None)
            df.index = idx.normalize()
            df = df[~df.index.duplicated(keep="last")]
        else:
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            df.index = idx.tz_convert(IST)
            tod = np.array([ts.time() for ts in df.index])
            df = df[(tod >= MARKET_OPEN) & (tod <= LAST_BAR_START)]
        if not df.empty:
            out[t] = df.sort_index()
    return out


def split_days(intra: pd.DataFrame | None) -> dict[date, pd.DataFrame]:
    if intra is None or intra.empty:
        return {}
    return {d: g for d, g in intra.groupby(intra.index.date)}


def drop_unclosed_bars(intra: dict[str, pd.DataFrame], now: datetime | None = None) -> dict[str, pd.DataFrame]:
    """During market hours Yahoo includes the candle that is still forming. Using it would
    peek at prices that don't exist yet, so live checks only use fully closed candles."""
    now = now or now_ist()
    cut = pd.Timestamp(now) - pd.Timedelta(minutes=BAR_MINUTES)
    return {t: df[df.index <= cut] for t, df in intra.items()}


# ==========================================
# PRE-MARKET FEATURES (daily candles, no look-ahead)
# ==========================================
def match_pattern(values) -> tuple[str, float]:
    """Original V1 shape matcher: MSE between min-max normalised last 10 closes and templates."""
    arr = np.asarray(values, dtype=float)[-10:]
    if len(arr) < 10 or np.isnan(arr).any():
        return "General Momentum", 0.5
    lo, hi = arr.min(), arr.max()
    norm = np.full_like(arr, 0.5) if hi == lo else (arr - lo) / (hi - lo)
    best, best_v = "General Momentum", float("inf")
    for name, tpl in TEMPLATE_PATTERNS.items():
        v = float(np.mean((norm - np.asarray(tpl)) ** 2))
        if v < best_v:
            best, best_v = name, v
    return best, round(best_v, 4)


def is_v1_bullish(pattern) -> bool:
    p = str(pattern or "")
    return ("Bull" in p) or ("ORB" in p) or ("V-Bottom" in p)


def compute_daily_features(daily: pd.DataFrame | None) -> pd.DataFrame | None:
    """Every feature in row t uses only data up to and including day t.
    To trade day D we always read the last row strictly BEFORE D (see row_before)."""
    if daily is None or len(daily) < 30:
        return None
    d = daily.copy()
    c = d["Close"]
    d["ema20"] = c.ewm(span=20, adjust=False).mean()
    d["ema50"] = c.ewm(span=50, adjust=False).mean()
    prev_c = c.shift(1)
    tr = pd.concat(
        [d["High"] - d["Low"], (d["High"] - prev_c).abs(), (d["Low"] - prev_c).abs()], axis=1
    ).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()
    d["atr_pct"] = d["atr14"] / c * 100
    # Original RVOL definition: last day's volume / 20-day mean (including that day)
    vol = d["Volume"].replace(0, np.nan)
    d["rvol_v1"] = vol / vol.rolling(20, min_periods=15).mean()
    d["ret_20d"] = c.pct_change(20) * 100
    vals = c.to_numpy(dtype=float)
    pats, pvars = [], []
    for i in range(len(vals)):
        if i < 9:
            pats.append(None)
            pvars.append(np.nan)
        else:
            p, v = match_pattern(vals[i - 9:i + 1])
            pats.append(p)
            pvars.append(v)
    d["pattern"] = pats
    d["pattern_var"] = pvars
    return d


def row_before(feats: pd.DataFrame | None, d: date) -> pd.Series | None:
    """Last completed daily row strictly before trading day d."""
    if feats is None or feats.empty:
        return None
    pos = feats.index.searchsorted(pd.Timestamp(d)) - 1
    if pos < 0:
        return None
    if (pd.Timestamp(d) - feats.index[pos]).days > 7:
        return None
    return feats.iloc[pos]


def select_v1(rows: dict[str, pd.Series | None], prof: dict) -> list[dict]:
    """Original V1 selection: bullish template, variance <= max, RVOL >= min; top 10 by lowest variance."""
    out = []
    for t, r in rows.items():
        if r is None:
            continue
        var, rv, pat = _f(r["pattern_var"]), _f(r["rvol_v1"]), r["pattern"]
        if np.isnan(var) or np.isnan(rv) or pat is None:
            continue
        if var <= prof["v1_max_variance"] and rv >= prof["v1_min_rvol"] and is_v1_bullish(pat):
            out.append({
                "strategy": "V1", "ticker": t, "as_of": str(r.name.date()),
                "prev_close": round(_f(r["Close"]), 2), "atr": round(_f(r["atr14"]), 2),
                "atr_pct": round(_f(r["atr_pct"]), 2), "pattern": pat, "pattern_var": var,
                "rvol_prev_day": round(rv, 2), "rel_strength_20d": np.nan,
            })
    out.sort(key=lambda x: x["pattern_var"])
    return out[:10]


def select_v2(rows: dict[str, pd.Series | None], nifty_row: pd.Series | None, prof: dict) -> list[dict]:
    """V2 pre-market watchlist: stocks in a clean daily uptrend, not over-extended, with
    enough daily range to be worth trading, ranked by 20-day strength vs Nifty."""
    if prof["v2_require_nifty_uptrend"] and nifty_row is not None:
        if not (_f(nifty_row["Close"]) > _f(nifty_row["ema20"])):
            return []   # market in a downtrend -> no long setups today
    n_ret = _f(nifty_row["ret_20d"], 0.0) if nifty_row is not None else 0.0
    out = []
    for t, r in rows.items():
        if r is None:
            continue
        c, e20, e50 = _f(r["Close"]), _f(r["ema20"]), _f(r["ema50"])
        atr_pct, ret20 = _f(r["atr_pct"]), _f(r["ret_20d"])
        if any(np.isnan(x) for x in (c, e20, e50, atr_pct, ret20)):
            continue
        if not (c > e20 > e50):
            continue
        if not (prof["v2_min_atr_pct"] <= atr_pct <= prof["v2_max_atr_pct"]):
            continue
        if (c / e20 - 1) * 100 > 2.5 * atr_pct:     # too stretched above the 20 EMA
            continue
        pat, var = r["pattern"], _f(r["pattern_var"])
        out.append({
            "strategy": "V2", "ticker": t, "as_of": str(r.name.date()),
            "prev_close": round(c, 2), "atr": round(_f(r["atr14"]), 2), "atr_pct": round(atr_pct, 2),
            "pattern": pat, "pattern_var": var, "rvol_prev_day": round(_f(r["rvol_v1"]), 2),
            "rel_strength_20d": round(ret20 - n_ret, 2),
        })
    out.sort(key=lambda x: x["rel_strength_20d"], reverse=True)
    return out[: prof["v2_watchlist_size"]]


def orb_volume_baseline(day_groups: dict[date, pd.DataFrame], orb_minutes: int,
                        lookback: int = 10, min_days: int = 5) -> dict[date, float]:
    """Average opening-range volume over the PREVIOUS `lookback` sessions (same time of day)."""
    if not day_groups:
        return {}
    orb_end = _add_minutes(MARKET_OPEN, orb_minutes)
    days = sorted(day_groups)
    vols = []
    for d in days:
        g = day_groups[d]
        m = np.array([ts.time() < orb_end for ts in g.index])
        vols.append(float(g["Volume"].to_numpy()[m].sum()) if m.any() else np.nan)
    s = pd.Series(vols, index=days).replace(0, np.nan)
    base = s.shift(1).rolling(lookback, min_periods=min_days).mean()
    return {d: _f(v) for d, v in base.items()}


# ==========================================
# INTRADAY SIMULATION
# ==========================================
def _arrays(bars: pd.DataFrame):
    return (
        bars.index,
        np.array([ts.time() for ts in bars.index]),
        bars["Open"].to_numpy(float), bars["High"].to_numpy(float),
        bars["Low"].to_numpy(float), bars["Close"].to_numpy(float),
        bars["Volume"].to_numpy(float),
    )


def day_is_complete(bars: pd.DataFrame | None) -> bool:
    if bars is None or bars.empty:
        return False
    return any(ts.time() >= SQUARE_OFF_BAR_START for ts in bars.index)


def _resolve(times, tod, o, h, l, c, start: int, entry: float, stop: float, target: float,
             complete: bool) -> dict:
    """Walk 5-min candles IN TIME ORDER from `start`. First barrier touched wins.
    If one candle touches both, we assume the stop (conservative).
    Exit at 15:15 close if neither is hit."""
    last = None
    for i in range(start, len(times)):
        if tod[i] > SQUARE_OFF_BAR_START:
            break
        last = i
        hit_sl, hit_tp = l[i] <= stop, h[i] >= target
        if hit_sl:
            px = min(stop, o[i]) if i > start else stop   # gap through stop -> fill at open
            return {"exit_time": times[i], "exit_price": px, "exit_reason": "STOP_LOSS_HIT",
                    "ambiguous_bar": bool(hit_tp)}
        if hit_tp:
            return {"exit_time": times[i], "exit_price": target, "exit_reason": "TARGET_HIT",
                    "ambiguous_bar": False}
    if last is None:
        return {"exit_time": None, "exit_price": entry, "exit_reason": "OPEN", "ambiguous_bar": False}
    return {"exit_time": times[last], "exit_price": c[last],
            "exit_reason": "SQUARED_OFF" if complete else "OPEN", "ambiguous_bar": False}


def _result(status: str, reason: str, *, entry_time=None, entry=np.nan, stop=np.nan,
            target=np.nan, exit: dict | None = None, cost_pct: float = 0.0, **extra) -> dict:
    out = {
        "status": status, "reason": reason, "entry_time": entry_time,
        "entry": round(_f(entry), 2), "stop": round(_f(stop), 2), "target": round(_f(target), 2),
        "exit_time": None, "exit_price": np.nan, "exit_reason": "", "gross_pct": np.nan,
        "net_pct": np.nan, "r_multiple": np.nan, "ambiguous_bar": False,
    }
    if exit is not None and entry and not np.isnan(_f(entry)):
        out.update(exit)
        gross = (exit["exit_price"] - entry) / entry * 100
        net = gross - cost_pct
        risk = entry - stop
        out["exit_price"] = round(_f(exit["exit_price"]), 2)
        out["gross_pct"] = round(gross, 3)
        out["net_pct"] = round(net, 3)
        out["r_multiple"] = round((net / 100 * entry) / risk, 3) if risk > 0 else np.nan
    out.update(extra)
    return out


def simulate_v1_as_logged(bars: pd.DataFrame, prev_close: float, sl_pct: float, tp_pct: float) -> dict:
    """Exact replica of the old evaluate_intraday_squareoff(): entry = previous close,
    stop checked before target using the whole day's low/high, no costs."""
    if bars is None or bars.empty:
        return _result("NO_DATA", "No intraday bars")
    entry = prev_close
    stop, target = entry * (1 - sl_pct), entry * (1 + tp_pct)
    day_low, day_high = float(bars["Low"].min()), float(bars["High"].max())
    close = float(bars["Close"].iloc[-1])
    if day_low <= stop:
        ex = {"exit_price": stop, "exit_reason": "STOP_LOSS_HIT"}
    elif day_high >= target:
        ex = {"exit_price": target, "exit_reason": "TARGET_HIT"}
    else:
        ex = {"exit_price": close, "exit_reason": "SQUARED_OFF"}
    ex.update({"exit_time": None, "ambiguous_bar": False})
    gap = (float(bars["Open"].iloc[0]) / prev_close - 1) * 100
    return _result("TRIGGERED", "Old logic: bought at previous close", entry_time="prev close",
                   entry=entry, stop=stop, target=target, exit=ex, cost_pct=0.0, gap_pct=round(gap, 2))


def simulate_open_entry(bars: pd.DataFrame, prev_close: float, sl_pct: float, tp_pct: float,
                        cost_pct: float) -> dict:
    """Buy at the 9:15 open, fixed % bracket, chronological resolution."""
    if bars is None or bars.empty:
        return _result("NO_DATA", "No intraday bars")
    times, tod, o, h, l, c, _ = _arrays(bars)
    entry = o[0]
    stop, target = entry * (1 - sl_pct), entry * (1 + tp_pct)
    ex = _resolve(times, tod, o, h, l, c, 0, entry, stop, target, day_is_complete(bars))
    gap = (entry / prev_close - 1) * 100 if prev_close else np.nan
    return _result("TRIGGERED", "Bought at 9:15 open", entry_time=times[0], entry=entry, stop=stop,
                   target=target, exit=ex, cost_pct=cost_pct, gap_pct=round(gap, 2))


def simulate_v2(bars: pd.DataFrame, prev_close: float, atr: float, orb_avg_vol: float,
                nifty_bars: pd.DataFrame | None, prof: dict, cost_pct: float) -> dict:
    """V2 entry logic. Every condition is checked on candles that have already closed
    at the moment of entry (no look-ahead), so it works identically live and in backtest."""
    info = {"gap_pct": np.nan, "orb_high": np.nan, "orb_low": np.nan, "orb_rvol": np.nan}
    if bars is None or bars.empty:
        return _result("WAITING", "No candles yet today", **info)
    complete = day_is_complete(bars)
    times, tod, o, h, l, c, v = _arrays(bars)
    orb_end = _add_minutes(MARKET_OPEN, prof["v2_orb_minutes"])
    orb_idx = np.where(tod < orb_end)[0]
    need = prof["v2_orb_minutes"] // BAR_MINUTES
    if len(orb_idx) < need:
        if complete:
            return _result("NO_DATA", "Opening-range candles missing", **info)
        return _result("WAITING", f"Opening range forms until {orb_end.strftime('%H:%M')}", **info)

    day_open = o[orb_idx[0]]
    orb_high, orb_low, orb_close = h[orb_idx].max(), l[orb_idx].min(), c[orb_idx[-1]]
    gap_pct = (day_open / prev_close - 1) * 100 if prev_close else np.nan
    orb_avg = _f(orb_avg_vol)
    orb_rvol = v[orb_idx].sum() / orb_avg if orb_avg > 0 else np.nan
    info.update(gap_pct=round(gap_pct, 2), orb_high=round(orb_high, 2), orb_low=round(orb_low, 2),
                orb_rvol=round(orb_rvol, 2) if not np.isnan(orb_rvol) else np.nan)

    def reject(msg):
        return _result("REJECTED", msg, **info)

    if np.isnan(gap_pct) or abs(gap_pct) > prof["v2_max_gap_pct"]:
        return reject(f"Gap {gap_pct:+.2f}% outside ±{prof['v2_max_gap_pct']}% (big gaps tend to fade)")
    rng = orb_high - orb_low
    if rng <= 0:
        return reject("Flat opening range")
    pos = (orb_close - orb_low) / rng
    if pos < prof["v2_orb_close_pos"]:
        return reject(f"Opening range closed weak ({pos:.0%} of range)")
    if np.isnan(orb_rvol):
        return reject("Not enough history for opening RVOL")
    if orb_rvol < prof["v2_min_orb_rvol"]:
        return reject(f"Opening RVOL {orb_rvol:.2f}x < {prof['v2_min_orb_rvol']}x")
    atr = _f(atr)
    if not atr > 0:
        return reject("ATR unavailable")
    if rng > prof["v2_max_risk_atr"] * atr:
        return reject(f"Opening range too wide ({rng / atr:.2f}x ATR) - stop would be too far")

    stop = min(orb_low, orb_high - prof["v2_min_risk_atr"] * atr)
    typical = (h + l + c) / 3
    cum_v = np.cumsum(v)
    vwap = np.where(cum_v > 0, np.cumsum(typical * v) / np.maximum(cum_v, 1e-9), c)

    n_close, n_open = None, None
    if nifty_bars is not None and not nifty_bars.empty:
        n_close = nifty_bars["Close"]
        n_open = float(nifty_bars["Open"].iloc[0])

    def nifty_ok(ts) -> bool:
        if n_close is None:
            return True
        val = n_close.asof(ts)
        return True if pd.isna(val) else float(val) >= n_open

    cutoff = prof["v2_entry_cutoff"]
    rr = prof["v2_reward_risk"]
    max_chase = 0.5 * (orb_high - stop)
    for j in range(orb_idx[-1] + 1, len(times)):
        if tod[j] > cutoff:
            break
        if h[j] <= orb_high:
            continue
        if c[j - 1] < vwap[j - 1]:          # last closed candle must be above VWAP
            continue
        if not nifty_ok(times[j - 1]):      # Nifty must be above its own open
            continue
        entry = max(o[j], orb_high)
        if entry - orb_high > max_chase:    # already ran away - don't chase
            continue
        risk = entry - stop
        target = entry + rr * risk
        ex = _resolve(times, tod, o, h, l, c, j, entry, stop, target, complete)
        return _result("TRIGGERED", "ORB breakout above VWAP with RVOL and Nifty support",
                       entry_time=times[j], entry=entry, stop=stop, target=target, exit=ex,
                       cost_pct=cost_pct, **info)

    if not complete and tod[-1] < cutoff:
        return _result("WAITING", f"Setup valid; waiting for break above {orb_high:.2f} "
                                  f"until {cutoff.strftime('%H:%M')}", **info)
    return _result("NO_TRIGGER", "No valid breakout before cutoff", **info)


# ==========================================
# LIVE: PRE-MARKET SCAN + WATCHLIST EVALUATION
# ==========================================
def premarket_scan(profile_name: str, trade_date: date, daily: dict[str, pd.DataFrame],
                   universe: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Builds V1 picks and V2 watchlist for `trade_date` from daily candles before that date."""
    prof = PROFILES[profile_name]
    universe = universe or MASTER_UNIVERSE
    feats = {t: compute_daily_features(daily.get(t)) for t in list(universe) + [NIFTY_TICKER]}
    missing = [t for t in universe if feats.get(t) is None]
    rows = {t: row_before(feats[t], trade_date) for t in universe}
    nifty_row = row_before(feats.get(NIFTY_TICKER), trade_date)
    picks = select_v1(rows, prof) + select_v2(rows, nifty_row, prof)
    df = pd.DataFrame(picks)
    if not df.empty:
        df.insert(2, "company_name", df["ticker"].map(universe))
    return df, missing


def evaluate_watchlist(watch: pd.DataFrame, trade_date: date, profile_name: str, cost_pct: float,
                       intra: dict[str, pd.DataFrame], universe: dict | None = None) -> pd.DataFrame:
    """Runs the entry + exit simulation for a saved watchlist on `trade_date`.
    Works on a partial day (live entry check) and a full day (square-off)."""
    prof = PROFILES[profile_name]
    universe = universe or MASTER_UNIVERSE
    nifty_bars = split_days(intra.get(NIFTY_TICKER)).get(trade_date)
    out, v2_hits = [], []
    for _, w in watch.iterrows():
        t = str(w["ticker"])
        groups = split_days(intra.get(t))
        bars = groups.get(trade_date)
        prev_close = _f(w["prev_close"])
        base = {"date_id": str(trade_date), "strategy": w["strategy"], "ticker": t,
                "company_name": universe.get(t, w.get("company_name", t))}
        if str(w["strategy"]) == "V1":
            if bars is None or bars.empty:
                out.append({**base, **_result("WAITING" if trade_date >= today_ist() else "NO_DATA",
                                                "No intraday candles yet")})
                continue
            res = simulate_open_entry(bars, prev_close, prof["v1_stop_loss_pct"],
                                      prof["v1_target_pct"], cost_pct)
            old = simulate_v1_as_logged(bars, prev_close, prof["v1_stop_loss_pct"], prof["v1_target_pct"])
            res["v1_logged_exit_reason"] = old["exit_reason"] if day_is_complete(bars) else ""
            res["v1_logged_net_pct"] = old["net_pct"] if day_is_complete(bars) else np.nan
            out.append({**base, **res})
        else:
            orb_base = orb_volume_baseline(groups, prof["v2_orb_minutes"]).get(trade_date, np.nan)
            res = simulate_v2(bars, prev_close, _f(w["atr"]), orb_base, nifty_bars, prof, cost_pct)
            row = {**base, **res}
            out.append(row)
            if res["status"] == "TRIGGERED":
                v2_hits.append(row)
    # Daily trade cap for V2: keep the earliest triggers
    v2_hits.sort(key=lambda r: r["entry_time"])
    for r in v2_hits[prof["v2_max_trades_per_day"]:]:
        r["status"] = "SKIPPED_LIMIT"
        r["reason"] = f"Daily cap of {prof['v2_max_trades_per_day']} V2 trades reached"
    return pd.DataFrame(out)


# ==========================================
# BACKTEST
# ==========================================
def run_backtest(profile_name: str, cost_pct: float, daily: dict[str, pd.DataFrame],
                 intra: dict[str, pd.DataFrame], universe: dict | None = None,
                 warmup_sessions: int = 10, progress_cb=None) -> dict:
    """Day-by-day walk-forward simulation of all strategies over the available 5-min history.
    The AI news filter cannot be backtested (no historical news snapshot) - it is measured
    in the forward test instead."""
    prof = PROFILES[profile_name]
    universe = universe or MASTER_UNIVERSE
    tickers = list(universe)
    feats = {t: compute_daily_features(daily.get(t)) for t in tickers + [NIFTY_TICKER]}
    days_by_t = {t: split_days(intra.get(t)) for t in tickers if intra.get(t) is not None}
    nifty_days = split_days(intra.get(NIFTY_TICKER))
    orb_base = {t: orb_volume_baseline(g, prof["v2_orb_minutes"]) for t, g in days_by_t.items()}
    missing = [t for t in tickers if feats.get(t) is None or t not in days_by_t]

    all_days = sorted(set().union(*[set(g) for g in days_by_t.values()])) if days_by_t else []
    test_days = [d for d in all_days[warmup_sessions:]]
    trades, rejections = [], []

    def add(strategy, d, t, res, **extra):
        trades.append({"date": str(d), "strategy": strategy, "ticker": t,
                       "company_name": universe.get(t, t), **extra, **res})

    for k, d in enumerate(test_days):
        rows = {t: row_before(feats.get(t), d) for t in tickers}
        nifty_row = row_before(feats.get(NIFTY_TICKER), d)

        def bars_for(t):
            b = days_by_t.get(t, {}).get(d)
            return b if day_is_complete(b) else None    # skip incomplete sessions (e.g. today)

        # ---- V1 (original picks) ----
        for p in select_v1(rows, prof):
            b = bars_for(p["ticker"])
            if b is None:
                continue
            extra = {"pattern": p["pattern"], "rvol_prev_day": p["rvol_prev_day"]}
            add("V1_AS_LOGGED", d, p["ticker"],
                simulate_v1_as_logged(b, p["prev_close"], prof["v1_stop_loss_pct"], prof["v1_target_pct"]), **extra)
            add("V1_OPEN_ENTRY", d, p["ticker"],
                simulate_open_entry(b, p["prev_close"], prof["v1_stop_loss_pct"], prof["v1_target_pct"], cost_pct), **extra)

        # ---- Baseline: every stock ----
        for t in tickers:
            b, r = bars_for(t), rows.get(t)
            if b is None or r is None:
                continue
            add("UNIVERSE_OPEN", d, t, simulate_open_entry(
                b, _f(r["Close"]), prof["v1_stop_loss_pct"], prof["v1_target_pct"], cost_pct))

        # ---- V2 ----
        hits = []
        for w in select_v2(rows, nifty_row, prof):
            t = w["ticker"]
            b = bars_for(t)
            if b is None:
                continue
            res = simulate_v2(b, w["prev_close"], w["atr"], orb_base.get(t, {}).get(d, np.nan),
                              nifty_days.get(d), prof, cost_pct)
            if res["status"] == "TRIGGERED":
                hits.append((res, w))
            else:
                rejections.append({"date": str(d), "ticker": t, "status": res["status"], "reason": res["reason"]})
        hits.sort(key=lambda x: x[0]["entry_time"])
        for i, (res, w) in enumerate(hits):
            if i >= prof["v2_max_trades_per_day"]:
                rejections.append({"date": str(d), "ticker": w["ticker"], "status": "SKIPPED_LIMIT",
                                   "reason": "Daily trade cap reached"})
                continue
            add("V2_ORB", d, w["ticker"], res, rel_strength_20d=w["rel_strength_20d"])

        if progress_cb:
            progress_cb((k + 1) / max(1, len(test_days)))

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df[trades_df["status"] == "TRIGGERED"].reset_index(drop=True)
    return {
        "trades": trades_df,
        "summary": summarize_trades(trades_df) if not trades_df.empty else pd.DataFrame(),
        "equity": equity_curve(trades_df) if not trades_df.empty else pd.DataFrame(),
        "rejections": pd.DataFrame(rejections),
        "missing": missing,
        "period": (str(test_days[0]), str(test_days[-1])) if test_days else ("", ""),
        "sessions": len(test_days),
    }


# ==========================================
# METRICS
# ==========================================
def summarize_trades(trades: pd.DataFrame, group_col: str = "strategy", date_col: str = "date") -> pd.DataFrame:
    rows = []
    if trades is None or trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    for col in ("net_pct", "r_multiple"):
        t[col] = pd.to_numeric(t[col], errors="coerce")
    t = t.dropna(subset=["net_pct"])
    order = [k for k in STRATEGY_LABELS if k in set(t[group_col])] + \
            [k for k in t[group_col].unique() if k not in STRATEGY_LABELS]
    for name in order:
        g = t[t[group_col] == name]
        n = len(g)
        if n == 0:
            continue
        net = g["net_pct"]
        wins = int((net > 0).sum())
        lo, hi = wilson_interval(wins, n)
        pos, neg = net[net > 0].sum(), -net[net < 0].sum()
        daily = g.groupby(date_col)["net_pct"].mean()
        eq = daily.sort_index().cumsum()
        dd = float((eq - eq.cummax()).min()) if len(eq) else np.nan
        reasons = g["exit_reason"].astype(str)
        rows.append({
            "Strategy": STRATEGY_LABELS.get(name, name),
            "Trades": n,
            "Days_Traded": int(g[date_col].nunique()),
            "Win_Rate_Pct": round(wins / n * 100, 1),
            "Win_Rate_95pct_Range": f"{lo:.0f}-{hi:.0f}%",
            "Target_Hit_Pct": round((reasons == "TARGET_HIT").mean() * 100, 1),
            "Stop_Hit_Pct": round((reasons == "STOP_LOSS_HIT").mean() * 100, 1),
            "Squared_Off_Pct": round((reasons == "SQUARED_OFF").mean() * 100, 1),
            "Avg_Net_Return_Pct": round(net.mean(), 3),
            "Avg_Win_Pct": round(net[net > 0].mean(), 3) if wins else 0.0,
            "Avg_Loss_Pct": round(net[net <= 0].mean(), 3) if wins < n else 0.0,
            "Profit_Factor": round(pos / neg, 2) if neg > 0 else np.inf,
            "Expectancy_R": round(g["r_multiple"].mean(), 3),
            "Cum_Return_Pct": round(float(eq.iloc[-1]), 2) if len(eq) else 0.0,
            "Max_Drawdown_Pct": round(dd, 2),
        })
    return pd.DataFrame(rows)


def equity_curve(trades: pd.DataFrame, group_col: str = "strategy", date_col: str = "date") -> pd.DataFrame:
    """Cumulative % return, equal capital split across that day's trades for each strategy."""
    t = trades.copy()
    t["net_pct"] = pd.to_numeric(t["net_pct"], errors="coerce")
    daily = t.pivot_table(index=date_col, columns=group_col, values="net_pct", aggfunc="mean")
    daily = daily.sort_index().fillna(0.0).cumsum()
    daily.columns = [STRATEGY_LABELS.get(c, c) for c in daily.columns]
    return daily
