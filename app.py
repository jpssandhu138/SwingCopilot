"""
SwingCopilot AI | Intraday Alpha Engine  (v2 - paper-trading / evaluation build)

Daily workflow
  ~8:30 AM   Tab 1  Pre-market scan  -> V1 picks + V2 watchlist + Gemini news veto
  9:30-11:30 Tab 2  Entry check      -> which setups actually triggered, with entry/stop/target
  after 3:20 Tab 3  Square-off        -> resolves every signal candle-by-candle, logs forward test
  anytime    Tab 4  Backtest Lab      -> same logic replayed on the last ~60 days of 5-min data

All trading logic lives in engine.py so the backtest and the live forward test are identical.
No orders are placed anywhere - this app only measures how good the signals are.
"""
import os
import re
import json
import uuid
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import date
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import gspread
from PIL import Image
from google.oauth2.service_account import Credentials
from google import genai
from google.genai import types

import engine as eng

# ==========================================
# 1. PAGE CONFIG & SHEET SCHEMA
# ==========================================
st.set_page_config(page_title="SwingCopilot AI | Intraday Alpha Engine", page_icon="⚡", layout="wide")

# New tab names so your old data in intraday_recommendations / intraday_decisions is untouched
WATCH_TAB = "intraday_watchlist_v2"
DEC_TAB = "intraday_decisions_v2"
RESULTS_TAB = "intraday_results_v2"
BACKTEST_TAB = "backtest_runs"

WATCH_COLS = [
    "date_id", "generated_at", "risk_profile", "strategy", "ticker", "company_name", "as_of",
    "prev_close", "atr", "atr_pct", "pattern", "pattern_var", "rvol_prev_day", "rel_strength_20d",
    "news_count", "ai_status", "ai_sentiment_score", "ai_risk_detected", "ai_catalyst",
    "ai_catalyst_strength", "ai_thesis_summary", "passed_ai_filter",
]
DEC_COLS = [
    "decision_id", "date_id", "decision_timestamp", "risk_profile", "strategy", "ticker",
    "company_name", "entry", "stop", "target", "human_decision",
]
RESULT_COLS = [
    "date_id", "evaluated_at", "risk_profile", "strategy", "ticker", "company_name",
    "passed_ai_filter", "human_decision", "status", "reason", "entry_time", "entry", "stop",
    "target", "exit_time", "exit_price", "exit_reason", "gross_pct", "net_pct", "r_multiple",
    "ambiguous_bar", "gap_pct", "orb_rvol", "v1_logged_exit_reason", "v1_logged_net_pct",
]
CLOSED_REASONS = ("TARGET_HIT", "STOP_LOSS_HIT", "SQUARED_OFF")

# ==========================================
# 2. GOOGLE SHEETS HELPERS
# ==========================================
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]


@st.cache_resource
def get_spreadsheet():
    creds = Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]), scopes=SCOPES)
    return gspread.authorize(creds).open_by_url(st.secrets["GSHEET_URL"])


def read_tab_df(tab_name: str, expected_cols: list[str]) -> pd.DataFrame:
    sh = get_spreadsheet()
    try:
        records = sh.worksheet(tab_name).get_all_records()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=100, cols=len(expected_cols))
        ws.append_row(expected_cols)
        return pd.DataFrame(columns=expected_cols)
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=expected_cols)
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""
    return df[expected_cols]


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float) and not np.isfinite(v):
        return ""
    if isinstance(v, (pd.Timestamp,)):
        return v.strftime("%Y-%m-%d %H:%M")
    return str(v)


def write_tab_df(tab_name: str, df: pd.DataFrame):
    """Overwrites a tab. Resizes the grid first - the old code created tabs with 1000 rows,
    which the watchlist would have outgrown in ~6 weeks."""
    sh = get_spreadsheet()
    rows = [df.columns.tolist()] + [[_cell(v) for v in r] for r in df.itertuples(index=False)]
    try:
        ws = sh.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=len(rows) + 50, cols=max(1, len(df.columns)))
    ws.clear()
    ws.resize(rows=max(len(rows) + 50, 100), cols=max(len(df.columns), 1))
    ws.update(range_name="A1", values=rows)


def upsert_rows(tab: str, cols: list[str], new_df: pd.DataFrame, match: dict):
    """Replace all rows matching `match` (e.g. same date + profile) with new_df."""
    existing = read_tab_df(tab, cols)
    if not existing.empty:
        mask = np.ones(len(existing), dtype=bool)
        for k, v in match.items():
            mask &= existing[k].astype(str).values == str(v)
        existing = existing[~mask]
    combined = pd.concat([existing, new_df.reindex(columns=cols)], ignore_index=True)
    write_tab_df(tab, combined)


# ==========================================
# 3. MARKET DATA (cached)
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def cached_download(tickers: tuple, period: str, interval: str):
    return eng.download_ohlcv(list(tickers), period, interval)


def live_intraday(tickers: list[str], period: str, trade_date: date):
    """Fresh (uncached) 5-min data; drops the still-forming candle if trade_date is today."""
    intra = eng.download_ohlcv(list(dict.fromkeys(tickers + [eng.NIFTY_TICKER])), period, "5m")
    if trade_date >= eng.today_ist():
        intra = eng.drop_unclosed_bars(intra)
    return intra


# ==========================================
# 4. NEWS + GEMINI (used as a VETO, not as a price forecast)
# ==========================================
def fetch_recent_news(company_name: str, max_items: int = 6, max_age_hours: int = 36):
    """Google News RSS restricted to the last ~2 days, with publish times.
    The old version had no date filter, so Gemini was often reading week-old headlines."""
    name = re.sub(r"\(.*?\)", "", company_name).strip()
    q = urllib.parse.quote(f"{name} share when:2d")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        res = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(res.content)
    except Exception as e:
        return [], f"news fetch failed: {e}"
    items, now = [], eng.now_ist()
    for it in root.findall(".//item"):
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        try:
            ts = parsedate_to_datetime(it.findtext("pubDate")).astimezone(eng.IST)
        except Exception:
            ts = None
        if ts is not None and (now - ts).total_seconds() > max_age_hours * 3600:
            continue
        items.append({"title": title, "published": ts.strftime("%d %b %H:%M IST") if ts else "unknown"})
        if len(items) >= max_items:
            break
    return items, None


def _parse_json(text: str) -> dict:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    data = json.loads(t)
    if isinstance(data, list):
        data = data[0] if data else {}
    return data


def evaluate_with_gemini(api_key: str, model: str, ticker: str, company: str, setup: dict,
                         trade_date: str) -> dict:
    headlines, err = fetch_recent_news(company)
    base = {"news_count": len(headlines), "ai_sentiment_score": 0.0, "ai_risk_detected": False,
            "ai_catalyst": "None", "ai_catalyst_strength": 0}
    if err:
        return {**base, "ai_status": "ERROR", "ai_thesis_summary": err}
    if not headlines:
        return {**base, "ai_status": "NO_NEWS",
                "ai_thesis_summary": "No stock-specific news in the last 36h - technical setup only."}
    news_text = "\n".join(f"- [{h['published']}] {h['title']}" for h in headlines)
    prompt = f"""You are screening the Indian stock {company} ({ticker}) for a possible LONG intraday trade on {trade_date}.

Technical context from yesterday's daily candles (already computed, do not re-derive):
{json.dumps(setup, default=str)}

Headlines from the last ~36 hours, with publish times:
{news_text}

Your only job: decide whether the NEWS makes a long trade today riskier, or supports it.
Rules:
- Use only facts present in the headlines. Do not invent numbers or events.
- Generic market wraps, "stocks to watch" lists and price-movement headlines are NOT catalysts.
- News published before yesterday's close is probably already in the price; weight it lower.

Return strict JSON with exactly these keys:
{{
  "sentiment_score": float from -1.0 to 1.0,
  "risk_detected": true if fraud, regulatory action, downgrade, weak results, promoter/block selling, or anything likely to cause a gap-down,
  "catalyst": one of ["Earnings", "Order/Contract Win", "Regulatory", "Corporate Action", "Management/Guidance", "Sector/Macro", "None"],
  "catalyst_strength": integer 0-3 (0 = nothing, 3 = major company-specific event),
  "summary": one sentence
}}"""
    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
        )
        data = _parse_json(resp.text)
        return {
            **base,
            "ai_status": "OK",
            "ai_sentiment_score": round(float(np.clip(float(data.get("sentiment_score", 0.0)), -1, 1)), 2),
            "ai_risk_detected": bool(data.get("risk_detected", False)),
            "ai_catalyst": str(data.get("catalyst", "None")),
            "ai_catalyst_strength": int(data.get("catalyst_strength", 0) or 0),
            "ai_thesis_summary": str(data.get("summary", "")),
        }
    except Exception as e:
        # The old code silently returned a fake "sentiment 0.1" here, which hid API failures.
        return {**base, "ai_status": "ERROR", "ai_thesis_summary": f"Gemini call failed: {str(e)[:180]}"}


def ai_passes(ai: dict, min_sentiment: float) -> bool:
    if ai.get("ai_status") not in ("OK", "NO_NEWS"):
        return False
    return (not ai.get("ai_risk_detected", False)) and float(ai.get("ai_sentiment_score", 0)) >= min_sentiment


def as_bool(v) -> bool:
    return str(v).strip().lower() == "true"


# ==========================================
# 5. WORKFLOWS
# ==========================================
def run_premarket_scan(api_key: str, model: str, profile: str, trade_date: date):
    tickers = list(eng.MASTER_UNIVERSE) + [eng.NIFTY_TICKER]
    with st.spinner("Downloading 1 year of daily candles..."):
        daily = eng.download_ohlcv(tickers, "1y", "1d")
    watch, missing = eng.premarket_scan(profile, trade_date, daily)
    if watch.empty:
        return 0, missing
    prof = eng.PROFILES[profile]
    uniq = list(dict.fromkeys(watch["ticker"]))
    bar = st.progress(0.0, text="Checking news with Gemini...")
    ai_by_ticker = {}
    for i, t in enumerate(uniq):
        row = watch[watch["ticker"] == t].iloc[0]
        bar.progress((i + 1) / len(uniq), text=f"News check {i + 1}/{len(uniq)}: {row['company_name']}")
        setup = {k: row[k] for k in ("strategy", "prev_close", "atr_pct", "pattern", "rvol_prev_day", "rel_strength_20d")}
        ai_by_ticker[t] = evaluate_with_gemini(api_key, model, t, row["company_name"], setup, str(trade_date))
    bar.empty()
    out = []
    stamp = eng.now_ist().strftime("%Y-%m-%d %H:%M:%S")
    for _, r in watch.iterrows():
        ai = ai_by_ticker[r["ticker"]]
        out.append({**r.to_dict(), **ai, "date_id": str(trade_date), "generated_at": stamp,
                    "risk_profile": profile, "passed_ai_filter": ai_passes(ai, prof["min_sentiment"])})
    upsert_rows(WATCH_TAB, WATCH_COLS, pd.DataFrame(out), {"date_id": str(trade_date), "risk_profile": profile})
    return len(out), missing


def get_watch(trade_date: date, profile: str) -> pd.DataFrame:
    w = read_tab_df(WATCH_TAB, WATCH_COLS)
    if w.empty:
        return w
    return w[(w["date_id"].astype(str) == str(trade_date)) & (w["risk_profile"].astype(str) == profile)].reset_index(drop=True)


def get_decisions(trade_date: date, profile: str) -> pd.DataFrame:
    d = read_tab_df(DEC_TAB, DEC_COLS)
    if d.empty:
        return d
    return d[(d["date_id"].astype(str) == str(trade_date)) & (d["risk_profile"].astype(str) == profile)]


def log_decision(row: dict, profile: str, decision: str):
    entry = pd.DataFrame([{
        "decision_id": str(uuid.uuid4())[:8], "date_id": row["date_id"],
        "decision_timestamp": eng.now_ist().strftime("%Y-%m-%d %H:%M:%S"), "risk_profile": profile,
        "strategy": row["strategy"], "ticker": row["ticker"], "company_name": row["company_name"],
        "entry": row["entry"], "stop": row["stop"], "target": row["target"], "human_decision": decision,
    }])
    upsert_rows(DEC_TAB, DEC_COLS, entry, {"date_id": row["date_id"], "risk_profile": profile,
                                           "strategy": row["strategy"], "ticker": row["ticker"]})


def evaluate_day(trade_date: date, profile: str, cost_pct: float, period: str) -> pd.DataFrame:
    watch = get_watch(trade_date, profile)
    if watch.empty:
        return pd.DataFrame()
    intra = live_intraday(list(watch["ticker"].unique()), period, trade_date)
    res = eng.evaluate_watchlist(watch, trade_date, profile, cost_pct, intra)
    ai_cols = watch[["strategy", "ticker", "passed_ai_filter", "ai_catalyst", "ai_thesis_summary", "ai_status"]]
    res = res.merge(ai_cols, on=["strategy", "ticker"], how="left")
    dec = get_decisions(trade_date, profile)
    if not dec.empty:
        res = res.merge(dec[["strategy", "ticker", "human_decision"]], on=["strategy", "ticker"], how="left")
    else:
        res["human_decision"] = ""
    res["human_decision"] = res["human_decision"].fillna("")
    # make sure every column the UI / sheet expects exists (V1-only days have no ORB columns etc.)
    extra = ["orb_high", "orb_low", "ai_catalyst", "ai_thesis_summary", "ai_status"]
    return res.reindex(columns=list(dict.fromkeys(RESULT_COLS + list(res.columns) + extra)))


def forward_test_cohorts(results: pd.DataFrame) -> pd.DataFrame:
    """Turns logged results into cohorts: all signals / AI-approved / human YES, plus old V1 scoring."""
    r = results.copy()
    for c in ("net_pct", "r_multiple", "v1_logged_net_pct"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    closed = r[(r["status"].astype(str) == "TRIGGERED") & (r["exit_reason"].isin(CLOSED_REASONS))]
    parts = []
    for strat, label in (("V1", "V1 realistic"), ("V2", "V2 ORB")):
        s = closed[closed["strategy"].astype(str) == strat]
        parts.append(s.assign(cohort=f"{label} - all signals"))
        parts.append(s[s["passed_ai_filter"].map(as_bool)].assign(cohort=f"{label} - AI approved"))
        parts.append(s[s["human_decision"].astype(str) == "YES"].assign(cohort=f"{label} - Human YES"))
        if strat == "V1":
            old = s[s["v1_logged_exit_reason"].astype(str) != ""].copy()
            old["net_pct"] = old["v1_logged_net_pct"]
            old["exit_reason"] = old["v1_logged_exit_reason"]
            old["r_multiple"] = np.nan
            parts.insert(0, old.assign(cohort="V1 old scoring (what the old app reported)"))
    allc = pd.concat(parts, ignore_index=True)
    if allc.empty:
        return pd.DataFrame()
    out = eng.summarize_trades(allc, group_col="cohort", date_col="date_id")
    return out.rename(columns={"Strategy": "Cohort"})


# ==========================================
# 6. UI HELPERS
# ==========================================
def price_card(label: str, price_val, accent: str = "#FFFFFF", sub: str = "") -> str:
    try:
        num = float(price_val)
        txt = f"₹{num:,.1f}" if num >= 10000 else f"₹{num:,.2f}"
    except (ValueError, TypeError):
        txt = f"₹{price_val}"
    badge = f'<div style="font-size:0.72rem;color:{accent};opacity:0.85;margin-top:2px;">{sub}</div>' if sub else ""
    return f"""<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
        border-radius:8px;padding:8px 10px;"><div style="font-size:0.78rem;color:#9CA3AF;">{label}</div>
        <div style="font-size:1.2rem;font-weight:700;color:{accent};white-space:nowrap;">{txt}</div>{badge}</div>"""


def fmt_time(v) -> str:
    try:
        return pd.Timestamp(v).strftime("%H:%M")
    except Exception:
        return str(v or "")


st.markdown("<style>.block-container{padding-top:1.5rem;} header[data-testid='stHeader']{background:transparent;}</style>",
            unsafe_allow_html=True)
if os.path.exists("logo.png"):
    img = Image.open("logo.png")
    w, h = img.size
    st.image(img.crop((int(w * 0.08), int(h * 0.25), int(w * 0.90), int(h * 0.72))), width=420)
else:
    st.title("⚡ SwingCopilot AI — Intraday Signal Lab")
st.caption("Paper-trading evaluation build · no orders are placed · Google Sheets database")

# ==========================================
# 7. SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    gemini_api_key = st.text_input("Gemini API Key", value=st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", "")),
                                   type="password")
    gemini_model = st.text_input("Gemini model", value=st.secrets.get("GEMINI_MODEL", "gemini-3.8-flash"),
                                 help="If calls fail, the error now shows in the ai_status column instead of being hidden.")
    date_str = st.text_input("Trading Date (YYYY-MM-DD)", value=str(eng.today_ist()))
    try:
        active_date = date.fromisoformat(date_str.strip())
    except ValueError:
        st.error("Date must be YYYY-MM-DD")
        st.stop()
    if "GSHEET_URL" in st.secrets:
        st.link_button("🔗 Open Google Sheet DB", st.secrets["GSHEET_URL"], use_container_width=True)
    st.divider()
    selected_risk = st.select_slider("Risk profile", options=list(eng.PROFILES), value="Balanced")
    cost_pct = st.number_input("Round-trip cost + slippage (%)", 0.0, 1.0, 0.10, 0.01,
                               help="Brokerage, STT, exchange fees, GST, stamp duty and slippage for a buy+sell. ~0.08-0.12% is typical for intraday.")
    p = eng.PROFILES[selected_risk]
    v1_be = (p["v1_stop_loss_pct"] * 100 + cost_pct) / ((p["v1_stop_loss_pct"] + p["v1_target_pct"]) * 100)
    st.info(
        f"**V1 (original):** SL -{p['v1_stop_loss_pct']*100:.1f}% / TP +{p['v1_target_pct']*100:.1f}% · "
        f"break-even win rate ≈ {v1_be:.0%}\n\n"
        f"**V2 (ORB):** gap ≤ ±{p['v2_max_gap_pct']}% · opening RVOL ≥ {p['v2_min_orb_rvol']}x · "
        f"stop = opening-range low · target = {p['v2_reward_risk']}R · entries until "
        f"{p['v2_entry_cutoff'].strftime('%H:%M')} · max {p['v2_max_trades_per_day']}/day · "
        f"break-even win rate ≈ {1 / (1 + p['v2_reward_risk']):.0%} + costs"
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌅 1. Pre-Market Scan", "⏱️ 2. Entry Check", "🏁 3. Square-Off & Forward Test",
    "🧪 4. Backtest Lab", "🗂️ 5. Sheets Inspector",
])

# ------------------------------------------
# TAB 1: PRE-MARKET SCAN
# ------------------------------------------
with tab1:
    c_h, c_b = st.columns([3, 1])
    with c_h:
        st.subheader(f"Pre-market scan for {active_date} ({selected_risk})")
        st.write("Builds the **watchlist** from yesterday's daily candles and checks overnight news. "
                 "Nothing here is an entry yet - entries are confirmed after the open in Tab 2.")
    with c_b:
        if st.button("🚀 Run Pre-Market Scan", type="primary", use_container_width=True):
            if not gemini_api_key:
                st.error("Please configure your Gemini API key.")
            else:
                n, missing = run_premarket_scan(gemini_api_key, gemini_model, selected_risk, active_date)
                st.success(f"Saved {n} watchlist rows to `{WATCH_TAB}`.")
                if missing:
                    st.warning("No Yahoo data for: " + ", ".join(missing) + " (renamed or delisted symbol?)")
    watch_df = get_watch(active_date, selected_risk)
    if watch_df.empty:
        st.warning("No scan saved for this date/profile yet.")
    else:
        errs = (watch_df["ai_status"].astype(str) == "ERROR").sum()
        if errs:
            st.error(f"{errs} Gemini/news checks failed - see ai_thesis_summary. Those rows are marked as not AI-approved.")
        show = ["ticker", "company_name", "prev_close", "atr_pct", "pattern", "pattern_var", "rvol_prev_day",
                "rel_strength_20d", "ai_status", "ai_sentiment_score", "ai_risk_detected", "ai_catalyst",
                "passed_ai_filter", "ai_thesis_summary"]
        for strat, title in (("V1", "V1 picks (original pattern logic)"), ("V2", "V2 watchlist (trend + strength)")):
            sub = watch_df[watch_df["strategy"].astype(str) == strat]
            st.markdown(f"#### {title} — {len(sub)} stocks")
            st.dataframe(sub[show], width="stretch", hide_index=True)

# ------------------------------------------
# TAB 2: ENTRY CHECK
# ------------------------------------------
with tab2:
    st.subheader("Live entry check")
    st.write("Run any time between **9:30 and the entry cutoff**. V1 picks are entered at the 9:15 open. "
             "V2 stocks only trigger when the opening-range high breaks with price above VWAP, strong "
             "opening volume and Nifty above its open. Uses closed 5-min candles only.")
    if st.button("⏱️ Check triggers now", type="primary"):
        with st.spinner("Fetching 5-min candles..."):
            sig = evaluate_day(active_date, selected_risk, cost_pct, "1mo")
        st.session_state["signals"] = {"key": (str(active_date), selected_risk), "df": sig,
                                       "at": eng.now_ist().strftime("%H:%M:%S")}
    s = st.session_state.get("signals")
    if not s or s["key"] != (str(active_date), selected_risk):
        st.info("Press the button to check triggers.")
    elif s["df"].empty:
        st.warning("No watchlist for this date/profile. Run the pre-market scan first.")
    else:
        sig = s["df"]
        st.caption(f"Last checked {s['at']} IST")
        trig = sig[sig["status"] == "TRIGGERED"]
        st.markdown(f"### 🎯 Triggered setups ({len(trig)})")
        for i, row in trig.iterrows():
            risk = float(row["entry"]) - float(row["stop"])
            reward = float(row["target"]) - float(row["entry"])
            p_tp = eng.no_edge_tp_probability(risk, reward)
            with st.container(border=True):
                c1, c2, c3 = st.columns([2.2, 2.6, 1.4])
                with c1:
                    ai_ok = as_bool(row.get("passed_ai_filter"))
                    st.markdown(f"#### {row['company_name']} (`{row['ticker']}`) · {row['strategy']}")
                    st.caption(f"Entry {fmt_time(row['entry_time'])} · gap {row.get('gap_pct')}% · "
                               f"opening RVOL {row.get('orb_rvol')}x · AI: {'✅ approved' if ai_ok else '⚠️ veto/unknown'} "
                               f"({row.get('ai_catalyst', '')})")
                    st.write(row.get("ai_thesis_summary", ""))
                    if row["exit_reason"]:
                        st.write(f"Current status: **{row['exit_reason']}** ({row['net_pct']}% net)")
                with c2:
                    m1, m2, m3 = st.columns(3)
                    m1.markdown(price_card("Entry", row["entry"], "#F9FAFB", fmt_time(row["entry_time"])), unsafe_allow_html=True)
                    m2.markdown(price_card("Stop", row["stop"], "#F87171", f"-{risk / float(row['entry']) * 100:.2f}%"), unsafe_allow_html=True)
                    m3.markdown(price_card("Target", row["target"], "#34D399", f"no-edge P(TP) {p_tp:.0%}"), unsafe_allow_html=True)
                with c3:
                    st.markdown(f"**Paper trade?** {row.get('human_decision') or 'PENDING'}")
                    b1, b2 = st.columns(2)
                    if b1.button("YES ✅", key=f"y_{row['strategy']}_{row['ticker']}", use_container_width=True):
                        log_decision(row.to_dict(), selected_risk, "YES")
                        sig.loc[i, "human_decision"] = "YES"
                        st.rerun()
                    if b2.button("NO ❌", key=f"n_{row['strategy']}_{row['ticker']}", use_container_width=True):
                        log_decision(row.to_dict(), selected_risk, "NO")
                        sig.loc[i, "human_decision"] = "NO"
                        st.rerun()
        rest = sig[sig["status"] != "TRIGGERED"]
        with st.expander(f"Waiting / rejected ({len(rest)})", expanded=True):
            st.dataframe(rest[["strategy", "ticker", "company_name", "status", "reason", "gap_pct", "orb_rvol",
                               "orb_high", "orb_low"]], width="stretch", hide_index=True)

# ------------------------------------------
# TAB 3: SQUARE-OFF & FORWARD TEST
# ------------------------------------------
with tab3:
    st.subheader("Square-off & forward-test log")
    st.write("Run after 3:20 PM. Every signal is resolved candle by candle (stop vs target in the order they "
             "actually happened), squared off at 15:15, and costs are deducted. Works for any date in the last ~55 days.")
    if st.button("🏁 Run square-off for this date", type="primary"):
        with st.spinner("Resolving trades..."):
            res = evaluate_day(active_date, selected_risk, cost_pct, "60d")
        if res.empty:
            st.error("No watchlist saved for this date/profile.")
        else:
            res["risk_profile"] = selected_risk
            res["evaluated_at"] = eng.now_ist().strftime("%Y-%m-%d %H:%M:%S")
            upsert_rows(RESULTS_TAB, RESULT_COLS, res, {"date_id": str(active_date), "risk_profile": selected_risk})
            still_open = (res["exit_reason"] == "OPEN").sum()
            st.success(f"Logged {len(res)} rows to `{RESULTS_TAB}`." +
                       (f" {still_open} trades still open - re-run after close." if still_open else ""))
            st.dataframe(res[["strategy", "ticker", "status", "entry_time", "entry", "stop", "target", "exit_time",
                              "exit_reason", "net_pct", "r_multiple", "passed_ai_filter", "human_decision",
                              "v1_logged_exit_reason", "reason"]], width="stretch", hide_index=True)

    st.markdown("### 📈 Cumulative forward test")
    all_res = read_tab_df(RESULTS_TAB, RESULT_COLS)
    all_res = all_res[all_res["risk_profile"].astype(str) == selected_risk] if not all_res.empty else all_res
    if all_res.empty:
        st.info("No forward-test results yet.")
    else:
        cohorts = forward_test_cohorts(all_res)
        st.caption(f"{all_res['date_id'].nunique()} trading days logged. "
                   "The 95% range shows how uncertain the win rate still is - it narrows as trades accumulate.")
        st.dataframe(cohorts, width="stretch", hide_index=True)

# ------------------------------------------
# TAB 4: BACKTEST LAB
# ------------------------------------------
with tab4:
    st.subheader("Backtest Lab")
    st.write("Replays every strategy day by day over the last ~60 days of 5-min data (Yahoo's limit). "
             "Signals for each day use only data available before that moment. "
             "The Gemini news filter can't be backtested (no historical news snapshot) - it's measured in Tab 3.")
    if st.button("🧪 Run backtest", type="primary"):
        tickers = tuple(list(eng.MASTER_UNIVERSE) + [eng.NIFTY_TICKER])
        with st.spinner("Downloading daily + 5-min history (cached for 1 hour)..."):
            daily = cached_download(tickers, "1y", "1d")
            intra = cached_download(tickers, "60d", "5m")
        prog = st.progress(0.0, text="Simulating...")
        bt = eng.run_backtest(selected_risk, cost_pct, daily, intra,
                              progress_cb=lambda x: prog.progress(x, text=f"Simulating... {x:.0%}"))
        prog.empty()
        st.session_state["bt"] = {"profile": selected_risk, "cost": cost_pct, "res": bt}

    bt_state = st.session_state.get("bt")
    if bt_state:
        bt = bt_state["res"]
        st.caption(f"Profile **{bt_state['profile']}** · cost {bt_state['cost']}% · {bt['sessions']} sessions "
                   f"({bt['period'][0]} → {bt['period'][1]})")
        if bt["missing"]:
            st.warning("No data for: " + ", ".join(bt["missing"]))
        if bt["summary"].empty:
            st.error("No trades generated.")
        else:
            st.markdown("#### Strategy comparison")
            st.dataframe(bt["summary"], width="stretch", hide_index=True)
            st.caption("Judge strategies on **Expectancy_R** and **Avg_Net_Return_Pct** (after costs), not win rate alone. "
                       "A strategy only has an edge if it beats the 'all stocks' baseline.")
            st.markdown("#### Equity curve (cumulative %, equal capital per day)")
            st.line_chart(bt["equity"])
            tr = bt["trades"]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Exit reasons (% of trades)")
                st.dataframe((pd.crosstab(tr["strategy"], tr["exit_reason"], normalize="index") * 100).round(1), width="stretch")
            with c2:
                rej = bt["rejections"]
                if not rej.empty:
                    st.markdown("#### Why V2 skipped watchlist stocks")
                    reasons = rej["reason"].str.replace(r"[-+]?\d+(\.\d+)?", "#", regex=True)
                    st.dataframe(reasons.value_counts().rename("count"), width="stretch")
            st.markdown("#### Trade log")
            pick = st.selectbox("Strategy", ["All"] + list(tr["strategy"].unique()))
            view = tr if pick == "All" else tr[tr["strategy"] == pick]
            st.dataframe(view, width="stretch", hide_index=True)
            st.download_button("⬇️ Download trades CSV", view.to_csv(index=False).encode(),
                               file_name=f"backtest_{bt_state['profile'].replace(' ', '')}.csv", mime="text/csv")
            if st.button("💾 Save summary to Google Sheet"):
                sm = bt["summary"].copy()
                sm.insert(0, "period", f"{bt['period'][0]} to {bt['period'][1]}")
                sm.insert(0, "cost_pct", bt_state["cost"])
                sm.insert(0, "risk_profile", bt_state["profile"])
                sm.insert(0, "run_at", eng.now_ist().strftime("%Y-%m-%d %H:%M"))
                old = read_tab_df(BACKTEST_TAB, list(sm.columns))
                write_tab_df(BACKTEST_TAB, pd.concat([old, sm], ignore_index=True))
                st.success(f"Saved to `{BACKTEST_TAB}`.")

# ------------------------------------------
# TAB 5: INSPECTOR
# ------------------------------------------
with tab5:
    for tab, cols in ((WATCH_TAB, WATCH_COLS), (DEC_TAB, DEC_COLS), (RESULTS_TAB, RESULT_COLS)):
        st.markdown(f"### 📄 `{tab}`")
        st.dataframe(read_tab_df(tab, cols), width="stretch", hide_index=True)
