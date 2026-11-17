import os
import math
import uuid
import json
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import gspread
from PIL import Image
from google.oauth2.service_account import Credentials
from google import genai
from google.genai import types

# ==========================================
# 1. PAGE CONFIG & CONSTANTS
# ==========================================
st.set_page_config(
    page_title="SwingCopilot AI | Intraday Alpha Engine",
    page_icon="⚡",
    layout="wide"
)

INTRADAY_REC_TAB = "intraday_recommendations"
INTRADAY_DEC_TAB = "intraday_decisions"

INTRADAY_REC_COLS = [
    "date_id", "generated_at", "risk_profile", "ticker", "company_name",
    "matched_pattern", "pattern_variance", "rvol_score", "prob_hit_sl_pct", "prob_hit_tp_pct",
    "buy_price", "stop_loss_price", "target_price",
    "ai_sentiment_score", "ai_risk_detected", "ai_catalyst", "ai_thesis_summary", "passed_ai_filter"
]

INTRADAY_DEC_COLS = [
    "decision_id", "date_id", "decision_timestamp", "risk_profile",
    "ticker", "company_name", "buy_price", "stop_loss_price", "target_price", "human_decision"
]

# ==========================================
# 2. ONE-TIME PRE-DEFINED TECHNICAL PATTERNS (LOCAL LIBRARY)
# ==========================================
# Normalized price arrays (0 to 1 scale over 10 periods) for shape matching / variance calculation
TEMPLATE_PATTERNS = {
    "Bull Flag Breakout": [0.3, 0.5, 0.8, 0.9, 0.85, 0.82, 0.83, 0.84, 0.92, 1.0],
    "Opening Range Breakout (ORB)": [0.5, 0.52, 0.51, 0.53, 0.65, 0.75, 0.8, 0.88, 0.95, 1.0],
    "Cup and Handle Continuation": [0.8, 0.6, 0.4, 0.42, 0.6, 0.8, 0.82, 0.78, 0.85, 1.0],
    "V-Bottom Reversal": [1.0, 0.8, 0.5, 0.2, 0.1, 0.3, 0.6, 0.8, 0.9, 1.0],
    "Consolidation Range": [0.5, 0.52, 0.48, 0.51, 0.49, 0.5, 0.52, 0.49, 0.51, 0.5],
    "Bearish Breakdown": [0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
}

def match_institutional_pattern(price_series: pd.Series) -> tuple[str, float]:
    """
    Computes variance (Euclidean distance) between normalized stock price action
    and pre-defined institutional templates. Returns the best-matching pattern and variance score.
    """
    try:
        arr = price_series.tail(10).values
        if len(arr) < 10:
            return "General Momentum", 0.5
        
        # Normalize between 0 and 1
        min_v, max_v = np.min(arr), np.max(arr)
        if max_v == min_v:
            norm_arr = np.ones_like(arr) * 0.5
        else:
            norm_arr = (arr - min_v) / (max_v - min_v)

        best_match = "General Momentum"
        min_variance = float("inf")

        for pat_name, template_vals in TEMPLATE_PATTERNS.items():
            template_arr = np.array(template_vals)
            variance = float(np.mean((norm_arr - template_arr) ** 2))
            if variance < min_variance:
                min_variance = variance
                best_match = pat_name

        return best_match, round(min_variance, 4)
    except Exception:
        return "General Momentum", 0.5

# ==========================================
# 3. GOOGLE SHEETS CLOUD DATABASE HELPERS
# ==========================================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

@st.cache_resource
def get_spreadsheet():
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=SCOPES
    )
    client = gspread.authorize(creds)
    return client.open_by_url(st.secrets["GSHEET_URL"])

def read_tab_df(tab_name: str, expected_cols: list[str]) -> pd.DataFrame:
    sh = get_spreadsheet()
    try:
        ws = sh.worksheet(tab_name)
        records = ws.get_all_records()
        if not records:
            return pd.DataFrame(columns=expected_cols)
        df = pd.DataFrame(records)
        for col in expected_cols:
            if col not in df.columns:
                df[col] = ""
        return df[expected_cols]
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=1000, cols=len(expected_cols))
        ws.append_row(expected_cols)
        return pd.DataFrame(columns=expected_cols)

def write_tab_df(tab_name: str, df: pd.DataFrame):
    sh = get_spreadsheet()
    try:
        ws = sh.worksheet(tab_name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=1000, cols=max(1, len(df.columns)))

    clean_df = df.fillna("")
    rows = [clean_df.columns.tolist()] + clean_df.astype(str).values.tolist()
    ws.update(range_name="A1", values=rows)

def write_squareoff_report_tab(tab_name: str, summary_df: pd.DataFrame, audit_df: pd.DataFrame):
    sh = get_spreadsheet()
    try:
        ws = sh.worksheet(tab_name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=200, cols=16)

    rows = [["INTRADAY COHORT PERFORMANCE SUMMARY"]]
    rows.append(summary_df.columns.tolist())
    rows.extend(summary_df.fillna("").astype(str).values.tolist())
    rows.append([])
    rows.append(["INTRADAY TRADE-BY-TRADE RESOLUTION AUDIT"])
    rows.append(audit_df.columns.tolist())
    rows.extend(audit_df.fillna("").astype(str).values.tolist())

    ws.update(range_name="A1", values=rows)

# ==========================================
# 4. MASTER UNIVERSE & INTRADAY RISK PROFILES
# ==========================================
MASTER_UNIVERSE = {
    # --- Large Cap & Index Heavyweights ---
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "LT.NS": "Larsen & Toubro",
    "ITC.NS": "ITC Limited",
    "AXISBANK.NS": "Axis Bank",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ASIANPAINT.NS": "Asian Paints",
    "MARUTI.NS": "Maruti Suzuki",
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "TITAN.NS": "Titan Company",
    "BAJFINANCE.NS": "Bajaj Finance",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "TATAMOTORS.NS": "Tata Motors",
    "M&M.NS": "Mahindra & Mahindra",
    "HCLTECH.NS": "HCL Technologies",
    "WIPRO.NS": "Wipro Limited",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "NTPC.NS": "NTPC Limited",
    "POWERGRID.NS": "Power Grid Corp",
    "ONGC.NS": "Oil & Natural Gas Corp",
    "COALINDIA.NS": "Coal India",
    "CIPLA.NS": "Cipla Limited",
    "NESTLEIND.NS": "Nestle India",
    "BRITANNIA.NS": "Britannia Industries",
    "TECHM.NS": "Tech Mahindra",
    "DRREDDY.NS": "Dr. Reddy's Laboratories",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "EICHERMOT.NS": "Eicher Motors",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "DIVISLAB.NS": "Divi's Laboratories",
    "JSWSTEEL.NS": "JSW Steel",
    "TATASTEEL.NS": "Tata Steel",
    "HINDALCO.NS": "Hindalco Industries",
    "TATACONSUM.NS": "Tata Consumer Products",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "ADANIPORTS.NS": "Adani Ports & SEZ",
    "ADANIENT.NS": "Adani Enterprises",
    "SHRIRAMFIN.NS": "Shriram Finance",
    "LTIM.NS": "LTIMindtree",

    # --- Mid-Cap Compounders & Growth Leaders ---
    "ZOMATO.NS": "Zomato",
    "DLF.NS": "DLF Limited",
    "HAL.NS": "Hindustan Aeronautics",
    "BEL.NS": "Bharat Electronics",
    "TRENT.NS": "Trent Limited",
    "SIEMENS.NS": "Siemens India",
    "INDHOTEL.NS": "Indian Hotels Co",
    "CHOLAFIN.NS": "Cholamandalam Investment",
    "PIDILITIND.NS": "Pidilite Industries",
    "ABB.NS": "ABB India",
    "HAVELLS.NS": "Havells India",
    "GODREJCP.NS": "Godrej Consumer Products",
    "DABUR.NS": "Dabur India",
    "MARICO.NS": "Marico Limited",
    "AMBUJACEM.NS": "Ambuja Cements",
    "BANKBARODA.NS": "Bank of Baroda",
    "PNB.NS": "Punjab National Bank",
    "CANBK.NS": "Canara Bank",
    "TVSMOTOR.NS": "TVS Motor Company",
    "MOTHERSON.NS": "Samvardhana Motherson",
    "PERSISTENT.NS": "Persistent Systems",
    "COFORGE.NS": "Coforge Limited",
    "DIXON.NS": "Dixon Technologies",
    "POLYCAB.NS": "Polycab India",
    "KEI.NS": "KEI Industries",
    "KPITTECH.NS": "KPIT Technologies",
    "FEDERALBNK.NS": "Federal Bank",
    "IDFCFIRSTB.NS": "IDFC First Bank",
    "CUMMINSIND.NS": "Cummins India",
    "MAXHEALTH.NS": "Max Healthcare",
    "LUPIN.NS": "Lupin Limited",
    "AUROPHARMA.NS": "Aurobindo Pharma",
    "ASHOKLEY.NS": "Ashok Leyland",
    "BHARATFORG.NS": "Bharat Forge",
    "ASTRAL.NS": "Astral Limited",
    "VOLTAS.NS": "Voltas Limited",
    "JUBLFOOD.NS": "Jubilant FoodWorks",
    "OBEROIRLTY.NS": "Oberoi Realty",
    "PRESTIGE.NS": "Prestige Estates",
    "MPHASIS.NS": "Mphasis Limited",

    # --- High-Beta Small-Cap, PSU & Momentum Breakouts ---
    "SUZLON.NS": "Suzlon Energy",
    "CDSL.NS": "Central Depository Services",
    "BSE.NS": "BSE Limited",
    "MCX.NS": "Multi Commodity Exchange",
    "ANGELONE.NS": "Angel One",
    "SONACOMS.NS": "Sona BLW Precision",
    "KAYNES.NS": "Kaynes Technology",
    "DATAPATTNS.NS": "Data Patterns India",
    "TEJASNET.NS": "Tejas Networks",
    "IEX.NS": "Indian Energy Exchange",
    "IRCTC.NS": "IRCTC Limited",
    "RVNL.NS": "Rail Vikas Nigam",
    "MAZDOCK.NS": "Mazagon Dock Shipbuilders",
    "COCHINSHIP.NS": "Cochin Shipyard",
    "TITAGARH.NS": "Titagarh Rail Systems",
    "KALYANKJIL.NS": "Kalyan Jewellers",
    "NYKAA.NS": "FSN E-Commerce Ventures",
    "PAYTM.NS": "One 97 Communications",
    "POLICYBZR.NS": "PB Fintech"
}

INTRADAY_RISK_PROFILES = {
    "Conservative": {
        "stop_loss_pct": 0.005,   # -0.5% tight intraday SL
        "target_pct": 0.012,      # +1.2% intraday TP
        "max_variance": 0.12,     # Strict pattern fit required
        "min_rvol": 1.5,          # High volume confirmation
        "min_sentiment": 0.40
    },
    "Balanced": {
        "stop_loss_pct": 0.007,   # -0.7% SL
        "target_pct": 0.018,      # +1.8% TP
        "max_variance": 0.20,
        "min_rvol": 1.2,
        "min_sentiment": 0.20
    },
    "Aggressive Breakout": {
        "stop_loss_pct": 0.010,   # -1.0% SL
        "target_pct": 0.025,      # +2.5% TP
        "max_variance": 0.35,
        "min_rvol": 1.0,
        "min_sentiment": 0.00
    }
}

def get_current_date_id() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# ==========================================
# 5. NEWS SCRAPER & GEMINI DRIFT EXTRACTION
# ==========================================
def fetch_indian_stock_news(company_name: str) -> list[str]:
    try:
        query = urllib.parse.quote(f"{company_name} share news India")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        res = requests.get(rss_url, timeout=6)
        root = ET.fromstring(res.content)
        return [item.find('title').text for item in root.findall('.//item')[:5] if item.find('title') is not None]
    except Exception:
        return []

def evaluate_intraday_with_gemini(api_key: str, ticker: str, company_name: str, pattern: str, variance: float, rvol: float) -> dict:
    headlines = fetch_indian_stock_news(company_name)
    news_text = "\n".join([f"- {h}" for h in headlines]) if headlines else "No major overnight news."

    client = genai.Client(api_key=api_key)
    prompt = f"""
    You are an AI Intraday Quantitative Trader evaluating Indian stock '{company_name}' ({ticker}) for next-day intraday trading.
    Matched Technical Pattern: {pattern} (Variance score: {variance})
    Relative Volume (RVOL): {rvol}x
    
    Overnight/Recent Headlines:
    {news_text}

    Return strict JSON with these exact keys:
    {{
        "sentiment_score": float between -1.0 and 1.0,
        "risk_detected": boolean (true if news indicates regulatory penalties, fraud, or negative gap-down risk),
        "catalyst": string (e.g., "Earnings Beat", "Major Contract Win", "Sector Momentum", "Technical Breakout"),
        "news_drift_shift_pct": float between -2.0 and 2.5 (expected opening gap/drift contribution),
        "summary": string (1 crisp sentence explaining intraday thesis)
    }}
    """
    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        return {
            "sentiment_score": 0.1,
            "risk_detected": False,
            "catalyst": "Pattern Momentum",
            "news_drift_shift_pct": 0.0,
            "summary": f"Fallback intraday setup based on {pattern} structure."
        }

# ==========================================
# 6. WORKFLOW: NIGHTLY INTRADAY SCANNER
# ==========================================
def run_intraday_scan(api_key: str, selected_risk: str, date_id: str) -> int:
    profile = INTRADAY_RISK_PROFILES[selected_risk]
    sl_pct = profile["stop_loss_pct"]
    tp_pct = profile["target_pct"]
    max_var = profile["max_variance"]
    min_rvol = profile["min_rvol"]
    min_sent = profile["min_sentiment"]

    tickers = list(MASTER_UNIVERSE.keys())
    progress = st.progress(0.1, text=f"Downloading intraday data and computing pattern variance across {len(tickers)} stocks...")

    batch_df = yf.download(tickers, period="1mo", interval="1d", group_by="ticker", progress=False)

    qualified_set = []
    for ticker in tickers:
        try:
            df = batch_df[ticker].dropna()
            if len(df) < 15:
                continue

            close_series = df["Close"]
            vol_series = df["Volume"]
            curr_price = float(close_series.iloc[-1])
            
            # Compute RVOL (Today's volume vs 20-day average volume)
            avg_vol = float(vol_series.tail(20).mean())
            curr_vol = float(vol_series.iloc[-1])
            rvol = round(curr_vol / max(1.0, avg_vol), 2)

            # Pattern Matching
            matched_pat, variance = match_institutional_pattern(close_series)

            if variance <= max_var and rvol >= min_rvol and ("Bull" in matched_pat or "ORB" in matched_pat or "V-Bottom" in matched_pat):
                qualified_set.append((ticker, curr_price, matched_pat, variance, rvol, close_series))
        except Exception:
            continue

    qualified_set.sort(key=lambda x: x[3]) # Sort by lowest variance (best pattern match)
    qualified_set = qualified_set[:10]

    new_rows = []
    total_q = len(qualified_set)
    for idx, (ticker, curr_price, matched_pat, variance, rvol, close_series) in enumerate(qualified_set):
        comp_name = MASTER_UNIVERSE[ticker]
        progress.progress(
            0.2 + (0.8 * ((idx + 1) / max(1, total_q))),
            text=f"Evaluating AI News & Drift ({idx + 1}/{total_q}): {comp_name} ({matched_pat})..."
        )

        ai_res = evaluate_intraday_with_gemini(api_key, ticker, comp_name, matched_pat, variance, rvol)
        sent_score = float(ai_res.get("sentiment_score", 0.0))
        risk_flag = bool(ai_res.get("risk_detected", False))
        drift_shift = float(ai_res.get("news_drift_shift_pct", 0.0))

        passed_ai = (not risk_flag) and (sent_score >= min_sentiment_check(min_sent))

        # Intraday volatility & barrier touch probability calculation
        log_rets = np.log(close_series / close_series.shift(1)).dropna()
        intraday_vol = float(log_rets.std()) / math.sqrt(6.5) # Scaled for intraday hours
        intraday_vol = max(intraday_vol, 0.002)

        # Adjusted drift incorporating news shift
        mu_adj = (drift_shift / 100.0)
        prob_sl = min(99.0, round(2.0 * norm_cdf((-sl_pct - mu_adj) / intraday_vol) * 100.0, 1))
        prob_tp = min(99.0, round(2.0 * norm_cdf((-tp_pct + mu_adj) / intraday_vol) * 100.0, 1))

        buy_price = round(curr_price, 2)
        stop_loss = round(buy_price * (1 - sl_pct), 2)
        target = round(buy_price * (1 + tp_pct), 2)

        new_rows.append({
            "date_id": date_id,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "risk_profile": selected_risk,
            "ticker": ticker,
            "company_name": comp_name,
            "matched_pattern": matched_pat,
            "pattern_variance": variance,
            "rvol_score": rvol,
            "prob_hit_sl_pct": prob_sl,
            "prob_hit_tp_pct": prob_tp,
            "buy_price": buy_price,
            "stop_loss_price": stop_loss,
            "target_price": target,
            "ai_sentiment_score": sent_score,
            "ai_risk_detected": risk_flag,
            "ai_catalyst": ai_res.get("catalyst", "Pattern Breakout"),
            "ai_thesis_summary": ai_res.get("summary", ""),
            "passed_ai_filter": passed_ai
        })

    progress.empty()

    if new_rows:
        existing_df = read_tab_df(INTRADAY_REC_TAB, INTRADAY_REC_COLS)
        if not existing_df.empty:
            existing_df = existing_df[~((existing_df["date_id"].astype(str) == str(date_id)) & (existing_df["risk_profile"].astype(str) == str(selected_risk)))]
        updated_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
        write_tab_df(INTRADAY_REC_TAB, updated_df)
        return len(new_rows)
    return 0

def min_sentiment_check(val):
    return val

# ==========================================
# 7. HUMAN IN THE LOOP DECISION LOGGING
# ==========================================
def log_intraday_decision(row: pd.Series, decision: str):
    dec_df = read_tab_df(INTRADAY_DEC_TAB, INTRADAY_DEC_COLS)
    if not dec_df.empty:
        mask = (
            (dec_df["date_id"].astype(str) == str(row["date_id"])) &
            (dec_df["ticker"].astype(str) == str(row["ticker"])) &
            (dec_df["risk_profile"].astype(str) == str(row["risk_profile"]))
        )
        dec_df = dec_df[~mask]

    new_entry = pd.DataFrame([{
        "decision_id": str(uuid.uuid4())[:8],
        "date_id": row["date_id"],
        "decision_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "risk_profile": row["risk_profile"],
        "ticker": row["ticker"],
        "company_name": row["company_name"],
        "buy_price": row["buy_price"],
        "stop_loss_price": row["stop_loss_price"],
        "target_price": row["target_price"],
        "human_decision": decision
    }])

    updated = pd.concat([dec_df, new_entry], ignore_index=True)
    write_tab_df(INTRADAY_DEC_TAB, updated)

# ==========================================
# 8. SAME-DAY SQUARE-OFF EVALUATOR
# ==========================================
def evaluate_intraday_squareoff(ticker: str, buy_price: float, stop_loss: float, target: float):
    try:
        # Pull today's intraday high/low action
        df = yf.download(ticker, period="1d", interval="5m", progress=False)
        if df.empty:
            df = yf.download(ticker, period="2d", interval="1d", progress=False)
            
        lows = df["Low"].iloc[:, 0] if isinstance(df["Low"], pd.DataFrame) else df["Low"]
        highs = df["High"].iloc[:, 0] if isinstance(df["High"], pd.DataFrame) else df["High"]
        closes = df["Close"].iloc[:, 0] if isinstance(df["Close"], pd.DataFrame) else df["Close"]

        day_low = round(float(lows.min()), 2)
        day_high = round(float(highs.max()), 2)
        close_price = round(float(closes.iloc[-1]), 2)

        if day_low <= stop_loss:
            exit_price = stop_loss
            reason = "STOP_LOSS_HIT"
        elif day_high >= target:
            exit_price = target
            reason = "TARGET_HIT"
        else:
            exit_price = close_price
            reason = "AUTO_SQUAREOFF_CLOSE"

        net_ret_pct = round(((exit_price - buy_price) / buy_price) * 100, 2)
        return day_low, day_high, exit_price, reason, net_ret_pct
    except Exception:
        return buy_price, buy_price, buy_price, "NO_DATA", 0.0

def run_squareoff_report(date_id: str, risk_profile: str):
    rec_df = read_tab_df(INTRADAY_REC_TAB, INTRADAY_REC_COLS)
    dec_df = read_tab_df(INTRADAY_DEC_TAB, INTRADAY_DEC_COLS)

    day_rec = rec_df[(rec_df["date_id"].astype(str) == str(date_id)) & (rec_df["risk_profile"].astype(str) == str(risk_profile))].copy()
    if day_rec.empty:
        return None, None

    day_dec = dec_df[(dec_df["date_id"].astype(str) == str(date_id)) & (dec_df["risk_profile"].astype(str) == str(risk_profile))]
    yes_tickers = set(day_dec[day_dec["human_decision"].astype(str) == "YES"]["ticker"].tolist()) if not day_dec.empty else set()

    audit_rows = []
    for _, row in day_rec.iterrows():
        b_price = float(row["buy_price"])
        sl_price = float(row["stop_loss_price"])
        tp_price = float(row["target_price"])
        passed_flag = str(row["passed_ai_filter"]).strip().lower() == "true"

        d_low, d_high, ex_price, ex_reason, net_ret = evaluate_intraday_squareoff(
            row["ticker"], b_price, sl_price, tp_price
        )
        audit_rows.append({
            "Ticker": row["ticker"],
            "Company": row["company_name"],
            "Pattern": row["matched_pattern"],
            "Passed_AI_Filter": passed_flag,
            "Human_Approved_YES": row["ticker"] in yes_tickers,
            "Buy_Price": b_price,
            "Stop_Loss": sl_price,
            "Target": tp_price,
            "Day_Low": d_low,
            "Day_High": d_high,
            "Exit_Price": ex_price,
            "Exit_Reason": ex_reason,
            "Net_Return_Pct": net_ret
        })

    audit_df = pd.DataFrame(audit_rows)

    def summarize_cohort(df_sub: pd.DataFrame, name: str):
        if df_sub.empty:
            return {"Cohort": name, "Trades": 0, "Win_Rate_Pct": 0.0, "Avg_Return_Pct": 0.0, "SL_Hits": 0}
        wins = len(df_sub[df_sub["Net_Return_Pct"] > 0])
        sl_hits = len(df_sub[df_sub["Exit_Reason"] == "STOP_LOSS_HIT"])
        return {
            "Cohort": name,
            "Trades": len(df_sub),
            "Win_Rate_Pct": round((wins / len(df_sub)) * 100, 1),
            "Avg_Return_Pct": round(df_sub["Net_Return_Pct"].mean(), 2),
            "SL_Hits": sl_hits
        }

    c_no_ai = summarize_cohort(audit_df, "1. Without AI (Pattern Baseline)")
    c_with_ai = summarize_cohort(audit_df[audit_df["Passed_AI_Filter"] == True], "2. With AI (Model Filtered)")
    c_human = summarize_cohort(audit_df[audit_df["Human_Approved_YES"] == True], "3. Human Confirmed (YES)")

    summary_df = pd.DataFrame([c_no_ai, c_with_ai, c_human])
    base_ret = c_no_ai["Avg_Return_Pct"]
    summary_df["Alpha_vs_Baseline_Pct"] = round(summary_df["Avg_Return_Pct"] - base_ret, 2)

    report_tab_name = f"SQ_{date_id}_{risk_profile.replace(' ', '')}"[:31]
    write_squareoff_report_tab(report_tab_name, summary_df, audit_df)

    return summary_df, audit_df

# ==========================================
# 9. DYNAMIC UI COMPONENT HELPERS
# ==========================================
def format_dynamic_price_card(label: str, price_val, accent_color: str = "#FFFFFF", sub_badge: str = "") -> str:
    try:
        num = float(price_val)
        formatted = f"₹{num:,.1f}" if num >= 10000 else f"₹{num:,.2f}"
    except (ValueError, TypeError):
        formatted = f"₹{price_val}"

    badge_html = f'<div style="font-size: 0.72rem; color: {accent_color}; opacity: 0.85; margin-top: 2px;">{sub_badge}</div>' if sub_badge else ""

    return f"""
    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
                border-radius: 8px; padding: 8px 10px; text-align: left;">
        <div style="font-size: 0.78rem; color: #9CA3AF; margin-bottom: 2px;">{label}</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: {accent_color};
                    white-space: nowrap; letter-spacing: -0.3px;">{formatted}</div>
        {badge_html}
    </div>
    """

# ==========================================
# 10. STREAMLIT UI LAYOUT
# ==========================================
st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; }
        header[data-testid="stHeader"] { background: transparent; }
    </style>
""", unsafe_allow_html=True)

if os.path.exists("logo.png"):
    img = Image.open("logo.png")
    w, h = img.size
    cropped = img.crop((int(w * 0.08), int(h * 0.25), int(w * 0.90), int(h * 0.72)))
    st.image(cropped, width=420)
else:
    st.title("⚡ SwingCopilot AI — Intraday Alpha & Pattern Engine")

st.caption("Nightly Intraday Pattern Matching & Same-Day Square-Off Engine | Live Google Sheets Cloud Database")

default_gemini_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))

with st.sidebar:
    st.header("⚙️ System Configuration")
    gemini_api_key = st.text_input(
        "Gemini API Key",
        value=default_gemini_key,
        type="password"
    )
    active_date = st.text_input("Trading Date ID (YYYY-MM-DD)", value=get_current_date_id())

    if "GSHEET_URL" in st.secrets:
        st.link_button("🔗 Open Live Google Sheet DB", st.secrets["GSHEET_URL"], use_container_width=True)

    st.divider()
    st.subheader("⚡ Intraday Risk Profile")
    selected_risk = st.select_slider(
        "Select Profile:",
        options=["Conservative", "Balanced", "Aggressive Breakout"],
        value="Balanced"
    )

    risk_info = INTRADAY_RISK_PROFILES[selected_risk]
    st.info(
        f"**{selected_risk} Intraday Setup**\n\n"
        f"• **Target:** +{risk_info['target_pct']*100:.1f}%\n"
        f"• **Stop-Loss:** -{risk_info['stop_loss_pct']*100:.1f}%\n"
        f"• **Max Pattern Variance:** ≤{risk_info['max_variance']}\n"
        f"• **Min RVOL:** ≥{risk_info['min_rvol']}x"
    )

tab1, tab2, tab3 = st.tabs([
    "🌅 1. Nightly Scan & Intraday Approval",
    "📊 2. Same-Day Square-Off & Audit",
    "🗂️ 3. Google Sheets Live Inspector"
])

# ------------------------------------------
# TAB 1: NIGHTLY SCAN & APPROVAL
# ------------------------------------------
with tab1:
    c_h, c_b = st.columns([3, 1])
    with c_h:
        st.subheader(f"Pre-Market Intraday Screening ({selected_risk})")
        st.write("Scans technical pattern variance against pre-defined templates, calculates RVOL, extracts overnight news drift, and prepares intraday setups.")
    with c_b:
        if st.button("🚀 Run Nightly Intraday Scan", type="primary", use_container_width=True):
            if not gemini_api_key:
                st.error("Please configure your Gemini API Key.")
            else:
                count = run_intraday_scan(gemini_api_key, selected_risk, active_date)
                st.success(f"Scan complete! Synced {count} intraday setups to `{INTRADAY_REC_TAB}`.")

    st.divider()

    rec_df = read_tab_df(INTRADAY_REC_TAB, INTRADAY_REC_COLS)
    dec_df = read_tab_df(INTRADAY_DEC_TAB, INTRADAY_DEC_COLS)

    day_candidates = rec_df[
        (rec_df["date_id"].astype(str) == str(active_date)) &
        (rec_df["risk_profile"].astype(str) == str(selected_risk))
    ] if not rec_df.empty else pd.DataFrame()

    if day_candidates.empty:
        st.warning(f"No intraday recommendations generated yet for **{active_date}** under **{selected_risk}** profile.")
    else:
        passed_mask = day_candidates["passed_ai_filter"].astype(str).str.strip().str.lower() == "true"
        approved_df = day_candidates[passed_mask]

        st.markdown(f"### ✅ AI-Approved Intraday Setups ({len(approved_df)} Picks)")
        for _, row in approved_df.iterrows():
            existing_dec = dec_df[
                (dec_df["date_id"].astype(str) == str(active_date)) &
                (dec_df["ticker"].astype(str) == str(row["ticker"])) &
                (dec_df["risk_profile"].astype(str) == str(selected_risk))
            ] if not dec_df.empty else pd.DataFrame()

            current_status = existing_dec["human_decision"].iloc[-1] if not existing_dec.empty else "PENDING"

            with st.container(border=True):
                c1, c2, c3 = st.columns([2.2, 2.5, 1.4])
                with c1:
                    st.markdown(f"#### {row['company_name']} (`{row['ticker']}`)")
                    st.caption(
                        f"Pattern: **{row['matched_pattern']}** (Var: {row['pattern_variance']}) | "
                        f"RVOL: **{row['rvol_score']}x** | Catalyst: `{row['ai_catalyst']}`"
                    )
                    st.write(f"**Intraday Thesis:** {row['ai_thesis_summary']}")
                with c2:
                    m1, m2, m3 = st.columns(3)
                    m1.markdown(format_dynamic_price_card("Buy / Open", row["buy_price"], "#F9FAFB", "Entry"), unsafe_allow_html=True)
                    m2.markdown(format_dynamic_price_card("Stop-Loss", row["stop_loss_price"], "#F87171", f"P(SL): {row['prob_hit_sl_pct']}%"), unsafe_allow_html=True)
                    m3.markdown(format_dynamic_price_card("Target", row["target_price"], "#34D399", f"P(TP): {row['prob_hit_tp_pct']}%"), unsafe_allow_html=True)
                with c3:
                    st.markdown("**Action:**")
                    if current_status == "YES":
                        st.success("🟢 Active (YES)")
                    elif current_status == "NO":
                        st.error("🔴 Skipped (NO)")
                    else:
                        st.warning("⏳ Pending")

                    b_yes, b_no = st.columns(2)
                    if b_yes.button("YES ✅", key=f"iyes_{row['ticker']}_{selected_risk}", use_container_width=True):
                        log_intraday_decision(row, "YES")
                        st.rerun()
                    if b_no.button("NO ❌", key=f"ino_{row['ticker']}_{selected_risk}", use_container_width=True):
                        log_intraday_decision(row, "NO")
                        st.rerun()

# ------------------------------------------
# TAB 2: SAME-DAY SQUARE-OFF & AUDIT
# ------------------------------------------
with tab2:
    st.subheader("Market Close Square-Off & Performance Audit")
    st.write("Resolves whether intraday target or stop-loss was hit before market close (3:30 PM IST), auto-squares off positions, and logs performance to Google Sheets.")

    if st.button("🏁 Run Market Close Square-Off & Sync", type="primary"):
        summary_df, audit_df = run_squareoff_report(active_date, selected_risk)
        if summary_df is None:
            st.error(f"No intraday data found for {active_date}.")
        else:
            st.success("Square-off evaluation completed and synced to Google Sheets!")

    sh = get_spreadsheet()
    all_tabs = [ws.title for ws in sh.worksheets() if ws.title.startswith("SQ_")]

    if all_tabs:
        selected_sq = st.selectbox("Select Square-Off Report Tab:", options=all_tabs[::-1])
        if selected_sq:
            ws_vals = sh.worksheet(selected_sq).get_all_values()
            if len(ws_vals) >= 5:
                sum_headers = ws_vals[1]
                sum_rows = ws_vals[2:5]
                summary_view = pd.DataFrame(sum_rows, columns=sum_headers)
                st.markdown(f"#### 🏆 Intraday Cohort Summary (`{selected_sq}`)")
                st.dataframe(summary_view, use_container_width=True)

# ------------------------------------------
# TAB 3: LIVE INSPECTOR
# ------------------------------------------
with tab3:
    c_l, c_r = st.columns(2)
    with c_l:
        st.markdown(f"### 📄 `{INTRADAY_REC_TAB}`")
        st.dataframe(read_tab_df(INTRADAY_REC_TAB, INTRADAY_REC_COLS), use_container_width=True)
    with c_r:
        st.markdown(f"### 👤 `{INTRADAY_DEC_TAB}`")
        st.dataframe(read_tab_df(INTRADAY_DEC_TAB, INTRADAY_DEC_COLS), use_container_width=True)
