import os
import uuid
import json
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
import requests
import pandas as pd
import yfinance as yf
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from google import genai
from google.genai import types

# ==========================================
# 1. PAGE CONFIG & CONSTANTS
# ==========================================
st.set_page_config(
    # Place this right below the "st.set_page_config" call at the top of app.py
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=False, width=450)
    else:
        # Fallback to text title if the image hasn't been uploaded yet
        st.title("📈 SwingCopilot AI — Retail Equity Decision & Risk Engine")
)

AI_REC_TAB = "ai_recommendations"
HUMAN_DEC_TAB = "human_decisions"

AI_REC_COLS = [
    "week_id", "generated_at", "risk_profile", "ticker", "company_name",
    "market_cap_tier", "buy_price", "stop_loss_price", "target_price",
    "tech_5d_return_pct", "ai_sentiment_score", "ai_risk_detected",
    "ai_catalyst", "ai_thesis_summary", "passed_ai_filter"
]

HUMAN_DEC_COLS = [
    "decision_id", "week_id", "decision_timestamp", "risk_profile",
    "ticker", "company_name", "buy_price", "stop_loss_price",
    "target_price", "human_decision"
]

# ==========================================
# 2. GOOGLE SHEETS CLOUD DATABASE HELPERS
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

def write_friday_report_tab(tab_name: str, summary_df: pd.DataFrame, audit_df: pd.DataFrame):
    sh = get_spreadsheet()
    try:
        ws = sh.worksheet(tab_name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=200, cols=15)

    rows = [["COHORT PERFORMANCE SUMMARY"]]
    rows.append(summary_df.columns.tolist())
    rows.extend(summary_df.fillna("").astype(str).values.tolist())
    rows.append([])
    rows.append(["TRADE-BY-TRADE RESOLUTION AUDIT"])
    rows.append(audit_df.columns.tolist())
    rows.extend(audit_df.fillna("").astype(str).values.tolist())

    ws.update(range_name="A1", values=rows)

# ==========================================
# 3. 5-TIER RISK PROFILE & NSE UNIVERSE
# ==========================================
RISK_PROFILES = {
    "Very Low": {
        "description": "Top Safest Blue-Chips (Low-Beta Nifty Leaders). Zero tolerance for negative news.",
        "stop_loss_pct": 0.015,
        "target_pct": 0.030,
        "min_sentiment": 0.50,
        "ai_rule": "Extremely conservative. Flag risk_detected=true on ANY hint of margin pressure, regulatory scrutiny, sector headwinds, or mixed commentary.",
        "universe": {
            "TCS.NS": ("Tata Consultancy Services", "Blue-Chip"),
            "HDFCBANK.NS": ("HDFC Bank", "Blue-Chip"),
            "INFY.NS": ("Infosys", "Blue-Chip"),
            "ITC.NS": ("ITC Limited", "Blue-Chip"),
            "HINDUNILVR.NS": ("Hindustan Unilever", "Blue-Chip"),
            "RELIANCE.NS": ("Reliance Industries", "Blue-Chip"),
            "ICICIBANK.NS": ("ICICI Bank", "Blue-Chip"),
            "SBIN.NS": ("State Bank of India", "Blue-Chip"),
            "BHARTIARTL.NS": ("Bharti Airtel", "Blue-Chip"),
            "LT.NS": ("Larsen & Toubro", "Blue-Chip")
        }
    },
    "Low": {
        "description": "Nifty 50 Large Caps. Strict filter; vetoes regulatory, legal, or earnings miss news.",
        "stop_loss_pct": 0.020,
        "target_pct": 0.040,
        "min_sentiment": 0.35,
        "ai_rule": "Conservative. Flag risk_detected=true if there are legal disputes, regulatory notices, management exits, or weak demand signals.",
        "universe": {
            "SUNPHARMA.NS": ("Sun Pharmaceutical", "Large-Cap"),
            "MARUTI.NS": ("Maruti Suzuki", "Large-Cap"),
            "AXISBANK.NS": ("Axis Bank", "Large-Cap"),
            "KOTAKBANK.NS": ("Kotak Mahindra Bank", "Large-Cap"),
            "TITAN.NS": ("Titan Company", "Large-Cap"),
            "ULTRACEMCO.NS": ("UltraTech Cement", "Large-Cap"),
            "BAJFINANCE.NS": ("Bajaj Finance", "Large-Cap"),
            "M&M.NS": ("Mahindra & Mahindra", "Large-Cap"),
            "NTPC.NS": ("NTPC Limited", "Large-Cap"),
            "POWERGRID.NS": ("Power Grid Corp", "Large-Cap")
        }
    },
    "Medium": {
        "description": "Nifty 100 (Large + Upper Mid-Cap). Balanced risk-reward; allows neutral news if momentum is strong.",
        "stop_loss_pct": 0.025,
        "target_pct": 0.050,
        "min_sentiment": 0.20,
        "ai_rule": "Balanced swing trader. Flag risk_detected=true only for material adverse events like SEBI penalties, earnings misses, or major downgrades.",
        "universe": {
            "TATAMOTORS.NS": ("Tata Motors", "Large-Cap"),
            "ZOMATO.NS": ("Zomato", "Large-Cap"),
            "DLF.NS": ("DLF Limited", "Large-Cap"),
            "HAL.NS": ("Hindustan Aeronautics", "Large-Cap"),
            "BEL.NS": ("Bharat Electronics", "Large-Cap"),
            "TRENT.NS": ("Trent Limited", "Large-Cap"),
            "INDHOTEL.NS": ("Indian Hotels Co", "Mid-Cap"),
            "CHOLAFIN.NS": ("Cholamandalam Investment", "Mid-Cap"),
            "PIDILITIND.NS": ("Pidilite Industries", "Large-Cap"),
            "SIEMENS.NS": ("Siemens India", "Large-Cap")
        }
    },
    "High": {
        "description": "Nifty Midcap 150. Aggressive momentum; accepts volatility unless severe fraud/SEBI risk exists.",
        "stop_loss_pct": 0.035,
        "target_pct": 0.075,
        "min_sentiment": 0.10,
        "ai_rule": "Aggressive momentum trader. Ignore minor market noise; only flag risk_detected=true for severe earnings shocks, fraud, or regulatory bans.",
        "universe": {
            "PERSISTENT.NS": ("Persistent Systems", "Mid-Cap"),
            "COFORGE.NS": ("Coforge Limited", "Mid-Cap"),
            "DIXON.NS": ("Dixon Technologies", "Mid-Cap"),
            "POLYCAB.NS": ("Polycab India", "Mid-Cap"),
            "KEI.NS": ("KEI Industries", "Mid-Cap"),
            "KPITTECH.NS": ("KPIT Technologies", "Mid-Cap"),
            "FEDERALBNK.NS": ("Federal Bank", "Mid-Cap"),
            "IDFCFIRSTB.NS": ("IDFC First Bank", "Mid-Cap"),
            "CUMMINSIND.NS": ("Cummins India", "Mid-Cap"),
            "MAXHEALTH.NS": ("Max Healthcare", "Mid-Cap")
        }
    },
    "Very High": {
        "description": "Broad Market + Nifty Smallcap 250. High-beta breakouts; only vetoes catastrophic governance events.",
        "stop_loss_pct": 0.050,
        "target_pct": 0.100,
        "min_sentiment": 0.00,
        "ai_rule": "High-risk small-cap speculator. Only flag risk_detected=true for insolvency, auditor resignation, raids, or catastrophic corporate governance failure.",
        "universe": {
            "SUZLON.NS": ("Suzlon Energy", "Small-Cap"),
            "CDSL.NS": ("Central Depository Services", "Small-Cap"),
            "BSE.NS": ("BSE Limited", "Mid-Cap"),
            "MCX.NS": ("Multi Commodity Exchange", "Small-Cap"),
            "ANGELONE.NS": ("Angel One", "Small-Cap"),
            "SONACOMS.NS": ("Sona BLW Precision", "Mid-Cap"),
            "KAYNES.NS": ("Kaynes Technology", "Small-Cap"),
            "DATA-PATTERNS.NS": ("Data Patterns India", "Small-Cap"),
            "TEJASNET.NS": ("Tejas Networks", "Small-Cap"),
            "JYOTHYLAB.NS": ("Jyothy Labs", "Small-Cap")
        }
    }
}

def get_current_week_id() -> str:
    now = datetime.now()
    return f"{now.year}-W{now.isocalendar()[1]:02d}"

# ==========================================
# 4. NEWS SCRAPER & GEMINI RISK ENGINE
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

def evaluate_with_gemini(api_key: str, ticker: str, company_name: str, risk_level: str) -> dict:
    headlines = fetch_indian_stock_news(company_name)
    if not headlines:
        return {
            "sentiment_score": 0.25,
            "risk_detected": False,
            "catalyst": "Technical Momentum (No recent news)",
            "summary": "No major news catalysts found; trade relies purely on positive price action."
        }

    client = genai.Client(api_key=api_key)
    risk_cfg = RISK_PROFILES[risk_level]
    news_text = "\n".join([f"- {h}" for h in headlines])

    prompt = f"""
    You are an AI Equity Risk Analyst evaluating Indian stock '{company_name}' ({ticker}) for a 1-week swing trade (Monday Open to Friday Close).
    
    ACTIVE USER RISK PROFILE: {risk_level}
    RISK MANDATE: {risk_cfg['ai_rule']}
    
    Recent Headlines:
    {news_text}

    Return strict JSON with these exact keys:
    {{
        "sentiment_score": float between -1.0 and 1.0,
        "risk_detected": boolean (true if news violates the Risk Mandate above),
        "catalyst": string (One of: "Earnings", "Regulatory", "Macro/Sector", "Order Win/Expansion", "Corporate Governance", "General Momentum"),
        "summary": string (1 crisp sentence explaining the bull/bear catalyst)
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
            "sentiment_score": 0.0,
            "risk_detected": False,
            "catalyst": "API Fallback",
            "summary": f"AI evaluation skipped or rate-limited ({str(e)[:40]})."
        }

# ==========================================
# 5. WORKFLOW 1: SUNDAY EVENING SCREENER
# ==========================================
def run_sunday_workflow(api_key: str, selected_risk: str, week_id: str) -> int:
    profile = RISK_PROFILES[selected_risk]
    universe = profile["universe"]
    sl_pct = profile["stop_loss_pct"]
    tp_pct = profile["target_pct"]
    min_sent = profile["min_sentiment"]

    new_rows = []
    progress = st.progress(0, text="Scanning NSE universe & running Gemini risk audit...")
    total = len(universe)

    for idx, (ticker, (comp_name, cap_tier)) in enumerate(universe.items()):
        progress.progress((idx + 1) / total, text=f"Evaluating {comp_name} ({ticker})...")
        try:
            df = yf.download(ticker, period="2mo", interval="1d", progress=False)
            if len(df) < 20:
                continue

            close_series = df["Close"].iloc[:, 0] if isinstance(df["Close"], pd.DataFrame) else df["Close"]
            curr_price = float(close_series.iloc[-1])
            sma_20 = float(close_series.rolling(window=20).mean().iloc[-1])
            price_5d_ago = float(close_series.iloc[-5])
            ret_5d_pct = round(((curr_price - price_5d_ago) / price_5d_ago) * 100, 2)

            if curr_price > sma_20 and ret_5d_pct > 0:
                ai_res = evaluate_with_gemini(api_key, ticker, comp_name, selected_risk)
                sent_score = float(ai_res.get("sentiment_score", 0.0))
                risk_flag = bool(ai_res.get("risk_detected", False))
                passed_ai = (not risk_flag) and (sent_score >= min_sent)

                buy_price = round(curr_price, 2)
                stop_loss = round(buy_price * (1 - sl_pct), 2)
                target = round(buy_price * (1 + tp_pct), 2)

                new_rows.append({
                    "week_id": week_id,
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "risk_profile": selected_risk,
                    "ticker": ticker,
                    "company_name": comp_name,
                    "market_cap_tier": cap_tier,
                    "buy_price": buy_price,
                    "stop_loss_price": stop_loss,
                    "target_price": target,
                    "tech_5d_return_pct": ret_5d_pct,
                    "ai_sentiment_score": sent_score,
                    "ai_risk_detected": risk_flag,
                    "ai_catalyst": ai_res.get("catalyst", "Momentum"),
                    "ai_thesis_summary": ai_res.get("summary", ""),
                    "passed_ai_filter": passed_ai
                })
        except Exception:
            continue

    progress.empty()

    if new_rows:
        existing_df = read_tab_df(AI_REC_TAB, AI_REC_COLS)
        if not existing_df.empty:
            existing_df = existing_df[~((existing_df["week_id"].astype(str) == str(week_id)) & (existing_df["risk_profile"].astype(str) == str(selected_risk)))]
        updated_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
        write_tab_df(AI_REC_TAB, updated_df)
        return len(new_rows)
    return 0

# ==========================================
# 6. HUMAN DECISION LOGGING
# ==========================================
def log_human_decision(row: pd.Series, decision: str):
    dec_df = read_tab_df(HUMAN_DEC_TAB, HUMAN_DEC_COLS)
    if not dec_df.empty:
        mask = (
            (dec_df["week_id"].astype(str) == str(row["week_id"])) &
            (dec_df["ticker"].astype(str) == str(row["ticker"])) &
            (dec_df["risk_profile"].astype(str) == str(row["risk_profile"]))
        )
        dec_df = dec_df[~mask]

    new_entry = pd.DataFrame([{
        "decision_id": str(uuid.uuid4())[:8],
        "week_id": row["week_id"],
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
    write_tab_df(HUMAN_DEC_TAB, updated)

# ==========================================
# 7. WORKFLOW 2: FRIDAY PERFORMANCE EVALUATOR
# ==========================================
def evaluate_single_trade(ticker: str, buy_price: float, stop_loss: float, target: float):
    try:
        df = yf.download(ticker, period="5d", interval="1d", progress=False)
        if df.empty:
            raise ValueError("Empty price dataframe")

        lows = df["Low"].iloc[:, 0] if isinstance(df["Low"], pd.DataFrame) else df["Low"]
        highs = df["High"].iloc[:, 0] if isinstance(df["High"], pd.DataFrame) else df["High"]
        closes = df["Close"].iloc[:, 0] if isinstance(df["Close"], pd.DataFrame) else df["Close"]

        week_low = round(float(lows.min()), 2)
        week_high = round(float(highs.max()), 2)
        fri_close = round(float(closes.iloc[-1]), 2)

        if week_low <= stop_loss:
            exit_price = stop_loss
            reason = "STOP_LOSS_HIT"
        elif week_high >= target:
            exit_price = target
            reason = "TARGET_HIT"
        else:
            exit_price = fri_close
            reason = "FRIDAY_CLOSE_EXIT"

        net_ret_pct = round(((exit_price - buy_price) / buy_price) * 100, 2)
        return week_low, week_high, fri_close, exit_price, reason, net_ret_pct
    except Exception:
        return buy_price, buy_price, buy_price, buy_price, "NO_DATA", 0.0

def run_friday_evaluation(week_id: str, risk_profile: str):
    ai_df = read_tab_df(AI_REC_TAB, AI_REC_COLS)
    dec_df = read_tab_df(HUMAN_DEC_TAB, HUMAN_DEC_COLS)

    week_ai = ai_df[(ai_df["week_id"].astype(str) == str(week_id)) & (ai_df["risk_profile"].astype(str) == str(risk_profile))].copy()
    if week_ai.empty:
        return None, None

    week_dec = dec_df[(dec_df["week_id"].astype(str) == str(week_id)) & (dec_df["risk_profile"].astype(str) == str(risk_profile))]
    yes_tickers = set(week_dec[week_dec["human_decision"].astype(str) == "YES"]["ticker"].tolist()) if not week_dec.empty else set()

    trade_audit_rows = []
    for _, row in week_ai.iterrows():
        b_price = float(row["buy_price"])
        sl_price = float(row["stop_loss_price"])
        tp_price = float(row["target_price"])
        passed_flag = str(row["passed_ai_filter"]).strip().lower() == "true"

        w_low, w_high, f_close, ex_price, ex_reason, net_ret = evaluate_single_trade(
            row["ticker"], b_price, sl_price, tp_price
        )
        trade_audit_rows.append({
            "Ticker": row["ticker"],
            "Company": row["company_name"],
            "Passed_AI_Filter": passed_flag,
            "Human_Approved_YES": row["ticker"] in yes_tickers,
            "Buy_Price": b_price,
            "Stop_Loss": sl_price,
            "Target": tp_price,
            "Week_Low": w_low,
            "Week_High": w_high,
            "Friday_Close": f_close,
            "Exit_Price": ex_price,
            "Exit_Reason": ex_reason,
            "Net_Return_Pct": net_ret
        })

    audit_df = pd.DataFrame(trade_audit_rows)

    def summarize_cohort(df_sub: pd.DataFrame, cohort_name: str):
        if df_sub.empty:
            return {
                "Cohort": cohort_name,
                "Total_Trades": 0,
                "Win_Rate_Pct": 0.0,
                "Avg_Return_Pct": 0.0,
                "Max_Drawdown_Pct": 0.0,
                "Stop_Loss_Hits": 0
            }
        wins = len(df_sub[df_sub["Net_Return_Pct"] > 0])
        sl_hits = len(df_sub[df_sub["Exit_Reason"] == "STOP_LOSS_HIT"])
        return {
            "Cohort": cohort_name,
            "Total_Trades": len(df_sub),
            "Win_Rate_Pct": round((wins / len(df_sub)) * 100, 1),
            "Avg_Return_Pct": round(df_sub["Net_Return_Pct"].mean(), 2),
            "Max_Drawdown_Pct": round(df_sub["Net_Return_Pct"].min(), 2),
            "Stop_Loss_Hits": sl_hits
        }

    c_no_ai = summarize_cohort(audit_df, "1. Without AI (Pure Technical Baseline)")
    c_with_ai = summarize_cohort(audit_df[audit_df["Passed_AI_Filter"] == True], "2. With AI (Model Recommendations)")
    c_human = summarize_cohort(audit_df[audit_df["Human_Approved_YES"] == True], "3. Human Confirmed (YES Triggers)")

    summary_df = pd.DataFrame([c_no_ai, c_with_ai, c_human])
    baseline_ret = c_no_ai["Avg_Return_Pct"]
    summary_df["Alpha_vs_No_AI_Pct"] = round(summary_df["Avg_Return_Pct"] - baseline_ret, 2)

    report_tab_name = f"{week_id}_{risk_profile.replace(' ', '')}"[:31]
    write_friday_report_tab(report_tab_name, summary_df, audit_df)

    return summary_df, audit_df

# ==========================================
# 8. STREAMLIT UI LAYOUT
# ==========================================
st.title("📈 SwingCopilot AI — Retail Equity Decision & Risk Engine")
st.caption("Human-in-the-Loop Weekly NSE Swing Trading System | Live Google Sheets Cloud Database")

default_gemini_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))

with st.sidebar:
    st.header("⚙️ System Configuration")
    gemini_api_key = st.text_input(
        "Gemini API Key",
        value=default_gemini_key,
        type="password",
        help="Auto-loaded from Streamlit Secrets if configured"
    )
    active_week = st.text_input("Active Week ID", value=get_current_week_id())

    if "GSHEET_URL" in st.secrets:
        st.link_button("🔗 Open Live Google Sheet DB", st.secrets["GSHEET_URL"], use_container_width=True)

    st.divider()
    st.subheader("🛡️ Dynamic Risk Filter")
    selected_risk = st.select_slider(
        "Select Portfolio Risk Appetite:",
        options=["Very Low", "Low", "Medium", "High", "Very High"],
        value="Medium"
    )

    risk_info = RISK_PROFILES[selected_risk]
    st.info(
        f"**{selected_risk} Risk Profile**\n\n"
        f"• **Universe:** {risk_info['description']}\n"
        f"• **Stop-Loss:** -{risk_info['stop_loss_pct']*100:.1f}%\n"
        f"• **Profit Target:** +{risk_info['target_pct']*100:.1f}%\n"
        f"• **Min AI Sentiment:** {risk_info['min_sentiment']}"
    )

    with st.expander("View Parent NSE Stock List"):
        for t, (c_name, tier) in risk_info["universe"].items():
            st.markdown(f"- `{t}`: {c_name} *({tier})*")

tab1, tab2, tab3 = st.tabs([
    "🌅 1. Sunday Screening & Human Approval",
    "📊 2. Friday Performance & AI vs. Non-AI Audit",
    "🗂️ 3. Google Sheets Live Data Inspector"
])

# ------------------------------------------
# TAB 1: SUNDAY WORKFLOW & YES/NO CARDS
# ------------------------------------------
with tab1:
    col_head, col_btn = st.columns([3, 1])
    with col_head:
        st.subheader(f"Sunday Pre-Market Screening ({selected_risk} Risk)")
        st.write("Screens parent universe for technical momentum, audits weekend news via Gemini, and syncs recommendations to Google Sheets.")
    with col_btn:
        if st.button("🚀 Run Sunday Scan Now", type="primary", use_container_width=True):
            if not gemini_api_key:
                st.error("Please configure your Gemini API Key in Streamlit Secrets or the sidebar.")
            else:
                count = run_sunday_workflow(gemini_api_key, selected_risk, active_week)
                st.success(f"Scanned universe! Synced {count} technical candidates to `{AI_REC_TAB}` tab in Google Sheets.")

    st.divider()

    ai_df = read_tab_df(AI_REC_TAB, AI_REC_COLS)
    dec_df = read_tab_df(HUMAN_DEC_TAB, HUMAN_DEC_COLS)

    current_candidates = ai_df[
        (ai_df["week_id"].astype(str) == str(active_week)) &
        (ai_df["risk_profile"].astype(str) == str(selected_risk))
    ] if not ai_df.empty else pd.DataFrame()

    if current_candidates.empty:
        st.warning(f"No recommendations generated yet for **{active_week}** under **{selected_risk}** risk. Click **Run Sunday Scan Now** above.")
    else:
        passed_mask = current_candidates["passed_ai_filter"].astype(str).str.strip().str.lower() == "true"
        recommended_df = current_candidates[passed_mask]
        vetoed_df = current_candidates[~passed_mask]

        st.markdown(f"### ✅ AI-Approved Trade Thesis Cards ({len(recommended_df)} Picks)")
        if recommended_df.empty:
            st.info("All technical candidates were vetoed by the Gemini Risk Filter for this risk tier.")

        for _, row in recommended_df.iterrows():
            existing_dec = dec_df[
                (dec_df["week_id"].astype(str) == str(active_week)) &
                (dec_df["ticker"].astype(str) == str(row["ticker"])) &
                (dec_df["risk_profile"].astype(str) == str(selected_risk))
            ] if not dec_df.empty else pd.DataFrame()

            current_status = existing_dec["human_decision"].iloc[-1] if not existing_dec.empty else "PENDING"

            with st.container(border=True):
                c1, c2, c3 = st.columns([2.5, 2, 1.5])
                with c1:
                    st.markdown(f"#### {row['company_name']} (`{row['ticker']}`)")
                    st.caption(f"Tier: **{row['market_cap_tier']}** | Catalyst: `{row['ai_catalyst']}` | 5D Momentum: **+{row['tech_5d_return_pct']}%**")
                    st.write(f"**AI Thesis:** {row['ai_thesis_summary']}")
                with c2:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Buy Price", f"₹{row['buy_price']}")
                    m2.metric("Stop-Loss", f"₹{row['stop_loss_price']}")
                    m3.metric("Target", f"₹{row['target_price']}")
                    sent_val = float(row['ai_sentiment_score']) if row['ai_sentiment_score'] != "" else 0.0
                    st.progress(max(0.0, min(1.0, (sent_val + 1) / 2)), text=f"AI Sentiment Score: {sent_val}")
                with c3:
                    st.markdown("**Human Confirmation:**")
                    if current_status == "YES":
                        st.success("🟢 Triggered (YES)")
                    elif current_status == "NO":
                        st.error("🔴 Rejected (NO)")
                    else:
                        st.warning("⏳ Awaiting Decision")

                    b_yes, b_no = st.columns(2)
                    if b_yes.button("YES ✅", key=f"yes_{row['ticker']}_{selected_risk}", use_container_width=True):
                        log_human_decision(row, "YES")
                        st.rerun()
                    if b_no.button("NO ❌", key=f"no_{row['ticker']}_{selected_risk}", use_container_width=True):
                        log_human_decision(row, "NO")
                        st.rerun()

        with st.expander(f"🛡️ Counterfactual Audit: {len(vetoed_df)} Stocks Vetoed by AI Risk Filter"):
            if not vetoed_df.empty:
                st.dataframe(vetoed_df[[
                    "ticker", "company_name", "buy_price", "tech_5d_return_pct",
                    "ai_sentiment_score", "ai_risk_detected", "ai_thesis_summary"
                ]], use_container_width=True)
            else:
                st.write("No stocks were vetoed in this batch.")

# ------------------------------------------
# TAB 2: FRIDAY EVALUATION & COHORT ANALYSIS
# ------------------------------------------
with tab2:
    st.subheader("Friday Post-Market Performance & A/B Cohort Benchmark")
    st.write("Compares **1. Pure Technical Baseline (Without AI)** vs. **2. AI Recommendations** vs. **3. Human Confirmed Trades (`YES`)** and writes to a new weekly tab in Google Sheets.")

    if st.button("📈 Run Friday Evaluation & Sync to Google Sheets", type="primary"):
        summary_df, audit_df = run_friday_evaluation(active_week, selected_risk)
        if summary_df is None:
            st.error(f"No Sunday data found for {active_week} ({selected_risk}). Run the Sunday scan first.")
        else:
            st.success(f"Evaluation synced to Google Sheet tab `{active_week}_{selected_risk.replace(' ', '')}`!")

    sh = get_spreadsheet()
    all_tabs = [ws.title for ws in sh.worksheets() if ws.title not in [AI_REC_TAB, HUMAN_DEC_TAB, "Sheet1"]]

    if all_tabs:
        selected_tab = st.selectbox("Select Historical Weekly Report Tab from Google Sheets:", options=all_tabs[::-1])
        if selected_tab:
            ws_vals = sh.worksheet(selected_tab).get_all_values()
            if len(ws_vals) >= 5:
                sum_headers = ws_vals[1]
                sum_rows = ws_vals[2:5]
                summary_view = pd.DataFrame(sum_rows, columns=sum_headers)

                st.markdown(f"#### 🏆 Cohort Performance Summary (`{selected_tab}`)")
                st.dataframe(summary_view, use_container_width=True)

                k1, k2, k3 = st.columns(3)
                no_ai_ret = float(summary_view.iloc[0]["Avg_Return_Pct"])
                ai_ret = float(summary_view.iloc[1]["Avg_Return_Pct"])
                hum_ret = float(summary_view.iloc[2]["Avg_Return_Pct"])

                k1.metric("1. Without AI (Technical Baseline)", f"{no_ai_ret}%", f"Win Rate: {summary_view.iloc[0]['Win_Rate_Pct']}%")
                k2.metric("2. With AI (Gemini Filtered)", f"{ai_ret}%", f"{round(ai_ret - no_ai_ret, 2)}% Alpha vs No-AI")
                k3.metric("3. Human Approved (YES Picks)", f"{hum_ret}%", f"{round(hum_ret - ai_ret, 2)}% vs Raw AI")

            if len(ws_vals) >= 8:
                st.markdown("#### 🔍 Trade-by-Trade Resolution Audit")
                aud_headers = ws_vals[7]
                aud_rows = ws_vals[8:]
                audit_view = pd.DataFrame(aud_rows, columns=aud_headers)
                st.dataframe(audit_view, use_container_width=True)

# ------------------------------------------
# TAB 3: RAW GOOGLE SHEETS INSPECTOR
# ------------------------------------------
with tab3:
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown(f"### 📄 Tab 1: `{AI_REC_TAB}`")
        df1 = read_tab_df(AI_REC_TAB, AI_REC_COLS)
        st.dataframe(df1, use_container_width=True)

    with c_right:
        st.markdown(f"### 👤 Tab 2: `{HUMAN_DEC_TAB}`")
        df2 = read_tab_df(HUMAN_DEC_TAB, HUMAN_DEC_COLS)
        st.dataframe(df2, use_container_width=True)
