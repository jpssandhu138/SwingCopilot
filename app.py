import os
import uuid
import json
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import requests
import pandas as pd
import yfinance as yf
import streamlit as st
from google import genai
from google.genai import types

# ==========================================
# 1. PAGE CONFIG & CONSTANTS
# ==========================================
st.set_page_config(
    page_title="SwingCopilot AI | NSE Decision Engine",
    page_icon="📈",
    layout="wide"
)

AI_REC_CSV = "ai_recommendations.csv"
HUMAN_DEC_CSV = "human_decisions.csv"
PERF_XLSX = "weekly_performance.xlsx"

# ==========================================
# 2. 5-TIER RISK PROFILE & NSE UNIVERSE
# ==========================================
RISK_PROFILES = {
    "Very Low": {
        "description": "Top Safest Blue-Chips (Low-Beta Nifty Leaders). Zero tolerance for negative news.",
        "stop_loss_pct": 0.015,   # -1.5%
        "target_pct": 0.030,      # +3.0%
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
        "stop_loss_pct": 0.020,   # -2.0%
        "target_pct": 0.040,      # +4.0%
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
        "stop_loss_pct": 0.025,   # -2.5%
        "target_pct": 0.050,      # +5.0%
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
        "stop_loss_pct": 0.035,   # -3.5%
        "target_pct": 0.075,      # +7.5%
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
        "stop_loss_pct": 0.050,   # -5.0%
        "target_pct": 0.100,      # +10.0%
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

# ==========================================
# 3. STORAGE INITIALIZATION HELPERS
# ==========================================
def init_csv_files():
    if not os.path.exists(AI_REC_CSV):
        pd.DataFrame(columns=[
            "week_id", "generated_at", "risk_profile", "ticker", "company_name",
            "market_cap_tier", "buy_price", "stop_loss_price", "target_price",
            "tech_5d_return_pct", "ai_sentiment_score", "ai_risk_detected",
            "ai_catalyst", "ai_thesis_summary", "passed_ai_filter"
        ]).to_csv(AI_REC_CSV, index=False)

    if not os.path.exists(HUMAN_DEC_CSV):
        pd.DataFrame(columns=[
            "decision_id", "week_id", "decision_timestamp", "risk_profile",
            "ticker", "company_name", "buy_price", "stop_loss_price",
            "target_price", "human_decision"
        ]).to_csv(HUMAN_DEC_CSV, index=False)

init_csv_files()

def get_current_week_id() -> str:
    now = datetime.now()
    return f"{now.year}-W{now.isocalendar()[1]:02d}"

# ==========================================
# 4. NEWS SCRAPER & GEMINI RISK ENGINE
# ==========================================
def fetch_indian_stock_news(company_name: str) -> list[str]:
    """Fetches top 5 recent Google News RSS headlines for an Indian stock."""
    try:
        query = urllib.parse.quote(f"{company_name} share news India")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        res = requests.get(rss_url, timeout=6)
        root = ET.fromstring(res.content)
        return [item.find('title').text for item in root.findall('.//item')[:5] if item.find('title') is not None]
    except Exception:
        return []

def evaluate_with_gemini(api_key: str, ticker: str, company_name: str, risk_level: str) -> dict:
    """Evaluates recent news using Gemini 2.5 Flash calibrated to the selected Risk Profile."""
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
            model="gemini-2.5-flash",
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
def run_sunday_workflow(api_key: str, selected_risk: str, week_id: str):
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

            close_series = df["Close"]
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]

            curr_price = float(close_series.iloc[-1])
            sma_20 = float(close_series.rolling(window=20).mean().iloc[-1])
            price_5d_ago = float(close_series.iloc[-5])
            ret_5d_pct = round(((curr_price - price_5d_ago) / price_5d_ago) * 100, 2)

            # Technical Filter: Price > 20 SMA and positive 5-day momentum
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
        existing_df = pd.read_csv(AI_REC_CSV)
        # Remove prior runs for the same week_id + risk_profile to prevent duplicate clutter
        existing_df = existing_df[~((existing_df["week_id"] == week_id) & (existing_df["risk_profile"] == selected_risk))]
        updated_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
        updated_df.to_csv(AI_REC_CSV, index=False)
        return len(new_rows)
    return 0

# ==========================================
# 6. HUMAN DECISION LOGGING
# ==========================================
def log_human_decision(row: pd.Series, decision: str):
    dec_df = pd.read_csv(HUMAN_DEC_CSV)
    # Remove existing decision for same week + ticker + risk_profile if user changes mind
    mask = (
        (dec_df["week_id"] == row["week_id"]) &
        (dec_df["ticker"] == row["ticker"]) &
        (dec_df["risk_profile"] == row["risk_profile"])
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
    updated.to_csv(HUMAN_DEC_CSV, index=False)

# ==========================================
# 7. WORKFLOW 2: FRIDAY PERFORMANCE EVALUATOR
# ==========================================
def evaluate_single_trade(ticker: str, buy_price: float, stop_loss: float, target: float, sim_mode: bool = False):
    """Fetches last 5 trading days of high/low/close to resolve weekly trade outcome."""
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

        # Conservative resolution: check Stop-Loss first, then Target, else Friday Close
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
    ai_df = pd.read_csv(AI_REC_CSV)
    dec_df = pd.read_csv(HUMAN_DEC_CSV)

    week_ai = ai_df[(ai_df["week_id"] == week_id) & (ai_df["risk_profile"] == risk_profile)].copy()
    if week_ai.empty:
        return None, None

    week_dec = dec_df[(dec_df["week_id"] == week_id) & (dec_df["risk_profile"] == risk_profile)]
    yes_tickers = set(week_dec[week_dec["human_decision"] == "YES"]["ticker"].tolist())

    trade_audit_rows = []
    for _, row in week_ai.iterrows():
        w_low, w_high, f_close, ex_price, ex_reason, net_ret = evaluate_single_trade(
            row["ticker"], row["buy_price"], row["stop_loss_price"], row["target_price"]
        )
        trade_audit_rows.append({
            "Ticker": row["ticker"],
            "Company": row["company_name"],
            "Passed_AI_Filter": bool(row["passed_ai_filter"]),
            "Human_Approved_YES": row["ticker"] in yes_tickers,
            "Buy_Price": row["buy_price"],
            "Stop_Loss": row["stop_loss_price"],
            "Target": row["target_price"],
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

    # Cohort 1: Without AI (All stocks that passed Sunday technicals, ignoring AI veto)
    c_no_ai = summarize_cohort(audit_df, "1. Without AI (Pure Technical Baseline)")
    # Cohort 2: With AI (Only stocks where passed_ai_filter == True)
    c_with_ai = summarize_cohort(audit_df[audit_df["Passed_AI_Filter"] == True], "2. With AI (Model Recommendations)")
    # Cohort 3: Human + AI (Only stocks where user clicked YES)
    c_human = summarize_cohort(audit_df[audit_df["Human_Approved_YES"] == True], "3. Human Confirmed (YES Triggers)")

    summary_df = pd.DataFrame([c_no_ai, c_with_ai, c_human])
    baseline_ret = c_no_ai["Avg_Return_Pct"]
    summary_df["Alpha_vs_No_AI_Pct"] = round(summary_df["Avg_Return_Pct"] - baseline_ret, 2)

    # Save to multi-tab Excel file (1 tab per week_id + risk_profile)
    sheet_name = f"{week_id}_{risk_profile.replace(' ', '')}"[:31]
    mode = "a" if os.path.exists(PERF_XLSX) else "w"
    writer_kwargs = {"path": PERF_XLSX, "engine": "openpyxl", "mode": mode}
    if mode == "a":
        writer_kwargs["if_sheet_exists"] = "replace"

    with pd.ExcelWriter(**writer_kwargs) as writer:
        summary_df.to_excel(writer, sheet_name=sheet_name, startrow=1, index=False)
        audit_df.to_excel(writer, sheet_name=sheet_name, startrow=len(summary_df) + 5, index=False)

    return summary_df, audit_df

# ==========================================
# 8. STREAMLIT UI LAYOUT
# ==========================================
st.title("📈 SwingCopilot AI — Retail Equity Decision & Risk Engine")
st.caption("Human-in-the-Loop Weekly NSE Swing Trading System | Powered by Gemini 2.5 Flash")

with st.sidebar:
    st.header("⚙️ System Configuration")
    gemini_api_key = st.text_input(
        "Gemini API Key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="Get a free key from aistudio.google.com"
    )
    active_week = st.text_input("Active Week ID", value=get_current_week_id())

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
    "🗂️ 3. Raw CSV & Excel Data Logs"
])

# ------------------------------------------
# TAB 1: SUNDAY WORKFLOW & YES/NO CARDS
# ------------------------------------------
with tab1:
    col_head, col_btn = st.columns([3, 1])
    with col_head:
        st.subheader(f"Sunday Pre-Market Screening ({selected_risk} Risk)")
        st.write("Screens parent universe for technical momentum, audits weekend news via Gemini, and generates trade cards for human confirmation.")
    with col_btn:
        if st.button("🚀 Run Sunday Scan Now", type="primary", use_container_width=True):
            if not gemini_api_key:
                st.error("Please enter your Gemini API Key in the sidebar first.")
            else:
                count = run_sunday_workflow(gemini_api_key, selected_risk, active_week)
                st.success(f"Scanned universe! Logged {count} technical candidates to `{AI_REC_CSV}`.")

    st.divider()

    ai_df = pd.read_csv(AI_REC_CSV)
    dec_df = pd.read_csv(HUMAN_DEC_CSV)

    current_candidates = ai_df[
        (ai_df["week_id"] == active_week) &
        (ai_df["risk_profile"] == selected_risk)
    ]

    if current_candidates.empty:
        st.warning(f"No recommendations generated yet for **{active_week}** under **{selected_risk}** risk. Click **Run Sunday Scan Now** above.")
    else:
        recommended_df = current_candidates[current_candidates["passed_ai_filter"] == True]
        vetoed_df = current_candidates[current_candidates["passed_ai_filter"] == False]

        st.markdown(f"### ✅ AI-Approved Trade Thesis Cards ({len(recommended_df)} Picks)")
        if recommended_df.empty:
            st.info("All technical candidates were vetoed by the Gemini Risk Filter for this risk tier.")

        for _, row in recommended_df.iterrows():
            # Check if human already decided
            existing_dec = dec_df[
                (dec_df["week_id"] == active_week) &
                (dec_df["ticker"] == row["ticker"]) &
                (dec_df["risk_profile"] == selected_risk)
            ]
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
                    st.progress(max(0.0, min(1.0, ( float(row['ai_sentiment_score']) + 1 ) / 2)), text=f"AI Sentiment Score: {row['ai_sentiment_score']}")
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

        with st.expander(f"🛡️ Counterfactual Audit: {len(vetoed_df)} Stocks Vetoed by AI Risk Filter (Tracked for Friday 'Without AI' Comparison)"):
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
    st.write("Compares **1. Pure Technical Baseline (Without AI)** vs. **2. AI Recommendations** vs. **3. Human Confirmed Trades (`YES`)** and writes to a dedicated weekly tab in `weekly_performance.xlsx`.")

    if st.button("📈 Run Friday Evaluation & Update Excel Sheet", type="primary"):
        summary_df, audit_df = run_friday_evaluation(active_week, selected_risk)
        if summary_df is None:
            st.error(f"No Sunday data found for {active_week} ({selected_risk}). Run the Sunday scan first.")
        else:
            st.success(f"Saved evaluation to `{PERF_XLSX}` under tab `{active_week}_{selected_risk.replace(' ', '')}`!")

    if os.path.exists(PERF_XLSX):
        excel_file = pd.ExcelFile(PERF_XLSX)
        selected_tab = st.selectbox("Select Historical Week Tab from Excel Workbook:", options=excel_file.sheet_names[::-1])

        if selected_tab:
            raw_sheet = pd.read_excel(PERF_XLSX, sheet_name=selected_tab)
            st.markdown(f"#### 🏆 Cohort Performance Summary (`{selected_tab}`)")
            # Extract top summary rows (first 3 rows after header)
            summary_view = pd.read_excel(PERF_XLSX, sheet_name=selected_tab, skiprows=1, nrows=3)
            st.dataframe(summary_view, use_container_width=True)

            # KPI Metrics Row
            if len(summary_view) == 3:
                k1, k2, k3 = st.columns(3)
                no_ai_ret = summary_view.iloc[0]["Avg_Return_Pct"]
                ai_ret = summary_view.iloc[1]["Avg_Return_Pct"]
                hum_ret = summary_view.iloc[2]["Avg_Return_Pct"]

                k1.metric("1. Without AI (Technical Baseline)", f"{no_ai_ret}%", f"Win Rate: {summary_view.iloc[0]['Win_Rate_Pct']}%")
                k2.metric("2. With AI (Gemini Filtered)", f"{ai_ret}%", f"{round(ai_ret - no_ai_ret, 2)}% Alpha vs No-AI")
                k3.metric("3. Human Approved (YES Picks)", f"{hum_ret}%", f"{round(hum_ret - ai_ret, 2)}% vs Raw AI")

            st.markdown("#### 🔍 Trade-by-Trade Resolution Audit")
            audit_view = pd.read_excel(PERF_XLSX, sheet_name=selected_tab, skiprows=7)
            st.dataframe(audit_view, use_container_width=True)

            with open(PERF_XLSX, "rb") as f:
                st.download_button("📥 Download Full Multi-Tab Excel Report (.xlsx)", f, file_name=PERF_XLSX)

# ------------------------------------------
# TAB 3: RAW CSV INSPECTOR
# ------------------------------------------
with tab3:
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("### 📄 Sheet 1: `ai_recommendations.csv`")
        df1 = pd.read_csv(AI_REC_CSV)
        st.dataframe(df1, use_container_width=True)
        st.download_button("Download AI Recommendations CSV", df1.to_csv(index=False), file_name=AI_REC_CSV)

    with c_right:
        st.markdown("### 👤 Sheet 2: `human_decisions.csv`")
        df2 = pd.read_csv(HUMAN_DEC_CSV)
        st.dataframe(df2, use_container_width=True)
        st.download_button("Download Human Decisions CSV", df2.to_csv(index=False), file_name=HUMAN_DEC_CSV)