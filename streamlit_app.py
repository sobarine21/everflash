"""
app.py - Real-Time Investment Compliance Monitoring System (single-file, prod-ready)

Usage:
    1. pip install -r requirements.txt
    2. Set environment variables (recommended) or use .streamlit/secrets.toml:
         KITE_API_KEY, KITE_API_SECRET, KITE_REDIRECT_URI (optional)
    3. streamlit run app.py

Notes:
 - This app expects portfolio uploads with at least these logical columns:
     Symbol (or Tradingsymbol), Quantity (or Units), Sector (optional but recommended), AssetClass (optional), Issuer (optional)
 - SID parsing is heuristic: for PDFs it extracts numeric percentages and labelled sector/asset-class tables.
 - The app batches LTP requests to Kite (KiteConnect.ltp accepts list) and caches results (1 minute).
 - If you don't want real Kite access, leave API keys blank and the app will still allow file uploads & offline compliance checks.

Author: ChatGPT (GPT-5 Thinking mini)
"""

import os
import io
import re
import time
import math
import json
import logging
import tempfile
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st

from datetime import datetime, timedelta
from functools import lru_cache

# PDF parsing
from PyPDF2 import PdfReader

# Kite
from kiteconnect import KiteConnect, exceptions as kite_exceptions

# Utils for export
import base64

# ---------------------------
# Configuration & Constants
# ---------------------------
APP_TITLE = "Invsion — Real-Time Investment Compliance Monitor"
LOG_LEVEL = logging.INFO

# Default tolerances
DEFAULT_TOLERANCE_PERCENT = 10.0  # ±10% tolerance by default for sector limits

# Configure logging
logger = logging.getLogger("compliance_app")
logger.setLevel(LOG_LEVEL)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(ch)

# ---------------
# Helper utils
# ---------------
def human(x):
    try:
        return f"{x:,.2f}"
    except Exception:
        return str(x)

def read_portfolio(file_buffer) -> pd.DataFrame:
    """Attempt to read a portfolio file: excel or csv. Normalize column names."""
    try:
        if hasattr(file_buffer, "name") and file_buffer.name.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(file_buffer, engine="openpyxl")
        else:
            file_buffer.seek(0)
            df = pd.read_csv(file_buffer)
    except Exception as e:
        logger.exception("Failed to read portfolio file")
        raise

    # Normalize columns (lowercase)
    df.columns = [c.strip() for c in df.columns]
    lower_cols = {c: c.lower() for c in df.columns}
    df.rename(columns=lower_cols, inplace=True)

    # Try to map common names
    col_map = {}
    for c in df.columns:
        if c in ("symbol", "tradingsymbol", "ticker"):
            col_map[c] = "symbol"
        if c in ("qty", "quantity", "units"):
            col_map[c] = "quantity"
        if c in ("sector", "industry"):
            col_map[c] = "sector"
        if c in ("assetclass", "asset_class", "asset type", "asset_type"):
            col_map[c] = "asset_class"
        if c in ("issuer", "issuer_name", "issue"):
            col_map[c] = "issuer"
        if c in ("cost", "cost_basis", "avg_cost", "avg_price"):
            col_map[c] = "cost_basis"
        if c in ("isin",):
            col_map[c] = "isin"
        if c in ("marketvalue", "market_value"):
            col_map[c] = "market_value"
    df.rename(columns=col_map, inplace=True)

    # Ensure required columns exist
    if "symbol" not in df.columns:
        raise ValueError("Portfolio file must contain a 'symbol' (or 'tradingsymbol') column.")
    if "quantity" not in df.columns and "market_value" not in df.columns:
        raise ValueError("Portfolio file must contain either 'quantity' (units) or 'market_value' column.")

    # If only market_value provided, set quantity as 0 (we'll work with market_value)
    if "market_value" not in df.columns:
        # quantity -> numeric
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    else:
        df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce").fillna(0)
        if "quantity" not in df.columns:
            df["quantity"] = 0

    # Fill optional columns
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    if "asset_class" not in df.columns:
        df["asset_class"] = "Unknown"
    if "issuer" not in df.columns:
        df["issuer"] = np.nan

    return df

def read_sid_excel(file_buffer) -> Dict:
    """Read SID/Excel that contains sector/asset class limits."""
    try:
        df = pd.read_excel(file_buffer, engine="openpyxl")
    except Exception:
        try:
            file_buffer.seek(0)
            df = pd.read_csv(file_buffer)
        except Exception as e:
            logger.exception("Failed reading SID excel")
            raise

    # Attempt to find tables with 'industry' / 'sector' and '%' columns
    sid = {"sector_allocations": {}, "assetclass_allocations": {}, "raw_tables": {}}
    df.columns = [str(c).strip() for c in df.columns]

    # common column heuristics
    lower = [c.lower() for c in df.columns]
    # find possible % column
    pct_cols = [c for c in df.columns if "%" in str(c) or "percent" in str(c).lower() or "to nav" in str(c).lower()]
    name_cols = [c for c in df.columns if any(k in c.lower() for k in ("sector", "industry", "issuer", "asset", "class"))]

    # If we find sector-like table rows
    if name_cols and pct_cols:
        for ncol in name_cols:
            for pcol in pct_cols:
                try:
                    for _, row in df[[ncol, pcol]].dropna().iterrows():
                        key = str(row[ncol]).strip()
                        val = float(str(row[pcol]).strip().replace("%", "").replace(",", ""))
                        if len(key) > 0 and val >= 0:
                            # naive guess: sector rows usually short keys
                            if len(key) < 60:
                                sid["sector_allocations"][key] = val
                except Exception:
                    continue

    # fallback: attempt to map any column with numeric fraction summing to ~100
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in numeric_cols:
        s = df[c].sum()
        if abs(s - 100) < 10 or (0.9 <= s <= 120):
            # might be percents
            # take non-null strings from first text column
            txt_cols = [col for col in df.columns if df[col].dtype == object]
            if txt_cols:
                keys = df[txt_cols[0]].astype(str).fillna("").tolist()
                vals = df[c].fillna(0).tolist()
                for k, v in zip(keys, vals):
                    if k.strip():
                        sid["sector_allocations"][k.strip()] = float(v)
                sid["raw_tables"][c] = "guessed_percent_table"

    # try to find asset class (Equity/Debt/Cash)
    for k in ["Equity", "Debt", "Government Securities", "Cash", "Cash Equivalents", "Others"]:
        for col in df.columns:
            matches = df[df[col].astype(str).str.contains(k, case=False, na=False)]
            if not matches.empty:
                # look in same row for numeric columns
                for idx, r in matches.iterrows():
                    for nc in numeric_cols:
                        v = float(df.loc[idx, nc]) if not pd.isna(df.loc[idx, nc]) else None
                        if v is not None:
                            sid["assetclass_allocations"][k] = v

    return sid

def parse_sid_pdf(file_buffer) -> Dict:
    """
    Heuristic PDF parser for SID/Factsheet PDFs. Extracts sector and asset-class %
    Returns a dict: { 'sector_allocations': {sector: pct}, 'assetclass_allocations': {...}, 'raw_text': str }
    """
    reader = PdfReader(file_buffer)
    text = []
    for p in reader.pages:
        try:
            t = p.extract_text()
        except Exception:
            t = ""
        if t:
            text.append(t)
    txt = "\n".join(text)
    sid = {"sector_allocations": {}, "assetclass_allocations": {}, "raw_text": txt}

    # Extract sector blocks by looking for lines like "SectorName  12.34"
    # We'll use regex to find "Name   number%" patterns
    pattern = re.compile(r"([A-Za-z &\-\/\.]{3,60})\s+([0-9]{1,2}(?:\.[0-9]+)?)\s*(?:%|$)", re.MULTILINE)
    for match in pattern.finditer(txt):
        name = match.group(1).strip()
        pct = float(match.group(2))
        # Filter out long unrelated lines
        if len(name) < 60 and pct >= 0:
            # common sector words to prioritize
            if any(k.lower() in name.lower() for k in ("services", "financial", "consumer", "technology", "energy", "power", "bank", "telecom", "health", "it", "infra", "realty", "construction", "metals", "chem")):
                sid["sector_allocations"][name] = pct
            else:
                # also add as sector candidate
                sid["sector_allocations"].setdefault(name, pct)

    # Extract asset-class percentages (Equity/Debt/Cash)
    asset_patterns = [
        (r"Equity\s+Shares\s*[:\-]?\s*([0-9]{1,2}(?:\.[0-9]+)?)\s*%", "Equity"),
        (r"Government Securities\s*[:\-]?\s*([0-9]{1,2}(?:\.[0-9]+)?)\s*%", "Government Securities"),
        (r"Cash(?:, Cash Equivalents and Others)?\s*[:\-]?\s*([0-9]{1,2}(?:\.[0-9]+)?)\s*%", "Cash"),
        (r"Debt\s*[:\-]?\s*([0-9]{1,2}(?:\.[0-9]+)?)\s*%", "Debt"),
    ]
    for pat, name in asset_patterns:
        m = re.search(pat, txt, flags=re.IGNORECASE)
        if m:
            sid["assetclass_allocations"][name] = float(m.group(1))

    # As a fallback, scan the first 1000 characters for "Portfolio Classification by Industry" table
    # (we already captured a lot with the generic pattern)
    return sid

# -------------------------
# Kite Helpers (cached)
# -------------------------
@st.cache_resource(ttl=600)
def init_kite_client(api_key: str) -> KiteConnect:
    """Return a KiteConnect instance (unauthenticated until set_access_token)."""
    return KiteConnect(api_key=api_key)

@st.cache_data(ttl=60)
def get_instruments(kite: KiteConnect, exchange: Optional[str] = "NSE") -> pd.DataFrame:
    """Load instruments list (cached)."""
    try:
        inst = kite.instruments(exchange=exchange) if exchange else kite.instruments()
    except Exception as e:
        logger.warning(f"Failed to load instruments: {e}")
        return pd.DataFrame()
    df = pd.DataFrame(inst)
    # normalize
    for c in ("instrument_token",):
        if c in df.columns:
            df[c] = df[c].astype("int64")
    return df

@st.cache_data(ttl=60)
def get_ltp_batch(kite: KiteConnect, symbols: List[str], exchange: str = "NSE") -> Dict[str, float]:
    """Batch request LTP for a list of symbols. Returns mapping symbol -> last_price (or NaN)."""
    out = {}
    if not symbols:
        return out
    try:
        # Build exchange-prefixed list
        keys = [f"{exchange.upper()}:{s.strip().upper()}" for s in symbols]
        # kite.ltp accepts list of tokens/keys
        data = kite.ltp(keys)
        for k in keys:
            val = data.get(k)
            if val and isinstance(val, dict):
                out[k.split(":", 1)[1]] = float(val.get("last_price", np.nan))
            else:
                out[k.split(":", 1)[1]] = np.nan
    except kite_exceptions.NetworkException as e:
        logger.warning(f"Kite network issue while fetching LTP: {e}")
        # Fallback: assign NaN for each
        for s in symbols:
            out[s] = np.nan
    except Exception as e:
        logger.exception("Error fetching LTP batch")
        for s in symbols:
            out[s] = np.nan
    return out

@st.cache_data(ttl=3600)
def get_historical_data(kite: KiteConnect, token: int, from_dt: datetime, to_dt: datetime, interval: str = "day") -> pd.DataFrame:
    """Fetch Kite historical data for a single instrument token."""
    try:
        # Kite expects date strings or datetime objects
        data = kite.historical_data(instrument_token=token, from_date=from_dt, to_date=to_dt, interval=interval)
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        return df
    except Exception as e:
        logger.exception("historical fetch failed")
        return pd.DataFrame()

# -------------------------
# Compliance engine
# -------------------------
def compute_market_values(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Market Value column exists. If live prices present and quantity > 0 use them."""
    df = df.copy()
    if "live_price" in df.columns and df["live_price"].notna().any():
        df["market_value"] = df["quantity"].fillna(0) * df["live_price"].fillna(0)
    # If market_value not set but cost_basis and quantity exist, use them
    if "market_value" not in df.columns or df["market_value"].isna().all():
        if "cost_basis" in df.columns:
            df["market_value"] = df["cost_basis"].fillna(0) * df["quantity"].fillna(0)
        else:
            df["market_value"] = 0.0
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce").fillna(0)
    return df

def compute_exposures(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    df = df.copy()
    total = df["market_value"].sum()
    if total == 0:
        # avoid division by zero
        total = 0.0
    # By sector
    sector = df.groupby("sector", dropna=False)["market_value"].sum().reset_index()
    sector["pct_of_portfolio"] = sector["market_value"] / (total if total else 1) * 100

    # By asset class
    asset = df.groupby("asset_class", dropna=False)["market_value"].sum().reset_index()
    asset["pct_of_portfolio"] = asset["market_value"] / (total if total else 1) * 100

    # By issuer
    issuer = df.groupby("issuer", dropna=False)["market_value"].sum().reset_index()
    issuer["pct_of_portfolio"] = issuer["market_value"] / (total if total else 1) * 100

    exposures = {"sector": sector, "asset": asset, "issuer": issuer}
    return exposures, total

def validate_against_sid(exposures: Dict[str, pd.DataFrame], sid_limits: Dict, tolerance_pct: float = DEFAULT_TOLERANCE_PERCENT) -> Dict:
    """
    Compare exposures vs sid limits. tolerance_pct is absolute percentage points tolerance.
    sid_limits expected format:
      { 'sector_allocations': { 'Banking': 20.0, ... },
        'assetclass_allocations': { 'Equity': 65.0, 'Debt': 30.0, ... },
        'issuer_limits': { 'HDFC Bank': 10.0, ... }  # optional
      }
    Returns dict of dataframes and violation lists.
    """
    results = {"sector": None, "asset": None, "issuer": None, "violations": []}

    # Sector check
    sector_df = exposures["sector"].copy()
    sector_df["limit_pct"] = sector_df["sector"].map(lambda s: sid_limits.get("sector_allocations", {}).get(s, np.nan))
    sector_df["limit_pct"] = sector_df["limit_pct"].astype(float, errors="ignore")
    sector_df["deviation_pct"] = sector_df["pct_of_portfolio"] - sector_df["limit_pct"]
    # Mark compliant: if limit exists, absolute deviation <= tolerance OR if limit missing, compliant by default
    def sector_compliant(row):
        if math.isnan(row["limit_pct"]):
            # No limit in SID; can't validate -> mark as Unknown
            return "Unknown"
        return "Compliant" if abs(row["deviation_pct"]) <= tolerance_pct else "Violation"

    sector_df["status"] = sector_df.apply(sector_compliant, axis=1)
    results["sector"] = sector_df

    # Asset class check
    asset_df = exposures["asset"].copy()
    asset_df["limit_pct"] = asset_df["asset_class"].map(lambda s: sid_limits.get("assetclass_allocations", {}).get(s, np.nan))
    asset_df["limit_pct"] = asset_df["limit_pct"].astype(float, errors="ignore")
    asset_df["deviation_pct"] = asset_df["pct_of_portfolio"] - asset_df["limit_pct"]
    def asset_compliant(row):
        if math.isnan(row["limit_pct"]):
            return "Unknown"
        return "Compliant" if abs(row["deviation_pct"]) <= tolerance_pct else "Violation"
    asset_df["status"] = asset_df.apply(asset_compliant, axis=1)
    results["asset"] = asset_df

    # Issuer check - optional limits in sid_limits.get('issuer_limits')
    issuer_df = exposures["issuer"].copy()
    issuer_limits = sid_limits.get("issuer_limits", {})
    issuer_df["limit_pct"] = issuer_df["issuer"].map(lambda s: issuer_limits.get(s, np.nan))
    issuer_df["deviation_pct"] = issuer_df["pct_of_portfolio"] - issuer_df["limit_pct"]
    def issuer_compliant(row):
        if math.isnan(row["limit_pct"]):
            return "Unknown"
        return "Compliant" if row["pct_of_portfolio"] <= row["limit_pct"] + tolerance_pct else "Violation"
    issuer_df["status"] = issuer_df.apply(issuer_compliant, axis=1)
    results["issuer"] = issuer_df

    # Collate violations
    for df_name in ("sector", "asset", "issuer"):
        dfi = results[df_name]
        if dfi is not None:
            if "status" in dfi.columns:
                vio = dfi[dfi["status"] == "Violation"]
                for _, r in vio.iterrows():
                    results["violations"].append({
                        "type": df_name,
                        "name": r.get(dfi.columns[0]),
                        "pct": r.get("pct_of_portfolio"),
                        "limit_pct": r.get("limit_pct"),
                        "deviation_pct": r.get("deviation_pct")
                    })
    return results

# -------------------------
# Streamlit App UI
# -------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.markdown("Built for **real-time** compliance monitoring using Zerodha KiteConnect. Upload portfolio + SID/KIM and connect Kite for live pricing.")

# Sidebar: Kite credentials and login flow
st.sidebar.header("KiteConnect Credentials & Login")
KITE_API_KEY_ENV = os.environ.get("KITE_API_KEY", "")
KITE_API_SECRET_ENV = os.environ.get("KITE_API_SECRET", "")
KITE_REDIRECT_URI_ENV = os.environ.get("KITE_REDIRECT_URI", "")  # optional

api_key_input = st.sidebar.text_input("Kite API Key", value=KITE_API_KEY_ENV, type="password")
api_secret_input = st.sidebar.text_input("Kite API Secret", value=KITE_API_SECRET_ENV, type="password")
redirect_uri_input = st.sidebar.text_input("Kite Redirect URI (if using login flow)", value=KITE_REDIRECT_URI_ENV)

# instantiate unauthenticated client for login URL
kite_unauth = None
if api_key_input:
    try:
        kite_unauth = init_kite_client(api_key_input)
    except Exception as e:
        logger.exception("Failed creating Kite client")
        kite_unauth = None

if kite_unauth:
    try:
        login_url = kite_unauth.login_url()
    except Exception:
        login_url = None
else:
    login_url = None

st.sidebar.markdown("**Login options**")
col1, col2 = st.sidebar.columns([2, 1])
if login_url:
    if col1.button("Open Kite Login (Browser)"):
        st.sidebar.markdown(f"[Open Kite Login]({login_url})")
st.sidebar.write("Or paste a request_token below after login (if you use the manual flow).")
request_token_input = st.sidebar.text_input("Request token (from redirect)", value="")

# Exchange selection
exchange = st.sidebar.selectbox("Exchange for LTP", ["NSE", "BSE", "NFO"], index=0)

# Kite authenticated instance holder
kite_client = None
if "kite_access_token" not in st.session_state:
    st.session_state["kite_access_token"] = None
if "kite_login_response" not in st.session_state:
    st.session_state["kite_login_response"] = None

# If user pasted request token, exchange for access token
if request_token_input and api_key_input and api_secret_input:
    if st.sidebar.button("Exchange request token for access token"):
        try:
            kite_tmp = KiteConnect(api_key=api_key_input)
            session_data = kite_tmp.generate_session(request_token_input.strip(), api_secret_input.strip())
            access_token = session_data.get("access_token")
            kite_tmp.set_access_token(access_token)
            st.session_state["kite_access_token"] = access_token
            st.session_state["kite_login_response"] = session_data
            st.sidebar.success("Kite token obtained and cached in session.")
        except Exception as e:
            st.sidebar.error(f"Failed to exchange request token: {e}")

# If session has access token, create authenticated client
if st.session_state.get("kite_access_token") and api_key_input:
    try:
        kite_client = KiteConnect(api_key=api_key_input)
        kite_client.set_access_token(st.session_state["kite_access_token"])
    except Exception as e:
        logger.exception("Failed initializing authenticated kite client")
        kite_client = None

# File uploads
st.header("1) Upload Portfolio & SID / KIM")
col_file_1, col_file_2 = st.columns([2, 2])
portfolio_file = col_file_1.file_uploader("Upload Portfolio (Excel/CSV)", type=["xlsx", "xls", "csv"], key="portfolio_file")
sid_file = col_file_2.file_uploader("Upload SID / Factsheet / Monthly Portfolio (Excel/PDF/CSV)", type=["xlsx", "xls", "csv", "pdf"], key="sid_file")

# Tolerance controls
st.sidebar.header("Validation Controls")
tolerance_pct = st.sidebar.number_input("Absolute tolerance (percentage points)", min_value=0.0, max_value=100.0, value=DEFAULT_TOLERANCE_PERCENT, step=1.0)
min_total_value_warn = st.sidebar.number_input("Min portfolio total (₹) to run live checks", min_value=0.0, value=100.0, step=100.0)

# Upload handling and parsing
portfolio_df = None
sid_limits = {"sector_allocations": {}, "assetclass_allocations": {}, "issuer_limits": {}}
sid_raw_text = ""

if portfolio_file:
    try:
        portfolio_df = read_portfolio(portfolio_file)
        st.success(f"Portfolio loaded — {len(portfolio_df)} rows.")
        st.dataframe(portfolio_df.head(50))
    except Exception as e:
        st.error(f"Failed to load portfolio: {e}")
        st.stop()
else:
    st.info("Please upload a portfolio file to proceed.")
    st.stop()

if sid_file:
    try:
        if sid_file.name.lower().endswith(".pdf"):
            sid_parsed = parse_sid_pdf(sid_file)
            sid_limits["sector_allocations"] = sid_parsed.get("sector_allocations", {})
            sid_limits["assetclass_allocations"] = sid_parsed.get("assetclass_allocations", {})
            sid_raw_text = sid_parsed.get("raw_text", "")
        else:
            # Excel / CSV parsing
            sid_parsed = read_sid_excel(sid_file)
            sid_limits["sector_allocations"] = sid_parsed.get("sector_allocations", {})
            sid_limits["assetclass_allocations"] = sid_parsed.get("assetclass_allocations", {})
            sid_raw_text = json.dumps(sid_parsed.get("raw_tables", {}), default=str)
        st.success("SID / KIM loaded and parsed.")
        st.subheader("Extracted SID Limits (guesses)")
        st.write("Sector allocations (top items):")
        st.dataframe(pd.DataFrame(list(sid_limits["sector_allocations"].items()), columns=["Sector", "% to NAV"]).head(50))
        st.write("Asset class allocations (guesses):")
        st.dataframe(pd.DataFrame(list(sid_limits["assetclass_allocations"].items()), columns=["AssetClass", "%"]).head(10))
    except Exception as e:
        st.error(f"Failed to parse SID: {e}")
else:
    st.info("Please upload a SID / Factsheet file to enable compliance rules (PDF/Excel).")

# Small UI: allow user adjustments to SID limits (manual override)
st.header("2) Review / Edit Extracted Compliance Limits")
with st.form("edit_sid_limits"):
    st.markdown("If auto-extraction missed anything, edit or add limits below. Values are percentages of NAV.")
    # Show sector table editable
    sec_df = pd.DataFrame(list(sid_limits["sector_allocations"].items()), columns=["Sector", "LimitPct"])
    sec_df = sec_df.sort_values(by="LimitPct", ascending=False).reset_index(drop=True) if not sec_df.empty else pd.DataFrame(columns=["Sector", "LimitPct"])
    # For convenience, allow user to paste a small CSV of sector,pct
    sid_text = st.text_area("Paste additional sector limits (sector,pct per line) — optional", height=100, placeholder="Financial Services, 30\nInformation Technology, 10")
    if sid_text:
        for line in sid_text.splitlines():
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) >= 2:
                try:
                    sid_limits["sector_allocations"][parts[0]] = float(parts[1])
                except Exception:
                    continue
    # manual edit for top 20
    if not sec_df.empty:
        edited = st.experimental_data_editor(sec_df, num_rows="dynamic")
        # reconstruct
        sid_limits["sector_allocations"] = {r["Sector"]: float(r["LimitPct"]) for _, r in edited.iterrows() if str(r["Sector"]).strip()}
    # assetclasses
    ac_df = pd.DataFrame(list(sid_limits["assetclass_allocations"].items()), columns=["AssetClass", "LimitPct"])
    ac_df = ac_df if not ac_df.empty else pd.DataFrame(columns=["AssetClass", "LimitPct"])
    if not ac_df.empty:
        edited_ac = st.experimental_data_editor(ac_df, num_rows="dynamic")
        sid_limits["assetclass_allocations"] = {r["AssetClass"]: float(r["LimitPct"]) for _, r in edited_ac.iterrows() if str(r["AssetClass"]).strip()}
    submit_btn = st.form_submit_button("Save SID Limits")

# -------------------------
# Live price fetch and compute
# -------------------------
st.header("3) Fetch Live Quotes & Compute Exposures")

# Prepare symbol list for fetching
symbols = portfolio_df["symbol"].astype(str).str.replace(" ", "").str.upper().unique().tolist()
st.write(f"Preparing to fetch LTP for {len(symbols)} unique symbols.")

# If Kite available & portfolio size above threshold, fetch live prices
if kite_client and portfolio_df is not None:
    # fetch in batches of 50 (Kite supports many but be conservative)
    batch_size = 50
    all_prices = {}
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        try:
            ltp_map = get_ltp_batch(kite_client, batch, exchange=exchange)
            all_prices.update(ltp_map)
        except Exception as e:
            logger.exception("LTP batch failed")
            for s in batch:
                all_prices[s] = np.nan
        # small sleep to be nice to API
        time.sleep(0.1)
    # map prices back to portfolio_df
    portfolio_df["symbol_upper"] = portfolio_df["symbol"].astype(str).str.upper().str.replace(" ", "")
    portfolio_df["live_price"] = portfolio_df["symbol_upper"].map(lambda s: all_prices.get(s, np.nan))
    # compute market values
    portfolio_df = compute_market_values(portfolio_df)
    total_val = portfolio_df["market_value"].sum()
    st.success(f"Live prices fetched — total portfolio market value ≈ ₹{human(total_val)}")
    st.dataframe(portfolio_df[["symbol", "quantity", "live_price", "market_value", "sector", "asset_class", "issuer"]].head(200))
else:
    # If kite not configured, try to use market_value column only
    portfolio_df = compute_market_values(portfolio_df)
    total_val = portfolio_df["market_value"].sum()
    st.warning("Kite client not configured or not authenticated — running offline compliance checks using provided market_value/cost_basis.")
    st.info(f"Total portfolio market value (provided data) ≈ ₹{human(total_val)}")
    st.dataframe(portfolio_df[["symbol", "quantity", "market_value", "sector", "asset_class", "issuer"]].head(200))

if total_val < min_total_value_warn:
    st.warning(f"Total portfolio value ₹{human(total_val)} is below the min threshold ₹{human(min_total_value_warn)}. Live checks may be noisy or unnecessary.")

# Compute exposures
exposures, total_val = compute_exposures(portfolio_df)
st.header("4) Exposures Overview")
colA, colB, colC = st.columns(3)
colA.metric("Total Market Value", f"₹{human(total_val)}")
colB.metric("Distinct Holdings", len(portfolio_df))
colC.metric("Distinct Sectors", exposures["sector"].shape[0] if exposures["sector"] is not None else 0)

# Visual (table) exposures
st.subheader("Sector Exposures")
st.dataframe(exposures["sector"].sort_values("pct_of_portfolio", ascending=False).reset_index(drop=True).style.format({"pct_of_portfolio": "{:.2f}"}))

st.subheader("Asset Class Exposures")
st.dataframe(exposures["asset"].sort_values("pct_of_portfolio", ascending=False).reset_index(drop=True).style.format({"pct_of_portfolio": "{:.2f}"}))

st.subheader("Top Issuer Exposures")
st.dataframe(exposures["issuer"].sort_values("pct_of_portfolio", ascending=False).reset_index(drop=True).head(50).style.format({"pct_of_portfolio": "{:.2f}"}))

# -------------------------
# Validation
# -------------------------
st.header("5) Validate Portfolio vs SID Limits")
validation = validate_against_sid(exposures, sid_limits, tolerance_pct=float(tolerance_pct))

st.subheader("Sector Validation")
if validation["sector"] is not None:
    sec_df = validation["sector"].copy()
    sec_df_display = sec_df[["sector", "market_value", "pct_of_portfolio", "limit_pct", "deviation_pct", "status"]].rename(columns={"sector": "Sector", "market_value": "MarketValue", "pct_of_portfolio": "% of Portfolio", "limit_pct": "Limit (%)", "deviation_pct": "Deviation (%)", "status": "Status"})
    st.dataframe(sec_df_display.sort_values("% of Portfolio", ascending=False).reset_index(drop=True).style.format({"MarketValue": "{:,.2f}", "% of Portfolio": "{:.2f}", "Limit (%)": "{:.2f}", "Deviation (%)": "{:.2f}"}))
else:
    st.info("No sector exposure data available.")

st.subheader("Asset Class Validation")
if validation["asset"] is not None:
    asset_display = validation["asset"][["asset_class", "market_value", "pct_of_portfolio", "limit_pct", "deviation_pct", "status"]].rename(columns={"asset_class": "AssetClass", "market_value": "MarketValue", "pct_of_portfolio": "% of Portfolio", "limit_pct": "Limit (%)", "deviation_pct": "Deviation (%)", "status": "Status"})
    st.dataframe(asset_display.sort_values("% of Portfolio", ascending=False).reset_index(drop=True).style.format({"MarketValue": "{:,.2f}", "% of Portfolio": "{:.2f}", "Limit (%)": "{:.2f}", "Deviation (%)": "{:.2f}"}))
else:
    st.info("No asset-class exposure data available.")

st.subheader("Issuer Validation")
if validation["issuer"] is not None:
    issuer_display = validation["issuer"][["issuer", "market_value", "pct_of_portfolio", "limit_pct", "status"]].rename(columns={"issuer": "Issuer", "market_value": "MarketValue", "pct_of_portfolio": "% of Portfolio", "limit_pct": "Limit (%)", "status": "Status"})
    st.dataframe(issuer_display.sort_values("% of Portfolio", ascending=False).reset_index(drop=True).style.format({"MarketValue": "{:,.2f}", "% of Portfolio": "{:.2f}", "Limit (%)": "{:.2f}"}))
else:
    st.info("No issuer exposure data available.")

# Summary of violations
st.header("6) Violations & Actions")
violations = validation.get("violations", [])
if violations:
    st.error(f"{len(violations)} violation(s) detected:")
    vf = pd.DataFrame(violations)
    st.dataframe(vf)
    st.markdown("**Suggested actions (examples)**:")
    st.markdown("- Reduce positions in violating sectors/issuers to meet SID limits.\n- Rebalance asset class allocation to match fund mandate.\n- Contact portfolio manager for re-allocation plan.")
else:
    st.success("No violations detected (within current tolerance).")

# -------------------------
# Exports (CSV / Excel)
# -------------------------
st.header("7) Export Reports / Download")

def to_excel_bytes(df_dict: Dict[str, pd.DataFrame]) -> bytes:
    """Write multiple DataFrames to an in-memory Excel file (multiple sheets)."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in df_dict.items():
            # limit sheet name length
            safe_name = sheet_name[:31]
            try:
                df.to_excel(writer, sheet_name=safe_name, index=False)
            except Exception:
                # fall back to converting objects to str
                dff = df.copy()
                for c in dff.columns:
                    if dff[c].dtype == object:
                        dff[c] = dff[c].astype(str)
                dff.to_excel(writer, sheet_name=safe_name, index=False)
        writer.save()
    return buffer.getvalue()

export_dict = {
    "portfolio": portfolio_df,
    "exposure_sector": exposures["sector"],
    "exposure_asset": exposures["asset"],
    "exposure_issuer": exposures["issuer"],
    "validation_sector": validation["sector"],
    "validation_asset": validation["asset"],
    "validation_issuer": validation["issuer"],
}

excel_bytes = to_excel_bytes(export_dict)
b64 = base64.b64encode(excel_bytes).decode()
href = f'<a href="data:application/octet-stream;base64,{b64}" download="compliance_report_{datetime.now().strftime("%Y%m%d_%H%M")}.xlsx">📥 Download Excel report</a>'
st.markdown(href, unsafe_allow_html=True)

csv_buffer = io.StringIO()
portfolio_df.to_csv(csv_buffer, index=False)
csv_b64 = base64.b64encode(csv_buffer.getvalue().encode()).decode()
st.markdown(f'<a href="data:file/csv;base64,{csv_b64}" download="portfolio_export.csv">📥 Download portfolio CSV</a>', unsafe_allow_html=True)

# -------------------------
# Advanced: Quick Historical Chart (optional)
# -------------------------
st.header("8) Optional: Historical Data Viewer (single symbol)")
with st.expander("Historical price fetch (Kite)"):
    symbol_for_hist = st.text_input("Symbol for history (e.g. RELIANCE)", value=symbols[0] if symbols else "")
    from_date = st.date_input("From date", value=(datetime.now().date() - timedelta(days=365)))
    to_date = st.date_input("To date", value=datetime.now().date())
    interval = st.selectbox("Interval", ["day", "5minute", "minute", "30minute", "week", "month"], index=0)
    if st.button("Fetch historical data"):
        if kite_client:
            # need instrument token: fetch instruments
            try:
                inst_df = get_instruments(kite_client, exchange=exchange)
                token = None
                if not inst_df.empty:
                    hits = inst_df[(inst_df["tradingsymbol"].str.upper() == symbol_for_hist.strip().upper()) & (inst_df["exchange"].str.upper() == exchange.upper())]
                    if not hits.empty:
                        token = int(hits.iloc[0]["instrument_token"])
                if not token:
                    st.error("Instrument token not found for symbol on selected exchange.")
                else:
                    df_hist = get_historical_data(kite_client, token, datetime.combine(from_date, datetime.min.time()), datetime.combine(to_date, datetime.min.time()), interval)
                    if not df_hist.empty:
                        st.line_chart(df_hist["close"])
                        st.dataframe(df_hist.tail(200))
                    else:
                        st.info("No historical data returned.")
            except Exception as e:
                st.error(f"Error fetching historical: {e}")
        else:
            st.error("Kite client not authenticated/available for historical fetch.")

# -------------------------
# Footer / deployment notes
# -------------------------
st.markdown("---")
st.markdown("#### Deployment & Production tips")
st.markdown("""
- Use environment variables for Kite credentials in production. Avoid checking secrets into VCS.
- For multi-user deployment, put Kite tokens in a DB keyed to user_id (e.g. Supabase/Postgres) and restrict access.
- Add rate-limiting around Kite LTP calls for large portfolios. The app batches LTP calls but for thousands of symbols use a queue.
- Use Streamlit sharing/Streamlit Cloud, Docker, or a VPS. A sample `Dockerfile` and `requirements.txt` are below.
""")

st.markdown("**requirements.txt**")
st.code("""
streamlit
pandas
numpy
openpyxl
PyPDF2
kiteconnect
openpyxl
""", language="text")

st.markdown("**Sample Dockerfile**")
st.code("""
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONUNBUFFERED=1
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
""", language="dockerfile")

st.info("If you want, I can add: issuer-level auto-limits, email alerts on violations, scheduled checks, or Supabase integration for storing user tokens and reports. Tell me which next and I'll add it directly into this single-file app.")
